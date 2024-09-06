"""Unitless/Unsortable linear time-series SCM and data generation"""

# Author: Dr. Rebecca Jean Herman <rebecca.herman@tu-dresden.de>

import numpy as np
from sympy import Matrix, Symbol, symbols, re, im, Abs, Float
from sympy.solvers import solve, nsolve

import sys
import io
import signal
from contextlib import redirect_stdout, contextmanager
import warnings

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

class UnstableError(Exception): pass
class ConvergenceError(Exception): pass
class GenerationError(Exception): pass
class TimeoutException(Exception): pass
@contextmanager
def time_lim(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class Graph(object):
    r"""Data-generation object, and methods for creating and manipulating them.
    Always contains a causal graph, and becomes and SCM after calling GEN_COEFFICIENTS.
    May also hold generated data after calling GEN_DATA.
    
    Parameters
    _______________________________
    N : int
        Number of random variables
    tau_max : int
        Maximum delay between a cause and its effect
    A : (N x N x tau_max+1) np.array
        A[i,j,v] is the effect of X_i(t-v) on X_j(t).
        When entries are 0 and 1, this is an adjacency matrix.
        Otherwise, it is a causal coeficients matrix.
    s2 : np.array of length N
        Noise variances (if set or calculated)
    topo_order : np.array of length N
        topological order of the variables
    components : list of 1-D np.arrays. 
        Variable indices are distributed among the arrays such that 
        each array is a component with feedback, and the arrays are 
        ordered according to a valid topological ordering on the summary graph.
    cov : (2*tau_max+1 x N x N) np.array
        Theoretical covariance matrix (if coeficients are generated).
        cov[i,j,v] is the covariance of X_i(t) and X_j(t+v), 
        where v can be negative.
    data : TimeSeries object
        data generated from this SCM (if generated)
    variables : iterable over variable indices
    lags : iterable over lags in the adjacency matrix
    shape = A.shape = (N x N x tau_max+1)
    style : SCM generation strategy

    Class Methods and Constants
    ___________________________
    AXIS_LABELS : dictionary
        names of the dimensions of the adjacency array and the dimension index.
    specified : shortcut for initializing graphs from a specified adjacency array
    gen_unitless_time_series : wrapper function for generating a large amount
                               of random data and associated ground-truth SCMs.
    ccopy : creates a copy of the current SCM
    select_vars : select a subset of the adjacency array showing direct effects
                  between the specified variables.
    """
    
    AXIS_LABELS = {'source': 0, 'sink': 1, 'time': 2, None:None}
    
    def __init__(self, N, tau_max, 
                 init_type='random', p=.5, p_auto=.8,
                 init=None, noise=None, labels=None, topo_order=None):
        """
        Optional Parameters
        ___________________
        init_type : string (default: 'random')
            Method for generating the adjacency matrix. Options include:
                'connected': a fully-connected acyclic time series DAG
                'random': randomly include edges from the connected graph 
                          such that the probability of a corresponding edge
                          in the summary graph is p (or p_auto for auto-dependence)
                'no_feedback': as with 'random', but the summary graph is also acyclic
                'disconnected': a graph with no edges
                'specified': A causal graph with adjacency matrix INIT
        p : float (Default: 0.5)
            Probability of a directed causal dependence between distinct variables 
            during random generation
        p_auto : float (Default: 0.8)
            Probability of dependence from a variable's past to its present 
            during random generation
        init : N x N x tau_max np.array (Default: None)
            Adjacency matrix for specified initialization.
            If the entries are not 1 and 0, 
            then non-zero values are interpreted as causal coefficients.
        noise : np.array of length N (Default: None)
            Noise variances for each random variable for specified initialization.
        labels : list of strings (deault: None)
            Names of the random variables during specified initialization.
        topo_order : np.array of length N (default: None)
            Topological order for specified generation.
        """
        types_ = ['random', 'connected', 'disconnected', 'specified', 'no_feedback']
        Graph._check_option('init_type', types_, init_type)

        #helper function definitions
        def adjust_p(p, tm):
            return 1 - (1-p)**(1/tm)
        def rand_edge(p):
            return np.random.choice([1,0], p=[p, 1-p])

        #user-specified initializations
        self.N = N
        self.variables = range(self.N)
        self.tau_max = tau_max
        self.lags = range(self.tau_max+1)
        if labels is not None:
            assert len(labels)==self.N, ("Length of labels {} "
                 "must match the number of variables {}").format(len(labels), self.N)
            self.labels = labels
        else:
            self.labels = ["$X_{}$".format(i) for i in self.variables]
        self.shape = tuple((self.N, self.N, len(self.lags)))

        #empty initializations for later replacement
        self.s2 = noise
        self.data = None
        self.cov = None
        self.topo_order = np.arange(self.N)
        self.style = None

        #initializion of the adjacency matrix and topological order
        if init_type=='specified':
            Graph._check_given('adjacency matrix', init)
            Graph._check_given('topological order', topo_order)
            assert init.shape==self.shape, ("initialization matrix shape {} "
                 "not consistent with expected shape {}").format(init.shape, self.shape)
            self.A = init
            self.topo_order = topo_order
        else:
            #Make a fully-connected graph
            self.A = np.ones(shape=self.shape)
            #remove contemporaneous cycles
            self.A[:,:,0] *= (np.tril(self.A[:,:,0])==0)
            if init_type=='disconnected':
                #make empty graph
                self.A = self.A*0
            elif init_type!='connected': #random or no_feedback
                Graph._check_given('edge probabilities', p)
                if p_auto is None:
                    p_auto=p
                p = adjust_p(p, self.tau_max+1)
                p_auto = adjust_p(p_auto, self.tau_max)
                #set contemporaneous adjacencies
                for i in range(self.N-1):
                    for j in range(i+1, self.N):
                        self[i,j,0] = rand_edge(p)
                #set lagged adjacencies
                for t in self.lags[1:]:
                    for i in self.variables:
                        for j in self.variables:
                            self[i,j,t] = rand_edge(p_auto) if i==j else rand_edge(p)
                if init_type=='no_feedback':
                    #removed lagged edges in reverse-topological order
                    self.A = self.triu()
            #randomize the order of appearance of the variables
            new_order = np.arange(self.N)
            np.random.shuffle(new_order)
            self.A = Graph.select_vars(self.A, new_order)
            self.topo_order = np.argsort(new_order)
        #lists feedback loops by summary-graph topological order.
        #Additionally updates topo_order to match this where possible.
        self.components = self.summary_order()

    #Public class methods
    @classmethod
    def specified(cls, init, topo_order, noise=None, labels=None):
        '''Helper function for initializing a graph from a specified adjacency matrix'''
        return cls(init.shape[0], init.shape[-1], 
                   init_type='specified', init=init, topo_order=topo_order,
                   labels=labels,noise=noise)
    @classmethod
    def ccopy(cls, G):
        '''Creates a new graph equivalent to the input graph G.'''
        return cls(G.N, G.tau_max, labels=G.labels, 
                   init_type='specified', init = G.A, topo_order=G.topo_order)

    @classmethod
    def gen_unitless_time_series(cls, N, tau_max, p=None, p_auto=None, T=1000, B=100, 
                                 time_limit=5, init_type='no_feedback'):
        r'''Method for generating data from many random SCMs.

        Parameters
        _________
        T : int, optional (Default: 1000)
            Number of observations in each generated time series
        B : int, optional (Default: 100)
            Number of SCMs (and associated data) to generate
        time_limit : int, optional (Default: 5)
            Maximum number of seconds to spend on each attempt at generating an SCM
        N, tau_max, p, p_auto, init_type: parameters for graph initialization

        Returns
        _______
        Gs : list of B Graph objects
        Ds : list of B TimeSeries objects
        text_trap: printed output from graph and data generation.
        
        '''
        no_converge = 0
        unstable = 0
        diverge = 0
        TO = 0
    
        Gs = []
        Ds = []

        text_trap = io.StringIO()
        while len(Gs)<B:
            all_errors = no_converge+unstable+diverge+TO
            Graph._progress_message("{:.0%} completed ({} discarded)".format(
                                    len(Ds)/B, all_errors))
            try:
                with time_lim(time_limit):
                    with redirect_stdout(text_trap):
                        G = Graph(N, tau_max, init_type=init_type)
                        G = G.gen_coefficients(convergence_attempts=2)
                        D = G.gen_data()
            except ConvergenceError:
                no_converge += 1
                continue
            except UnstableError:
                unstable += 1
                continue
            except GenerationError:
                diverge += 1
                continue
            except TimeoutException:
                TO += 1
                continue
            Gs+=[G]
            Ds+=[D]
    
        sys.stdout.write('\r')
        if all_errors>0:
            print(("Discarded {} system{}: "
                   "{} that did not converge, "
                   "{} that were analytically unstable, "
                   "{} that computationally diverged, "
                   "and {} that timed out.").format(
                all_errors, 's' if all_errors>1 else '', 
                no_converge, unstable, diverge, TO))
    
        return Gs, Ds, text_trap
    @classmethod
    def select_vars(cls, A, V):
        r'''Direct effects between a subset of the variables.
        
        Parameters
        __________
        A : an adjacency matrix
        V : np.array of variable indices

        Returns an array of shape (len(V), len(V)) if A is 2-D, 
        or an array of shape (len(V), len(V), A.shape[-1]) if A is 3-D
        '''
        if len(A.shape)==3:
            return A[V,:,:][:,V,:]
        elif len(A.shape)==2:
            return A[V,:][:,V]
        else:
            raise ValueError(("A must be 2- or 3-dimensional, "
                              "not {}-dimensional").format(len(A.shape)))

    #Private class methods
    @classmethod
    def _sim_graph(cls, G, A):
        r'''Creates a new graph 
        with the same number of variables and topological order as the input graph G, 
        but with a modified agencency matrix A.'''
        return cls(G.N, A.shape[-1]-1, labels=G.labels, 
                   init_type='specified', init = A, topo_order=G.topo_order)
    @classmethod
    def _remove_diagonal(cls, M):
        '''removes the diagonal from a 2D array M'''
        return M & ~np.diag(np.diag(M))
    @classmethod
    def _check_option(cls, name, options, chosen):
        '''checks that a valid keyword is chosen'''
        if chosen not in options:
            raise ValueError("Valid choices for {} include {}".format(name, options))
    @classmethod
    def _check_given(cls, name, value):
        '''Checks that an optional input is specified'''
        if value is None:
            raise ValueError("Please specify {}".format(name))
    @classmethod
    def _progress_message(cls, msg):
        '''Progress update that modifies in place'''
        sys.stdout.write('\r')
        sys.stdout.write(msg)
        sys.stdout.flush()
        
    def gen_coefficients(self, convergence_attempts=10, style='standardized'):
        r'''Generate a random SCM from a causal graph.

        Parameters
        __________
        convergence_attempts : int, optional (Default: 10)
            Number of attempts to define a random SCM from this adjacency matrix.
        style : string, optional (Default: 'standardized')
            Data generation technique; options are 'standardized' and 'unit-variance-noise'.

        Style = 'standardized':
        Given the adjacency matrix self.A, this method:
        1. randomly draws parameters that represent the relative strength of 
           different parent processes and noise in the absence of covariance
           among the parents. 
        2. uses sympy's numerical solver to scale these parameters so that the 
           total variance of each variable should be 1. 
        3. checks the stability of the potential solution. 
        If the numerical solver fails to converge, this raises a ConvergenceError. 
        If the potential solution is unstable, this raises an UnstableError. 
        The user may specify how many times the method should attempt this process.
        Modifies self.A, and sets self.s2, self.cov, and self.style.

        Style = 'unit-variance-noise':
        This method is included for comparison. As is typical of previous data-
        generation techniques, coefficients are drawn from a uniform distribution 
        over [-2,-0.2]U[0.2,2], and all noise variances are set to 1. The method
        discards solutions that are unstable, and tries at most 
        convergence_attempts times to find a stable solution.
        
        '''
        style_options = ['standardized', 'unit-variance-noise']
        Graph._check_option('style', style_options, style)
        self.style = style
        CO = self.components
        Bp_init = self.summary().astype(float)
        P = self.get_num_parents()
        
        stable = False
        discarded_u = 0
        discarded_c = 0

        def R_symbol(i,j,l):
            return Symbol("R{}.{}({})".format(i,j,l))
        def check_vars(now_vars, s_exp):
            missing_vars = [v for v in Matrix(s_exp).free_symbols if not Matrix(now_vars).has(v)]
            if len(missing_vars)>0:
                print("MISSING VARIABLES! {}".format(missing_vars))
            return len(missing_vars)==0

        A_init = (self.A != 0).astype(float)
        while not stable:
            self.A = A_init.copy()
            if (discarded_c >= convergence_attempts 
                or discarded_u >= convergence_attempts):
                if discarded_u>discarded_c:
                    raise UnstableError(("No stable solution found for above graph {}; "
                                         "tried {}x").format(id(self), discarded_u))
                raise ConvergenceError(("No solution found for above graph {}; "
                                        "tried {}x").format(id(self), discarded_c))

            Graph._progress_message("Attempt {}/{}".format(discarded_c+discarded_u+1, 
                                                           convergence_attempts))

            if self.style=='standardized':
                #initial draws
                self.A*=np.random.normal(size=self.shape)
                Bp=Bp_init*np.random.normal(size=Bp_init.shape)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = np.random.uniform(size=(self.N,))**(1/P)
    
                #first update
                B = self.sum(axis='time')
                for i in self.variables:
                    for j in self.variables:
                        if B[i,j]!=0:
                            self[i,j,:]*=Bp[i,j]/B[i,j]
    
                #construct symbols
                Cs = symbols(["C"+str(i) for i in self.variables])
                RHOs = (np.ones((2*self.tau_max+1, self.N, self.N))*np.nan).astype(object)
                for i in self.variables:
                    RHOs[0,i,i] = 1
                for i in range(self.N-1):
                    for j in range(i+1,self.N):
                        [i_t, j_t] = self.topo_order[[i,j]]
                        RHOs[0,i_t, j_t] = R_symbol(i_t,j_t,0)
                        RHOs[0,j_t, i_t] = RHOs[0,i_t, j_t]
                RHOs[1:(self.tau_max+1),:,:] = np.array([[[R_symbol(j,k,t) for k in self.variables] 
                                                          for j in self.variables] 
                                                         for t in self.lags[1:]])
                RHOs[self.tau_max,:,self.topo_order[-1]]=0
                for t in range(-self.tau_max, 0):
                    RHOs[t,:,:] = RHOs[-t,:,:].T
    
                Bps = np.array([np.sum(Bp[:,i]**2) for i in self.variables])
                Bps[Bps==0]=1
                            
                try:
                    for idc, c in enumerate(CO):
                        calc_dicts = []
                        #construct system of equations
                        anc = self.ancestry()
                        Ps = np.arange(self.N)[anc[:,c].any(axis=Graph.AXIS_LABELS['sink'])]
                        def make_sigma_expression(t,j,i):
                            return (-RHOs[t,j,i] 
                                    + r[i]/Cs[i]*np.sum(np.array([[self[k,i,v]*RHOs[t-v,j,k] 
                                                                   for k in self.variables] 
                                                                  for v in self.lags])))
                        last = np.max(np.array([self.order(i) for i in list(c)]))
                        so_far = CO[:idc] #+1
                        if len(so_far)>0:
                            so_far = np.concatenate(so_far) #+1
                            now_look =np.concatenate([so_far,c])
                        else:
                            now_look = c
                        def to_include(j,i,t):
                            if t==0 and self.order(i)<=self.order(j):
                                return False
                            if t==self.tau_max and self.order(i)>=last:
                                return False
                            if j in c:
                                return i in Ps
                            elif i in c:
                                return j in Ps
                            return False
                        def to_calc(j,i,t):
                            if t==self.tau_max and self.order(i)==last and self.order(i)<self.N-1:
                                return True
                            if j in c:
                                return i not in Ps
                            elif i in c:
                                return j not in Ps
                            return False
                        s_exp = [-Cs[i]**2 
                                 + r[i]**2*np.sum(np.array([[[[self[j,i,t]*self[k,i,v]*RHOs[t-v,j,k]
                                                               for j in self.variables] 
                                                              for k in self.variables]
                                                             for t in self.lags]
                                                            for v in self.lags])) 
                                  + (1-r[i]**2)*Bps[i] for i in c]
                        s_exp += [make_sigma_expression(0,j,i) 
                                  for i in now_look
                                  for j in now_look
                                  if to_include(j,i,0)]
                        s_exp += [make_sigma_expression(self.tau_max, j, i) 
                                  for i in now_look 
                                  for j in now_look
                                  if to_include(j,i,self.tau_max)]
                        s_exp += [make_sigma_expression(t,j,i) 
                                  for i in now_look
                                  for j in now_look
                                  for t in self.lags[1:]
                                  if to_include(j,i,t)]
                        cxs = np.array([n in now_look for n in self.variables])
                        rho_loc = list(set(RHOs[:,cxs,:][:,:,cxs][:self.tau_max,:,:].flatten()))
                        cxs_small = np.array([n in now_look if self.order(n)<last else False 
                                              for n in self.variables])
                        rho_loc += list(set(RHOs[:,cxs,:][:,:,cxs_small][self.tau_max,:,:].flatten()))
                        rho_loc_1 = [rho for rho in rho_loc if (isinstance(rho, type(Symbol('test'))) 
                                                                and Matrix(s_exp).has(rho))]
                        rho_loc_2 = [rho for rho in rho_loc if (isinstance(rho, type(Symbol('test'))) 
                                                                and not Matrix(s_exp).has(rho))]
                        now_vars = [Cs[cx] for cx in c] + rho_loc_1
                        
                        #solve
                        check_vars(now_vars, s_exp)
                        SSSS = nsolve(s_exp, now_vars, [1 for i in c]+[0 for i in rho_loc_1])
                        S_dict_local = {k: SSSS[i] for i, k in enumerate(now_vars)}
    
                        #second update
                        for i in c:
                            Cs[i] = Cs[i].subs(S_dict_local)
                            self[:,i,:]=self[:,i,:]*r[i]/Cs[i]
                            RHOs[:,:,i] = np.array(Matrix(RHOs[:,:,i]).subs(S_dict_local))
                            RHOs[:,i,:] = np.array(Matrix(RHOs[:,i,:]).subs(S_dict_local))
                        
                        #calculate some more!
                        calc_dict = {}
                        last_c = c[np.array([self.order(ci) for ci in c]).squeeze()==last][0]
                        for j in now_look:
                            for i in now_look:
                                for t in self.lags:
                                    if to_calc(j,i,t) and isinstance(RHOs[t,j,i], type(Symbol('test'))):
                                        calc_dict[RHOs[t,j,i]]=r[i]/Cs[i] \
                                                               *np.sum(np.array([[self[k,i,v]*RHOs[t-v,j,k]
                                                                                  for k in now_look]
                                                                                 for v in self.lags]))
                        exp_here = [-k + v for k, v in calc_dict.items()]
                        ks = list(calc_dict.keys())
                        if len(exp_here)>0:
                            check_vars(ks, exp_here)
                            SH = nsolve(exp_here, ks, [0 for i in ks])
                            calc_dict = {k: SH[i] for i, k in enumerate(ks)}
                            calc_dicts+=[calc_dict]
                            for i in c:
                                RHOs[:,i,:] = np.array(Matrix(RHOs[:,i,:]).subs(calc_dict))
                                RHOs[:,:,i] = np.array(Matrix(RHOs[:,:,i]).subs(calc_dict))

                    if not check_vars([], Matrix(np.sum(RHOs, axis=2))):
                        for cd in calc_dicts:
                            for k, v in cd.items():
                                print("{}: {}".format(k,v))
    
                except ValueError:
                    discarded_c +=1
                    continue

            else:
                self.A*=np.random.uniform(low=.2, high=2, size=self.shape)
                for i in self.variables:
                    for j in self.variables:
                        for k in self.lags:
                            self[i,j,k]*=np.random.choice(np.array([1,-1]))
            
            #stability check
            z = Symbol("z")
            M = Matrix(
                np.linalg.inv(self[:,:,0]+np.diag(np.ones((self.N,))))
                - np.sum(self[:,:,1:]*np.array([[[z**i for i in self.lags[1:]]]]), 
                         axis=Graph.AXIS_LABELS['time'])
            ).det()
            S = solve(M)
            stable = (np.array([Abs(s) for s in S])>1).all()
            if not stable:
                discarded_u += 1

        if style=='standardized':
            self.cov = RHOs
            s2 = np.array([(1-r[i]**2)/Cs[i]**2*np.sum(Bp[:,i]**2) for i in self.variables])
            s2[s2==0]=1
        else:
            s2 = np.array([1 for i in self.variables])
        self.s2 = s2

        sys.stdout.write('\r')
        if discarded_u + discarded_c > 0:
            if discarded_u == 0:
                print("discarded {} solution{} that did not converge".format(
                    discarded_c, 's' if discarded_c>1 else ''))
            elif discarded_c == 0:
                print("discarded {} unstable solution{}".format(
                    discarded_u, 's' if discarded_u>1 else ''))
            else:
                print("discarded {} solutions: {} unstable and {} that did not converge".format(
                    discarded_u+discarded_c, discarded_u, discarded_c))
        else:
            print('                             ')

        return self
        
    def gen_data(self, T=1000):
        r'''
        Generates time series data from the SCM given by causal coefficients self.A 
        and noises self.s2. T (an integer) determines the length of the generated
        time series (default = 1000). The resulting TimeSeries object is saved under
        self.data and returned. If the variance of any of the variables in the time
        series is too small (less than .2) or large (more than 2 for 'standardized'
        generation, or more than 1000 for 'unit-variance-noise' generation), this
        raises a GenerationError.
        '''
        U = np.random.normal(size=(self.N, T+len(self.lags)))
        X = np.zeros(U.shape)
        prelim_steps = self.N*len(self.lags)-1
        RHOp = np.zeros((prelim_steps,prelim_steps))
        def get_loc(idx):
            return self.topo_order[idx%self.N],idx//self.N
        if self.cov is not None:
            for idx in range(prelim_steps):
                i,t = get_loc(idx)
                for jdx in range(idx+1):
                    j,v = get_loc(jdx)
                    RHOp[jdx, idx] = self.cov[t-v, j, i]
        for idx in range(prelim_steps):
            xloc = get_loc(idx)
            if self.cov is not None:
                to_parents = RHOp[:idx,idx]
                between_parents = RHOp[:idx,:idx]
                coefs = np.array(symbols(["a{}".format(j) for j in range(len(to_parents))]))
                expressions = []
                for j, p_ji in enumerate(to_parents):
                    expressions += [-p_ji + np.sum(np.array([a_k*between_parents[k,j] 
                                                             for k, a_k in enumerate(coefs)]))]
                if len(coefs)>0:
                    Ap = np.array(nsolve(expressions, coefs, [1 for i in coefs]))[:,0]
                    s2 = 1 - np.sum(np.array([[a_j*a_k*between_parents[j,k] 
                                               for j, a_j in enumerate(Ap)] 
                                              for k, a_k in enumerate(Ap)]))
                    for i, a in enumerate(Ap):
                        X[xloc]+=a*X[get_loc(i)]
                else:
                    s2=1
                if s2<0:
                    s2=0
                X[xloc]+=s2**.5*U[xloc]
            else:
                X[xloc]=U[xloc]
        for idx in range(prelim_steps, self.N*T):
            i, t = get_loc(idx)
            s2i = self.s2[i] if self.s2 is not None else 1
            X[i,t] = (s2i**.5*U[i,t]
                      + np.sum(np.array([[self[j,i,v]*X[j,t-v]
                                          for v in self.lags]
                                         for j in self.variables])))
        TS = TimeSeries(self.N, T, self.labels, X[:,len(self.lags):])
        V = TS.var()
        if self.style=='standardized':
            cutoff=2
        else:
            cutoff=1000
        if ((V>cutoff) | (V<.2)).any():
            raise GenerationError("generated data has variance too var from 1: {}".format(V))
        self.data = TS
        return self.data

    #functions acting on the adjacency matrix...
    #returning a matrix or element
    def _check_tuple(self, tpl):
        if len(tpl)!=3:
            raise ValueError("tuple length must be 3")
    def __getitem__(self, tpl):
        self._check_tuple(tpl)
        return self.A.__getitem__(tpl)
    def __setitem__(self, tpl, v):
        self._check_tuple(tpl)
        return self.A.__setitem__(tpl,v)
    def __eq__(self, G):
        return (
            isinstance(G, Graph) 
            and self.N==G.N and self.tau_max==G.tau_max 
            and (Graph.select_vars(self.A,self.topo_order)==Graph.select_vars(G.A, G.topo_order)).all()
        )
    def _pass_on_solo(self, func, axis=None):
        if type(axis) is tuple:
            return func(self.A, axis=tuple(Graph.AXIS_LABELS[a] for a in axis))
        return func(self.A, axis=Graph.AXIS_LABELS[axis])
    def sum(self, axis=None):
        return self._pass_on_solo(np.sum, axis)
    def any(self, axis=None):
        return self._pass_on_solo(np.any, axis)
    def inv(self):
        return self._pass_on_solo(np.linalg.inv)
    def triu(self):
        A_new = np.zeros(self.shape)
        for i in self.lags:
            A_new[:,:,i] = np.triu(self[:,:,i])
        return A_new

    #returning a new graph
    def _pass_on(self, other, func):
        if type(other) in [np.ndarray, float, int]:
            new = func(self.A,other)
        elif type(other) is Graph:
            new = func(self.A,other.A)
        else:
            raise TypeError("{} is not supported for type Graph and type {}".format(
                func,type(other)))
        return Graph._sim_graph(self, new)
    def __abs__(self):
        return Graph._sim_graph(self, self.A.__abs__())
    def __add__(self, other):
        return self._pass_on(other, lambda x,y : x+y)
    def __sub__(self, other):
        return self._pass_on(other, lambda x,y : x-y)
    def __mul__(self, other):
        return self._pass_on(other, lambda x,y : x*y)
    def __truediv__(self, other):
        return self._pass_on(other, lambda x,y : x/y)

    #functions with graph logic
    def order(self, i):
        '''returns the placement of the variable at index i in the topological order'''
        return np.where(self.topo_order==i)[0]
    def summary(self):
        r'''Returns an N x N summary-graph adjacency matrix.
        The i,j-th entry represents an effect of X_i on X_j.
        '''
        return self.any(axis='time').astype(int)
    def get_num_parents(self):
        '''Returns an np.array of length N containing the number of parent processes 
        of each variable (in the summary graph)'''
        S = self.summary()
        return np.sum(S, axis=Graph.AXIS_LABELS['source'])
    
    def ancestry(self):
        r'''Returns an N x N matrix summarizing ancestries in the summary graph.
        The i,j-th is True if X_i is an ancestor of X_j and False otherwise.
        '''
        E = self.summary() != 0
        Ek = E.copy()
        all_paths = Ek.copy()*False
        for path_len in range(self.N - 1):
            all_paths = all_paths | Ek
            Ek = Ek.dot(E)
        return all_paths
    def summary_order(self):
        r'''Detects cycles in the adjacency matrix self.A. 
        Two variables X_i and X_j are in a cycle if X_i is an ancestor of X_j and
        X_j is also an ancestor of X_i. Returns a list of arrays, where each array
        is a cycle in the summary graph, with component variables listed according
        to the time-series DAG topological order. The cycles are ordered according
        to the summary-graph topological ordering between them. The time-seires DAG
        topological order is updated to an equally-valid alternative that reflects
        the summary-graph ordering.
        '''
        idn = np.arange(self.N)
        anc = self.ancestry()
        cycles = np.triu(Graph._remove_diagonal(anc*anc.T))
        collected_cycles = []
        id_used = idn.astype(bool)*False
        for i in idn:
            if id_used[i]:
                continue
            c1 = np.insert(idn[cycles[i,:]],0,i)
            collected_cycles+=[c1]
            id_used = np.array([any([j in c for c in collected_cycles])
                                for j in idn]).squeeze()
        idc = np.array([c[0] for c in collected_cycles])
        c_anc = Graph._remove_diagonal(Graph.select_vars(anc, idc))
        num_ancestors = np.sum(c_anc, axis=Graph.AXIS_LABELS['source'])
        num_descendents = np.sum(c_anc, axis=Graph.AXIS_LABELS['sink'])
        cycle_order = np.argsort(num_ancestors)
        summary_order = [self.topo_order[np.sort(np.concatenate([self.order(e)
                                                                 for e in collected_cycles[i]]))]
                         for i in cycle_order]
        self.topo_order = np.concatenate(summary_order)
        return summary_order
    def sortability(self, func='var', tol=1e-9):
        '''Calculates time-series sortability of variables in the SCM according to a 
        funtion of the user's choice on the generated data. The code is based on code 
        found at <https://github.com/Scriddie/Varsortability>, in reference to 
        Reisach, A. G., Seiler, C., & Weichwald, S. (2021). "Beware of the Simulated DAG! 
        Causal Discovery Benchmarks May Be Easy To Game" (arXiv:2102.13647). Reisach's 
        definition of sortability has been modified to avoid double-counting the pair-wise 
        sortability of two variables with multiple causal paths between them, and has been 
        further modified to accept time series data in the manner described by Christopher 
        Lohse and Jonas Wahl in "Sortability of Time Series Data" (Submitted to the Causal 
        Inference for Time Series Data Workshop at the 40th Conference on Uncertainty in 
        Artificial Intelligence.). 

        Function options include:
            'var' : variance over time, as in Lohse and Wahl; analogous to Reisach et al.
            'R2_summary' : Predictability from the past and present of distinct processes,
                           as in Lohse and Wahl.
            'R2' : Predictability from distinct processes and the process's own past.
                   Introduced here; analogous to Reisach et al.
        These functions are defined in the TimeSeries class.
        '''
        func_options = ['var', 'R2', 'R2_summary']
        E = self.summary() != 0
        anc = self.ancestry()
        Ek = E.copy()
        if func=='var':
            var = self.data.var()
        elif func=='R2':
            var = self.data.R2(self.tau_max, summary=False)
        elif func=='R2_summary':
            var = self.data.R2(self.tau_max, summary=True)
        else:
            raise ValueError("For func, please choose between {}".format(func_options))

        n_paths = 0
        n_correctly_ordered_paths = 0
        checked_paths = Ek.copy()*False

        for path_len in range(self.N - 1):
            check_now = (Ek 
                         & ~ checked_paths # to avoid double counting
                         & ~ anc.T) #to avoid comparison within a cycle
            n_paths += (check_now).sum()
            n_correctly_ordered_paths += (check_now * var.T / var > 1 + tol).sum()
            n_correctly_ordered_paths += 1/2*(
                (check_now * var.T / var <= 1 + tol) *
                (check_now * var.T / var >=  1 - tol)).sum()
            checked_paths = checked_paths | check_now
            Ek = Ek.dot(E) #examine paths of path_len+=1

        return n_correctly_ordered_paths / n_paths

    def __repr__(self):
        '''Displays a summary graph, and a table detailing all adjacencies'''
        ax = plt.figure(figsize=(7,3), layout="constrained").subplots(1,2)
        ax[0].set_axis_off()
        artists = []
        for i, label in enumerate(self.labels):
            angle = 2*np.pi/self.N*i
            radius = .25
            artist = mpatches.Circle((np.cos(angle)*radius+.5,np.sin(angle)*radius+.5), .05, ec="none")
            artist.set(color="black")
            ax[0].add_artist(artist)
            ax[0].annotate(label, (.5,.5), xycoords=artist, c='w', ha='center', va='center')
            artists +=[artist]
        S = self.summary()
        summary_edges = np.sum(S)
        if summary_edges > 0: #necessary because of weirdness in matplotlib.table -- can't make an empty table
            rep = np.zeros((summary_edges, len(self.lags))).astype(object)
            h = np.array(["Lag {}".format(i) for i in self.lags])
            ri = np.zeros((summary_edges,)).astype(object)
            r=0
            for i in self.variables:
                for j in self.variables:
                    if S[i,j]!=0:
                        posA=artists[i].center
                        posB=artists[j].center
                        if i<j:
                            connectionstyle="arc3,rad=.5"
                        elif i==j:
                            connectionstyle="arc3,rad=2"
                            posA = artists[i].get_corners()[0]
                            posB = artists[i].get_corners()[1]
                        else:
                            connectionstyle="arc3"
                        arrow = mpatches.FancyArrowPatch(posA, posB, patchA=artists[i], patchB=artists[j], 
                                                         arrowstyle='->', mutation_scale=15, color='k', 
                                                         connectionstyle=connectionstyle)
                        ax[0].add_artist(arrow)
                        ri[r] = "{}-->{}".format(i,j)
                        rep[r,:]=np.array([str(round(a,3)) for a in self[i,j,:]])
                        if self.order(j)<=self.order(i):
                            rep[r,0]=np.nan
                        r+=1
            cs = [['0.8']*rep.shape[1],['1']*rep.shape[1]]*((rep.shape[0])//2)
            if rep.shape[0]%2==1:
                cs+=[['0.8']*(rep.shape[1])]
            ax[1].table(cellText=rep, loc='center', rowLabels=ri, colLabels=h, cellLoc='center', cellColours=cs)
        ax[1].axis("off")
        return "Graph {}".format(id(self))

class TimeSeries(object):
    r""" Time Series Data object.

    Parameters
    _________
    N : int
        Number of variables
    T : int
        Number of observations
    labels : array-like, length N
        Names of the Variables
    data : N x T array
        Time series data

    Class Constants
    _______________
    AXIS_LABELS : dict
        Names of dimensions and their indices
    """
    AXIS_LABELS = {'variables': 0, 'time': 1}

    def __init__(self, N, T, labels, data):
        self.N = N
        self.T = T
        self.labels = labels
        self.data = data
        
    def var(self):
        '''Variance of each variable over time'''
        return np.var(self.data, axis=TimeSeries.AXIS_LABELS['time'], keepdims=True)
        
    def R2(self, tau_max, summary=False):
        r'''R2 predictability as detailed in Reisach, Tami, Seiler, Chambaz, and Weichwald (2023). 
        "A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models." 
        (In Advances in Neural Information Processing Systems, Vol. 36. 785–807.) modified to fit
        time series. This provides an upper-bound for predictability from causal parents. 

        Set summary=True for timeseries R2-sortability defined in Lohse and Wahl.
        '''
        R2s = np.ones((self.N,1))*np.nan
        X = self[:,tau_max:]
        for tau in range(1, tau_max+1):
            X = np.append(X, self[:,tau_max-tau:-tau], axis=TimeSeries.AXIS_LABELS['variables'])
        n_rows = (tau_max+1)*self.N
        idr = np.arange(n_rows)
        for i in range(self.N):
            if summary:
                X_i = X[idr%self.N!=i,:]
            else:
                X_i = X[idr!=i,:]
            Xi = X[i,:]
            _, resid, _, _ = np.linalg.lstsq(X_i.T, Xi.T,rcond=None)
            R2s[i] = 1 - resid[0]/self.T/np.var(Xi)
        return R2s
    def __getitem__(self, tpl):
        return self.data.__getitem__(tpl)
    def __repr__(self):
        plt.figure(layout='constrained', figsize=(4,3))
        v = self.var().squeeze()
        vo = np.argsort(v)
        vo = vo[::-1]
        plt.plot(self.data.T[:,vo])
        plt.legend(list(np.array(self.labels)[vo]))
        plt.xlim([0,self.T-1])
        plt.xlabel("Time")
        plt.ylabel("Standardized Value")
        return "TimeSeries {}".format(id(self))

if __name__ == '__main__':
    '''Beta. to be updated.'''
    Gs, Ds, text_trap = Graph.gen_unitless_time_series(10, 1, B=10)
    print(text_trap.getvalue())
    axs = plt.figure(figsize=(7,6), layout="constrained").subplots(2,2)
    plt.subplot(2,2,1)
    plt.hist(np.array([d.var() for d in Ds]).flatten())
    plt.title("Variance")
    plt.subplot(2,2,2)
    plt.hist(np.array([d.sortability() for d in Gs]).flatten())
    plt.xlim([0,1])
    plt.title("Varsortability")
    plt.subplot(2,2,3)
    plt.hist(np.array([d.sortability('R2') for d in Gs]).flatten())
    plt.xlim([0,1])
    plt.title("R2-sortability")
    plt.subplot(2,2,4)
    plt.hist(np.array([d.sortability('R2_summary') for d in Gs]).flatten())
    plt.xlim([0,1])
    plt.title("R2'-sortability")