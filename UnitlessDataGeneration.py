"""Unitless/Unsortable linear (time-series) SCM and data generation"""

# Author: Dr. Rebecca Jean Herman <rebecca.herman@tu-dresden.de>

import numpy as np
from sympy import Matrix, Symbol, symbols, re, im, Abs, Float
from sympy.solvers import solve, nsolve

import sys
import io
import signal
from contextlib import redirect_stdout, contextmanager
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import re

#Checks, warnings, and errors
class UnstableError(Exception): pass
class ConvergenceError(Exception): pass
class GenerationError(Exception): pass
class TimeoutException(Exception): pass
class OptionError(Exception): pass
@contextmanager
def _time_lim(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
def _check_given(name, value):
    '''Checks that an optional input is specified'''
    if value is None:
        raise OptionError("Please specify {}".format(name))
def _check_option(name, options, chosen):
    '''checks that a valid keyword is chosen'''
    if chosen not in options:
        raise OptionError("Valid choices for {} include {}".format(name, options))
def _progress_message(msg):
    '''Progress update that modifies in place'''
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()
def _clear_progress_message():
    sys.stdout.write('\r')
def _check_vars(now_vars, s_exp):
    missing_vars = [v for v in Matrix(s_exp).free_symbols if not Matrix(now_vars).has(v)]
    if len(missing_vars)>0:
        print("MISSING VARIABLES! {}".format(missing_vars))
    return len(missing_vars)==0

#user-available helper function
def remove_diagonal(M):
    '''removes the diagonal from a 2D array M'''
    return M & ~np.diag(np.diag(M))

def gen_unitless_time_series(N, tau_max, p=None, p_auto=None, T=1000, B=100, 
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
        _progress_message("{:.0%} completed ({} discarded)".format(
                                len(Ds)/B, all_errors))
        try:
            with _time_lim(time_limit):
                with redirect_stdout(text_trap):
                    G = tsGraph(N, tau_max, init_type=init_type)
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

    _clear_progress_message()
    if all_errors>0:
        print(("Discarded {} system{}: "
               "{} that did not converge, "
               "{} that were analytically unstable, "
               "{} that computationally diverged, "
               "and {} that timed out.").format(
            all_errors, 's' if all_errors>1 else '', 
            no_converge, unstable, diverge, TO))

    return Gs, Ds, text_trap

class Graph(object):
    r"""Data-generation object, and methods for creating and manipulating them.
    Always contains a causal graph, and becomes and SCM after calling GEN_COEFFICIENTS.
    May also hold generated data after calling GEN_DATA.
    
    Parameters
    _______________________________
    N : int
        Number of random variables
    A : (N x N) np.array
        A[i,j] is the effect of X_i on X_j.
        When entries are 0 and 1, this is an adjacency matrix.
        Otherwise, it is a causal coeficients matrix.
    s2 : np.array of length N
        Noise variances (if set or calculated)
    topo_order : np.array of length N
        topological order of the variables
    cov : (N x N) np.array
        Theoretical covariance matrix (if coeficients are generated).
        cov[i,j] is the covariance of X_i(t) and X_j(t+v), 
        where v can be negative.
    data : Data object
        data generated from this SCM (if generated)
    variables : iterable over variable indices
    shape = A.shape = (N x N)
    style : SCM generation strategy

    Class Constants
    _______________
    AXIS_LABELS : dictionary
        names of the dimensions of the adjacency array and the dimension index.
    graph_types_ : list
        accepted inputs for init_type
    generation_options_ : list
        accepted options for gen_coefficients
    
    specified : shortcut for initializing graphs from a specified adjacency array
    ccopy : creates a copy of the current SCM
    select_vars : select a subset of the adjacency array showing direct effects
                  between the specified variables.
    """
    #constants
    AXIS_LABELS = {'source': 0, 'sink': 1, None:None}
    graph_types_ = ['random', 'connected', 'disconnected', 'specified']
    generation_options_ = ['standardized', 'unit-variance-noise']

    def __init__(self, N, init_type='random', p=.5, 
                 init=None, noise=None, labels=None):
        #user-specified initializations
        self.N = N
        self.variables = range(self.N)
        if labels is not None:
            assert len(labels)==self.N, ("Length of labels {} "
                 "must match the number of variables {}").format(len(labels), self.N)
            self.labels = labels
        else:
            self.labels = ["$X_{}$".format(i) for i in self.variables]
        self._make_shape()
        _check_option('init_type', self.graph_types_, init_type)
        self.init_type = init_type
        
        #empty initializations for later replacement
        self.s2 = noise
        self.data = None
        self.cov = None
        self.topo_order = np.arange(self.N)
        self.style = None

        #initializion of the adjacency matrix and topological order
        if init_type=='specified':
            self._make_specified(init)
        else:
            #Make a fully-connected DAG
            self.A = np.ones(shape=self.shape)
            self._remove_cycles()
            if init_type=='disconnected':
                self *= 0
            elif init_type!='connected': #random or no_feedback
                self._make_random(p)
            #randomize the order of appearance of the variables
            self.shuffle()

    @classmethod
    def specified(cls, init, noise=None, labels=None):
        '''Helper function for initializing a graph from a specified adjacency matrix'''
        return cls(init.shape[cls.AXIS_LABELS['source']], 
                   init_type='specified', init=init, labels=labels, noise=noise)

    #User-available retrieval functions
    def get_adjacencies(self):
        '''returns an NxN boolean matrix where A[i,j]=True if X_i --> X_j'''
        return self.A != 0
    def order(self, i):
        '''returns the placement of the variable at index i in the topological order'''
        return np.where(self.topo_order==i)[0]
    def get_num_parents(self):
        '''Returns an np.array of length N containing the number of parent processes 
        of each variable (in the summary graph)'''
        return self.sum(matrix=self.get_adjacencies(), axis='source')
    def ancestry(self):
        r'''Returns an N x N matrix summarizing ancestries in the summary graph.
        The i,j-th is True if X_i is an ancestor of X_j and False otherwise.
        '''
        E = self.get_adjacencies()
        Ek = E.copy()
        all_paths = Ek.copy()*False
        for path_len in range(self.N - 1):
            all_paths = all_paths | Ek
            Ek = Ek.dot(E)
        return all_paths
    def select_vars(self, V, A=None):
        r'''Subset and/or reorder the adjacency array or an alternative array A
        via indices V'''
        if A is None:
            A = self.A
        return A[V,:][:,V]
    def copy(self):
        '''Creates a new graph equivalent to the input graph G.'''
        return deepcopy(self)

    #user-available modification functions
    def set_topo_order(self):
        '''Discovers and sets a topological ordering consistent with the adjacency matrix.'''
        anc = self.ancestry()
        num_ancestors = self.sum(matrix=anc, axis='source')
        self.topo_order = np.argsort(num_ancestors)
    def shuffle(self):
        '''Randomly shuffles the order of the variables.'''
        new_order = np.arange(self.N)
        np.random.shuffle(new_order)
        self.A = self.select_vars(new_order)
        self.topo_order = np.argsort(new_order)
        return
    def gen_coefficients(self, style='standardized'):
        r'''Creates an SCM from the graph using one of two generation styles:
            standardized : Procudes unsortable standardized SCMs. introduced here, recommended.
            unit-variance-noise : Draws coefficients from a uniform distribution and sets all
                                  noise variances to 1. typically used, here for comparison.
        '''
        _check_option('style', self.generation_options_, style)
        self.style = style
        self._reset_adjacency_matrix()
        self._reset_cov()
        self._reset_s2()
        if self.style=='standardized':
            self._gen_coefficients_standardized()
        else:
            self._gen_coefficients_UVN()
        return self
    def gen_data(self, P):
        '''Generates and returns a dataset with P observations from the current SCM'''
        #calculate noises
        X = np.random.normal(scale=self.s2.reshape(self.N,1), size=(self.N,P))
        #add dependencies
        for i in self.topo_order[1:]:
            X[[i],:]+=np.matmul(self[:,[i]].T,X)
        self.data = Data(self.N, P, self.labels, X)
        return self.data

    #user-available analysis functions
    def sortability(self, func='var', tol=1e-9):
        '''Calculates sortability of variables in the SCM according to a 
        funtion of the user's choice on the generated data. The code is based on code 
        found at <https://github.com/Scriddie/Varsortability>, in reference to 
        Reisach, A. G., Seiler, C., & Weichwald, S. (2021). "Beware of the Simulated DAG! 
        Causal Discovery Benchmarks May Be Easy To Game" (arXiv:2102.13647). Reisach's 
        definition of sortability has been modified to avoid double-counting the pair-wise 
        sortability of two variables with multiple causal paths between them, and has been 
        further modified to accept graphs with cycles in the manner described by Christopher 
        Lohse and Jonas Wahl in "Sortability of Time Series Data" (Submitted to the Causal 
        Inference for Time Series Data Workshop at the 40th Conference on Uncertainty in 
        Artificial Intelligence.). 

        Function options include:
            'var' : variance 
            'R2' : Predictability from other variables, as in Reisach et al.
        These functions are defined in the Data class.
        '''
        _check_option('func', self.data.analysis_options, func)
        E = self.get_adjacencies()
        anc = self.ancestry()
        Ek = E.copy()
        F = getattr(self.data, func)
        M = F()
        n_paths = 0
        n_correctly_ordered_paths = 0
        checked_paths = Ek.copy()*False

        for path_len in range(self.N - 1):
            check_now = (Ek 
                         & ~ checked_paths # to avoid double counting
                         & ~ anc.T) #to avoid comparison within a cycle
            n_paths += (check_now).sum()
            n_correctly_ordered_paths += (check_now * M.T / M > 1 + tol).sum()
            n_correctly_ordered_paths += 1/2*(
                (check_now * M.T / M <= 1 + tol) *
                (check_now * M.T / M >=  1 - tol)).sum()
            checked_paths = checked_paths | check_now
            Ek = Ek.dot(E) #examine paths of path_len+=1

        if n_paths == 0:
            return 0.5
        else:
            return n_correctly_ordered_paths / n_paths
    
    #Initialization Helper Functions
    def _rand_edges(self, p, size=None):
        return np.random.choice(a=[1,0], size=size, p=[p, 1-p])
    def _make_shape(self):
        self.shape = tuple((self.N, self.N))
        return
    def _remove_cycles(self):
        self *= (np.tril(self.A)==0)
    def _make_random(self, p):
        self*=self._rand_edges(p, size=self.shape)
    def _make_specified(self, init):
        _check_given('adjacency matrix', init)
        assert init.shape==self.shape, ("initialization matrix shape {} "
             "not consistent with expected shape {}").format(init.shape, self.shape)
        self.A = init
        self.set_topo_order()
        return

    #gen_coefficients helper functions
    def _gen_coefficients_UVN(self, low=.2, high=2):
        self *= np.random.uniform(low=low, high=high, size=self.shape)
        self *= np.random.choice(a=[-1,1], size=self.shape)
    def _gen_coefficients_standardized(self):
        P = self.get_num_parents()
        A, r = self._initial_draws(P)
        self *= A
        self._rescale_coefficients(r, P)
    def _reset_adjacency_matrix(self):
        self.A = self.A != 0
    def _reset_cov(self):
        self.cov = np.diag(np.ones((self.N,)))
    def _reset_s2(self):
        self.s2 = np.ones((self.N,))  
    def _initial_draws(self, P):
        A = np.random.normal(size=self.shape) #coefficient draws -- a'
        r = np.random.uniform(size=(self.N,)) #starting draws -- r
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r=r**(1/P)
        return A, r
    def _re_sort(self, matrix=None):
        if matrix is None:
            matrix=self.A
        return self.select_vars(np.argsort(self.topo_order), A=matrix)
    def _rescale_coefficients(self, r, P):
        #Need to go through by topological order!
        ind_length = (self**2).sum(axis='source') #constant when parents are independent
        loc_cov = self.select_vars(self.topo_order, A=self.cov)
        A_loc = self.select_vars(self.topo_order)
        for it, i in enumerate(self.topo_order):
            if P[i]==0:
                continue
            Ap = A_loc[:it,[it]]
            r_i = r[i]
            norm = ind_length[i]
            R_i = loc_cov[:it,:it]
            Ci = ((r_i**2*np.matmul(np.matmul(Ap.T, R_i), Ap) + (1-r_i**2)*norm)**.5)[0,0]
            A_loc[:,it] *= r_i/Ci
            self.s2[i] = ((1-r_i**2)*norm)**.5/Ci
            loc_cov[:it,[it]] = np.matmul(loc_cov, A_loc[:,[it]])[:it,:]
            loc_cov[it,:] = loc_cov[:,it]
        self.cov = self._re_sort(loc_cov)
        self.A = self._re_sort(A_loc)

    #Display helper functions for overwriting
    def _get_num_cols(self):
        return 1
    def _make_table_titles(self):
        return np.array(["Coefficient"])
    def _get_coefficients(self, i, j):
        return [self[i,j]]

    #Magic Functions
    #returning a matrix or element
    def __getitem__(self, tpl):
        return self.A.__getitem__(tpl)
    def __setitem__(self, tpl, v):
        return self.A.__setitem__(tpl,v)
    def __eq__(self, G):
        return (
            isinstance(G, Graph) 
            and self.N==G.N
            and (self.select_vars(self.topo_order)==G.select_vars(G.topo_order)).all()
        )
    def _pass_on_solo(self, func, axis=None, matrix=None):
        if matrix is None:
            matrix=self.A.copy()
        if type(axis) is tuple:
            return func(matrix, axis=tuple(self.AXIS_LABELS[a] for a in axis))
        return func(matrix, axis=self.AXIS_LABELS[axis])
    def sum(self, axis=None, matrix=None):
        return self._pass_on_solo(np.sum, axis, matrix)
    def any(self, axis=None, matrix=None):
        return self._pass_on_solo(np.any, axis, matrix)
    def inv(self, matrix=None):
        return self._pass_on_solo(np.linalg.inv, matrix=matrix)

    #returning a new graph
    def _pass_on(self, func, other=None):
        G_new = deepcopy(self)
        if other is None:
            G_new.A = func(self.A)
        elif type(other) in [np.ndarray, float, int]:
            G_new.A = func(self.A,other)
        elif type(other) is Graph:
            G_new.A = func(self.A,other.A)
        else:
            raise TypeError("{} is not supported for type Graph and type {}".format(
                func,type(other)))
        return G_new
    def __abs__(self):
        return self._pass_on(lambda x : x.__abs__())
    def triu(self):
        G_new = self*0
        G_new.i_triu()
        return G_new
    def __add__(self, other):
        return self._pass_on(lambda x,y : x+y, other)
    def __sub__(self, other):
        return self._pass_on(lambda x,y : x-y, other)
    def __mul__(self, other):
        return self._pass_on(lambda x,y : x*y, other)
    def __truediv__(self, other):
        return self._pass_on(lambda x,y : x/y, other)
    def __pow__(self, other):
        return self._pass_on(lambda x,y : x**y, other)

    #modifying the graph in place
    def _i_pass_on(self, func, other=None):
        if other is None:
            self.A = func(self.A)
        elif type(other) in [np.ndarray, float, int]:
            self.A = func(self.A,other)
        elif type(other) is Graph:
            self.A = func(self.A,other.A)
        else:
            raise TypeError("{} is not supported for type Graph and type {}".format(
                func,type(other)))
        return self
    def i_triu(self):
        for i in self.lags:
            self[:,:,i] = np.triu(self[:,:,i])
    def __iadd__(self, other):
        return self._i_pass_on(lambda x,y : x+y, other)
    def __isub__(self, other):
        return self._i_pass_on(lambda x,y : x-y, other)
    def __imul__(self, other):
        return self._i_pass_on(lambda x,y : x*y, other)
    def __itruediv__(self,other):
        return self._i_pass_on(lambda x,y : x/y, other)
    def __ipow__(self, other):
        return self._i_pass_on(lambda x,y : x**y, other)

    def __repr__(self):
        '''Displays a summary graph, and a table detailing all adjacencies'''
        #helper function definitions
        def label_len(label):
            digits_ = 0
            valid_digits = ['[0-9]', '[A-Z]', '[a-z]']
            for digit_type in valid_digits:
                digits_ += len(re.findall(digit_type,label))
            return digits_
        def plot_table(table_id, rep, ri, h):
            cs = [['0.8']*rep.shape[1],['1']*rep.shape[1]]*((rep.shape[0])//2)
            if rep.shape[0]%2==1:
                cs+=[['0.8']*(rep.shape[1])]
            subplot_ = table_id+1
            ax[subplot_].table(cellText=rep, loc='center', rowLabels=ri, colLabels=h, cellLoc='center', cellColours=cs)
            ax[subplot_].axis("off")
        def make_table_contents(table_id, summary_edges):
            remaining_edges = summary_edges - table_id*ROWS_PER_TABLE
            num_rows = min([remaining_edges, ROWS_PER_TABLE])
            rep = np.zeros((num_rows, self._get_num_cols())).astype(object)
            ri = np.zeros((num_rows,)).astype(object)
            r = 0
            return rep, ri, r
        def make_fig(N_CUTOFF, DAG_width, summary_edges):
            if self.N > N_CUTOFF:
                DAG_width *= self.N/N_CUTOFF
            MAX_ROWS = int(np.floor(5*DAG_width))
            Table_height = DAG_width/(MAX_ROWS+1)*(summary_edges+1)
            Table_width = .75*self._get_num_cols()+.5
            if summary_edges > 0:
                num_tables = int(np.ceil(Table_height/DAG_width))
                w_space_frac = .05
                ROWS_PER_TABLE = int(np.ceil(summary_edges/num_tables))
            else:
                num_tables = 0
                w_space_frac = 0
                ROWS_PER_TABLE = 0
            fig_width = (DAG_width+(Table_width)*num_tables)/(1-w_space_frac)
            fig_height = DAG_width*(1-w_space_frac)
            ax = plt.figure(figsize=(fig_width,fig_height), 
                            layout="constrained").subplots(1,num_tables+1,
                                                           width_ratios=[DAG_width]+[Table_width]*num_tables,
                                                           gridspec_kw = {'wspace':w_space_frac})
            return ax, num_tables, ROWS_PER_TABLE
        def add_variable(i, label, ax):
            angle = 2*np.pi/self.N*i
            radius = .35
            artist = mpatches.Ellipse((np.cos(angle)*radius+.5,np.sin(angle)*radius+.525),
                                      .025*label_len(label)+.05, .1, ec="none")
            artist.set(color="black")
            ax.add_artist(artist)
            ax.annotate(label, (.5,.5), xycoords=artist, c='w', ha='center', va='center')
            return artist
        def add_edge(i, j, artists, ax):
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
            ax.add_artist(arrow)

            
        S = self.get_adjacencies()
        summary_edges = np.sum(S)
        DAG_width = 3
        N_CUTOFF = 8
        ax, num_tables, ROWS_PER_TABLE = make_fig(N_CUTOFF, DAG_width, summary_edges)
        first_ax = ax if num_tables==0 else ax[0]
        first_ax.axis("off")
        artists = []
        for i, label in enumerate(self.labels):
            artists +=[add_variable(i, label, first_ax)]
        if summary_edges > 0: #necessary because of weirdness in matplotlib.table -- can't make an empty table
            h = self._make_table_titles()
            table_id = 0
            rep, ri, r = make_table_contents(table_id, summary_edges)
            for i in self.variables:
                for j in self.variables:
                    if S[i,j]!=0:
                        add_edge(i, j, artists, first_ax)
                        ri[r] = "{}-->{}".format(i,j)
                        rep[r,:]=np.array([str(round(a,3)) for a in self._get_coefficients(i, j)])
                        if self.order(j)<=self.order(i):
                            rep[r,0]=np.nan
                        r+=1
                        if r >= ROWS_PER_TABLE and table_id+1 < num_tables:
                            plot_table(table_id, rep, ri, h)
                            table_id+=1
                            rep, ri, r = make_table_contents(table_id, summary_edges)
            plot_table(table_id, rep, ri, h)
                            
        return "Graph {}".format(id(self))

class tsGraph(Graph):
    r"""Data-generation object, and methods for creating and manipulating them.
    Always contains a causal graph, and becomes and SCM after calling GEN_COEFFICIENTS.
    May also hold generated data after calling GEN_DATA.
    
    Parameters
    __________
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
        cov[v,i,j] is the covariance of X_i(t) and X_j(t+v), 
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
    graph_types_ : list
        accepted inputs for init_type
    specified : shortcut for initializing graphs from a specified adjacency array
    gen_unitless_time_series : wrapper function for generating a large amount
                               of random data and associated ground-truth SCMs.
    ccopy : creates a copy of the current SCM
    select_vars : select a subset of the adjacency array showing direct effects
                  between the specified variables.
    """
    AXIS_LABELS = Graph.AXIS_LABELS
    AXIS_LABELS['time'] = 2
    graph_types_ = Graph.graph_types_ + ['no_feedback']
    
    def __init__(self, N, tau_max, 
                 init_type='random', p=.5, p_auto=.8, #TODO change p_auto default to None?
                 init=None, noise=None, labels=None):
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
        self.tau_max = tau_max
        self.lags = range(self.tau_max+1)
        self.p_auto = p_auto
        super().__init__(N, init_type, p, 
                 init, noise, labels)
        #lists feedback loops by summary-graph topological order as "components"
        #Additionally updates topo_order to match this where possible.
        self.set_topo_order()

    @classmethod
    def specified(cls, init, noise=None, labels=None):
        '''Helper function for initializing a graph from a specified adjacency matrix'''
        return cls(init.shape[cls.AXIS_LABELS['source']], init.shape[cls.AXIS_LABELS['time']]-1, 
                   init_type='specified', init=init, labels=labels, noise=noise)

    #User-available retrieval functions
    #order, get_num_parents, ancestry, select_vars
    def get_num_lags(self):
        return len(self.lags)
    def get_adjacencies(self):
        r'''Returns an N x N summary-graph adjacency matrix.
        The i,j-th entry represents an effect of X_i on X_j.
        '''
        return self.any(axis='time')
    def cycle_ancestry(self):
        anc = self.ancestry()
        cycles = np.triu(remove_diagonal(anc*anc.T))
        collected_cycles = []
        idn = np.arange(self.N)
        id_used = idn.astype(bool)*False
        for i in idn:
            if id_used[i]:
                continue
            c1 = np.insert(idn[cycles[i,:]],0,i)
            collected_cycles+=[c1]
            id_used = np.array([any([j in c for c in collected_cycles])
                                for j in idn]).squeeze()
        idc = np.array([c[0] for c in collected_cycles])
        c_anc = remove_diagonal(self.select_vars(idc, A=anc))
        return collected_cycles, c_anc

    #user-available modification functions
    #shuffle    
    def set_topo_order(self):
        r'''Detects cycles in the adjacency matrix self.A. 
        Two variables X_i and X_j are in a cycle if X_i is an ancestor of X_j and
        X_j is also an ancestor of X_i. Returns a list of arrays, where each array
        is a cycle in the summary graph, with component variables listed according
        to the time-series DAG topological order. The cycles are ordered according
        to the summary-graph topological ordering between them. The time-seires DAG
        topological order is updated to an equally-valid alternative that reflects
        the summary-graph ordering.
        '''
        collected_cycles, c_anc = self.cycle_ancestry()
        num_ancestors = self.sum(matrix=c_anc, axis='source')
        cycle_order = np.argsort(num_ancestors)
        summary_order = [self.topo_order[np.sort(np.concatenate([self.order(e)
                                                                 for e in collected_cycles[i]]))]
                         for i in cycle_order]
        self.topo_order = np.concatenate(summary_order)
        self.components = summary_order

    def gen_coefficients(self, style='standardized', convergence_attempts=10):
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
        stable = False
        discarded_u = 0
        discarded_c = 0

        while not stable:
            self._gen_coefficients_cutoffs(discarded_c, discarded_u, convergence_attempts)
            try:
                super().gen_coefficients(style)
            except ValueError:
                discarded_c +=1
                continue
            stable = self._check_stability()
            if not stable:
                discarded_u += 1

        _clear_progress_message()
        self._summarize_discarded_solutions(discarded_c, discarded_u)
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
        U = np.random.normal(size=(self.N, T+self.get_num_lags()))
        X = np.zeros(U.shape)
        prelim_steps = self.N*self.get_num_lags()-1
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
        TS = TimeSeries(self.N, T, self.labels, X[:,self.get_num_lags():])
        V = TS.var()
        if self.style=='standardized':
            cutoff=2
        else:
            cutoff=1000
        if ((V>cutoff) | (V<.2)).any():
            raise GenerationError("generated data has variance too var from 1: {}".format(V))
        self.data = TS
        return self.data

    #user-available analysis functions
    def sortability(self, func='var', tol=1e-9):
        r'''
        Function options include:
            'var' : variance over time, as in Lohse and Wahl; analogous to Reisach et al.
            'R2_summary' : Predictability from (the past and present of) distinct processes,
                           as in Lohse and Wahl.
            'R2' : Predictability from distinct processes and the process's own past.
                   Introduced here; analogous to Reisach et al.
        These functions are defined in the TimeSeries class.
        '''
        self.data.tau_max = self.tau_max
        R = super().sortability(func=func, tol=tol)
        return R

    #Initialization Helper Functions
    #_rand_edges, _make_specified
    def _make_shape(self):
        self.shape = tuple((self.N, self.N, self.get_num_lags()))
        return
    def _remove_cycles(self):
        self[:,:,0] *= (np.tril(self[:,:,0])==0)
    def _make_random(self, p):
        #helper function definitions
        def adjust_p(p, tm):
            return 1 - (1-p)**(1/tm)

        if self.p_auto is None:
            self.p_auto=p
        super()._make_random(adjust_p(p, self.tau_max+1))
        self.p_auto = adjust_p(self.p_auto, self.tau_max)
        #set auto-dependencies using p_auto
        for t in self.lags[1:]:
            for i in self.variables:
                self[i,i,t] = self._rand_edges(self.p_auto)
        if self.init_type=='no_feedback':
            #removed lagged edges in reverse-topological order
            self.i_triu()

    #gen_coefficients helper functions
    #_reset_adjaceny_matrix, _reset_s2, _re_sort
    def _reset_cov(self):
        if self.style=='standardized':
            RHOs = (np.ones((2*self.tau_max+1, self.N, self.N))*np.nan).astype(object)
            for i in self.variables:
                RHOs[0,i,i] = 1
            for i in range(self.N-1):
                for j in range(i+1,self.N):
                    [i_t, j_t] = self.topo_order[[i,j]]
                    RHOs[0,i_t, j_t] = self._R_symbol(i_t,j_t,0)
                    RHOs[0,j_t, i_t] = RHOs[0,i_t, j_t]
            RHOs[1:(self.tau_max+1),:,:] = np.array([[[self._R_symbol(j,k,t) for k in self.variables] 
                                                      for j in self.variables] 
                                                     for t in self.lags[1:]])
            RHOs[self.tau_max,:,self.topo_order[-1]]=0
            for t in range(-self.tau_max, 0):
                RHOs[t,:,:] = RHOs[-t,:,:].T
            self.cov = RHOs
        else:
            self.cov = None
    def _initial_draws(self, P):
        A, r = super()._initial_draws(P)
        Bp_init = self.get_adjacencies().astype(float)
        Bp=Bp_init*np.random.normal(size=Bp_init.shape)
        B = self.sum(axis='time')
        for i in self.variables:
            for j in self.variables:
                if B[i,j]!=0:
                    A[i,j]*=Bp[i,j]/B[i,j]
        return A, r
        
        A = np.random.normal(size=self.shape) #coefficient draws -- a'
        r = np.random.uniform(size=(self.N,)) #starting draws -- r
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r=r**(1/P)
        return A, r
    def _rescale_coefficients(self, r, P):
        #construct symbols
        Cs = symbols(["C"+str(i) for i in self.variables])
        def make_sigma_expression(t,j,i):
            return (-self.cov[t,j,i] 
                    + r[i]/Cs[i]*np.sum(np.array([[self[k,i,v]*self.cov[t-v,j,k] 
                                                   for k in self.variables] 
                                                  for v in self.lags])))

        Bp = self.sum(axis='time')
        Bps = np.array([np.sum(Bp[:,i]**2) for i in self.variables])
        Bps[Bps==0]=1
                    
        for idc, c in enumerate(self.components):
            calc_dicts = []
            #construct system of equations
            anc = self.ancestry()
            Ps = np.arange(self.N)[self.any(matrix=anc[:,c], axis='sink')]
            last = np.max(np.array([self.order(i) for i in list(c)]))
            so_far = self.components[:idc] #+1
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
                     + r[i]**2*np.sum(np.array([[[[self[j,i,t]*self[k,i,v]*self.cov[t-v,j,k]
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
            rho_loc = list(set(self.cov[:,cxs,:][:,:,cxs][:self.tau_max,:,:].flatten()))
            cxs_small = np.array([n in now_look if self.order(n)<last else False 
                                  for n in self.variables])
            rho_loc += list(set(self.cov[:,cxs,:][:,:,cxs_small][self.tau_max,:,:].flatten()))
            rho_loc_1 = [rho for rho in rho_loc if (isinstance(rho, type(Symbol('test'))) 
                                                    and Matrix(s_exp).has(rho))]
            rho_loc_2 = [rho for rho in rho_loc if (isinstance(rho, type(Symbol('test'))) 
                                                    and not Matrix(s_exp).has(rho))]
            now_vars = [Cs[cx] for cx in c] + rho_loc_1
            
            #solve
            _check_vars(now_vars, s_exp)
            SSSS = nsolve(s_exp, now_vars, [1 for i in c]+[0 for i in rho_loc_1])
            S_dict_local = {k: SSSS[i] for i, k in enumerate(now_vars)}

            #update
            for i in c:
                Cs[i] = Cs[i].subs(S_dict_local)
                self[:,i] = self[:,i]*r[i]/Cs[i]
                self.cov[:,:,i] = np.array(Matrix(self.cov[:,:,i]).subs(S_dict_local))
                self.cov[:,i,:] = np.array(Matrix(self.cov[:,i,:]).subs(S_dict_local))
            
            #calculate some more!
            calc_dict = {}
            last_c = c[np.array([self.order(ci) for ci in c]).squeeze()==last][0]
            for j in now_look:
                for i in now_look:
                    for t in self.lags:
                        if to_calc(j,i,t) and isinstance(self.cov[t,j,i], type(Symbol('test'))):
                            calc_dict[self.cov[t,j,i]]=r[i]/Cs[i] \
                                                   *np.sum(np.array([[self[k,i,v]*self.cov[t-v,j,k]
                                                                      for k in now_look]
                                                                     for v in self.lags]))
            exp_here = [-k + v for k, v in calc_dict.items()]
            ks = list(calc_dict.keys())
            if len(exp_here)>0:
                _check_vars(ks, exp_here)
                SH = nsolve(exp_here, ks, [0 for i in ks])
                calc_dict = {k: SH[i] for i, k in enumerate(ks)}
                calc_dicts+=[calc_dict]
                for i in c:
                    self.cov[:,i,:] = np.array(Matrix(self.cov[:,i,:]).subs(calc_dict))
                    self.cov[:,:,i] = np.array(Matrix(self.cov[:,:,i]).subs(calc_dict))

        if not _check_vars([], Matrix(np.sum(self.cov, axis=2))):
            for cd in calc_dicts:
                for k, v in cd.items():
                    print("{}: {}".format(k,v))

        self.s2 = np.array([(1-r[i]**2)/Cs[i]**2*Bps[i] for i in self.variables])
        self.s2[self.s2==0]=1
        if (self.s2>1).any():
            raise ConvergenceError("Converged Cs produced s2>1")
        return
    def _check_stability(self):
        z = Symbol("z")
        M = Matrix(
            np.linalg.inv(self[:,:,0]+np.diag(np.ones((self.N,))))
            - np.sum(self[:,:,1:]*np.array([[[z**i for i in self.lags[1:]]]]), 
                     axis=tsGraph.AXIS_LABELS['time'])
        ).det()
        S = solve(M)
        return (np.array([Abs(s) for s in S])>1).all()
    def _R_symbol(self, i,j,l):
        return Symbol("R{}.{}({})".format(i,j,l))
    def _gen_coefficients_cutoffs(self, discarded_c, discarded_u, convergence_attempts):
        if (discarded_c >= convergence_attempts 
            or discarded_u >= convergence_attempts):
            if discarded_u>discarded_c:
                raise UnstableError(("No stable solution found for above graph {}; "
                                     "tried {}x").format(id(self), discarded_u))
            raise ConvergenceError(("No solution found for above graph {}; "
                                    "tried {}x").format(id(self), discarded_c))

        _progress_message("Attempt {}/{}".format(discarded_c+discarded_u+1, 
                                                       convergence_attempts))
    def _summarize_discarded_solutions(self, discarded_c, discarded_u):
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
    
    #Display helper functions
    def _get_num_cols(self):
        return self.get_num_lags()
    def _make_table_titles(self):
        return np.array(["Lag {}".format(i) for i in self.lags])
    def _get_coefficients(self, i, j):
        return self[i,j]

    #Magic Functions
    def __eq__(self, G):
        return (
            isinstance(G, tsGraph) 
            and self.tau_max==G.tau_max 
            and super().__eq__(self, G)
        )

class Data(object):
    AXIS_LABELS = {'variables': 0, 'observations': 1}
    analysis_options = ['var', 'R2']

    def __getitem__(self, tpl):
        return self.data.__getitem__(tpl)
        
    def __init__(self, N, O, labels, data):
        self.N = N
        self.P = O
        self.labels = labels
        self.data = data
        
    def var(self):
        '''Variance of each variable over time'''
        return np.var(self.data, axis=self.AXIS_LABELS['observations'], keepdims=True)

    def _get_regressors(self):
        return self.data

    def _remove_regressors(self, X, i):
        return X[np.arange(self.N) != i, :]
        
    def R2(self):
        r'''R2 predictability as detailed in Reisach, Tami, Seiler, Chambaz, and Weichwald (2023). 
        "A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models." 
        (In Advances in Neural Information Processing Systems, Vol. 36. 785–807.) modified to fit
        time series. This provides an upper-bound for predictability from causal parents. 

        Set summary=True for timeseries R2-sortability defined in Lohse and Wahl.
        '''
        R2s = np.ones((self.N,1))*np.nan
        X = self._get_regressors()
        for i in range(self.N):
            X_i = self._remove_regressors(X, i)
            Xi = X[i,:]
            _, resid, _, _ = np.linalg.lstsq(X_i.T, Xi.T,rcond=None)
            R2s[i] = 1 - resid[0]/self.P/np.var(Xi)
        return R2s

class TimeSeries(Data):
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
    AXIS_LABELS = Data.AXIS_LABELS
    AXIS_LABELS['time'] = AXIS_LABELS['observations']
    analysis_options = Data.analysis_options + ['R2_summary']

    def __init__(self, N, T, labels, data):
        super().__init__(N, T, labels, data)

    def _get_regressors(self):
        X = self[:,self.tau_max:]
        for tau in range(1, self.tau_max+1):
            X = np.append(X, self[:,self.tau_max-tau:-tau], axis=TimeSeries.AXIS_LABELS['variables'])
        return X

    def _remove_regressors(self, X, i):
        n_rows = (self.tau_max+1)*self.N
        idr = np.arange(n_rows)
        if self.summary:
            X_i = X[idr%self.N!=i,:]
        else:
            X_i = X[idr!=i,:]
        return X_i
        
    def R2(self, tau_max=None, summary=False):
        r'''R2 predictability as detailed in Reisach, Tami, Seiler, Chambaz, and Weichwald (2023). 
        "A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models." 
        (In Advances in Neural Information Processing Systems, Vol. 36. 785–807.) modified to fit
        time series. This provides an upper-bound for predictability from causal parents. 

        Set summary=True for timeseries R2-sortability defined in Lohse and Wahl.
        '''
        if tau_max is not None:
            self.tau_max = tau_max
        self.summary = summary
        R = super().R2()
        del self.tau_max
        del self.summary
        return R

    def R2_summary(self, tau_max=None):
        return self.R2(tau_max=tau_max, summary=True)

    def __repr__(self):
        plt.figure(layout='constrained', figsize=(4,3))
        v = self.var().squeeze()
        vo = np.argsort(v)
        vo = vo[::-1]
        plt.plot(self.data.T[:,vo])
        plt.legend(list(np.array(self.labels)[vo]))
        plt.xlim([0,self.P-1])
        plt.xlabel("Time")
        plt.ylabel("Standardized Value")
        return "TimeSeries {}".format(id(self))

if __name__ == '__main__':
    '''Beta. to be updated.'''
    Gs, Ds, text_trap = tsGraph.gen_unitless_time_series(10, 1, B=10)
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