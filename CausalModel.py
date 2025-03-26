"""Unitless Unrestricted Markov-Consistent (time-series) Linear Additive Gaussian SCM and Data Generation 
(https://doi.org/10.48550/arXiv.2503.17037)"""

# Author: Dr. Rebecca Jean Herman <rebecca.herman@tu-dresden.de>

from ChecksErrors import *
from Data import *

import numpy as np
from sympy import Matrix, Symbol, symbols, re, im, Abs, Float
from sympy.solvers import solve, nsolve

import io
from contextlib import redirect_stdout
import warnings
from copy import deepcopy
import time

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import re

from causallearn.graph.GraphClass import CausalGraph
from dao import corr

#user-available helper function
def remove_diagonal(M):
    '''removes the diagonal from a 2D array M'''
    return M * (np.diag(np.diag(M))==0)

#TODO create derivative MEC object?
class CausalModel(object):
    r"""Data-generation object, and methods for creating and manipulating them.
    Always contains a causal graph with adjacencies self.get_adjacencies() where a_{ji}=1 <=> X_j -> X_i
    Becomes a linear additive Gaussian SCM after calling GEN_COEFFICIENTS. Causal coefficients are given
    by self.A and noise standard deviations are given by self.s.
    May also hold generated data after calling GEN_DATA. 
    Adjacency/coefficient matrices can be manipulated and new CAUSALMODELs can be created using magic functions. 
    
    Attributes
    _______________________________
    N : int
        Number of random variables
    A : (N x N) np.array
        A[i,j] is the effect of X_i on X_j.
        When entries are 0 and 1, this is an adjacency matrix.
        Otherwise, it is a causal coeficients matrix.
    s : np.array of length N
        Noise standard deviations (if set or calculated)
    topo_order : np.array of length N
        topological order of the variables
    cov : (N x N) np.array
        Theoretical covariance matrix (if coefficients are generated).
        cov[i,j] is the covariance of X_i and X_j.
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
        accepted options for INIT_TYPE during CAUSALMODEL generation / initialization
    generation_options_ : list
        accepted options for STYLE in GEN_COEFFICIENTS()
    
    Initialization Options
    ______________________
    __init__ : see below
    specified : Class Method shortcut for initializing a CAUSALMODEL from a specified adjacency array
    from_causallearn : Class method to initialize CAUSALMODEL from a causal-learn cpdag

    Other Class Methods
    ___________________
    gen_dataset : create many graphs/SCMs with generated data

    Adjacency Matrix Manipulation Functions
    _______________________________________
    MAGIC FUNCTIONS
    G[i,j] : G.A[i,j] (can retrieve and set values this way)
    G == G' : G.A == G'.A
    abs(G) : returns a new CAUSALMODEL with G'.A = abs(G.A)
    the following functions may take 2 CAUSALMODELs or a CAUSALMODEL and a float.
    +, -, *, /, ** : Returns a new CAUSALMODEL. 
    +=, -=, *=, /=, **= : Modifies the CAUSALMODEL in place.
    NUMPY FUNCTIONS 
    sum, any, inv : takes a CAUSALMODEL and returns an array or value
    triu : returns a new CAUSALMODEL
    i_triu : modifies the CAUSALMODEL in place
    """
    #constants
    AXIS_LABELS = {'source': 0, 'sink': 1, None:None}
    graph_types_ = ['ER', 'connected', 'disconnected', 'specified']
    generation_options_ = ['UUMC', #https://doi.org/10.48550/arXiv.2503.17037
                           'unit-variance-noise', #https://doi.org/10.48550/arXiv.1803.01422
                           'iSCM', #https://arxiv.org/abs/2406.11601
                           'IPA',  #http://jmlr.org/papers/v21/17-123.html
                           '50-50', #https://proceedings.mlr.press/v177/squires22a.html
                           'DaO' #https://doi.org/10.48550/arXiv.2405.13100
                          ]

    def __init__(self, N, init_type='ER', p=.5, 
                 init=None, noise=None, labels=None):
        """
        Optional Parameters
        ___________________
        init_type : string (default: 'ER')
            Method for generating the adjacency matrix. Options include:
                'connected': a fully-connected acyclic time series DAG
                'ER': Erdös-Rényi random graph generation. Randomly include edges 
                      with probability P
                'disconnected': a graph with no edges
                'specified': A causal graph with adjacency matrix INIT
        p : float (Default: 0.5)
            Probability of an edge during ER graph generation
        init : N x N np.array (Default: None)
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

        #user-specified initializations
        self.N = N
        self.variables = range(self.N)
        if labels is not None:
            assert len(labels)==self.N, ("Length of labels {} "
                 "must match the number of variables {}").format(len(labels), self.N)
            self.labels = labels
        else:
            self.labels = ["$X_{{{}}}$".format(i) for i in self.variables]
        self._make_shape()
        _check_option('init_type', self.graph_types_, init_type)
        self.init_type = init_type
        
        #empty initializations for later replacement
        self.s = noise
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
            elif init_type!='connected': #ER or no_feedback
                self._make_random(p)
            #randomize the order of appearance of the variables
            self.shuffle()

    @classmethod
    def specified(cls, init, noise=None, labels=None):
        '''Helper function for initializing a CAUSALMODEL from a specified adjacency/coefficient matrix'''
        return cls(init.shape[cls.AXIS_LABELS['source']], 
                   init_type='specified', init=init, labels=labels, noise=noise)

    @classmethod
    def from_causallearn(cls, cpdag):
        '''creage a CAUSALMODEL object from a causal-learn cpdag'''
        return CausalModel.specified(cpdag.graph)

    @classmethod
    def gen_dataset(cls, N, O, B, init_args={}, coef_args={}, every=20):
        '''Generate a DATASET with data generated from B SCMs with N variables each and O (for 'observations') samples.'''
        Gs = []
        for i in range(B):
            if len(Gs)%every==0:
                _progress_message("{:.0%} completed".format(len(Gs)/B))
            g = cls(N, **init_args).gen_coefficients(**coef_args)
            g.gen_data(O)
            Gs += [g]
        _progress_message("{:.0%} completed\n".format(len(Gs)/B))
        return DataSet(Gs)

    #User-available retrieval functions
    #TODO add communication with more packages?
    def to_causallearn(self):
        '''Create a causal-learn CausalDag from current adjacency matrix.
        Credit to ZehaoJin: https://github.com/py-why/causal-learn/issues/167#issuecomment-1947214169'''
        adjacency_matrix = self.get_adjacencies()
        num_nodes = adjacency_matrix.shape[0]
        cg = CausalGraph(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge1 = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
                if edge1 is not None:
                    cg.G.remove_edge(edge1)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency_matrix[i,j] == 1:
                    cg.G.add_edge(Edge(cg.G.nodes[i], cg.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
        DAG = cg.G
        return DAG
    def get_adjacencies(self, **args):
        '''returns an NxN boolean matrix where A[i,j]=True iff X_i --> X_j'''
        return self.A != 0
    def order(self, i):
        '''returns the placement in the topological order of the variable at index i'''
        return np.where(self.topo_order==i)[0]
    def get_num_parents(self):
        '''Returns an np.array of length N containing the number of parent processes 
        of each variable (in the summary graph)'''
        return self.sum(matrix=self.get_adjacencies(include_auto=False), axis='source')
    def ancestry(self):
        r'''Returns an N x N matrix summarizing ancestries in the (summary) graph.
        The i,j-th entry is True if X_i is an ancestor of X_j and False otherwise.
        '''
        E = self.get_adjacencies(include_auto=True)
        Ek = E.copy()
        all_paths = Ek.copy()*False
        for path_len in range(self.N):
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
        '''Creates a new CAUSALMODEL equivalent to the input CAUSALMODEL G.'''
        return deepcopy(self)

    #user-available modification functions
    def gen_coefficients(self, style='UUMC', gen_args={}):
        r'''Creates an SCM from the graph using any of the options from GENERATION_OPTIONS_:
            UUMC : Procudes unitless, unrestricted, Markov-consistent SCMs. introduced here, recommended.
                   (https://doi.org/10.48550/arXiv.2503.17037)
            unit-variance-noise : Draws coefficients uniformly from [-HIGH, -LOW] U [LOW, HIGH], and sets all
                                  noise variances to 1. Defaults LOW=.5, HIGH=2. typically used.
                                  (https://doi.org/10.48550/arXiv.1803.01422)
            iSCM : Begins with UVN SCM generation. The SCM is not complete until calling GEN_DATA. 
                   During data generation, the coefficients (and data) for each variable are 
                   standardized by the sample standard deviation of the generated data before
                   moving on to the next variable in the topological order.
                   (https://arxiv.org/abs/2406.11601)
            IPA : Each variable is scaled down by the variance it would have had if its parents were independent.
                  (http://jmlr.org/papers/v21/17-123.html)
            50-50 : Begins with UVN SCM generation. The SCM is not complete until calling GEN_DATA.
                    During data generation, data for each variable is generated first without noise, 
                    then the coefficients and data are scaled down to have a variance of 1/2, and 
                    noise with variance 1/2 is added before moving on to the next variable in the 
                    topological order.
                    (https://proceedings.mlr.press/v177/squires22a.html)
            DaO : DAG Adaptation of the Onion Method; dao.py taken from https://github.com/bja43/DaO_simulation
                  (https://doi.org/10.48550/arXiv.2405.13100)

            gen_args : dict (optional; default = {})
                additional arguments for generation styles. 
                unit-variance-noise:
                    low : float (optional, default=0.5)
                    high : float (optional, default=2.)
                UUMC:
                    dist : function (optional, default=np.random.uniform)
                        The function from which to draw r^(#parents). 
                        Must be able to take size=(self.N,) as an input, and have support on (0,1).
                    
        '''
        _check_option('style', self.generation_options_, style)
        self.style = style
        self._reset_adjacency_matrix()
        self._reset_cov()
        self._reset_s2()
        if self.style=='UUMC':
            self._gen_coefficients_standardized(r_args = gen_args)
        elif self.style=='IPA':
            self._gen_coefficients_UVN(low=0.5, high=1.5)
            for i in self.topo_order[1:]:
                norm=np.sqrt(np.sum(self[:,i]**2)+1)
                self[:,i]/=norm
                self.s[i]/=norm
                self.cov = self._calc_cov()
        elif self.style=='50-50': #SCM modified during GEN_DATA
            self._gen_coefficients_UVN(low=0.25, high=1)
        elif self.style=='DaO': 
            self.cov, B, self.s = corr(self.A.T)
            self.A = B.T
        elif self.style=='iSCM': #SCM modified during GEN_DATA
            self._gen_coefficients_UVN()
        else: #UVN
            self._gen_coefficients_UVN(**gen_args)
        return self        
    def gen_data(self, P):
        '''Generates and returns a dataset with P observations from the current SCM'''
        
        #calculate noises
        par = self.get_num_parents()
        X = np.random.normal(scale=self.s.reshape(self.N,1), size=(self.N,P))

        #data generation rescaling of coefficients
        def rescale_iSCM(i):
            #scale so the total variance is 1
            sample_std = np.std(X[i,:])
            self[:,i]/=sample_std
            self.s[i]/=sample_std
            X[i,:]/=sample_std
        def rescale_50(i, to_add):
            #scale so half the variance is noise
            X[i,:]/=np.sqrt(2)
            sample_std = np.sqrt(2)*np.std(to_add)
            self[:,i]/=sample_std
            return to_add/sample_std            
            
        if self.style=='iSCM':
            rescale_iSCM(self.topo_order[0])

        #add causal variability
        for i in self.topo_order[1:]:
            #calculate causal variability
            to_add=np.matmul(self[:,[i]].T,X)
            
            if self.style=='50-50' and par[i] != 0:
                to_add = rescale_50(i, to_add)

            #complete data
            X[[i],:]+=to_add
            
            if self.style=='iSCM':
                rescale_iSCM(i)
        
        self.data = Data(self.N, P, self.labels, X)
        if self.style in ['50-50', 'iSCM']:
            self._calc_cov()
        return self.data
    def deduce_topo_order(self):
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
        self.deduce_topo_order()
        return

    #gen_coefficients helper functions
    def _gen_coefficients_UVN(self, low=.5, high=2):
        self *= np.random.uniform(low=low, high=high, size=self.shape)
        self *= np.random.choice(a=[-1,1], size=self.shape)
        self._calc_cov()
    def _gen_coefficients_standardized(self, r_args={}):
        self._P = self.get_num_parents()
        self._r = self._initial_draws_r(**r_args)
        self._initial_draws_A()
        self._rescale_coefficients() #sets s and cov
    def _reset_adjacency_matrix(self):
        self.A = self.A != 0
    def _reset_cov(self):
        self.cov = np.diag(np.ones((self.N,)))
    def _reset_s2(self):
        self.s = np.ones((self.N,))  
    def _initial_draws_A(self):
        self *= np.random.normal(size=self.shape) #coefficient draws -- a'
    def _initial_draws_r(self, dist=np.random.uniform):
        r = dist(size=(self.N,)) #starting draws -- r
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r=r**(1/self._P)
        return r
    def _re_sort(self, matrix=None):
        if matrix is None:
            matrix=self.A
        return self.select_vars(np.argsort(self.topo_order), A=matrix)
    def _calc_cov(self):
        P=self.get_num_parents()
        loc_cov = deepcopy(self.select_vars(self.topo_order, A=self.cov))
        A_loc = self.select_vars(self.topo_order)
        for it, i in enumerate(self.topo_order):
            loc_cov[it,it] = self.s[i]**2
            if P[i]!=0:
                A_curr = A_loc[:,[it]]
                loc_cov[:it,[it]] = np.matmul(loc_cov, A_curr)[:it,:]
                loc_cov[it,:] = loc_cov[:,it]
                loc_cov[it,it] += np.matmul(np.matmul(A_curr.T, loc_cov), A_curr)
        self.cov = self._re_sort(loc_cov)
        self.A = self._re_sort(A_loc)
    def _rescale_coefficients(self):
        r = self._r
        P = self._P
        ind_length = (self**2).sum(axis='source') #constant when parents are independent
        r[ind_length==0]=0
        ind_length[ind_length==0]=1
        
        loc_cov = deepcopy(self.select_vars(self.topo_order, A=self.cov))
        A_loc = self.select_vars(self.topo_order)
        
        Cs = np.ones((self.N,))
        for it, i in enumerate(self.topo_order):
            if P[i]!=0:
                Ap = A_loc[:it,[it]]
                r_i = r[i]
                norm = ind_length[i]
                R_i = loc_cov[:it,:it]
                Ci = ((r_i**2*np.matmul(np.matmul(Ap.T, R_i), Ap) + (1-r_i**2)*norm)**.5)[0,0]
                A_loc[:,it] *= r_i/Ci
                Cs[i]=Ci
                loc_cov[:it,[it]] = np.matmul(loc_cov, A_loc[:,[it]])[:it,:]
                loc_cov[it,:] = loc_cov[:,it]
                
        self.s = np.divide(((1-r**2)*ind_length)**.5,Cs)
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
            isinstance(G, CAUSALMODEL) 
            and self.N==G.N
            and (self.select_vars(self.topo_order)==G.select_vars(G.topo_order)).all()
        )
    def _pass_on_solo(self, func, axis=None, matrix=None):
        if matrix is None:
            matrix=self.A.copy()
        if type(axis) is tuple:
            axis = tuple((a if type(a) is int else self.AXIS_LABELS[a] for a in axis))
        elif type(axis) is not int:
            axis = self.AXIS_LABELS[axis]
        return func(matrix, axis=axis)
    def sum(self, axis=None, matrix=None, out=None):
        return self._pass_on_solo(np.sum, axis, matrix)
    def any(self, axis=None, matrix=None):
        return self._pass_on_solo(np.any, axis, matrix)
    def inv(self, matrix=None):
        return self._pass_on_solo(np.linalg.inv, matrix=matrix)

    #returning a new CausalModel
    def _pass_on(self, func, other=None):
        G_new = deepcopy(self)
        if other is None:
            G_new.A = func(self.A)
        elif type(other) in [np.ndarray, float, int]:
            G_new.A = func(self.A,other)
        elif type(other) is CausalModel:
            G_new.A = func(self.A,other.A)
        else:
            raise TypeError("{} is not supported for type CausalModel and type {}".format(
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

    #modifying the CausalModel in place
    def _i_pass_on(self, func, other=None):
        if other is None:
            self.A = func(self.A)
        elif type(other) in [np.ndarray, float, int]:
            self.A = func(self.A,other)
        elif type(other) is CausalModel:
            self.A = func(self.A,other.A)
        else:
            raise TypeError("{} is not supported for type CausalModel and type {}".format(
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

    def _ij_style(self,ref, sig):
        return "arc3"
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
            ref = (j-i)%self.N - self.N/2
            if i==j:
                connectionstyle="arc3,rad=2"
                posA = artists[i].get_corners()[0]
                posB = artists[i].get_corners()[1]
            elif (ref < 0) or ((ref==0) and (i<j)):
                sig = 1 if (((j-i)%self.N==1) or (ref==0)) else -1
                connectionstyle=self._ij_style((self.N/2 + ref)/self.N*2, 
                                               sig)
            else:
                connectionstyle="arc3"
            arrow = mpatches.FancyArrowPatch(posA, posB, patchA=artists[i], patchB=artists[j], 
                                             arrowstyle='->', mutation_scale=15, color='k', 
                                             connectionstyle=connectionstyle)
            ax.add_artist(arrow)

        S = self.get_adjacencies(include_auto=True)
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
                            
        return str(self)

    def __str__(self):
        return "CausalModel {}".format(id(self))

class tsCausalModel(CausalModel):
    r"""Time-series Data-generation object, and methods for creating and manipulating them.
    Always contains a time-series causal graph, and becomes and SCM after calling GEN_COEFFICIENTS.
    May also hold generated data after calling GEN_DATA.
    Adjacency matrices can be manipulated and new tsCausalModels can be created using magic functions. 
    
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
        accepted options for INIT_TYPE during tsCausalModel generation / initialization
    generation_options_ : list
        accepted options for STYLE in GEN_COEFFICIENTS()
    specified : shortcut for initializing tsCausalModels from a specified adjacency array
    gen_dataset : wrapper function for generating a large amount
                  of random data and associated ground-truth SCMs.

    Adjacency Matrix Manipulation Functions
    _______________________________________
    MAGIC FUNCTIONS
    G[i,j] : G.A[i,j] (can retrieve and set values this way)
    G == G' : G.A == G'.A
    abs(G) : returns a new tsCAUSALMODEL with G'.A = abs(G.A)
    the following functions may take 2 tsCAUSALMODELs or a tsCAUSALMODEL and a float.
    +, -, *, /, ** : Returns a new tsCAUSALMODEL. 
    +=, -=, *=, /=, **= : Modifies the tsCAUSALMODEL in place.
    NUMPY FUNCTIONS 
    sum, any, inv : takes a tsCAUSALMODEL and returns an array or value
    triu : returns a new tsCAUSALMODEL
    i_triu : modifies the tsCAUSALMODEL in place
    """
    AXIS_LABELS = CausalModel.AXIS_LABELS
    AXIS_LABELS['time'] = 2
    graph_types_ = CausalModel.graph_types_ + ['no_feedback']
    generation_options_ = CausalModel.generation_options_[:2] #UUMC and unit-variance-noise
    
    def __init__(self, N, tau_max, 
                 init_type='ER', p=.5, p_auto=.8, #TODO change p_auto default to None?
                 init=None, noise=None, labels=None):
        """
        Optional Parameters
        ___________________
        init_type : string (default: 'ER')
            Method for generating the adjacency matrix. Options include:
                'connected': a fully-connected acyclic time series DAG
                'ER': randomly include edges from the connected graph 
                          such that the probability of a corresponding edge
                          in the summary graph is p (or p_auto for auto-dependence)
                'no_feedback': as with 'ER', but the summary graph is also acyclic
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
        self.sort_topo_order()

    @classmethod
    def specified(cls, init, noise=None, labels=None):
        '''Helper function for initializing a tsCAUSALMODEL from a specified adjacency/coefficient matrix'''
        return cls(init.shape[cls.AXIS_LABELS['source']], init.shape[cls.AXIS_LABELS['time']]-1, 
                   init_type='specified', init=init, labels=labels, noise=noise)
    
    @classmethod
    def gen_dataset(cls, N, tau_max, T, B, init_args={}, coef_args={}, time_limit=5, verbose=False, text_trap=None):
        r'''Method for generating data from many random SCMs.
    
        Parameters
        _________
        N, tau_max, init_args: parameters for graph initialization
        T : int, optional (Default: 1000)
            Number of observations in each generated time series
        B : int, optional (Default: 100)
            Number of SCMs (and associated data) to generate
        coef_args : dictionary of inputs to gen_coefficients
        time_limit : int, optional (Default: 5)
            Maximum number of seconds to spend on each attempt at generating an SCM
        verbose : whether to print all output
        '''
        error_types = [ConvergenceError, UnstableError, GenerationError, TimeoutException]
        errors = {k: 0 for k in error_types}
        Gs = []
        if text_trap is None:
            text_trap = io.StringIO()

        while len(Gs)<B:
            all_errors = sum(list(errors.values()))
            _progress_message("{:.0%} completed ({} discarded)".format(
                                len(Gs)/B, all_errors))
            try:
                with _time_lim(time_limit):
                    with redirect_stdout(text_trap):
                        g = cls(N=N, tau_max=tau_max, **init_args).gen_coefficients(**coef_args)
                        g.gen_data(T)
            except tuple(errors.keys()) as E:
                errors[type(E)]+=1
                continue
            Gs += [g]

        if all_errors > 0:
            num_errors = len(errors.keys())
            _clear_progress_message("Discarded {} system{} due to the following errors: ".format(all_errors, 's' if all_errors>1 else ''))
            print((", and ".join([(", ".join(["{} {}{}"]*(num_errors-1))), "{} {}{}"])).format(*sum(((v, k.__name__, 's' if v!=1 else '') for k, v in errors.items()),())))
        else:
            _progress_message("{:.0%} completed ({} discarded)\n".format(
                                len(Gs)/B, all_errors))
        if verbose:
            print(text_trap.read())
        return DataSet(Gs)

    #User-available retrieval functions
    #order, get_num_parents, ancestry, select_vars
    def get_num_lags(self):
        r'''Returns tau_max + 1'''
        return len(self.lags)
    def get_adjacencies(self, include_auto=True):
        r'''Returns an N x N summary-graph adjacency matrix.
        The i,j-th entry represents an effect of X_i on X_j.
        '''
        adj = self.any(axis='time')
        if not include_auto:
            adj = remove_diagonal(adj)
        return adj
    def cycle_ancestry(self):
        r'''Returns a list of cycles and 
            a matrix describing the ancestral relationships between them.
            The i,j-th entry is True is collected_cycles[i] are ancestors of 
            collected_cycles[j] and False otherwise.
        '''
        anc = self.ancestry()
        cycles = np.triu(remove_diagonal(anc*anc.T))
        collected_cycles = []
        idn = np.arange(self.N)
        id_used = idn.astype(bool)*False
        for i in idn:
            if id_used[i]:
                continue
            c1 = np.insert(idn[cycles[i,:]==True],0,i)
            collected_cycles+=[c1]
            id_used = np.array([any([j in c for c in collected_cycles])
                                for j in idn]).squeeze()
        idc = np.array([c[0] for c in collected_cycles])
        c_anc = remove_diagonal(self.select_vars(idc, A=anc))
        return collected_cycles, c_anc

    #user-available modification functions
    #shuffle    
    def sort_topo_order(self):
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

    def gen_coefficients(self, style='UUMC', convergence_attempts=10):
        r'''Generate a random SCM from a causal graph.

        Parameters
        __________
        convergence_attempts : int, optional (Default: 10)
            Number of attempts to define a random SCM from this adjacency matrix.
        style : string, optional (Default: 'UUMC')
            Data generation technique; options are 'UUMC' and 'unit-variance-noise'.

        Style = 'UUMC':
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
        Modifies self.A, and sets self.s, self.cov, and self.style.

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
        discarded_psd = 0
        while not stable:
            self._gen_coefficients_cutoffs(discarded_c, discarded_u, discarded_psd, convergence_attempts)
            try:
                super().gen_coefficients(style=style)
            except ValueError:
                discarded_c +=1
                continue
            except ConvergenceError:
                discarded_psd +=1
                continue
            stable = self._check_stability()
            if not stable:
                discarded_u += 1
                
        self._summarize_discarded_solutions(discarded_c, discarded_u, discarded_psd)
        return self
        
    def gen_data(self, T=1000, generation_attempts=2):
        r'''
        Generates time series data from the SCM given by causal coefficients self.A 
        and noises self.s. T (an integer) determines the length of the generated
        time series (default = 1000). The resulting TimeSeries object is saved under
        self.data and returned. If the variance of any of the variables in the time
        series is too small (less than .2) or large (more than 2 for 'UUMC'
        generation, or more than 1000 for 'unit-variance-noise' generation), this
        raises a GenerationError.
        '''
        for a in range(generation_attempts):
            U = np.random.normal(size=(self.N, T+self.tau_max))
            X = np.zeros(U.shape)
            X[:,:self.tau_max] = self._gen_initial_values()
            def get_loc(idx):
                return self.topo_order[idx%self.N],idx//self.N
            for idx in range(self.N*self.tau_max, np.prod(X.shape)):
                i, t = get_loc(idx)
                si = self.s[i] if self.s is not None else 1
                X[i,t] = (si*U[i,t]
                          + np.sum(np.array([[self[j,i,v]*X[j,t-v]
                                              for v in self.lags]
                                             for j in self.variables])))
            TS = TimeSeries(self.N, T, self.labels, X[:,self.tau_max:])
            V = TS.var()
            if self.style=='UUMC':
                cutoff=2
            else:
                cutoff=1000
            if ((V>cutoff) | (V<.2)).any():
                continue
            self.data = TS
            return self.data
        raise GenerationError("generated data has variance too var from 1: {} (tried {}x)".format(V, generation_attempts))

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
        if self.p_auto is None:
            self.p_auto=p
        #if self.init_type=='AR(1)':
            
        #helper function definitions
        def adjust_p(p, tm):
            return 1 - (1-p)**(1/tm)
        super()._make_random(adjust_p(p, self.tau_max+1))
        if self.tau_max>0:
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
        if self.style!='unit-variance-noise':
            RHOs = (np.ones((self.N, self.N, 2*self.tau_max+1))*np.nan).astype(object)
            for i in self.variables:
                RHOs[i,i,0] = 1
            for i in range(self.N-1):
                for j in range(i+1,self.N):
                    [i_t, j_t] = self.topo_order[[i,j]]
                    RHOs[i_t, j_t,0] = self._R_symbol(i_t,j_t,0)
                    RHOs[j_t, i_t,0] = RHOs[i_t, j_t,0]
            if self.tau_max > 0:
                RHOs[:,:,1:(self.tau_max+1)] = np.array([[[self._R_symbol(j,k,t) for t in self.lags[1:]]
                                                           for k in self.variables] 
                                                          for j in self.variables])
                for t in range(-self.tau_max, 0):
                    RHOs[:,:,t] = RHOs[:,:,-t].T
            #RHOs[self.tau_max,:,self.topo_order[-1]]=0
            #RHOs[-self.tau_max,self.topo_order[-1],:]=0
            self.cov = RHOs
        else:
            self.cov = None
    def _convolve_squared(self, A=None, C=None):
        result = 0
        if A is None:
            A = self.A
        if C is None:
            C = self.cov
        for l in self.lags:
            for v in self.lags:
                result+=np.tensordot(np.tensordot(A[:,:,l].T, C[:,:,l-v],1), A[:,:,v],1)
        return result
    def _remove_vars(self, M):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                for t in range(M.shape[2]):
                    if isinstance(M[i,j,t], type(Symbol('test'))):
                        if i==j:
                            M[i,j,t]=1
                        else:
                            M[i,j,t]=0
        return M.astype(float)
    def _calc_C_guess(self, r, P, i, it):
        #return 1
        #Bp_auto = np.diag(self.sum(axis='time'))[i]**2
        #norm = self.sum(matrix=remove_diagonal(self.sum(axis='time'))[:,[i]]**2, axis='source')
        norm = self.sum(matrix=self.sum(axis='time')[:,[i]]**2, axis='source')[0]
        #if norm==0:
        #    norm=1.0-Bp_auto
        #else:
        #    norm=norm*(1-r**2)/(r**2)
        Ap = self.select_vars(self.topo_order[:(it+1)])[:,[self.order(i)[0]]]
        loc_cov = self._remove_vars(self.select_vars(self.topo_order[:(it+1)], A=self.cov))
        matmulres = self._convolve_squared(Ap, loc_cov)
        if matmulres<0 or norm<0:
            print("matmulres={}, norm={}".format(matmulres, norm))
        return matmulres[0,0]/norm
        #return np.real(((r**2*matmulres + (1-r**2)*norm)**.5)[0,0])
    def _rescale_coefficients(self):
        r = self._r
        P = self._P
        f = np.random.f(100,100)
        B = self.sum(axis='time')
        Bp_init = self.get_adjacencies().astype(float)
        Bp=remove_diagonal(Bp_init*np.random.normal(size=Bp_init.shape))
        Bp_diag_p = np.diag(Bp_init)*np.random.uniform(size=(self.N,))
        Bp2_sum = (self.sum(axis='source', matrix=Bp**2))**.5
        Bp_diag = Bp_diag_p**f
        for i in self.variables:
            if B[i,i] != 0:
                self[i,i]*=Bp_diag[i]/B[i,i]
                self.s[i]=np.sqrt(1-Bp_diag[i]**2)
            if P[i]!=0:
                for j in self.variables:
                    if j != i:
                        Bp[j,i]*=r[i]*self.s[i]/Bp2_sum[i]
                        if Bp_diag_p[j]-Bp_diag[i] != 0:
                            Bp[j,i]*=(Bp_diag[j]-Bp_diag[i])/(Bp_diag_p[j]-Bp_diag_p[i])
                        elif Bp_diag_p[i] != 0:
                            Bp[j,i]*=f*Bp_diag_p[i]**(f-1)
                        if B[j,i] != 0:
                            self[j,i,:]*=Bp[j,i]/B[j,i]
                self.s[i]*=np.sqrt(1-r[i]**2)
        
        #Bps[Bps==0]=1.0#-Bps_auto[Bps==0]
        Cs = np.array(symbols(["C"+str(i) for i in self.variables]))
        Cs[((P+Bp_diag)==0)]=1.0 # * (Bps_auto==0)
        def use_C(j,i):
            if j!=i:
                return Cs[i]
            return 1
        def make_sigma_expression(t,j,i):
            return (-self.cov[j,i,t] 
                    + sum([self[k,i,v]*self.cov[j,k,t-v]/use_C(k,i)
                           for k in self.variables 
                           for v in self.lags]))
        checks = []
        solve_time = []
        update_time = []
        for idc, c in enumerate(self.components):
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
                if t==self.tau_max and self.order(i)==last:
                    return True
                if j in c:
                    return i not in Ps
                elif i in c:
                    return j not in Ps
                return False


            #SLOW (.02s/iteration)
            checks+=[[time.time()]]
            #print("equation components: cause-explained={}, noise={}".format([sum([self[j,i,t]*self[k,i,v]*self.cov[j,k,t-v]/use_C(j,i)/use_C(k,i)
             #               for j in self.variables
             #               for k in self.variables
             #               for t in self.lags
             #               for v in self.lags]) for i in c], [(self.s[i]/Cs[i])**2 for i in c]))
            s_exp = [-1
                     + sum([self[j,i,t]*self[k,i,v]*self.cov[j,k,t-v]/use_C(j,i)/use_C(k,i)
                            for j in self.variables
                            for k in self.variables
                            for t in self.lags
                            for v in self.lags]) 
                     + (self.s[i]/Cs[i])**2 for i in c] #*multiplier_[i]    
            
            checks[-1]+=[time.time()]
            s_exp += [make_sigma_expression(0,j,i) 
                      for i in now_look
                      for j in now_look
                      if to_include(j,i,0)]
            if self.tau_max != 0:
                s_exp += [make_sigma_expression(self.tau_max, j, i) 
                          for i in now_look 
                          for j in now_look
                          if to_include(j,i,self.tau_max)]
            s_exp += [make_sigma_expression(t,j,i) 
                      for i in now_look
                      for j in now_look
                      for t in self.lags[1:]
                      if to_include(j,i,t)]
            
            checks[-1]+=[time.time()]
            cxs = np.array([n in now_look for n in self.variables])
            rho_loc = list(set(self.select_vars(cxs,A=self.cov)[:,:,:self.tau_max].flatten()))
            cxs_small = np.array([n in now_look if self.order(n)<last else False 
                                  for n in self.variables])
            rho_loc += list(set(self.cov[cxs,:,:][:,cxs_small,:][:,:,self.tau_max].flatten()))
            rho_loc_1 = [rho for rho in rho_loc if (isinstance(rho, type(Symbol('test'))) 
                                                    and Matrix(s_exp).has(rho))]
            rho_loc_2 = [rho for rho in rho_loc if (isinstance(rho, type(Symbol('test'))) 
                                                    and not Matrix(s_exp).has(rho))]
            now_cs = [Cs[cx] for cx in c if isinstance(Cs[cx], type(Symbol('test')))]
            C_guesses = [self._calc_C_guess(1, P[i], i, last) for i in c if isinstance(Cs[i], type(Symbol('test')))] #self.order(i)[0]
            now_vars = now_cs + rho_loc_1
            if len(now_vars)>0:
                _check_vars(now_vars, s_exp)
                if len(rho_loc_1)==0:
                    S_dict_local = {k: C_guesses[i] for i, k in enumerate(now_cs)}
                else:
                    #solve
                    SSSS = nsolve(s_exp, now_vars, C_guesses+[0 for i in rho_loc_1])
                    S_dict_local = {k: SSSS[i] for i, k in enumerate(now_vars)}

                #for i in range(len(now_cs)):
                    #print("{}: guessed {}, found {}".format(now_cs[i], C_guesses[i], S_dict_local[now_cs[i]]))

                #update
                for i in c:
                    Cs[i] = float(Matrix([Cs[i]]).subs(S_dict_local)[0])
                    for j in self.variables:
                        if j != i:
                            self[j,i] = self[j,i]/Cs[i]
                    self.cov[:,i,:] = np.array(Matrix(self.cov[:,i,:]).subs(S_dict_local))
                    self.cov[i,:,:] = np.array(Matrix(self.cov[i,:,:]).subs(S_dict_local))
                    self.s[i]/=Cs[i]
            
            #SLOW (.02s/iteration)
            checks[-1]+=[time.time()]
            #calculate some more!
            calc_dict = {}
            last_c = c[np.array([self.order(ci) for ci in c]).squeeze()==last][0]
            for j in now_look:
                for i in now_look:
                    for t in self.lags:
                        if to_calc(j,i,t) and isinstance(self.cov[j,i,t], type(Symbol('test'))):
                            calc_dict[self.cov[j,i,t]]= np.sum(np.array([[self[k,i,v]*self.cov[j,k,t-v]
                                                                      for k in now_look]
                                                                     for v in self.lags])) #*r[i]/Cs[i]
            
            
            #SLOW (.02s/iteration)
            checks[-1]+=[time.time()]

            update_now = {k: v for k, v in calc_dict.items() if not any([v.has(rho) for rho in calc_dict.keys()])}
            solved = deepcopy(update_now)
            while len(update_now)>0:
                calc_dict = {k: v.subs(update_now) for k, v in calc_dict.items() if k not in solved.keys()}
                update_now = {k: v for k, v in calc_dict.items() if not any([v.has(rho) for rho in calc_dict.keys()])}
                solved.update(update_now)
            
            ks = list(calc_dict.keys())
            if len(ks) > 0:
                start=time.time()
                exp_here = [-k + v for k, v in calc_dict.items()]
                _check_vars(ks, exp_here)
                SH = nsolve(exp_here, ks, [0 for i in ks])
                solved.update({k: SH[i] for i, k in enumerate(ks)})
                end=time.time()
                solve_time+=[end-start]
            if len(solved.keys())>0:
                start=time.time()
                for i in c:
                    self.cov[i,:,:] = np.array(Matrix(self.cov[i,:,:]).subs(solved))
                    self.cov[:,i,:] = np.array(Matrix(self.cov[:,i,:]).subs(solved))
                end=time.time()
                update_time+=[end-start]
            checks[-1]+=[time.time()]

        _check_vars([], Matrix(np.sum(self.cov, axis=2)))

        if self.tau_max!=0 and np.any(np.linalg.eigvals(self.cov[:,:,0].astype(float))<0):
            raise ConvergenceError("covariance matrix not positive semi-definite")
        #self.s = np.divide(((1-r**2)*Bps)**.5,Cs)#multiplier_
        if (self.s>1).any():
            raise ConvergenceError("Converged Cs produced s2>1: r={}, Cs={}, s2={}".format(r, Cs, self.s))
        self.cov = self.cov.astype(float)
        if (np.abs(self.cov)>1).any():
            raise ConvergenceError("Converged RHOs are sometimes > 1! {}".format(self.cov))
        checks=np.array(checks)
        checks=np.mean(checks[:,1:]-checks[:,:-1], axis=0)
        #print(("Elapsed time for... C expression construction={:.2f}, RHO expression construction={:.2f}, between={:.2f}, additional construction={:.2f}, additional solve={:.2f} (solve: {:.2f}. update: {:.2f})").format(*checks, np.mean(np.array(solve_time)), np.mean(np.array(update_time))))
        return
    def _check_stability(self):
        if self.tau_max==0:
            return True
        z = Symbol("z")
        M = Matrix(
            np.linalg.inv(self[:,:,0]+np.diag(np.ones((self.N,))))
            - np.sum(self[:,:,1:]*np.array([[[z**i for i in self.lags[1:]]]]), 
                     axis=tsCausalModel.AXIS_LABELS['time'])
        ).det()
        S = solve(M)
        return (np.array([Abs(s) for s in S])>1).all()
    def _R_symbol(self, i,j,l):
        return Symbol("R{}.{}({})".format(i,j,l))
    def _gen_coefficients_cutoffs(self, discarded_c, discarded_u, discarded_psd, convergence_attempts):
        bad_sols = discarded_u + discarded_psd
        if (discarded_c >= convergence_attempts 
            or bad_sols >= convergence_attempts):
            if bad_sols > discarded_c:
                raise UnstableError(("No stable solution with a positive semi-definite covariance matrix"
                                     " found for above graph {}; tried {}x "
                                     "({} unstable, {} not positive semi-definite)").format(
                    id(self), bad_sols, discarded_u, discarded_psd))
            raise ConvergenceError(("No solution found for above graph {}; "
                                    "tried {}x").format(id(self), discarded_c))

        _progress_message("Attempt {}/{}".format(discarded_c+discarded_u+1, 
                                                       convergence_attempts))
    def _summarize_discarded_solutions(self, discarded_c, discarded_u, discarded_psd):
        new_msg = None
        if discarded_u + discarded_c + discarded_psd> 0:
            if discarded_u + discarded_psd == 0:
                new_msg = "discarded {} solution{} that did not converge".format(
                    discarded_c, 's' if discarded_c>1 else '')
            elif discarded_c == 0:
                new_msg = "discarded {} unstable solution{} and {} solution{} producing a covariance matrix that was not positive semi-definite".format(
                    discarded_u, 's' if discarded_u!=1 else '', discarded_psd, 's' if discarded_psd!=1 else '')
            else:
                new_msg = "discarded {} solutions: {} unstable, {} producing a non-positive semi-definite covariance matrix, and {} that did not converge".format(
                    discarded_u+discarded_c+discarded_psd, discarded_u, discarded_psd, discarded_c)
        _clear_progress_message(new_msg)

    #generation helper funtions
    def _make_flat_cov_array(self):
        d = self.N*self.tau_max
        COV = (np.ones((d,d))*np.nan)
        def irange(l):
            return slice(self.N*l,self.N*(l+1))
        for l1 in range(self.tau_max):
            for l2 in range(self.tau_max):
                COV[irange(l1),irange(l2)]=self.cov[:,:,l2-l1]
        return COV
    def _gen_initial_values(self):
        COV = self._make_flat_cov_array()
        CSR = np.linalg.cholesky(COV)
        X = np.matmul(CSR, np.random.normal(size=(COV.shape[0], 1)))
        return X.reshape((self.N,self.tau_max), order='F')            
    
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
            isinstance(G, tsCausalModel) 
            and self.tau_max==G.tau_max 
            and super().__eq__(self, G)
        )

    def _ij_style(self, ref, sig):
        rad = sig*.4*4/self.N
        return "arc3,rad={}".format(rad)
