import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame as df

class Data(object):
    '''
    Data Object. 

    Attributes
    __________
    N : int
        Number of variables
    P : int
        Number of observations
    data : N x P numpy array
        data, where different rows are different variables and the columns are samples.
        May be indexed by calling self[i,j].
    labels : list of strings
        Names of the variables
    print_labels : array of strings
        a version of LABELS that is better for printing to stdout

    Class Constants
    _______________
    AXIS_LABELS : dictionary containing the axis index of the 'variables' and 'observations'
    analysis_options : list of built-in analysis functions

    Analysis Functions
    __________________
    var : returns the variance of each variable in an array
    R2 : returns the R2 regression score of each variable on all the others

    Export Functions
    ________________
    to_dataframe : casts the Data object as a pandas DataFrame
    '''
    AXIS_LABELS = {'variables': 0, 'observations': 1}
    analysis_options = ['var', 'R2']
        
    def __init__(self, N, P, labels, data, print_labels=None):
        '''
        Parameters
        __________
        N : int
            Number of variables
        P : int
            Number of samples
        labels : list of strings
            Names of the variables
        data : N x P numpy array
            data, where different rows are different variables and the columns are samples.
            May be indexed by calling self[i,j].
        print_labels : array of strings (optional; default = labels)
            A version of labels that does not require Latex rendering
        '''
        s = data.shape
        if (s[self.AXIS_LABELS['variables']]!=N) or (s[self.AXIS_LABELS['observations']]!= P):
            raise ValueError("data must be an N x O array")
        self.N = N
        self.P = P
        self.labels = labels
        self.data = data
        if print_labels is None:
            self.print_labels=self.labels
        else:
            self.print_labels=print_labels
        
    def var(self):
        '''Variance of each variable over time'''
        return np.var(self.data, axis=self.AXIS_LABELS['observations'], keepdims=True)
        
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
            norm = np.var(Xi, dtype=np.float32)*float(Xi.shape[0])
            if resid[0]>norm:
                R2s[i]=0
            else:
                R2s[i] = 1.0 - resid[0]/norm
        return R2s

    def to_dataframe(self):
        '''Casts the current Data object as a pandas DataFrame'''
        return _to_dataframe(interactive=True)

    def _get_regressors(self):
        return self.data

    def _remove_regressors(self, X, i):
        return X[np.arange(self.N) != i, :]

    def _to_dataframe(self, interactive=False):
        if interactive:
            labels = self.labels
        else:
            labels = self.print_labels
        return df(self.data.T, columns=self.print_labels)

    def __getitem__(self, tpl):
        return self.data.__getitem__(tpl)

    def _repr_html_(self):
        sns.set(style='ticks', font_scale=.7)
        g = sns.PairGrid(df({l: self.data[i] for i, l in enumerate(self.labels)}), 
                         diag_sharey=False, height=1, aspect=1.15, layout_pad=0.1)
        g.map_upper(sns.scatterplot, size=.5)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.kdeplot)
        return "Data object at {}".format(hex(id(self)))

    def __str__(self):
        return str(self._to_dataframe())

    def __repr__(self):
        return "Data object at {}: ".format(hex(id(self)))+'\n'+str(self)

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
    analysis_options : list of built-in analysis functions

    Analysis Functions
    __________________
    var : returns the variance of each variable in an array
    R2 : returns the R2 regression score of each variable on all the others
    R2_summary : regresses each variable on parent processes but not own past
    """
    AXIS_LABELS = Data.AXIS_LABELS
    AXIS_LABELS['time'] = AXIS_LABELS['observations']
    analysis_options = Data.analysis_options + ['R2_summary']

    def __init__(self, N, T, labels, data, print_labels=None):
        super().__init__(N, T, labels, data, print_labels=print_labels)

    def _get_regressors(self):
        X = self[:,self.tau_max:]
        for tau in range(1, self.tau_max+1):
            X = np.append(X, self[:,(self.tau_max-tau):-tau], axis=TimeSeries.AXIS_LABELS['variables'])
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

    def _repr_html_(self):
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