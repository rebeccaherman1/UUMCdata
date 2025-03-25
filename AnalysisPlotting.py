from DataGeneration import *
import scipy.stats

class SortabilityPlotting():
    @classmethod
    def plot_sortability_dist(cls, sort_list, labels, title, 
                              legend_title="Edge Likelihood", fontsize=None):
        for i, r2sort in enumerate(sort_list):
            _ = plt.hist(r2sort, density=True, cumulative=False, 
                         bins=20, alpha=.5, label=labels[i])
        plt.xlim([0,1])
        plt.legend(title=legend_title, loc="upper right", 
                   fontsize=fontsize, title_fontsize=fontsize)
        plt.title(title)
    @classmethod
    def plot_box(cls, d, pos, lc, fc, label, widths=.5):
        bp1 = plt.boxplot(d, positions = pos, manage_ticks = False, 
                          patch_artist=True, showfliers=False, widths=widths)
        for element in bp1.keys():
                plt.setp(bp1[element], color=lc)
        for patch in bp1['boxes']:
                patch.set(facecolor=fc)
        bp1['boxes'][0].set_label(label)
    @classmethod
    def plot_stat_dist(cls, r2s, title):
        plt.xlim([0,1])
        for i in range(r2s.shape[1]):
            _ = plt.hist(r2s[:,i], density=True, cumulative=False, 
                         bins=20, alpha=.5, label=str(i))
        plt.legend(title="Node", fontsize='small')
        plt.title(title)
        plt.xlabel("R2 score")

#Making analysis plots
def sortability_compare_collider_confounder(Ns = [i for i in range(3,22)], T = 500, B = 500):
    def make_collider_adj(N):
        A = np.zeros((N,N))
        A[:-1,-1] = np.ones((N-1,))
        return (A != 0)*1
    def make_confounder_adj(N):
        A = np.zeros((N,N))
        A[0,1:] = np.ones((N-1,))
        return (A != 0)*1
    adj_type_dict = {"collider": make_collider_adj, "confounder": make_confounder_adj}
        
    r2dict = {}
    keys = list(adj_type_dict.keys())
    
    for k in adj_type_dict.keys():
        print("\t{}s:".format(k))
        r2s  = np.empty((B,2, len(Ns)))*np.nan
        for n, N in enumerate(Ns):
            print('N = {}:'.format(N))
            Gs = Graph.gen_dataset(N,T,B,init_args = {'init_type': 'specified', 'init': adj_type_dict[k](N)})
            r2s[:,:,n] = np.array([g.data.R2()[[0,-1]] for g in Gs]).squeeze()
        r2dict[k] = r2s
    plt.figure(figsize=(4,3))
    SortabilityPlotting.plot_box(r2dict["confounder"][:,0,:], np.array(Ns)-.25-1, "blue", "skyblue", "confounder")
    SortabilityPlotting.plot_box(r2dict["collider"][:,1,:], np.array(Ns)+.25-1, "red", "peachpuff", "collider")
    plt.legend(loc="lower right")
    plt.title("R2 scores for the hub node")
    plt.xlabel("Number of Parents/Children")
    plt.savefig(fname="HubR2", bbox_inches='tight')

def sortability_compare_duple_types(O = 500, B = 50000, tau_max=1):
    adj_types = {}
    adj_types["one-way"] =np.array([[[0,1],[1,0]],[[0,0],[0,1]]]); 
    if tau_max is not None:
        if tau_max!=1:
            raise ValueError("this function only supports tau_max=1")
    
    r2dict = {}
    keys = list(adj_types.keys())

    for k in keys:
        print("{}s:".format(k))
        Gs = tsGraph.gen_dataset(N=2, tau_max=tau_max, T=O, B=B, init_args={'init_type': 'specified', 'init': adj_types[k]})
        r2dict[k] = np.array([g.data.R2(tau_max=tau_max) for g in Gs])
        

    k = r2dict.keys()
    F = plt.figure(figsize=(4,4))
    i=1
    for t, v in r2dict.items():
        SortabilityPlotting.plot_stat_dist(v, t)
        i+=1
    yl_max = 0
    #plt.subplot(2, len(k), 2*len(k))
    #SortabilityPlotting.plot_sortability_dist([r2sort, r2sort2], ["original", "modified"], "Random Graph", legend_title=None)
    #plt.xlabel("R2-sortability")
    #plt.legend(loc=(.6,.6))
    F.tight_layout()
    #plt.savefig(fname="triples")

def sortability_compare_triple_types(O = 500, B = 50000, tau_max=None):
    adj_types = {}
    adj_types["collider"] =np.array([[0,0,1],[0,0,1],[0,0,0]]); 
    adj_types["chain"] =np.array([[0,1,0],[0,0,1],[0,0,0]]); 
    adj_types["confounder"] =np.array([[0,1,1],[0,0,0],[0,0,0]]); 
    if tau_max is not None:
        if tau_max!=1:
            raise ValueError("this function only supports tau_max=1")
        auto = np.diag(np.ones((3,)))
        auto.shape+=(1,)
        for v in adj_types.values():
            v.shape+=(1,) 
        adj_types = {k:np.append(v, auto, axis=2) for k, v in adj_types.items()}
    
    r2dict = {}
    keys = list(adj_types.keys())

    for k in keys:
        print("{}s:".format(k))
        if tau_max is None:
            Gs = Graph.gen_dataset(N=3, O=O, B=B, init_args={'init_type': 'specified', 'init': adj_types[k]})
            r2dict[k] = np.array([g.data.R2() for g in Gs])
        else:
            Gs = tsGraph.gen_dataset(N=3, tau_max=tau_max, T=O, B=B, init_args={'init_type': 'specified', 'init': adj_types[k]})
            r2dict[k] = np.array([g.data.R2(tau_max=tau_max) for g in Gs])
        

    k = r2dict.keys()
    F = plt.figure(figsize=(7,4.5))
    ax = F.subplots(2,len(k))
    i=1
    for t, v in r2dict.items():
        plt.subplot(2, len(k), i)
        SortabilityPlotting.plot_stat_dist(v, t)
        i+=1
    i=0
    for t in list(reversed(list(r2dict.keys()))):
        plt.subplot(2,len(k),5)
        _ = plt.hist(r2dict[t][:,i], density=True, cumulative=False, bins=20, alpha=.5, label=t)
        plt.subplot(2,len(k),4)
        _ = plt.hist(r2dict[t][:,(i+1)%3], density=True, cumulative=False, bins=20, alpha=.5, label=t)
        i+=1
    ts = ["lowest", "highest"]
    yl_max = 0
    for p in [4,5]:
        yl_max = max(yl_max, plt.ylim()[1])
    for p in [4, 5]:
        plt.subplot(2,len(k),p)
        plt.xlim([0,1])
        plt.ylim([0, yl_max])
        plt.legend(fontsize='small')#loc = locs[p-1])
        plt.title(ts[p-4]+"-scoring variable")
        plt.xlabel("R2 score")
    #plt.subplot(2, len(k), 2*len(k))
    #SortabilityPlotting.plot_sortability_dist([r2sort, r2sort2], ["original", "modified"], "Random Graph", legend_title=None)
    #plt.xlabel("R2-sortability")
    #plt.legend(loc=(.6,.6))
    F.tight_layout()
    #plt.savefig(fname="triples")

def sortability_compare_p(N=20, ps=[i/10 for i in range(1,11)], O=100, B=5000, tau_max=None, further_init_args={}, coef_args={}):#'convergence_attempts': 5}):
    p_dict = {}
    for p in ps:
        print("p = {}:".format(p))
        further_init_args['p']=p
        if tau_max is None:
            Gs = Graph.gen_dataset(N=N, O=O, B=B, init_args=further_init_args, coef_args=coef_args)
        else:
            Gs = tsGraph.gen_dataset(N=N, tau_max=tau_max, T=O, B=B, init_args=further_init_args, coef_args=coef_args)
        varsort = [g.sortability() for g in Gs]
        r2sort2  = [g.sortability('R2') for g in Gs]
        p_dict[p] = (varsort, r2sort2)
    ds = []
    for i in range(2):
        for func in [np.nanmean, scipy.stats.skew]:
            ds += [np.array([scipy.stats.bootstrap((v[i],), func, n_resamples=99).bootstrap_distribution for _, v in p_dict.items()])]
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    SortabilityPlotting.plot_sortability_dist([v[0] for _, v in p_dict.items()], [k for k in p_dict.keys()], "Varsortability", fontsize="x-small")
    plt.xlabel("Sortability")
    plt.ylabel("Probability")
    plt.subplot(1,3,2)
    SortabilityPlotting.plot_sortability_dist([v[1] for _, v in p_dict.items()], [k for k in p_dict.keys()], "R2-sortability", fontsize="x-small")
    plt.xlabel("Sortability")
    plt.ylabel("Probability")
    #_ = plt.hist(varsort, density=True, cumulative=False)#, bins=3)"#int(scipy.special.factorial(N)))
    plt.subplot(1,3,3)
    names = ["varsortability mean", "varsortability skew", "R2-sortability mean", "R2-sortability skew"]
    colors = [("blue", "skyblue"), ("darkturquoise", "powderblue"), ("red", "peachpuff"), ("lightcoral", "mistyrose")]
    plt.hlines([0, .5], 0, 1, colors=None, linestyles='--', label='_nolegend_')
    for j in range(4):
        SortabilityPlotting.plot_box(ds[j].T, np.array(list(p_dict.keys())), colors[j][0], colors[j][1], names[j], widths=.05)
    plt.legend(loc=(.01, .6), fontsize="xx-small")
    plt.xlabel("Edge Likelihood")
    plt.title("Sortability Distribution Moments")
    #plt.savefig(fname="SortabilityStats", bbox_inches='tight')