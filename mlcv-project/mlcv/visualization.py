import numpy as np
from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
import matplotlib.pyplot as plt

def render_pcl(data, indexing, ax=None):
    """renders the point cloud of a set of points with 
    a corresponding partition

    :param data: nx3 array containing points
    :type data: np.array[n,3,dtype=float]
    :param indexing: n array containing index for each point
    :type indexing: np.array[n,dtype=int]
    :param ax: axes object with "projection='3d'"
    :type ax: matplotlib.axes.Axes, optional
    """    
    _indexing = np.zeros(indexing.shape, dtype=np.int)
    index_mapping, index_ctr = {}, 0
    for vertex, index in enumerate(indexing):
        if index not in index_mapping:
            index_mapping[index] = index_ctr
            index_ctr += 1
        _indexing[vertex] = index_mapping[index]
    if ax is None:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
    markers = [".","x","o","1","2","3","4"]
    for part_idx in range(index_ctr+1):
        samples = data[_indexing == part_idx]
        ax.scatter(samples[:,0], 
                   samples[:,1], 
                   samples[:,2], 
                   marker=markers[part_idx%len(markers)])
    return ax


def render_stats(stats, axes=None):
    """renders statistics of a local-search run on a set of axes

    :param stats: list of dictionaries containing stats per iteration, where every 
    dictionary is built as follows: 
        {"rc": float,                                   # how much the cost was reduced 
         "t_neighbours": float,                         # time for neighbourhood generation
         "t_rcs_mean": float,                           # time reduced costs (average)
         "t_rcs_sum": float,                            # time reduced costs (sum)
         "t_rcs_std": float },                          # time reduced costs (standard deviation)
         "partcount": int }                             # number of distinct partitions
    OR pandas dataframe with column names that correspond to the above dictionary keys
    :type stats: List[Dict] or pd.DataFrame
    :param axes: sequence of 6 axis objects
    :type axes: List[matplotlib.axes.Axes], optional
    """
    # transform into pandas dataframe if necessary
    if not isinstance(stats, pd.DataFrame):
        tmp = pd.DataFrame()
        for row in stats:
            tmp = tmp.append(row, ignore_index=True)
        stats = tmp

    if axes is None:
        _,axes = plt.subplots(6,1,figsize=(6,12))

    ax1,ax2,ax3,ax4,ax5,ax6 = axes
    iters = np.arange(len(stats))
    if ax1 is not None:
        ax1.plot(iters, stats["rc"], label="reduced costs")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("costs")
        ax1.legend()
    if ax2 is not None:
        ax2.plot(iters, stats["t_neighbours"], label="time/neighbourhood-generation" )
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("time [ms]")
        ax2.legend()
    if ax3 is not None:
        ax3.plot(iters, stats["t_rcs_mean"], label="average time/reduced-costs-computation")
        ax3.fill_between(iters, 
                        stats["t_rcs_mean"]-stats["t_rcs_std"]*0.5, 
                        stats["t_rcs_mean"]+stats["t_rcs_std"]*0.5,  
                        color='grey', alpha=0.2)
        ax3.set_xlabel("iteration")
        ax3.set_ylabel("time [ms]")
        ax3.legend()
    if ax4 is not None:
        ax4.bar(iters, stats["t_neighbours"], width=1, label="time/neighbourhood-generation")
        ax4.bar(iters, stats["t_rcs_sum"], width=1, 
                bottom=stats["t_neighbours"],
                label="time/reduced-costs-computation")
        ax4.set_xlabel("iteration")
        ax4.set_ylabel("time [ms]")
        ax4.legend()
    if ax5 is not None:
        ax5.plot(iters, stats["partcount"], label="partitions")
        ax5.set_xlabel("iteration")
        ax5.set_ylabel("#partitions")
        ax5.legend()
    if ax6 is not None:
        ax6.plot(iters, (stats["t_neighbours"] + stats["t_rcs_sum"]).cumsum()/1000, 
                 label="cumulative time")
        ax6.set_xlabel("iteration")
        ax6.set_ylabel("time [s]")
        ax6.legend()
    return ax1,ax2,ax3,ax4,ax5,ax6

def render_overlap(indexing1, indexing2, ax=None,
                   indexing1name="indexing 1",
                   indexing2name="indexing 2"):
    """renders the overlap between two differnt indexings

    :param indexing1: first indexing
    :type indexing1: np.array[n,dtype=int]
    :param indexing2: second indexing
    :type indexing2: np.array[n,dtype=int]
    :param ax: matplotlib axes object (optional)
    :type ax: matplotlib.axes.Axes, optional
    :param indexing1name: name given for first indexing, defaults to "indexing 1"
    :type indexing1name: str, optional
    :param indexing2name: name given for second indexing, defaults to "indexing 2"
    :type indexing2name: str, optional
    """                   
    x1 = np.unique(indexing1)
    x2 = np.unique(indexing2)
    overlap_mat = np.zeros((x2.shape[0], x1.shape[0]))
    for idx1, idx2 in enumerate(x1):
        overlap = indexing2[ indexing1 == idx2 ]
        for idx_pred, part_pred in enumerate(x2):
            overlap_idx = np.sum(overlap == part_pred)
            overlap_mat[ idx_pred, idx1 ] = overlap_idx

    if ax is None:
        _, ax = plt.subplots(figsize=(6,6))
    
    ax.matshow(overlap_mat)
    for (i, j), z in np.ndenumerate(overlap_mat):
        ax.text(j, i, "%d" % z, ha='center', va='center', color="grey")
    ax.set_xlabel("Partition (%s)" % indexing1name)
    ax.set_ylabel("Partition (%s)" % indexing2name)
    return ax