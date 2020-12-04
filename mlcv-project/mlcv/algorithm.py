import numpy as np
from collections import defaultdict
from time import time

def sample_noreplace(arr, n, k):
    """samples n k-ary subsets from a given iterable (with length property)

    :param arr: the iterable sampled from
    :type arr: iterable/list/np.array/...
    :param n: amount of subsets that should be sampled
    :type n: int
    :param k: arity of every subset, i.e. the amount of elements
    :type k: int
    :return: array with subsets
    :rtype: np.array[n,k]
    """    
    # code from https://www.iditect.com/how-to/58566613.html
    idx = np.random.randint(len(arr) - np.arange(k), size=[n, k])
    for i in range(k-1, 0, -1):
        idx[:,i:] += idx[:,i:] >= idx[:,i-1,None]
    return np.array(arr)[idx]


def partial_cost(data, indexing, combinations, cf, cf_prime):
    """computes the costs of a dataset with respect to some indexing,
    a selection of 3-ary subsets (of the dataset), and cost functions cf
    and cf_prime

    :param data: the dataset containing 3d-points for n vertices
    :type data: np.array[n,3,dtype=float]
    :param indexing: indexing that maps vertices to indices
    :type indexing: np.array[n,dtype=int]
    :param combinations: selection of m 3-ary subsets of vertices
    :type combinations: np.array[m,3,dtype=int]
    :param cf: cost function c that computes the cost of a dataset 
        w.r.t. a selection of 3-ary subsets
    :type cf: function with cf : np.array[n,3,dtype=float] x np.array[m,3,dtype=int] -> float
    :param cf_prime: cost function c' that computes the cost of a dataset 
        w.r.t. a selection of 3-ary subsets
    :type cf_prime: function cf_prime : np.array[n,3,dtype=float] x np.array[m,3,dtype=int] -> float
    :return: the costs of the dataset w.r.t. the indexing and the subset-selection
    :rtype: float
    """
    # selects indices that are mapped to by u,v and w vertices respectively
    part_us = indexing[combinations[:,0]]
    part_vs = indexing[combinations[:,1]]
    part_ws = indexing[combinations[:,2]]
    
    result = 0
    # cf_prime is applied to cf_prime_combs and cf to cf_combs respectively
    # the result is the sum over both functions
    if cf_prime is not None:
        # select pairings where u,v and w vertices share the same index (i.e. same partition)
        cf_prime_combs = combinations[(part_us == part_vs) & (part_us == part_ws)]
        if cf_prime_combs.shape[0] > 0:
            result += cf_prime(data, cf_prime_combs).sum()
    if cf is not None:
        # select pairings where u,v and w vertices have 
        # all distinct indices (i.e. distinct partitions) 
        cf_combs = combinations[(part_us != part_vs) & (part_us != part_ws) & (part_vs != part_ws)]
        if cf_combs.shape[0] > 0:
            result += cf(data, cf_combs).sum()
    
    # divide by the overall amount of selected pairings
    return result/combinations.shape[0]


def cost(data, indexing, cf, cf_prime, uvw_sample_count=None):
    """computes the cost of a dataset with respect to some indexing and 
    cost functions cf, cf_prime

    :param data: the dataset containing 3d-points for n vertices
    :type data: np.array[n,3,dtype=float]
    :param indexing: indexing that maps vertices to indices
    :type indexing: np.array[n,dtype=int]
    :param cf: cost function c that computes the cost of a dataset 
        w.r.t. a selection of 3-ary subsets
    :type cf: function with cf : np.array[n,3,dtype=float] x np.array[m,3,dtype=int] -> float
    :param cf_prime: cost function c' that computes the cost of a dataset 
        w.r.t. a selection of 3-ary subsets
    :type cf_prime: function cf_prime : np.array[n,3,dtype=float] x np.array[m,3,dtype=int] -> float
    :return: the costs of the dataset w.r.t. the indexing
    :rtype: float
    """
    points = None
    if uvw_sample_count is None:
        us = np.arange(data.shape[0])
        # compute all possible 3-ary pairings and remove duplicates (if interpreted as sets)
        points = np.array(np.meshgrid(us,us,us)).T.reshape(-1,3)
        points = points[ (points[:,0] < points[:,1]) & 
                         (points[:,1] < points[:,2]) ]
    else:
        points = sample_noreplace(np.arange(indexing.shape[0]), uvw_sample_count, 3)
    return partial_cost(data, indexing, points, cf, cf_prime)


def reduced_cost(data, indexing, cf, cf_prime, v, k, uw_sample_count=None):
    """computes the reduced cost of a dataset with respect to a given indexing if vertex v is moved to index k

    :param data: the dataset containing 3d-points for n vertices
    :type data: np.array[n,3,dtype=float]
    :param indexing: indexing that maps vertices to indices
    :type indexing: np.array[n,dtype=int]
    :param cf: cost function c that computes the cost of a dataset 
        w.r.t. a selection of 3-ary subsets
    :type cf: function with cf : np.array[n,3,dtype=float] x np.array[m,3,dtype=int] -> float
    :param cf_prime: cost function c' that computes the cost of a dataset 
        w.r.t. a selection of 3-ary subsets
    :type cf_prime: function cf_prime : np.array[n,3,dtype=float] x np.array[m,3,dtype=int] -> float
    :param v: the vertex that is selected
    :type v: int
    :param k: the index the vertex is moved to
    :type k: int
    :param uw_sample_count: if None, the reduced costs are computed explicitly for all pairings.
        otherwise, an amount of uw_sample_count of random vertices is selected and the sample mean
        of the costs is computed
    :type uw_sample_count: None or int
    :return: the reduced costs of the dataset w.r.t. the indexing and movement of v to k
    :rtype: float
    """
    # copy indexing and move v to k
    cpy = indexing.copy()
    cpy[v] = k
    vs = np.arange(data.shape[0])
    points = None
    if uw_sample_count is None:
        # compute all possible 3-ary subsets that contain v
        points = np.array(np.meshgrid(v,vs,vs)).T.reshape(-1,3)
        points = points[ (points[:,0] != points[:,1]) & 
                         (points[:,0] != points[:,2]) & 
                         (points[:,1]  < points[:,2]) ]
    else:
        # sample 3-ary subsets that contain v
        pointsv = np.ones((uw_sample_count,1), dtype=np.int)*v
        allowed = np.arange(indexing.shape[0])[np.arange(indexing.shape[0]) != v]
        pointsuw = sample_noreplace(allowed, uw_sample_count, 2)
        points = np.concatenate((pointsv, pointsuw), axis=1)
    # the result is the difference between the original indexing
    # and the indexing after the move-operation
    result = (partial_cost(data, indexing, points, cf, cf_prime) -
              partial_cost(data,      cpy, points, cf, cf_prime))
    return result


def compute_index_counts(indexing):
    """computes the amount of vertices that are mapped to each index

    :param indexing: indexing that maps vertices to indices
    :type indexing: np.array[n,dtype=int]
    :return: array that contains the vertex-count per index
    :rtype: np.array[n,dtype=int]
    """    
    counts = np.zeros(indexing.shape)
    for index in indexing:
        counts[index] += 1
    return counts


def compute_probability_weights(candidate_vertices, indexing, counts, image, ridx):
    countsgeq2 = len([c for c in counts if c >= 2])
    others = np.zeros_like(indexing)
    accsum = 0
    for vertex in candidate_vertices:
        index = indexing[vertex]
        if counts[index] == 1:
            others[vertex] = accsum
            accsum += 1
    others = np.max(others) - others

    weights = np.zeros(len(candidate_vertices))
    for i, vertex in enumerate(candidate_vertices):
        index = indexing[vertex]
        if counts[index] >= 3:
            weights[i] = len(image)
        elif counts[index] >= 2:
            U_v = ridx[index]
            offset = 1 if vertex==U_v[0] else 0
            weights[i] = len(image) - 1 + offset
        elif counts[index] == 1:
            weights[i] = countsgeq2 + others[vertex]
    return weights/np.sum(weights)

def neighbours(indexing, taboo=None, random_stream=True):
    """enumerates the neighbours of an indexing. the neighbourhood is defined
    by the set of possible moves of vertices to other indices.

    :param indexing: the indexing
    :type indexing: np.array[n,dtype=int]
    :param randomize: whether the neighbours should be returned in 
        randomized order, defaults to True
    :type randomize: bool, optional
    :yield: iterator over all neighbours, where a neighbour is encoded as a vertex/index-pair
    :rtype: iterator[Tuple[int,int]]
    """    
    # compute the number of vertices per index
    counts = compute_index_counts(indexing)
    # compute ridx := idx^-1 only when it's relevant: i.e. for sets
    # with size 2 or 1. the remainder can be done via counts
    ridx = defaultdict(list)
    for vertex, index in enumerate(indexing):
        if counts[index] == 1 or counts[index] == 2:
            # the lower vertex will always be on index 0, the greater on 1
            ridx[index].append(vertex)
    # find, if possible, an empty index
    empty = None
    for index,count in enumerate(counts):
        if count == 0:
            empty = index
            break
    # compute the image of indexing
    image = [idx for idx,count in enumerate(counts) if count != 0]
    
    def move_candidates(vertex, index, image, ridx, counts, empty):
        for k in image:
            if k == index:
                continue
            if counts[index] > 1 or counts[k] > 1:
                yield k
            elif len(U_k := ridx[k]) == 1 and vertex < U_k[0]:
                yield k
        if counts[index] > 2 or (len(U_v:=ridx[vertex]) == 2 and vertex==U_v[0]):
            yield empty

    candidate_vertices = [v for v in range(indexing.shape[0]) if taboo is None or not taboo[v]] 
    if random_stream:
        pweights = compute_probability_weights(candidate_vertices, indexing, counts, image, ridx)
        while True:
            vertex = np.random.choice(candidate_vertices, p=pweights)
            index = indexing[vertex]
            k_candidates = list(move_candidates(vertex, index, image, ridx, counts, empty))
            k = k_candidates[ np.random.choice(len(k_candidates)) ]
            yield vertex, k
    else:
        for vertex in candidate_vertices:
            index = indexing[vertex]
            for k in move_candidates(vertex, index, image, ridx, counts, empty):
                yield vertex, k

def best_move(data, 
              indexing, 
              cf, cf_prime, 
              N=20, 
              M=30,
              taboo=None):
    stats = {}

    timer = time()
    ns = []
    for i,(v,k) in enumerate(neighbours(indexing, taboo=taboo, random_stream=True)):
        if i >= N:
            break
        ns.append((v,k))
    stats["n_neighbours"] = len(ns)
    stats["t_neighbours"] = 1000*(time() - timer)

    dt_rcs = []
    bestpair, best_rcost = None, None
    for v,k in ns:
        timer = time()
        rc = reduced_cost(data, indexing, cf, cf_prime, v, k, uw_sample_count=M)
        dt_rcs.append(1000*(time() - timer))
        if bestpair is None or rc > best_rcost:
            bestpair = v,k
            best_rcost = rc
    stats["t_rcs"] = {"mean": np.mean(dt_rcs), "std": np.std(dt_rcs), "sum": np.sum(dt_rcs)}
    stats["rc"] = best_rcost
    return bestpair, best_rcost, stats

def greedy_search(data, indexing, cf, cf_prime, stop=1000, N=None, M=None):
    if isinstance(stop,int):
        nstop = stop
        stop = lambda i,indx: i>=nstop

    i = 0
    while not stop(i,indexing):
        (v,k), c, stats_bm = best_move(data, indexing, cf, cf_prime, N=N, M=M)
        if c > 0:
            indexing[v] = k
            yield indexing, v, k, stats_bm
        else:
            yield indexing, None, None, stats_bm
        i += 1


def taboo_search(data, indexing, cf, cf_prime, taboo_dur=20, stop=1000, N=None, M=None):
    if isinstance(stop,int):
        nstop = stop
        stop = lambda i,indx: i>=nstop

    i = 0
    taboo = np.zeros(indexing.shape)
    while not stop(i,indexing):
        (v,k), c, stats_bm = best_move(data, indexing, cf, cf_prime, N=N, M=M, taboo=(taboo!=0))
        
        taboo[v] += taboo_dur+1
        taboo -= 1
        taboo = taboo.clip(0, taboo+1)
        
        if c > 0:
            indexing[v] = k
            yield indexing, v, k, stats_bm
        else:
            yield indexing, None, None, stats_bm
        i += 1