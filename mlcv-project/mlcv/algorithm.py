import numpy as np
from time import time
import itertools
import random

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


def compute_probability_weights(indexing, 
                                counts, 
                                image, 
                                binary_set_mappings):
    """computes an array that contains the probability for each vertex 
    of being subject of a move-operation w.r.t. the given indexing

    :param indexing: the indexing
    :type indexing: np.array[n,dtype=int]
    :param counts: array containing number of vertices/index
    :type counts: np.array[n,dtype=int]
    :param image: list containing non-empty indices
    :type image: list
    :param binary_set_mappings: result of `compute_binary_set_mappings`
    :type binary_set_mappings: np.array[n,dtype=int]
    :return: the probabilities for each vertex
    :rtype: np.array[n,dtype=float]
    """    
    S_w_cardinalities = np.zeros_like(indexing)

    countsgeq2 = sum(c>=2 for c in counts) # compute amount of indices that have count>=2
    countseq1 = [v for v in range(indexing.shape[0]) if counts[indexing[v]]==1]
    K_cardinalities = np.zeros_like(indexing)
    for card,w in enumerate(countseq1[::-1]):
        K_cardinalities[w] = card

    for w,index in enumerate(indexing):
        if counts[index] >= 3:
            S_w_cardinalities[w] = len(image)
        elif counts[index] == 2:
            offset = 1 if w==binary_set_mappings[index] else 0
            S_w_cardinalities[w] = len(image) - 1 + offset
        elif counts[index] == 1:
            S_w_cardinalities[w] = countsgeq2 + K_cardinalities[w]

    return S_w_cardinalities/np.sum(S_w_cardinalities)

def find_empty(counts):
    """returns an index with not assigned vertices, if there is one

    :param counts: array containing the amount of vertices/index
    :type counts: np.array[n,dtype=int]
    :return: index with no assigned vertices or None
    :rtype: int/None
    """    
    for index,count in enumerate(counts):
        if count == 0:
            return index
    return None

def compute_binary_set_mappings(indexing, counts):
    """computes an array that contains a mapping from indices to vertices if
    these indices carry only two vertices. in particular, every index is only
    mapped to the lower index

    :param indexing: the indexing
    :type indexing: np.array[n,dtype=int]
    :param counts: the amount of vertices/index
    :type counts: np.array[n,dtype=int]
    :return: an array A that maps indices to vertices, such that
        A[i]=-1 if counts[i]!=2 and 
        A[i]= v if counts[i]==2, and indexing(v)=i, [v]_indexing={v,u} and v < u 
    :rtype: np.array[n,dtype=int]
    """    
    ret = np.zeros_like(indexing)-1
    for vertex,index in enumerate(indexing):
        if counts[index] == 2:
            if ret[index] == -1:
                ret[index] = vertex
    return ret

def compute_unary_set_mappings(indexing, counts):
    """computes an array that contains a mapping from indices to vertices if 
    these indices carry only one vertex

    :param indexing: the indexing
    :type indexing: np.array[n,dtype=int]
    :param counts: the amount of vertices/index
    :type counts: np.array[n,dtype=int]
    :return: an array A that maps indices to vertices, such that
        A[i]=-1 if counts[i]!=1 and 
        A[i]= v if indexing(v)=i, [v]_indexing={v} 
    :rtype: np.array[n,dtype=int]
    """    
    ret = np.zeros_like(indexing)-1
    for vertex,index in enumerate(indexing):
        if counts[index] == 1:
            ret[index] = vertex
    return ret

def neighbours(indexing, random_stream=None):
    """enumerates the neighbours of an indexing. the neighbourhood is defined
    by the set of possible moves of vertices to other indices

    :param indexing: the indexing
    :type indexing: np.array[n,dtype=int]
    :param random_stream: amount of neighbours that should be sampled from the neighbourhood.
        if None, then every neighbour is returned
    :type random_stream: int/None
    :yield: iterator over the neighbours, where a neighbour is encoded as a vertex/index-pair
    :rtype: iterator[Tuple[int,int]]
    """

    # pre-compute some necessary values
    counts = compute_index_counts(indexing)
    binary_sm = compute_binary_set_mappings(indexing, counts)
    unary_sm = compute_unary_set_mappings(indexing, counts)
    empty = find_empty(counts)
    image = [idx for idx,count in enumerate(counts) if count != 0]
    
    def candidates(vertex, index, image, binary_sm, unary_sm, counts, empty):
        """generates the set of possible target indices for a given vertex

        :param vertex: the vertex
        :type vertex: int
        :param index: the current index of the vertex
        :type index: int
        :param image: the image of the current indexing
        :type image: list
        :param binary_sm: result of `compute_binary_set_mappings`
        :type binary_sm: np.array[n,dtype=int]
        :param unary_sm: result of `compute_unary_set_mappings`
        :type unary_sm: np.array[n,dtype=int]
        :param counts: number of vertices/index
        :type counts: np.array[n,dtype=int]
        :param empty: an index that is assigned no vertex, None is also allowed
        :type empty: int/None
        :yield: iterator over target indices
        :rtype: Iterator[int]
        """
        for k in image:
            if k == index:
                continue
            if counts[index] > 1 or counts[k] > 1:
                yield k
            elif vertex < unary_sm[k]: # implicitly: counts[index]==1 and counts[k]==1
                yield k
        if counts[index] > 2 or (counts[index] == 2 and vertex==binary_sm[index]):
            yield empty
 
    if random_stream is not None:
        # Random Move-Enumeration
        pweights = compute_probability_weights(indexing, counts, image, binary_sm)
        vertices = np.random.choice(indexing.shape[0], random_stream, p=pweights)
        for vertex in vertices:
            index = indexing[vertex]
            ks = list(candidates(vertex, index, image, binary_sm, unary_sm, counts, empty))
            k = random.choice(ks)
            yield vertex, k
    else:
        # Move-Enumeration
        for vertex, index in enumerate(indexing):
            for k in candidates(vertex, index, image, binary_sm, unary_sm, counts, empty):
                yield vertex, k

def best_move(data, indexing, cf, cf_prime, N=20, M=30):
    stats = {}
    timer = time()
    ns = list(neighbours(indexing, random_stream=N))
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

    stats["t_rcs_mean"] = np.mean(dt_rcs)
    stats["t_rcs_std"]  = np.std(dt_rcs)
    stats["t_rcs_sum"]  = np.sum(dt_rcs)
    stats["rc"] = best_rcost
    stats["partcount"] = np.unique(indexing).shape[0]
    return bestpair, best_rcost, stats

def greedy_search(data, indexing, cf, cf_prime, stop=1000, N=None, M=None):
    if isinstance(stop,int):
        nstop = stop
        stop = lambda i,index: i>=nstop

    for i in itertools.count(0,1):
        if stop(i, indexing):
            break
        (v,k), c, stats_bm = best_move(data, indexing, cf, cf_prime, N=N, M=M)
        if c > 0:
            indexing[v] = k
            yield indexing, v, k, stats_bm
        else:
            yield indexing, None, None, stats_bm
            if M is None and N is None:
                break