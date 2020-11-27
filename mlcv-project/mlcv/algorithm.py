import numpy as np
from collections import defaultdict

def sample_noreplace(arr, n, k):
    # code from https://www.iditect.com/how-to/58566613.html
    idx = np.random.randint(len(arr) - np.arange(k), size=[n, k])
    for i in range(k-1, 0, -1):
        idx[:,i:] += idx[:,i:] >= idx[:,i-1,None]
    return np.array(arr)[idx]


def partial_cost(data, indexing, combinations, cf, cf_prime):
    part_us = indexing[combinations[:,0]]
    part_vs = indexing[combinations[:,1]]
    part_ws = indexing[combinations[:,2]]
    cf_prime_combs = combinations[(part_us == part_vs) & 
                                  (part_us == part_ws)]
    cf_combs = combinations[(part_us != part_vs) & 
                            (part_us != part_ws) & 
                            (part_vs != part_ws)]
    result = 0
    if cf_prime_combs.shape[0] > 0:
        result += cf_prime(data, cf_prime_combs).sum()
    if cf_combs.shape[0] > 0:
        result += cf(data, cf_combs).sum()
    
    if combinations.shape[0] > 0:
        return result/combinations.shape[0]
    else:
        return 0

def cost(data, indexing, cf, cf_prime):
    us = np.arange(data.shape[0])
    cartesian = np.array(np.meshgrid(us,us,us)).T.reshape(-1,3)
    cartesian = cartesian[ (cartesian[:,0] < cartesian[:,1]) & 
                           (cartesian[:,1] < cartesian[:,2]) ]
    return partial_cost(data, indexing, cartesian, cf, cf_prime)

def sample_cost(data, indexing, cf, cf_prime, uvw_sample_count=1000):
    points = sample_noreplace(np.arange(indexing.shape[0]), uvw_sample_count, 3)
    # pointsu = np.random.choice(         data.shape[0]-2, uvw_sample_count)
    # pointsv = __draw_uniform(pointsu+1, data.shape[0]-1, uvw_sample_count)
    # pointsw = __draw_uniform(pointsv+1, data.shape[0]  , uvw_sample_count)
    # points = np.vstack((pointsu,pointsv,pointsw)).T
    return partial_cost(data, indexing, points, cf, cf_prime)

def reduced_cost(data, indexing, cf, cf_prime, v, k):
    cpy = indexing.copy()
    cpy[v] = k
    vs = np.arange(data.shape[0])
    cartesian = np.array(np.meshgrid(v,vs,vs)).T.reshape(-1,3)
    cartesian = cartesian[ (cartesian[:,0] != cartesian[:,1]) & 
                           (cartesian[:,0] != cartesian[:,2]) & 
                           (cartesian[:,1]  < cartesian[:,2]) ]

    result = (partial_cost(data, indexing, cartesian, cf, cf_prime) -
              partial_cost(data,      cpy, cartesian, cf, cf_prime))
    return result

def sample_reduced_cost(data, indexing, cf, cf_prime, vs, ks, uw_sample_count=1000):
    # cpy_rp/indexing_rp.shape == (#moves, #vertices)
    indexing_rp = indexing.repeat(vs.shape[0]).reshape(indexing.shape[0],-1).T
    cpy_rp = indexing_rp.copy()
    cpy_rp[np.arange(vs.shape[0]), vs] = ks

    # pointsv/pointsu/pointsw.shape == (#moves, #samples)
    points = []
    for move in range(vs.shape[0]):
        pointsv = np.ones((uw_sample_count,1), dtype=np.int)*vs[move]
        allowed = np.arange(indexing.shape[0])[np.arange(indexing.shape[0]) != vs[move]]
        pointsuw = sample_noreplace(allowed, uw_sample_count, 2)
        points.append( np.concatenate((pointsv, pointsuw), axis=1) )
    points = np.stack(points, axis=0)

    results = []
    for move in range(vs.shape[0]):
        result = (partial_cost(data, indexing, points[move,:,:], cf, cf_prime) -
                  partial_cost(data, cpy_rp[move,:], points[move,:,:], cf, cf_prime))
        results.append(result)
    return results


def compute_index_counts(indexing):
    counts = np.zeros(indexing.shape)
    for _, index in enumerate(indexing):
        counts[index] += 1
    return counts


def neighbours(indexing, randomize=True):
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
    
    
    # then enter the main loop over all vertices.
    # this is equivalent to algorithm 1 in the 
    sequence = []
    for vertex,index in enumerate(indexing):
        for k in image:
            if k == index:
                continue
            if counts[index] > 1 or counts[k] > 1:
                if randomize:
                    sequence.append((vertex,k))
                else:
                    yield vertex, k
            elif len(U_k := ridx[k]) == 1 and vertex < U_k[0]:
                if randomize:
                    sequence.append((vertex,k))
                else:
                    yield vertex, k
        if empty is not None:
            if counts[index] > 2:
                if randomize:
                    sequence.append((vertex,empty))
                else:
                    yield vertex, empty
            elif len(U_v:=ridx[vertex]) == 2 and vertex==U_v[0]:
                if randomize:
                    sequence.append((vertex,empty))
                else:
                    yield vertex, empty

    if randomize:
        np.random.shuffle(sequence)
        for el in sequence:
            yield el