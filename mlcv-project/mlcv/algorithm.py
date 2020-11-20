import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from itertools import combinations
from collections import defaultdict
import random

def partial_cost(data, part_col, u, v, w, cf, cf_prime):
    part = data.loc[[u,v,w], part_col]
    if part[u] == part[v] and part[u] == part[w]:
        return cf_prime(data,u,v,w)
    elif part[u] != part[v] and part[u] != part[w] and part[v] != part[w]:
        return cf(data,u,v,w)
    else:
        return 0

def cost(data, part_col, cf, cf_prime):
    result = 0
    bin_coeff = 0
    for u,v,w in combinations(range(data.shape[0]), 3):
        result += partial_cost(data, part_col, u, v, w, cf, cf_prime)
        bin_coeff += 1
    return result/bin_coeff

def sample_cost(data, part_col, cf, cf_prime, uvw_sample_count=1000):
    result = 0
    points = np.random.choice(data.shape[0], (uvw_sample_count, 3))
    samples = 0
    for row in points:
        u,v,w = row[0], row[1], row[2]
        if u != v and v != w and u != w:
            result += partial_cost(data, part_col, u, v, w, cf, cf_prime)
            samples += 1
    return result/samples

def move_diff_cost(data, part_col, cf, cf_prime, u, new_part_idx):
    result = 0
    bin_coeff = 0
    cpy = data.copy()
    cpy.loc[u, part_col] = new_part_idx
    for v,w in combinations([i for i in range(data.shape[0]) if i != u], 2):
        result += partial_cost(data, part_col, u, v, w, cf, cf_prime)
        result -= partial_cost(cpy, part_col, u, v, w, cf, cf_prime)
        bin_coeff += 1
    return result/bin_coeff

def sample_move_diff_cost(data, part_col, cf, cf_prime, u, new_part_idx, vw_sample_count=1000):
    result = 0
    points = np.random.choice(data.shape[0], (vw_sample_count, 2))
    samples = 0
    cpy = data.copy()
    cpy.loc[u, part_col] = new_part_idx
    for row in points:
        v, w = row[0], row[1]
        if u != v and v != w and u != w:
            result += partial_cost(data, part_col, u, v, w, cf, cf_prime)
            result -= partial_cost(cpy, part_col, u, v, w, cf, cf_prime)
            samples += 1
    return result/samples

def compute_index_counts(indexing):
    counts = np.zeros_like(indexing)
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
    
    it = list(enumerate(indexing))
    if randomize:
        random.shuffle(it)

    # then enter the main loop over all vertices.
    # this is equivalent to algorithm 1
    for vertex,index in it:
        for k in image:
            if k == index:
                continue
            if counts[index] > 1 or counts[k] > 1:
                yield vertex, k
            elif len(U_k := ridx[k]) == 1 and vertex < U_k[0]:
                yield vertex, k
        if empty is not None:
            if counts[index] > 2:
                yield vertex, empty
            elif len(U_v:=ridx[vertex]) == 2 and vertex==U_v[0]:
                yield vertex, empty
