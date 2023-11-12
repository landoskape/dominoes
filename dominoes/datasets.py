import itertools
import numpy as np
import scipy as sp
import torch
from torch import nn
import torch.nn.functional as F
from . import functions as df

# code for generating a hand
def randomDominoeHand(numInHand, listDominoes, highestDominoe, batch_size=1, null_token=True, available_token=True):
    """method to produce an encoded random hand"""
    numDominoes = len(listDominoes)
        
    # choose dominoes from the batch, and get their value (in points)
    selection = np.stack([np.random.choice(numDominoes, numInHand, replace=False) for _ in range(batch_size)])
    if available_token:
        available = np.random.randint(0, highestDominoe+1, batch_size)
    else:
        available = [None]*batch_size
    
    # create tensor representations
    input = torch.stack([df.twohotDominoe(sel, listDominoes, highestDominoe, available=ava,
                                          available_token=available_token, null_token=null_token, with_batch=False) 
                         for sel,ava in zip(selection, available)])
    return input, selection, available
    
def getBestLine(dominoes, selection, highestDominoe):
    bestSequence = []
    bestDirection = []
    for sel in selection:
        cBestSeq = []
        cBestDir = []
        cBestVal = []
        for available in range(highestDominoe+1):
            cseq, cdir = df.constructLineRecursive(dominoes, sel, available)
            cval = [np.sum(dominoes[cs]) for cs in cseq]
            cidx = max(enumerate(cval), key=lambda x: x[1])[0]
            cBestSeq.append(cseq[cidx])
            cBestDir.append(cdir[cidx])
            cBestVal.append(cval[cidx])

        cBestIdx = max(enumerate(cBestVal), key=lambda x: x[1])[0]
        bestSequence.append(cBestSeq[cBestIdx])
        bestDirection.append(cBestDir[cBestIdx])

    return bestSequence, bestDirection

def getBestLineFromAvailable(dominoes, selection, highestDominoe, available):
    bestSequence = []
    bestDirection = []
    for sel, ava in zip(selection, available):
        cseq, cdir = df.constructLineRecursive(dominoes, sel, ava)
        cval = [np.sum(dominoes[cs]) for cs in cseq]
        cidx = max(enumerate(cval), key=lambda x: x[1])[0]
        bestSequence.append(cseq[cidx])
        bestDirection.append(cdir[cidx])
    return bestSequence, bestDirection
    
def convertToHandIndex(selection, bestSequence):
    indices = []
    for sel,seq in zip(selection, bestSequence):
        # look up table for current selection
        elementIdx = {element:idx for idx, element in enumerate(sel)}
        indices.append([elementIdx[element] for element in seq])
    return indices
    
def padBestLine(bestSequence, max_output, ignore_index=-1):
    for bs in bestSequence:
        bs += [ignore_index]*(max_output-len(bs))
    return bestSequence

def generateBatch(highestDominoe, dominoes, batch_size, numInHand, return_target=True,
                  available_token=False, null_token=False, ignore_index=-1, return_full=False):
    input, selection, available = randomDominoeHand(numInHand, dominoes, highestDominoe, batch_size=batch_size, null_token=null_token, available_token=available_token)

    mask_tokens = numInHand + (1 if null_token else 0) + (1 if available_token else 0)
    mask = torch.ones((batch_size, mask_tokens), dtype=torch.float)

    if return_target:
        # then measure best line and convert it to a "target" array
        if available_token:
            bestSequence, bestDirection = getBestLineFromAvailable(dominoes, selection, highestDominoe, available)
        else:
            bestSequence, bestDirection = getBestLine(dominoes, selection, highestDominoe)

        # convert sequence to hand index
        iseq = convertToHandIndex(selection, bestSequence)
        # create target and append null_index for ignoring impossible plays
        null_index = ignore_index if not(null_token) else numInHand
        target = torch.tensor(np.stack(padBestLine(iseq, numInHand+(1 if null_token else 0), ignore_index=null_index)), dtype=torch.long)
    else:
        # otherwise set these to None so we can use the same return structure
        target, bestSequence, bestDirection = None, None, None
        
    if return_full:
        return input, target, mask, bestSequence, bestDirection, selection, available
    return input, target, mask



def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.

    Parameters:
        dists: distance matrix

    Returns:
        A tuple, (cost, path).

    Credit to: https://github.com/CarlEkerot/held-karp/blob/master/held-karp.py
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def get_path(batch):
    """
    for batch of (batch, num_cities, 2), returns shortest path using
    held-karp algorithm that starts closest to origin and is clockwise
    """
    path = []
    for xy in batch:
        closest_to_origin = np.argmin(np.sum(xy**2, axis=1))
        dists = sp.spatial.distance.squareform(sp.spatial.distance.pdist(xy))
        _, cpath = held_karp(dists)
        shift = {val: idx for idx, val in enumerate(cpath)}[closest_to_origin]
        cpath = np.roll(cpath, -shift)
        check_points = xy[cpath[[1,-1]]] # second point and last point - check for clockwise travel
        angles = np.arctan(check_points[:,1]/check_points[:,0])
        if angles[1] > angles[0]:
            cpath = np.flip(np.roll(cpath, -1))
        path.append(cpath)
    return path

def tsp_batch(batch_size, num_cities):
    xy = np.random.random((batch_size, num_cities, 2))
    input = torch.tensor(xy, dtype=torch.float)
    target = torch.tensor(np.stack(get_path(xy)), dtype=torch.long)
    return input, target

