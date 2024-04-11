from copy import copy
import itertools
from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy as sp
import torch


def get_dominoe_set(highest_dominoe, as_torch=False):
    """
    Create a list of dominoes in a set with highest value of <highest_dominoe>

    The dominoes are paired values (combinations with replacement) of integers
    from 0 to <highest_dominoe>. This method returns either a numpy array or a
    torch tensor of the dominoes as integers.

    The shape will be (num_dominoes, 2) where the first column is the first value
    of the dominoe, and the second column is the second value of the dominoe.

    args:
        highest_dominoe: the highest value of a dominoe
        as_torch: return dominoes as torch tensor if True, otherwise return numpy array

    returns:
        dominoes: an array or tensor of dominoes in the set
    """
    # given a standard rule for how to organize the list of dominoes as one-hot arrays, list the dominoes present in a one hot array
    array_function = torch.tensor if as_torch else np.array
    stack_function = torch.stack if as_torch else np.stack
    dominoe_set = [array_function(quake, dtype=int) for quake in itertools.combinations_with_replacement(np.arange(highest_dominoe + 1), 2)]
    return stack_function(dominoe_set)


def get_best_line(dominoes, selection, highest_dominoe, value_method="dominoe"):
    # check value method
    if not (value_method == "dominoe" or value_method == "length"):
        raise ValueError("did not recognize value_method, it has to be either 'dominoe' or 'length'")

    bestSequence = []
    bestDirection = []
    for sel in selection:
        cBestSeq = []
        cBestDir = []
        cBestVal = []
        for available in range(highest_dominoe + 1):
            cseq, cdir = construct_line_recursive(dominoes, sel, available)
            if value_method == "dominoe":
                cval = [np.sum(dominoes[cs]) for cs in cseq]
            else:
                cval = [len(cs) for cs in cseq]
            cidx = max(enumerate(cval), key=lambda x: x[1])[0]
            cBestSeq.append(cseq[cidx])
            cBestDir.append(cdir[cidx])
            cBestVal.append(cval[cidx])

        cBestIdx = max(enumerate(cBestVal), key=lambda x: x[1])[0]
        bestSequence.append(cBestSeq[cBestIdx])
        bestDirection.append(cBestDir[cBestIdx])

    return bestSequence, bestDirection


def get_best_line_from_available(dominoes, selection, available, value_method="dominoe"):
    # check value method
    if not (value_method == "dominoe" or value_method == "length"):
        raise ValueError("did not recognize value_method, it has to be either 'dominoe' or 'length'")

    bestSequence = []
    bestDirection = []
    for sel, ava in zip(selection, available):
        cseq, cdir = construct_line_recursive(dominoes, sel, ava)
        if value_method == "dominoe":
            cval = [np.sum(dominoes[cs]) for cs in cseq]
        else:
            cval = [len(cs) for cs in cseq]
        cidx = max(enumerate(cval), key=lambda x: x[1])[0]
        bestSequence.append(cseq[cidx])
        bestDirection.append(cdir[cidx])
    return bestSequence, bestDirection


def construct_line_recursive(dominoes, myHand, available, previousSequence=[], previousDirection=[], maxLineLength=None):
    # this version of the function uses absolute dominoe numbers, rather than indexing based on which order they are in the hand
    # if there are too many dominoes in hand, constructing all possible lines takes way too long...
    if (maxLineLength is not None) and (len(previousSequence) == maxLineLength):
        return [previousSequence], [previousDirection]

    assert type(previousSequence) == list and type(previousDirection) == list, "previous sequence and direction must be lists"
    if len(previousSequence) > 0:
        # if a previous sequence was provided, make sure the end of it matches what is defined as available
        assert (
            dominoes[previousSequence[-1]][0 if previousDirection[-1] == 1 else 1] == available
        ), "the end of the last sequence doesn't match what is defined as available!"

    # recursively constructs all possible lines given a hand (value pairs in list), an available value to play on, and the previous played/direction dominoe index sequences
    hand = dominoes[myHand]
    possiblePlays = np.where(np.any(hand == available, axis=1) & ~np.isin(myHand, previousSequence))[0]

    # if there are no possible plays, the return the finished sequence
    if len(possiblePlays) == 0:
        return [previousSequence], [previousDirection]

    # otherwise, make new lines for each possible play
    sequence = []
    direction = []
    for idxPlay in possiblePlays:
        # if the first value of the possible play matches the available, then play it in the forward direction
        if hand[idxPlay][0] == available:
            # copy previousSequence and previousDirection, append new play in forward direction to it
            cseq = copy(previousSequence)
            cseq.append(myHand[idxPlay])
            cdir = copy(previousDirection)
            cdir.append(0)
            # then recursively construct line from this standpoint
            cSequence, cDirection = construct_line_recursive(
                dominoes, myHand, hand[idxPlay][1], previousSequence=cseq, previousDirection=cdir, maxLineLength=maxLineLength
            )
            # once lines are constructed, add them all to "sequence" and "direction", which will be a list of lists of all possible sequences
            for cns, cnd in zip(cSequence, cDirection):
                sequence.append(cns)
                direction.append(cnd)

        # if the second value of the possible play matches the available and it isn't a double, then play it in the reverse direction (all same except direction and next available)
        if (hand[idxPlay][0] != hand[idxPlay][1]) and (hand[idxPlay][1] == available):
            cseq = copy(previousSequence)
            cseq.append(myHand[idxPlay])
            cdir = copy(previousDirection)
            cdir.append(1)
            cSequence, cDirection = construct_line_recursive(
                dominoes, myHand, hand[idxPlay][0], previousSequence=cseq, previousDirection=cdir, maxLineLength=maxLineLength
            )
            for cns, cnd in zip(cSequence, cDirection):
                sequence.append(cns)
                direction.append(cnd)

    # return :)
    return sequence, direction


def dominoeUnevenBatch(batchSize, minSeq, maxSeq, listDominoes, dominoeValue, highestDominoe, ignoreIndex=-1, return_full=False):
    """
    retrieve a batch of dominoes and their target order given the value of each dominoe

    dominoes are paired values (combinations with replacement) of integers
    from 0 to <highestDominoe>. The total value of each dominoe is the sum of
    the two integers associated with that dominoe. For example, the dominoe
    (7|3) has value 10.

    Each element in the batch contains an input and target. The input is
    composed of a sequence of dominoes in a random order, transformed into a
    simple representation (explained below). The target is a list of the order
    of dominoes by the one with the highest value to the one with the lowest
    value. Note that many dominoes share the same value, but since the dominoe
    list is always the same, equal value dominoes will always be sorted in the
    same way.

    Each element can have a different sequence length, they will be padded
    with zeros to whatever the longest sequence is. The ignoreIndex is used to
    determine what to label targets for any padded elements (i.e. any place
    where no prediction is needed). The nll_loss function then accepts this as
    an input to ignore. This is part of the reason why pointer networks are
    awesome... the input and output can vary in size!!!

    The simple representation is a two-hot vector where the first
    <highestDominoe+1> elements represent the first value of the dominoe, and
    the second <highestDominoe+1> elements represent the second value of the
    dominoe. Here are some examples for highest dominoe = 3:

    (0 | 0): [1, 0, 0, 0, 1, 0, 0, 0]
    (0 | 1): [1, 0, 0, 0, 0, 1, 0, 0]
    (0 | 2): [1, 0, 0, 0, 0, 0, 1, 0]
    (0 | 3): [1, 0, 0, 0, 0, 0, 0, 1]
    (1 | 0): [0, 1, 0, 0, 1, 0, 0, 0]
    (2 | 1): [0, 0, 1, 0, 0, 1, 0, 0]

    """
    numDominoes = len(listDominoes)
    input_dim = 2 * (highestDominoe + 1)

    # choose how long each sequence in the batch will be
    seqLength = np.random.randint(minSeq, maxSeq + 1, batchSize)
    maxSeqLength = max(seqLength)  # max sequence length for padding

    # choose dominoes from the batch, and get their value (in points)
    selection = [np.random.choice(numDominoes, sl, replace=False).tolist() for sl in seqLength]
    value = [dominoeValue[sel] for sel in selection]

    # index of first and second value in two-hot representation
    pad = [[0] * (maxSeqLength - sl) for sl in seqLength]
    firstValue = np.stack([listDominoes[sel, 0].tolist() + p for p, sel in zip(pad, selection)])
    secondValue = np.stack([(listDominoes[sel, 1] + highestDominoe + 1).tolist() + p for p, sel in zip(pad, selection)])
    firstValue = torch.tensor(firstValue, dtype=torch.int64).unsqueeze(2)
    secondValue = torch.tensor(secondValue, dtype=torch.int64).unsqueeze(2)

    # create mask (used for scattering and also as an output)
    mask = 1.0 * (torch.arange(maxSeqLength).view(1, -1).expand(batchSize, -1) < torch.tensor(seqLength).view(-1, 1))

    # scatter data into two-hot vectors, except where sequence length is exceed where the mask is 0
    input = torch.zeros((batchSize, maxSeqLength, input_dim), dtype=torch.float)
    input.scatter_(2, firstValue, mask.float().unsqueeze(2))
    input.scatter_(2, secondValue, mask.float().unsqueeze(2))

    # sort and pad each list of dominoes by value
    def sortPad(val, padTo, ignoreIndex=-1):
        s = sorted(range(len(val)), key=lambda i: -val[i])
        p = [ignoreIndex] * (padTo - len(val))
        return s + p

    # create a padded sort index, then turn into a torch tensor as the target vector
    sortIdx = [sortPad(val, maxSeqLength, ignoreIndex) for val in value]  # pad with ignore index so nll_loss ignores them
    target = torch.stack([torch.LongTensor(idx) for idx in sortIdx])

    if return_full:
        return input, target, mask, selection
    else:
        return input, target, mask


def makeLines(input, dominoes, value_method="dominoe"):
    selection, available = input  # unpack
    cseq, cdir = construct_line_recursive(dominoes, selection, available)
    if value_method == "dominoe":
        cval = [np.sum(dominoes[cs]) for cs in cseq]
    else:
        cval = [len(cs) for cs in cseq]
    cidx = max(enumerate(cval), key=lambda x: x[1])[0]
    return cseq[cidx], cdir[cidx]


def getBestLineFromAvailablePool(dominoes, selection, available, value_method="dominoe", threads=18):
    # check value method
    if not (value_method == "dominoe" or value_method == "length"):
        raise ValueError("did not recognize value_method, it has to be either 'dominoe' or 'length'")
    p_makeLines = partial(makeLines, dominoes=dominoes, value_method=value_method)

    with Pool(threads) as p:
        lines = p.map(p_makeLines, zip(selection, available))
    bestSequence, bestDirection = map(list, zip(*lines))
    return bestSequence, bestDirection


def convertToHandIndex(selection, bestSequence):
    indices = []
    for sel, seq in zip(selection, bestSequence):
        # look up table for current selection
        elementIdx = {element: idx for idx, element in enumerate(sel)}
        indices.append([elementIdx[element] for element in seq])
    return indices


def padBestLine(bestSequence, max_output, null_index, ignore_index=-1):
    for bs in bestSequence:
        c_length = len(bs)
        append_null = [null_index] if max_output > c_length else []
        append_ignore = [ignore_index] * (max_output - (c_length + 1))
        bs += append_null + append_ignore
        # bs += [ignore_index]*(max_output-len(bs))
    return bestSequence


def generateBatch(
    highestDominoe,
    dominoes,
    batch_size,
    numInHand,
    return_target=True,
    value_method="dominoe",
    available_token=False,
    null_token=False,
    ignore_index=-1,
    return_full=False,
):

    input, selection, available = random_dominoe_hand(
        numInHand, dominoes, highestDominoe, batch_size=batch_size, null_token=null_token, available_token=available_token
    )

    mask_tokens = numInHand + (1 if null_token else 0) + (1 if available_token else 0)
    mask = torch.ones((batch_size, mask_tokens), dtype=torch.float)

    if return_target:
        # then measure best line and convert it to a "target" array
        if available_token:
            bestSequence, bestDirection = getBestLineFromAvailable(dominoes, selection, available, value_method=value_method)
        else:
            bestSequence, bestDirection = getBestLine(dominoes, selection, highestDominoe, value_method=value_method)

        # convert sequence to hand index
        iseq = convertToHandIndex(selection, bestSequence)

        # create target and append null_index once, then ignore_index afterwards
        # the idea is that the agent should play the best line, then indicate that the line is over, then anything else doesn't matter
        null_index = numInHand if null_token else ignore_index
        target = torch.tensor(np.stack(padBestLine(iseq, numInHand + 1, null_index, ignore_index=ignore_index)), dtype=torch.long)
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


def make_path(input):
    pos, distmat = input
    closest_to_origin = np.argmin(np.sum(pos**2, axis=1))
    dd = sp.spatial.distance.squareform(distmat)
    _, cpath = held_karp(dd)
    shift = {val: idx for idx, val in enumerate(cpath)}[closest_to_origin]
    cpath = np.roll(cpath, -shift)
    check_points = pos[cpath[[1, -1]]]  # second point and last point - check for clockwise travel
    angles = np.arctan(check_points[:, 1] / check_points[:, 0])
    if angles[1] > angles[0]:
        cpath = np.flip(np.roll(cpath, -1))
    # finally, move it so the origin is the last location
    return np.roll(cpath, -1)


def get_path(xy, dists):
    """
    for batch of (batch, num_cities, 2), returns shortest path using
    held-karp algorithm that ends closest to origin and is clockwise
    """
    return [make_path(input) for input in zip(xy, dists)]


def get_path_pool(xy, dists, threads=8):
    with Pool(threads) as p:
        path = list(p.map(make_path, zip(xy, dists)))
    return path


def tsp_batch(batch_size, num_cities, return_target=True, return_full=False, threads=1):
    """parallelized preparation of batch, better to use 1 thread if num_cities~<10 or batch_size<=256"""
    xy = np.random.random((batch_size, num_cities, 2))
    dists = np.stack([sp.spatial.distance.pdist(p) for p in xy])
    input = torch.tensor(xy, dtype=torch.float)
    if return_target:
        if threads > 1:
            target = torch.tensor(np.stack(get_path_pool(xy, dists, threads)), dtype=torch.long)
        else:
            target = torch.tensor(np.stack(get_path(xy, dists)), dtype=torch.long)
    else:
        target = None
    if return_full:
        torch_dists = torch.stack([torch.tensor(sp.spatial.distance.squareform(d)) for d in dists])
        return input, target, torch.tensor(xy), torch_dists
    else:
        return input, target
