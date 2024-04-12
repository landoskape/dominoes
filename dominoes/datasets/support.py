from copy import copy
import itertools
from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy as sp
import torch


def get_dominoes(highest_dominoe, as_torch=False):
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


def get_best_line(dominoes, available, value_method="dominoe"):
    """
    get the best line of dominoes given a set of dominoes and an available token

    args:
        dominoes: torch.tensor of shape (num_dominoes, 2)
        available: (int) the value that is available to play on
        value_method: (str) either "dominoe" or "length" to measure the value of the line
                      if "dominoe" the value is the sum of the dominoes in the line
                      if "length" the value is the length of the line

    returns:
        best_sequence: the best sequence of dominoes
        best_direction: the direction of each dominoe in the sequence

    """
    # check value method
    if not (value_method == "dominoe" or value_method == "length"):
        raise ValueError("did not recognize value_method, it has to be either 'dominoe' or 'length'")

    # get all possible lines with this set of dominoes and the available token
    allseqs, alldirs = construct_line_recursive(dominoes, available)

    # measure value with either dominoe method or length method
    if value_method == "dominoe":
        allval = [torch.sum(dominoes[seq]) for seq in allseqs]
    else:
        allval = [len(seq) for seq in allseqs]

    # get index to the best sequence
    best_idx = max(enumerate(allval), key=lambda x: x[1])[0]

    # return the best sequence and direction
    return allseqs[best_idx], alldirs[best_idx]


def pad_best_lines(best_seq, max_output, null_index, ignore_index=-1):
    """
    pad the best sequence of dominoes to a fixed length

    args:
        best_seq: the best sequence of dominoes
        max_output: the maximum length of the sequence
        null_index: the index of the null index (set to ignore_index if no null token)
        ignore_index: the index of the ignore index

    returns:
        padded_best_seq: the best sequence padded to a fixed length
    """
    padded_best_seq = []
    for seq in best_seq:
        c_length = len(seq)
        append_null = [null_index] if max_output > c_length else []
        append_ignore = [ignore_index] * (max_output - (c_length + 1))
        seq += append_null + append_ignore
        padded_best_seq.append(seq)
    return padded_best_seq


def padBestLine(bestSequence, max_output, null_index, ignore_index=-1):
    for bs in bestSequence:
        c_length = len(bs)
        append_null = [null_index] if max_output > c_length else []
        append_ignore = [ignore_index] * (max_output - (c_length + 1))
        bs += append_null + append_ignore
        # bs += [ignore_index]*(max_output-len(bs))
    return bestSequence


def construct_line_recursive(dominoes, available, hand_index=None, prev_seq=[], prev_dir=[], max_length=None):
    """
    recursively construct all possible lines given a set of dominoes, an available value to play on,
    and the previous played/direction dominoe index sequences.

    This method can be used in two ways:
        1. if hand_index is not provided, it will use all dominoes in the set and the resulting
           sequences will use the indices of the dominoes in the set provided in the first argument.
        2. if hand_index is provided, it will only use those dominoes in the set and the resulting
           sequences will use the indices of the dominoes in the hand_index list.

    args:
        dominoes: torch.tensor or numpy nd.array of shape (num_dominoes, 2)
        available: (int) the value that is available to play on
        hand_index: (optional, list[int]) the index of the dominoes in the hand
        prev_seq: the previous sequence of dominoes -- is used for recursion
        prev_dir: the previous direction of the dominoes -- is used for recursion
        max_length: the maximum length of the line

    returns:
        sequence: the list of all possible sequences of dominoes (with indices corresponding
                  to the dominoes in the set or hand_index)
        direction: the list of the direction each dominoe must be played within each sequence
    """
    # if the maximum length of the sequence is reached, return sequence up to this point
    if max_length is not None and len(prev_seq) == max_length:
        return [prev_seq], [prev_dir]

    # check if previous sequence end position matches the available value
    if len(prev_seq) > 0:
        msg = "the end of the last sequence doesn't match what is defined as available!"
        assert dominoes[prev_seq[-1]][0 if prev_dir[-1] == 1 else 1] == available, msg

    # convert dominoes to torch tensor if it is a numpy array
    if isinstance(dominoes, np.ndarray):
        dominoes = torch.tensor(dominoes)

    # if hand_index is not provided, use all dominoes in the set
    if hand_index is None:
        hand_index = torch.arange(len(dominoes))

    # set hand ("playable dominoes")
    hand = dominoes[hand_index]

    # find all dominoes in hand that can be played on the available token
    possible_plays = torch.where(torch.any(hand == available, axis=1) & ~torch.isin(hand_index, prev_seq))[0]

    # if no more plays are possible, return the finished sequence and direction
    if len(possible_plays) == 0:
        return [prev_seq], [prev_dir]

    # otherwise create new lines for each possible play
    sequence = []
    direction = []
    for idx_play in possible_plays:
        # if the first value of the possible play matches the available value
        if hand[idx_play][0] == available:
            # add to sequence
            cseq = copy(prev_seq)
            cseq.append(hand_index[idx_play])
            # play in forward direction
            cdir = copy(prev_dir)
            cdir.append(0)
            # construct sequence recursively from this new sequence
            cseq, cdir = construct_line_recursive(
                dominoes, hand[idx_play][1], hand_index=hand_index, prev_seq=cseq, prev_dir=cdir, max_length=max_length
            )
            # add all sequence/direction lists to possible sequences
            for cns, cnd in zip(cseq, cdir):
                sequence.append(cns)
                direction.append(cnd)

        # if the second value of the possible play matches the available and it isn't a double,
        # then play it in the reverse direction
        else:
            # add to sequence
            cseq = copy(prev_seq)
            cseq.append(hand_index[idx_play])
            # play in reverse direction
            cdir = copy(prev_dir)
            cdir.append(1)
            # construct sequence recursively from this new sequence
            cseq, cdir = construct_line_recursive(
                dominoes, hand[idx_play][0], hand_index=hand_index, prev_seq=cseq, prev_dir=cdir, max_length=max_length
            )
            # add all sequence/direction lists to possible sequences
            for cns, cnd in zip(cseq, cdir):
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
