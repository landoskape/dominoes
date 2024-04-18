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

    def as_tensor(seq):
        return torch.tensor(seq, dtype=torch.long)

    padded_best_seq = []
    for seq in best_seq:
        c_length = len(seq)
        append_null = [null_index] if max_output > c_length else []
        append_ignore = [ignore_index] * (max_output - (c_length + 1))
        seq = torch.cat((seq, as_tensor(append_null), as_tensor(append_ignore)))
        padded_best_seq.append(seq)
    return padded_best_seq


def construct_line_recursive(dominoes, available, hand_index=None, prev_seq=None, prev_dir=None, max_length=None):
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
    # if prev_seq and prev_dir are not provided, that means this is the first call
    if prev_seq is None:
        prev_seq = torch.tensor([], dtype=torch.long)
        prev_dir = torch.tensor([], dtype=torch.long)

    # if the maximum length of the sequence is reached, return sequence up to this point
    if max_length is not None and len(prev_seq) == max_length:
        return [prev_seq], [prev_dir]

    # convert dominoes to torch tensor if it is a numpy array
    if isinstance(dominoes, np.ndarray):
        dominoes = torch.tensor(dominoes)

    # if hand_index is not provided, use all dominoes in the set
    if hand_index is None:
        hand_index = torch.arange(len(dominoes), dtype=torch.long)

    # set hand ("playable dominoes")
    hand = dominoes[hand_index]

    # check if previous sequence end position matches the available value
    if len(prev_seq) > 0:
        msg = "the end of the last sequence doesn't match what is defined as available!"
        assert hand[prev_seq[-1]][0 if prev_dir[-1] == 1 else 1] == available, msg

    # find all dominoes in hand that can be played on the available token
    possible_plays = torch.where(torch.any(hand == available, dim=1) & ~torch.isin(hand_index, prev_seq))[0]

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
            cseq = torch.cat((prev_seq.clone().view(-1), hand_index[idx_play].view(1)))
            # play in forward direction
            cdir = torch.cat((prev_dir.clone().view(-1), torch.tensor(0, dtype=torch.long).view(1)))
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
            cseq = torch.cat((prev_seq.clone().view(-1), hand_index[idx_play].view(1)))
            # play in forward direction
            cdir = torch.cat((prev_dir.clone().view(-1), torch.tensor(1, dtype=torch.long).view(1)))
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
    for _ in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def make_path(coordinates, distances, idx_init):
    """
    for a set of coordinates, returns the shortest path that starts at the
    initial index and ends closest to the origin, and is clockwise

    args:
        coordinates: (num_cities, 2) tensor of coordinates
        distances: (num_cities, num_cities) tensor of distances between coordinates
        idx_init: index of the initial city

    returns:
        best_path: (num_cities) tensor of the best path
    """
    # use held_karp algorithm to get fastest path through coordinates
    best_path = torch.tensor(held_karp(distances)[1], dtype=torch.long)

    # shift the path so it starts at the initial index
    shift = {val.item(): idx for idx, val in enumerate(best_path)}[idx_init.item()]
    best_path = torch.roll(best_path, -shift)

    # make second point in path the second closest to origin
    check_points = coordinates[best_path[[1, -1]]]
    check_distance = torch.sum(check_points**2, dim=1)

    # flip the path such that the second point is the second closest to the origin
    if check_distance[1] < check_distance[0]:
        best_path = torch.flip(torch.roll(best_path, -1), dims=(0,))

    # finally, roll it once so the origin is the last location
    best_path = best_path[1:]  # torch.roll(best_path, -1)

    return best_path


def get_paths(coordinates, distances, idx_init, threads=1):
    """
    for batch of (batch, num_cities, 2), returns shortest path using
    held-karp algorithm that ends closest to origin and is clockwise
    """
    if threads > 1:
        with Pool(threads) as p:
            path = list(p.starmap(make_path, zip(coordinates, distances, idx_init)))
    else:
        path = [make_path(coord, dist, idx) for coord, dist, idx in zip(coordinates, distances, idx_init)]

    return torch.stack(path).long()
