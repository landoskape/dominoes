from warnings import warn
import numpy as np
import numpy as np
import torch
import matplotlib

from ..datasets.support import get_dominoes
from .classes import AttributeDict


def loadSavedExperiment(prmsPath, resPath, fileName, args=None):
    try:
        prms = np.load(prmsPath / (fileName + ".npy"), allow_pickle=True).item()
    except:
        raise ValueError(f"Failed to load parameter file at {prmsPath / (fileName+'.npy')}, this probably means it wasn't run yet.")

    if args is not None:
        assert (
            prms.keys() <= vars(args).keys()
        ), f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(args).keys())}"
        for ak in vars(args):
            if ak == "justplot":
                continue
            if ak == "nosave":
                continue
            if ak == "printargs":
                continue
            if ak in prms and prms[ak] != vars(args)[ak]:
                print(f"Requested argument {ak}={vars(args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(args, ak, prms[ak])
    else:
        args = AttributeDict(prms)

    results = np.load(resPath / (fileName + ".npy"), allow_pickle=True).item()

    return results, args


def averageGroups(var, numPerGroup, axis=0):
    """method for averaging variable across repeats within group on specified axis"""
    assert isinstance(var, np.ndarray), "This only works for numpy arrays"
    numGroups = var.shape[axis] / numPerGroup
    assert numGroups.is_integer(), f"numPerGroup provided is incorrect, this means there are {numGroups} groups..."
    numGroups = int(numGroups)
    exvar_shape = list(np.expand_dims(var, axis=axis + 1).shape)
    exvar_shape[axis] = numGroups
    exvar_shape[axis + 1] = numPerGroup
    exvar = var.reshape(exvar_shape)
    return np.mean(exvar, axis=axis + 1)


def ncmap(name="Spectral", vmin=0.0, vmax=1.0):
    cmap = matplotlib.cm.get_cmap(name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    def cm(val):
        return cmap(norm(val))

    return cm


def softmax(values):
    ev = np.exp(values - np.max(values))
    return ev / np.sum(ev)


def playDirection(available, dominoe):
    # available=the value available on a line
    # dominoe=the dominoe (value,value) pair being played
    # returns (the direction of the dominoe (forwards/backwards) and the next available value after play)
    if available == dominoe[0]:
        return 0, int(dominoe[1])
    if available == dominoe[1]:
        return 1, int(dominoe[0])
    raise ValueError(f"request dominoe ({dominoe}) cannot be played on value {available}!")


def numberDominoes(highestDominoe):
    return int((highestDominoe + 1) * (highestDominoe + 2) / 2)


def listDominoes(highestDominoe):
    """alias for new function that is deprecated"""
    warn("listDominoes is deprecated, use get_dominoes instead", DeprecationWarning, stacklevel=2)
    return get_dominoes(highestDominoe, as_torch=False)


def dominoesString(dominoe):
    return f"{dominoe[0]:>2}|{dominoe[1]:<2}"


def printDominoeList(options, dominoes, name=None, fullList=False):
    if name is None:
        nameFunc = lambda x: "options:"
    if name is not None and options.ndim == 2:
        nameFunc = lambda x: f"{name} {x}:"
    if name is not None and options.ndim == 1:
        nameFunc = lambda x: name
    if options.ndim == 1:
        options = np.reshape(options, (1, len(options)))
    dominoeList = []
    for player in range(options.shape[0]):
        if fullList:
            dlist = [dominoesString(dominoe) if opt else "---" for dominoe, opt in zip(dominoes, options[player])]
        else:
            dlist = [dominoesString(dominoe) for dominoe, opt in zip(dominoes, options[player]) if opt]
        print(f"{nameFunc(player)} {dlist}")


def handValue(dominoes, idxHand):
    return np.sum(dominoes[idxHand])


def gameSequenceToString(dominoes, sequence, direction, player=None, playNumber=None, labelLines=False):
    # take in game sequence and dominoes and convert to string, then print output
    # manage inputs --
    if len(sequence) == 0:
        print("no play")
        return
    input1d = not isinstance(sequence[0], list)
    if input1d:
        sequence = [sequence]  # np.reshape(sequence, (1,-1)) # make iterable in the expected way
    if input1d:
        direction = [direction]  # np.reshape(direction, (1,-1))
    if labelLines:
        if len(sequence) == 1:
            name = ["dummy: "]
        else:
            name = [f"player {idx}: " for idx in range(len(sequence))]
    else:
        name = [""] * len(sequence)

    assert all([len(seq) == len(direct) for seq, direct in zip(sequence, direction)]), "sequence and direction do not have same shape"
    if input1d and player is not None:
        player = [player]  # np.reshape(player, (1,-1))
    if player is not None:
        assert all([len(seq) == len(play) for seq, play in zip(sequence, player)]), "provided player is not same shape as sequence"
    if input1d and playNumber is not None:
        playNumber = [playNumber]  # np.reshape(playNumber, (1,-1))
    if playNumber is not None:
        assert all([len(seq) == len(play) for seq, play in zip(sequence, playNumber)]), "provided playNumber is not same shape as sequence"

    # now, for each sequence, print out dominoe list in correct direction
    for idx, seq in enumerate(sequence):
        sequenceString = [
            dominoesString(dominoes[domIdx]) if domDir == 0 else dominoesString(np.flip(dominoes[domIdx]))
            for domIdx, domDir in zip(seq, direction[idx])
        ]
        if player is not None:
            sequenceString = [seqString + f" Ag:{cplay}" for seqString, cplay in zip(sequenceString, player[idx])]
        if playNumber is not None:
            sequenceString = [seqString + f" P:{cplay}" for seqString, cplay in zip(sequenceString, playNumber[idx])]
        print(name[idx], sequenceString)


def uniqueSequences(lineSequence, lineDirection, updatedLine):
    seen = set()  # keep track of unique sequences here
    uqSequence = []
    uqDirection = []
    uqUpdated = []
    for subSeq, subDir, subUpdate in zip(lineSequence, lineDirection, updatedLine):
        subSeqTuple = tuple(subSeq)  # turn into tuple so we can add it to a set
        if subSeqTuple not in seen:
            # if it hasn't been seen yet, add it to the set, and add it to the unique list
            seen.add(subSeqTuple)
            uqSequence.append(subSeq)
            uqDirection.append(subDir)
            uqUpdated.append(subUpdate)
    return uqSequence, uqDirection, uqUpdated


def updateLine(lineSequence, lineDirection, nextPlay, onOwn):
    if nextPlay is None:
        return lineSequence, lineDirection  # if there wasn't a play, then don't change anything
    if lineSequence == [[]]:
        return lineSequence, lineDirection  # if there wasn't any lines, return them as they can't change

    newSequence, newDirection, updatedLine = [], [], []
    if onOwn:
        # if playing on own line, then the still-valid sequences can be truncated and some can be removed
        for pl, dr in zip(lineSequence, lineDirection):
            if pl[0] == nextPlay:
                # for sequences that started with the played dominoe, add them starting from the second dominoe
                newSequence.append(pl[1:])
                newDirection.append(dr[1:])
                updatedLine.append(True)
    else:
        # otherwise, update any sequences that included the played dominoe
        for pl, dr in zip(lineSequence, lineDirection):
            if nextPlay in pl:
                # if the sequence includes the played dominoe, include the sequence only up to the played dominoe
                idxInLine = np.where(pl == nextPlay)[0][0]
                if idxInLine > 0:
                    # only include it if there are dominoes left
                    newSequence.append(pl[:idxInLine])
                    newDirection.append(dr[:idxInLine])
                    updatedLine.append(True)
            else:
                # if the sequence doesn't include the played dominoe, add it unchanged
                newSequence.append(pl)
                newDirection.append(dr)
                updatedLine.append(False)

    # this helper function returns the unique sequences in a mostly optimized manner
    uqSequence, uqDirection, uqUpdated = uniqueSequences(newSequence, newDirection, updatedLine)

    # if there are no valid sequences, fast return
    if uqSequence == []:
        return [[]], [[]]

    # next, determine if any sequences are subsumed by other sequences (in which case they are irrelevant)
    subsumed = [False] * len(uqSequence)
    for idx, (seq, updated) in enumerate(zip(uqSequence, uqUpdated)):
        # for any sequence that has been updated --
        if updated:
            for icmp, scmp in enumerate(uqSequence):
                # compare it with all the other sequences that are longer than it
                if len(scmp) > len(seq):
                    # if they start the same way, delete the one that is smaller
                    if seq == scmp[: len(seq)]:
                        subsumed[idx] = True
                        continue

    # keep only unique and valid sequences, then return
    finalSequence = [uqSeq for (uqSeq, sub) in zip(uqSequence, subsumed) if not (sub)]
    finalDirection = [uqDir for (uqDir, sub) in zip(uqDirection, subsumed) if not (sub)]
    return finalSequence, finalDirection


def eloExpected(Ra, Rb):
    return 1 / (1 + 10 ** ((Rb - Ra) / 400))


def eloUpdate(S, E, k=32):
    return k * (S - E)


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_device(tensor):
    """simple method to get device of input tensor"""
    return "cuda" if tensor.is_cuda else "cpu"


# the following masked softmax methods are from allennlp
# https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#L243
def masked_softmax(
    vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32
) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If the input is completely masked anywere (across the requested dimension), then this will make it
    uniform instead of keeping it masked, which would lead to nans.
    """
    if mask is None:
        return torch.nn.functional.log_softmax(vector, dim=dim)
    mask = mask.float()
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
    with torch.no_grad():
        min_value = vector.min() - 50.0  # make sure it's lower than the lowest value
    vector = vector.masked_fill(mask == 0, min_value)
    # vector = vector + (mask + 1e-45).log()
    # vector = vector.masked_fill(mask==0, float('-inf'))
    # vector[torch.all(mask==0, dim=dim)]=1.0 # if the whole thing is masked, this is needed to prevent nans
    return torch.nn.functional.log_softmax(vector, dim=dim)
