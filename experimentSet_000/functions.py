from copy import copy
import itertools
import numpy as np

def softmax(values):
    ev = np.exp(values - np.max(values))
    return ev / np.sum(ev)

def playDirection(available, dominoe):
    # available=the value available on a line
    # dominoe=the dominoe (value,value) pair being played
    # returns (the direction of the dominoe (forwards/backwards) and the next available value after play)
    if available==dominoe[0]: 
        return 0, int(dominoe[1])
    if available==dominoe[1]: 
        return 1, int(dominoe[0])
    raise ValueError(f"request dominoe ({dominoe}) cannot be played on value {available}!")
    
def numberDominoes(highestDominoe):
        return int((highestDominoe+1)*(highestDominoe+2)/2)
    
def listDominoes(highestDominoe):
    # given a standard rule for how to organize the list of dominoes as one-hot arrays, list the dominoes present in a one hot array
    return np.array([np.array(quake) for quake in itertools.combinations_with_replacement(np.arange(highestDominoe+1), 2)], dtype=int)

def dominoesString(dominoe):
    return f"{dominoe[0]:>2}|{dominoe[1]:<2}"
    
def printDominoeList(options, dominoes, name=None, fullList=False):
    if name is None: nameFunc=lambda x: 'options:'
    if name is not None and options.ndim==2: nameFunc=lambda x: f"{name} {x}:"
    if name is not None and options.ndim==1: nameFunc=lambda x: name
    if options.ndim==1: options = np.reshape(options, (1, len(options)))
    dominoeList = []
    for player in range(options.shape[0]):
        if fullList: dlist = [dominoesString(dominoe) if opt else '---' for dominoe,opt in zip(dominoes, options[player])]
        else: dlist = [dominoesString(dominoe) for dominoe,opt in zip(dominoes, options[player]) if opt]
        print(f"{nameFunc(player)} {dlist}")
        
def handValue(dominoes, idxHand):
    return np.sum(dominoes[idxHand])

def gameSequenceToString(dominoes, sequence, direction, player=None, playNumber=None):
    # take in game sequence and dominoes and convert to string, then print output
    # manage inputs -- 
    if len(sequence)==0: 
        print('no play')
        return
    input1d = not isinstance(sequence[0],list)
    if input1d: sequence = [sequence] # np.reshape(sequence, (1,-1)) # make iterable in the expected way
    if input1d: direction = [direction] # np.reshape(direction, (1,-1)) 
    assert all([len(seq)==len(direct) for seq,direct in zip(sequence,direction)]), "sequence and direction do not have same shape"
    if input1d and player is not None:
        player = [player] # np.reshape(player, (1,-1))
    if player is not None: assert all([len(seq)==len(play) for seq,play in zip(sequence,player)]), "provided player is not same shape as sequence"
    if input1d and playNumber is not None:
        playNumber = [playNumber] # np.reshape(playNumber, (1,-1))
    if playNumber is not None: assert all([len(seq)==len(play) for seq,play in zip(sequence,playNumber)]), "provided playNumber is not same shape as sequence"
        
    # now, for each sequence, print out dominoe list in correct direction
    for idx,seq in enumerate(sequence):
        sequenceString = [dominoesString(dominoes[domIdx]) if domDir==0 else dominoesString(np.flip(dominoes[domIdx])) for domIdx,domDir in zip(seq,direction[idx])]
        if player is not None:
            sequenceString = [seqString+f" Ag:{cplay}" for seqString,cplay in zip(sequenceString,player[idx])]
        if playNumber is not None:
            sequenceString = [seqString+f" P:{cplay}" for seqString,cplay in zip(sequenceString,playNumber[idx])]
        print(sequenceString)

def constructLineRecursive(dominoes, myHand, available, previousSequence=[], previousDirection=[], maxLineLength=None):
    # this version of the function uses absolute dominoe numbers, rather than indexing based on which order they are in the hand
    # if there are too many dominoes in hand, constructing all possible lines takes way too long...
    if (maxLineLength is not None) and (len(previousSequence)==maxLineLength): 
        return [previousSequence], [previousDirection]
    
    assert type(previousSequence)==list and type(previousDirection)==list, "previous sequence and direction must be lists"
    if len(previousSequence)>0: 
        # if a previous sequence was provided, make sure the end of it matches what is defined as available
        assert dominoes[previousSequence[-1]][0 if previousDirection[-1]==1 else 1]==available, "the end of the last sequence doesn't match what is defined as available!"
    
    # recursively constructs all possible lines given a hand (value pairs in list), an available value to play on, and the previous played/direction dominoe index sequences
    hand = dominoes[myHand]
    possiblePlays = np.where(np.any(hand==available,axis=1) & ~np.isin(myHand, previousSequence))[0]
    
    # if there are no possible plays, the return the finished sequence
    if len(possiblePlays)==0: return [previousSequence], [previousDirection]
    
    # otherwise, make new lines for each possible play 
    sequence = []
    direction = []
    for idxPlay in possiblePlays:
        # if the first value of the possible play matches the available, then play it in the forward direction
        if hand[idxPlay][0]==available:
            # copy previousSequence and previousDirection, append new play in forward direction to it
            cseq = copy(previousSequence)
            cseq.append(myHand[idxPlay])
            cdir = copy(previousDirection)
            cdir.append(0)
            # then recursively construct line from this standpoint
            cSequence, cDirection = constructLineRecursive(dominoes, myHand, hand[idxPlay][1], previousSequence=cseq, previousDirection=cdir, maxLineLength=maxLineLength)
            # once lines are constructed, add them all to "sequence" and "direction", which will be a list of lists of all possible sequences
            for cns,cnd in zip(cSequence, cDirection):
                sequence.append(cns)
                direction.append(cnd)
                
        # if the second value of the possible play matches the available and it isn't a double, then play it in the reverse direction (all same except direction and next available)
        if (hand[idxPlay][0]!=hand[idxPlay][1]) and (hand[idxPlay][1]==available):
            cseq = copy(previousSequence)
            cseq.append(myHand[idxPlay])
            cdir = copy(previousDirection)
            cdir.append(1)
            cSequence, cDirection = constructLineRecursive(dominoes, myHand, hand[idxPlay][0], previousSequence=cseq, previousDirection=cdir, maxLineLength=maxLineLength)
            for cns,cnd in zip(cSequence, cDirection):
                sequence.append(cns)
                direction.append(cnd)
    
    # return :)
    return sequence, direction

def uniqueSequences(lineSequence, lineDirection, updatedLine):
    seen = set() # keep track of unique sequences here
    uqSequence = []
    uqDirection = []
    uqUpdated = []
    for subSeq, subDir, subUpdate in zip(lineSequence, lineDirection, updatedLine):
        subSeqTuple = tuple(subSeq) # turn into tuple so we can add it to a set
        if subSeqTuple not in seen:
            # if it hasn't been seen yet, add it to the set, and add it to the unique list
            seen.add(subSeqTuple)
            uqSequence.append(subSeq)
            uqDirection.append(subDir)
            uqUpdated.append(subUpdate)
    return uqSequence, uqDirection, uqUpdated

def updateLine(lineSequence, lineDirection, nextPlay, onOwn):
    if nextPlay is None: return lineSequence, lineDirection # if there wasn't a play, then don't change anything
    if lineSequence==[[]]: return lineSequence, lineDirection # if there wasn't any lines, return them as they can't change
    
    newSequence, newDirection, updatedLine = [], [], []
    if onOwn:
        # if playing on own line, then the still-valid sequences can be truncated and some can be removed
        for pl,dr in zip(lineSequence,lineDirection):
            if pl[0]==nextPlay:
                # for sequences that started with the played dominoe, add them starting from the second dominoe
                newSequence.append(pl[1:])
                newDirection.append(dr[1:])
                updatedLine.append(True)
    else:
        # otherwise, update any sequences that included the played dominoe
        for pl,dr in zip(lineSequence,lineDirection):
            if nextPlay in pl: 
                # if the sequence includes the played dominoe, include the sequence only up to the played dominoe
                idxInLine = np.where(pl==nextPlay)[0][0]
                if idxInLine>0:
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
    if uqSequence==[]: return [[]], [[]]
    
    # next, determine if any sequences are subsumed by other sequences (in which case they are irrelevant)
    subsumed = [False]*len(uqSequence)
    for idx, (seq, updated) in enumerate(zip(uqSequence, uqUpdated)):
        # for any sequence that has been updated --
        if updated:
            for icmp, scmp in enumerate(uqSequence):
                # compare it with all the other sequences that are longer than it
                if len(scmp)>len(seq):
                    # if they start the same way, delete the one that is smaller
                    if seq==scmp[:len(seq)]:
                        subsumed[idx]=True
                        continue
    
    # keep only unique and valid sequences, then return
    finalSequence = [uqSeq for (uqSeq, sub) in zip(uqSequence, subsumed) if not(sub)]
    finalDirection = [uqDir for (uqDir, sub) in zip(uqDirection, subsumed) if not(sub)]
    return finalSequence, finalDirection






























































# def constructLineRecursiveRelative(hand, available, previousSequence=[], previousDirection=[], maxLineLength=None):
#     # if there are too many dominoes in hand, constructing all possible lines takes way too long...
#     if (maxLineLength is not None) and (len(previousSequence)==maxLineLength): 
#         return [previousSequence], [previousDirection]
    
#     # recursively constructs all possible lines given a hand (value pairs in list), an available value to play on, and the previous played/direction dominoe index sequences
#     possiblePlays = np.where(np.any(hand==available,axis=1) & ~np.isin(np.arange(len(hand)), previousSequence))[0]
    
#     # if there are no possible plays, the return the finished sequence
#     if len(possiblePlays)==0: return [previousSequence], [previousDirection]
    
#     # otherwise, make new lines for each possible play 
#     sequence = []
#     direction = []
#     for idxPlay in possiblePlays:
#         # if the first value of the possible play matches the available, then play it in the forward direction
#         if hand[idxPlay][0]==available:
#             # copy previousSequence and previousDirection, append new play in forward direction to it
#             cseq = copy(previousSequence)
#             cseq.append(idxPlay)
#             cdir = copy(previousDirection)
#             cdir.append(0)
#             # then recursively construct line from this standpoint
#             cSequence, cDirection = constructLineRecursiveRelative(hand, hand[idxPlay][1], previousSequence=cseq, previousDirection=cdir, maxLineLength=maxLineLength)
#             # once lines are constructed, add them all to "sequence" and "direction", which will be a list of lists of all possible sequences
#             for cns,cnd in zip(cSequence, cDirection):
#                 sequence.append(cns)
#                 direction.append(cnd)
                
#         # if the second value of the possible play matches the available and it isn't a double, then play it in the reverse direction (all same except direction and next available)
#         if (hand[idxPlay][0]!=hand[idxPlay][1]) and (hand[idxPlay][1]==available):
#             cseq = copy(previousSequence)
#             cseq.append(idxPlay)
#             cdir = copy(previousDirection)
#             cdir.append(1)
#             cSequence, cDirection = constructLineRecursiveRelative(hand, hand[idxPlay][0], previousSequence=cseq, previousDirection=cdir, maxLineLength=maxLineLength)
#             for cns,cnd in zip(cSequence, cDirection):
#                 sequence.append(cns)
#                 direction.append(cnd)
    
#     # return :)
#     return sequence, direction


# def updateLineRelative(lineSequence, lineDirection, nextPlay, onOwn, updatePlayIdx):
#     if nextPlay is None: return lineSequence, lineDirection # if there wasn't a play, then don't change anything
#     if lineSequence==[[]]: return lineSequence, lineDirection # if there wasn't any lines, return them as they can't change
    
#     newSequence, newDirection, updatedLine = [], [], []
#     if onOwn:
#         # if playing on own line, then the still-valid sequences can be truncated and some can be removed
#         for pl,dr in zip(lineSequence,lineDirection):
#             if pl[0]==nextPlay:
#                 # for sequences that started with the played dominoe, add them starting from the second dominoe
#                 newSequence.append([updatePlayIdx[pp] for pp in pl[1:]])
#                 newDirection.append(dr[1:])
#                 updatedLine.append(True)
#     else:
#         # otherwise, update any sequences that included the played dominoe
#         for pl,dr in zip(lineSequence,lineDirection):
#             if nextPlay in pl: 
#                 # if the sequence includes the played dominoe, include the sequence only up to the played dominoe
#                 idxInLine = np.where(pl==nextPlay)[0][0]
#                 if idxInLine>0:
#                     # only include it if there are dominoes left
#                     newSequence.append([updatePlayIdx[pp] for pp in pl[:idxInLine]])
#                     newDirection.append(dr[:idxInLine])
#                     updatedLine.append(True)
#             else:
#                 # if the sequence doesn't include the played dominoe, add it unchanged
#                 newSequence.append([updatePlayIdx[pp] for pp in pl])
#                 newDirection.append(dr)
#                 updatedLine.append(False)
    
#     # this helper function returns the unique sequences in a mostly optimized manner
#     uqSequence, uqDirection, uqUpdated = uniqueSequences(newSequence, newDirection, updatedLine)
    
#     # if there are no valid sequences, fast return
#     if uqSequence==[]: return [[]], [[]]
    
#     # next, determine if any sequences are subsumed by other sequences (in which case they are irrelevant)
#     subsumed = [False]*len(uqSequence)
#     for idx, (seq, updated) in enumerate(zip(uqSequence, uqUpdated)):
#         # for any sequence that has been updated --
#         if updated:
#             for icmp, scmp in enumerate(uqSequence):
#                 # compare it with all the other sequences that are longer than it
#                 if len(scmp)>len(seq):
#                     # if they start the same way, delete the one that is smaller
#                     if seq==scmp[:len(seq)]:
#                         subsumed[idx]=True
#                         continue
    
#     # keep only unique and valid sequences, then return
#     finalSequence = [uqSeq for (uqSeq, sub) in zip(uqSequence, subsumed) if not(sub)]
#     finalDirection = [uqDir for (uqDir, sub) in zip(uqDirection, subsumed) if not(sub)]
#     return finalSequence, finalDirection



