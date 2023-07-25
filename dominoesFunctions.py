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
        
def constructLineRecursive(hand, available, previousPlayed=[], previousDirection=[], maxLineLength=None):
    # if there are too many dominoes in hand, constructing all possible lines takes way too long...
    if (maxLineLength is not None) and (len(previousPlayed)==maxLineLength): 
        return [previousPlayed], [previousDirection]
    
    # recursively constructs all possible lines given a hand (value pairs in list), an available value to play on, and the previous played/direction dominoe index sequences
    possiblePlays = np.where(np.any(hand==available,axis=1) & ~np.isin(np.arange(len(hand)), previousPlayed))[0]
    
    # if there are no possible plays, the return the finished sequence
    if len(possiblePlays)==0: return [previousPlayed], [previousDirection]
    
    # otherwise, make new lines for each possible play 
    played = []
    direction = []
    for idxPlay in possiblePlays:
        # if the first value of the possible play matches the available, then play it in the forward direction
        if hand[idxPlay][0]==available:
            # copy previousPlayed and previousDirection, append new play in forward direction to it
            cplay = copy(previousPlayed)
            cplay.append(idxPlay)
            cdir = copy(previousDirection)
            cdir.append(0)
            # then recursively construct line from this standpoint
            cPlayed, cDirection = constructLineRecursive(hand, hand[idxPlay][1], previousPlayed=cplay, previousDirection=cdir, maxLineLength=maxLineLength)
            # once lines are constructed, add them all to "played" and "direction", which will be a list of lists of all possible sequences
            for cnp,cnd in zip(cPlayed, cDirection):
                played.append(cnp)
                direction.append(cnd)
                
        # if the second value of the possible play matches the available and it isn't a double, then play it in the reverse direction (all same except direction and next available)
        if (hand[idxPlay][0]!=hand[idxPlay][1]) and (hand[idxPlay][1]==available):
            cplay = copy(previousPlayed)
            cplay.append(idxPlay)
            cdir = copy(previousDirection)
            cdir.append(1)
            cPlayed, cDirection = constructLineRecursive(hand, hand[idxPlay][0], previousPlayed=cplay, previousDirection=cdir, maxLineLength=maxLineLength)
            for cnp,cnd in zip(cPlayed, cDirection):
                played.append(cnp)
                direction.append(cnd)
    
    # return :)
    return played, direction





