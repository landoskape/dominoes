import sys
import os

# add path that contains the dominoes package
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

# standard imports
from copy import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.cuda as torchCuda

# dominoes package
from dominoes import functions as df
from dominoes import transformers

device = 'cuda' if torchCuda.is_available() else 'cpu'

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(network=False):
    fname = "pointerSequencer"
    if args.use_rl:
        fname = fname + '_rl'
    if network:
        fname = fname + '.pt'
    return fname

def gamma_type(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a floating-point literal")

    if (x <= 0.0) or (x > 1.0):
        raise argparse.ArgumentTypeError(f"{x} is not in range (0.0, 1.0]")
    return x

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer dominoe sequencing experiment.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('-hs','--hand-size', type=int, default=12, help='the maximum tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=128, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=20000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=1000, help='the number of testing epochs')

    parser.add_argument('--embedding_dim', type=int, default=96, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=8, help='the number of heads in transformer layers')
    parser.add_argument('--expansion', type=int, default=2, help='the expansion at the MLP part of the transformer')
    parser.add_argument('--encoding-layers', type=int, default=2, help='the number of stacked transformers in the encoder')
    parser.add_argument('--greedy', default=False, action='store_true', help='if used, will generate greedy predictions of each step rather than probability-weighted predictions')
    parser.add_argument('--temperature', type=int, default=1, help='temperature of choice during training')
    parser.add_argument('--use-rl', default=False, action='store_true', help='if used, will learn by reinforcement learning, otherwise supervised')
    parser.add_argument('--gamma', type=gamma_type, default=0.3, help='the gamma value used for discounting in RL')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    return args

@torch.no_grad()
def measureReward(available, hands, choices, normalize=True, return_direction=False, verbose=None):
    assert choices.ndim==2, f"choices should be a (batch_size, max_output) tensor of indices, it is: {choices.shape}"
    batch_size, max_output = choices.shape
    num_in_hand = hands.shape[1]
    device = transformers.get_device(choices)

    # check verbose
    if verbose is not None:
        debug = True
        assert 0 <= verbose < batch_size, "verbose should be an index corresponding to one of the batch elements"
    else:
        debug = False
    
    # initialize these tracker variables
    next_available = torch.tensor(available, dtype=torch.float).to(device)
    havent_played = torch.ones((batch_size, num_in_hand+1), dtype=torch.bool).to(device) # True until dominoe has been played (include null for easier coding b/c out_choices includes idx to null
    hands = torch.tensor(hands, dtype=torch.float).to(device)
    handsOriginal = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1) # keep track of original values
    handsUpdates = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1) # Add null choice as [-1,-1]
    
    rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
    if return_direction: 
        direction = -torch.ones((batch_size, max_output), dtype=torch.float).to(device)

    if debug:
        print("Original hand:\n", hands[verbose])
        
    # then for each output:
    for idx in range(max_output):
        # idx of outputs that chose the null token
        idx_null = choices[:,idx] == num_in_hand

        if debug:
            print('')
            print("\nNew loop in measure reward:\n")
            print("next_available:", next_available[verbose])
            print("Choice: ", choices[verbose,idx])
            print("IdxNull: ", idx_null[verbose])
        
        # idx of outputs that have not already been chosen
        idx_not_played = torch.gather(havent_played, 1, choices[:, idx].view(-1,1)).squeeze(1)
        havent_played.scatter_(1, choices[:,idx].view(-1,1), torch.zeros((batch_size,1), dtype=torch.bool).to(device))

        # idx of play (not the null token and a dominoe that hasn't been played yet)
        idx_play = ~idx_null & idx_not_played
        
        # the dominoe that is chosen (including the null token)
        next_play = torch.gather(handsOriginal, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

        # figure out whether play was valid, whcih direction, and next available
        valid_play = torch.any(next_play.T == next_available, 0) # figure out if available matches a value in each dominoe
        next_value_idx = 1*(next_play[:,0]==next_available) # if true, then 1 is next value, if false then 0 is next value (for valid plays)
        new_available = torch.gather(next_play, 1, next_value_idx.view(-1,1)).squeeze(1) # get next available value
        next_available[valid_play] = new_available[valid_play] # update available for all valid plays

        # after getting next play, set any dominoe that has been played to [-1, -1]
        insert_values = next_play.clone()
        insert_values[valid_play] = -1
        handsUpdates.scatter_(1, choices[:, idx].view(-1,1,1).expand(-1,-1,2), insert_values.unsqueeze(1))

        if return_direction:
            play_direction = 1.0*(next_value_idx==0)
            direction[idx_play & valid_play, idx] = play_direction[idx_play & valid_play].float()
            
        # if dominoe that is a valid_play and hasn't been played yet is selected, add reward
        idx_good_play = ~idx_null & idx_not_played & valid_play
        rewards[idx_good_play, idx] += (torch.sum(next_play[idx_good_play], dim=1) + 1)
    
        # if a dominoe is chosen but is invalid, subtract points
        idx_bad_play = ~idx_null & (~idx_not_played | ~valid_play)
        rewards[idx_bad_play, idx] -= (torch.sum(next_play[idx_bad_play], dim=1) + 1)

        # determine which hands still have playable dominoes
        idx_still_playable = torch.any((handsUpdates == next_available.view(-1, 1, 1)).view(handsUpdates.size(0), -1), dim=1)

        # if the null is chosen and no other dominoes are possible, give some reward, otherwise negative reward
        rewards[idx_null & ~idx_still_playable, idx] += 1.0
        rewards[idx_null & idx_still_playable, idx] -= 1.0

        if debug:
            print("Next play: ", next_play[verbose])
            if return_direction:
                print("play_direction:", play_direction[verbose])
            print("next available: ", next_available[verbose])
            print("valid_play:", valid_play[verbose])
            print("idx_still_playable:", idx_still_playable[verbose])
            print("idx_play: ", idx_play[verbose])
            print("Hands updated:\n", handsUpdates[verbose])
            print("Rewards[verbose,idx]:", rewards[verbose, idx])
    
    if normalize:
        rewards /= (highestDominoe+1) 
        
    if return_direction:
        return rewards, direction
    else:        
        return rewards

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

def get_gamma_transform(gamma, N, B):
    exponent = torch.arange(N).view(-1,1) - torch.arange(N).view(1,-1)
    gamma_transform = (gamma ** exponent * (exponent >= 0)).unsqueeze(0).expand(B, -1, -1)
    return gamma_transform
    
def trainTestModel():
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)

    handSize = args.hand_size
    batchSize = args.batch_size
    null_token=True
    available_token=True
    num_output = handSize + (1 if null_token else 0)
    ignore_index = -1
    
    input_dim = (2 if not(available_token) else 3)*(highestDominoe+1) + (1 if null_token else 0)
    embedding_dim = args.embedding_dim
    heads = args.heads
    expansion = args.expansion
    encoding_layers = args.encoding_layers
    greedy = args.greedy
    temperature = args.temperature
    contextual_encoder = True # use available token as a context

    use_rl = args.use_rl
    return_target = not(use_rl) # if not using RL, then using supervised learning and we need the target

    if use_rl:
        gamma = args.gamma
        gamma_transform = get_gamma_transform(gamma, num_output, batchSize).to(device)
        
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs

    alpha = 1e-3
    weight_decay = 1e-5
    
    # Create a pointer network
    pnet = transformers.PointerNetwork(input_dim, embedding_dim, encoding_layers=encoding_layers, heads=heads, expansion=expansion, kqnorm=True, 
                                       contextual_encoder=contextual_encoder, decode_with_gru=False, greedy=greedy, temperature=temperature)
    pnet = pnet.to(device)
    pnet.train()

    # Create an optimizer, Adam with weight decay is pretty good
    optimizer = torch.optim.Adam(pnet.parameters(), lr=alpha, weight_decay=weight_decay)
    
    # Train network
    print("Training network...")
    trainValue = torch.zeros(trainEpochs)
    for epoch in tqdm(range(trainEpochs)):
        # zero gradients
        optimizer.zero_grad()
    
        # generate input batch
        batch = generateBatch(highestDominoe, listDominoes, batchSize, handSize, return_target=return_target,
                              null_token=null_token, available_token=available_token, ignore_index=ignore_index, return_full=True)

        # unpack batch tuple
        input, target, _, _, _, selection, available = batch

        # move to correct device
        input = input.to(device)
        if return_target:
            target = target.to(device)

        # divide input into main input and context
        x, context = input[:, :-1], input[:, [-1]]
        input = (x, context)
        
        # propagate it through the network
        out_scores, out_choices = pnet(input, max_output=num_output)
        
        # measure loss and do backward pass
        if use_rl:
            # measure rewards for each sequence
            rewards = measureReward(available, listDominoes[selection], out_choices, normalize=False)
            G = torch.bmm(rewards.unsqueeze(1), gamma_transform).squeeze(1)
            logprob_policy = torch.gather(out_scores, 2, out_choices.unsqueeze(2)).squeeze(2) # log-probability for each chosen dominoe
            
            # do backward pass on J
            J = -torch.sum(logprob_policy * G)
            J.backward()

            c_train_value = torch.mean(torch.sum(G, dim=1))
        else:
            unrolled = out_scores.view(batchSize * num_output, -1)
            loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignore_index)
            loss.backward()

            c_train_value = loss.item()
    
        # update network
        optimizer.step()
    
        trainValue[epoch] = c_train_value
    
    # Test network - same thing as in testing but without updates to model
    with torch.no_grad():
        print("Testing network...")
        pnet.eval()
        pnet.temperature = 1
        
        testValue = torch.zeros(testEpochs)
        for epoch in tqdm(range(testEpochs)):
            # generate input batch
            batch = generateBatch(highestDominoe, listDominoes, batchSize, handSize, return_target=return_target,
                                  null_token=null_token, available_token=available_token, ignore_index=ignore_index, return_full=True)
    
            # unpack batch tuple
            input, target, _, _, _, selection, available = batch
    
            # move to correct device
            input = input.to(device)
            if return_target:
                target = target.to(device)
    
            # divide input into main input and context
            x, context = input[:, :-1], input[:, [-1]]
            input = (x, context)
            
            # propagate it through the network
            out_scores, out_choices = pnet(input, max_output=num_output)
            
            # measure loss and do backward pass
            if use_rl:
                # measure rewards for each sequence
                rewards = measureReward(available, listDominoes[selection], out_choices, normalize=False)
                G = torch.bmm(rewards.unsqueeze(1), gamma_transform).squeeze(1)
                c_test_value = torch.mean(torch.sum(G, dim=1))
            else:
                unrolled = out_scores.view(batchSize * num_output, -1)
                loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignore_index)
                c_test_value = loss.item()
        
            testValue[epoch] = c_test_value
            

    results = {
        'trainValue': trainValue,
        'testValue': testValue,
    }

    return results, pnet


def plotResults(results, args):
    ylabel = 'G' if args.use_rl else 'Loss'
    title = lambda x : x + ('Reward' if args.use_rl else 'Loss')
    fig, ax = plt.subplots(1,2,figsize=(8,4), layout='constrained')
    ax[0].plot(range(args.train_epochs), results['trainValue'], color='k', lw=1)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel(ylabel)
    if not(args.use_rl):
        ax[0].set_ylim(0)
    yMin, yMax = ax[0].get_ylim()
    ax[0].set_title(title('Training'))
    
    ax[1].plot(range(args.test_epochs), results['testValue'], color='b', lw=1)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel(ylabel)
    ax[1].set_ylim(yMin, yMax)
    ax[1].set_title(title('Testing'))

    # ax[2].scatter(results['testSeqLength'], results['testLoss'], color='b', alpha=0.5)
    # ax[2].set_xlabel('Sequence Length')
    # ax[2].set_ylabel('Testing Loss')
    # # ax[2].set_ylim(yMin, yMax)
    # ax[2].set_title('Test Loss vs. Sequence Length')

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()
    
if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        # train and test pointerNetwork 
        results, pnet = trainTestModel()
        
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
            torch.save(pnet, savePath / getFileName(network=True))
            np.save(prmsPath / getFileName(), vars(args))
            np.save(resPath / getFileName(), results)
        
    else:
        prms = np.load(prmsPath / (getFileName()+'.npy'), allow_pickle=True).item()
        assert prms.keys() <= vars(args).keys(), f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(args).keys())}"
        for (pk,pi), (ak,ai) in zip(prms.items(), vars(args).items()):
            if pk=='justplot': continue
            if pk=='nosave': continue
            if prms[pk] != vars(args)[ak]:
                print(f"Requested argument {ak}={ai} differs from saved, which is: {pk}={pi}. Using saved...")
                setattr(args,pk,pi)
        
        results = np.load(resPath / (getFileName()+'.npy'), allow_pickle=True).item()
        
    plotResults(results, args)
    
    
    






