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
    return "pointerSequencer"+('.pt' if network else '')

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
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    return args

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

def generateBatch(highestDominoe, dominoes, batch_size, numInHand, available_token=False, null_token=False, ignore_index=-1, return_full=False):
    input, selection, available = randomDominoeHand(numInHand, dominoes, highestDominoe, batch_size=batch_size, null_token=null_token, available_token=available_token)
    if available_token:
        bestSequence, bestDirection = getBestLineFromAvailable(dominoes, selection, highestDominoe, available)
        mask = torch.ones((batch_size, numInHand+1), dtype=torch.float)
        mask[:,-1]=0
    else:
        bestSequence, bestDirection = getBestLine(dominoes, selection, highestDominoe)
        mask = torch.ones((batch_size, numInHand+1))
    iseq = convertToHandIndex(selection, bestSequence)
    null_index = ignore_index if not(null_token) else numInHand
    target = torch.tensor(np.stack(padBestLine(iseq, numInHand+(1 if null_token else 0), ignore_index=null_index)), dtype=torch.long)
    if return_full:
        return input, target, mask, bestSequence, bestDirection, selection, available
    return input, target, mask
    
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
    
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs

    alpha = 1e-3
    weight_decay = 1e-5
    
    # Create a pointer network
    pnet = transformers.PointerNetwork(input_dim, embedding_dim, encoding_layers=encoding_layers, heads=heads, expansion=expansion, kqnorm=True, decode_with_gru=False, greedy=greedy, temperature=temperature)
    pnet = pnet.to(device)
    pnet.train()

    # Create an optimizer, Adam with weight decay is pretty good
    optimizer = torch.optim.Adam(pnet.parameters(), lr=alpha, weight_decay=weight_decay)
    
    # Train network
    print("Training network...")
    trainLoss = torch.zeros(trainEpochs)
    for epoch in tqdm(range(trainEpochs)):
        # zero gradients
        optimizer.zero_grad()
    
        # generate input batch
        input, target, mask = generateBatch(highestDominoe, listDominoes, batchSize, handSize, null_token=null_token, available_token=available_token, ignore_index=ignore_index)
        input, target, mask = input.to(device), target.to(device), mask.to(device)
    
        # propagate it through the network
        out_scores, out_choices = pnet(input, max_output=num_output)
        
        # measure loss and do backward pass
        unrolled = out_scores.view(batchSize * num_output, -1)
        loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignore_index)
        loss.backward()
    
        # update network
        optimizer.step()
    
        trainLoss[epoch] = loss.item()

    # Test network - same thing as in testing but without updates to model
    with torch.no_grad():
        print("Testing network...")
        pnet.eval()
        pnet.temperature = 1
        
        testLoss = torch.zeros(testEpochs)
        for epoch in tqdm(range(testEpochs)):
            # generate input batch
            input, target, mask = generateBatch(highestDominoe, listDominoes, batchSize, handSize, null_token=null_token, available_token=available_token, ignore_index=ignore_index)
            input, target, mask = input.to(device), target.to(device), mask.to(device)
        
            # propagate it through the network
            out_scores, out_choices = pnet(input, max_output=num_output)
            
            # measure loss and do backward pass
            unrolled = out_scores.view(batchSize * num_output, -1)
            loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignore_index)
            
            testLoss[epoch] = loss.item()
            

    results = {
        'trainLoss': trainLoss,
        'testLoss': testLoss,
    }

    return results, pnet


def plotResults(results, args):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(range(args.train_epochs), results['trainLoss'], color='k', lw=1)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim(0)
    ax[0].set_title('Training Loss')
    yMin, yMax = ax[0].get_ylim()
    
    ax[1].plot(range(args.test_epochs), results['testLoss'], color='b', lw=1)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_ylim(yMin, yMax)
    ax[1].set_title('Testing Loss')

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
            torch.save(pnet, savePath / getFileName())
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
    
    
    






