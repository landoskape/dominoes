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
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.cuda as torchCuda

# dominoes package
from dominoes import functions as df
from dominoes import datasets
from dominoes import training
from dominoes import transformers

device = 'cuda' if torchCuda.is_available() else 'cpu'

# general variables for experiment
POINTER_METHODS = ['PointerStandard', 'PointerDot', 'PointerDotLean', 'PointerDotNoLN', 'PointerAttention', 'PointerTransformer']

# can edit this for each machine it's being used on
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

for path in (resPath, prmsPath, figsPath):
    if not(path.exists()):
        path.mkdir()
        
# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "sl_pointerArchitectureComparison"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer demonstration.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('--train-fraction', type=float, default=0.75, help='the fraction of dominoes in the set to train with')
    parser.add_argument('-mn','--min-seq-length', type=int, default=4, help='the minimum tokens per sequence')
    parser.add_argument('-mx','--max-seq-length', type=int, default=12, help='the maximum tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=512, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=8000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=100, help='the number of testing epochs')

    parser.add_argument('--embedding_dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=1, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--greedy', default=False, action='store_true', help='if used, will generate greedy predictions of each step rather than probability-weighted predictions')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    assert args.min_seq_length <= args.max_seq_length, "min seq length has to be less than or equal to max seq length"
    assert args.train_fraction > 0 and args.train_fraction <= 1, "train fraction must be greater than 0 and less than or equal to 1"
    
    return args
    

def trainTestDominoes(baseDominoes, trainFraction):
    # figure out which dominoes need to be represented in both ways (ones with different values)
    doubleDominoes = baseDominoes[:,0] == baseDominoes[:,1]
    nonDoubleReverse = baseDominoes[~doubleDominoes][:,[1,0]] # represent each dominoe in both orders

    # list of full set of dominoe representations and value of each
    fullDominoes = np.concatenate((baseDominoes, nonDoubleReverse), axis=0)
    
    # subset dominoes for training
    trainNumber = int(len(fullDominoes)*trainFraction)
    trainIndex = np.sort(np.random.permutation(len(fullDominoes))[:trainNumber]) # keep random fraction (but sort in same way)
    trainDominoes = fullDominoes[trainIndex]

    return fullDominoes, trainDominoes, trainIndex
    
    
def trainTestModel():
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    baseDominoes = df.listDominoes(highestDominoe)
    
    # other batch parameters
    ignoreIndex = -1 # this is only used when doing uneven batch sizes, which I'm not in this experiment
    minSeqLength = args.min_seq_length
    maxSeqLength = args.max_seq_length
    batchSize = args.batch_size
    
    # network parameters
    input_dim = 2*(highestDominoe+1)
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    greedy = True 
    temperature = 1
    
    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs

    numNets = len(POINTER_METHODS)
    fullDominoes, trainDominoes, trainIndex = trainTestDominoes(baseDominoes, args.train_fraction)
    trainValue = np.sum(trainDominoes, axis=1)
    fullValue = np.sum(fullDominoes, axis=1)

    # create pointer networks with different pointer methods
    nets = [transformers.PointerNetwork(input_dim, embedding_dim, temperature=temperature, pointer_method=POINTER_METHOD, 
                                         thompson=False, encoding_layers=encoding_layers, heads=heads, kqnorm=True, 
                                         decoder_method='transformer', greedy=greedy)
             for POINTER_METHOD in POINTER_METHODS]
    nets = [net.to(device) for net in nets]
    
    # Create an optimizer, Adam with weight decay is pretty good
    optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) for net in nets]
    
    print(f"Doing training...")
    trainLoss = torch.zeros((trainEpochs, numNets))
    testLoss = torch.zeros((testEpochs, numNets))
    trainPositionError = torch.full((trainEpochs, maxSeqLength, numNets), torch.nan) # keep track of where there was error
    trainMaxScore = torch.full((trainEpochs, maxSeqLength, numNets), torch.nan) # keep track of confidence of model
    testMaxScore = torch.full((testEpochs, maxSeqLength, numNets), torch.nan) 
    for epoch in tqdm(range(trainEpochs)):
        # generate batch
        input, target, mask = df.dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, trainDominoes, trainValue, highestDominoe, ignoreIndex=ignoreIndex)
        input, target, mask = input.to(device), target.to(device), mask.to(device)

        # zero gradients, get output of network
        for opt in optimizers: opt.zero_grad()
        log_scores, choices = map(list, zip(*[net(input) for net in nets]))

        # measure loss with negative log-likelihood
        unrolled = [log_score.view(-1, log_score.size(-1)) for log_score in log_scores]
        loss = [torch.nn.functional.nll_loss(unroll, target.view(-1), ignore_index=ignoreIndex) for unroll in unrolled]
        assert all([not np.isnan(l.item()) for l in loss]), "model diverged :("

        # update networks
        for l in loss: l.backward()
        for opt in optimizers: opt.step()
            
        # save training data
        for i, l in enumerate(loss):
            trainLoss[epoch, i] = l.item()

        # measure position dependent error 
        with torch.no_grad():
            # start by getting score for target at each position 
            target_noignore = target.clone().masked_fill_(target==-1, 0)
            target_score = [torch.gather(unroll, dim=1, index=target_noignore.view(-1,1)).view(batchSize, maxSeqLength) for unroll in unrolled]
            # then get max score for each position (which would correspond to the score of the actual choice)
            max_score = [torch.max(unroll, dim=1)[0].view(batchSize, maxSeqLength) for unroll in unrolled]
            # then calculate position error
            pos_error = [ms - ts for ms, ts in zip(max_score, target_score)] # high if the chosen score is much bigger than the target score
            # now remove locations where it is masked out
            for pe in pos_error:
                pe.masked_fill_(mask==0, torch.nan)
                
            # add to accounting
            for i, (pe, ms) in enumerate(zip(pos_error, max_score)):
                trainPositionError[epoch,:,i] = torch.nansum(pe, dim=0)
                trainMaxScore[epoch,:,i] = torch.nanmean(ms, dim=0)
                
    
    with torch.no_grad():
        print('Testing network...')
        for epoch in tqdm(range(testEpochs)):
            # generate batch
            input, target, mask = df.dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, fullDominoes, fullValue, highestDominoe, ignoreIndex=ignoreIndex)
            input, target, mask = input.to(device), target.to(device), mask.to(device)
    
            log_scores, choices = map(list, zip(*[net(input) for net in nets]))
    
            # measure loss with negative log-likelihood
            unrolled = [log_score.view(-1, log_score.size(-1)) for log_score in log_scores]
            loss = [torch.nn.functional.nll_loss(unroll, target.view(-1), ignore_index=ignoreIndex) for unroll in unrolled]
            assert all([not np.isnan(l.item()) for l in loss]), "model diverged :("

            # get max score 
            max_score = [torch.max(unroll, dim=1)[0].view(batchSize, maxSeqLength) for unroll in unrolled]
            
            # save training data
            for i, (l, ms) in enumerate(zip(loss, max_score)):
                testLoss[epoch, i] = l.item()
                testMaxScore[epoch,:,i] = torch.nanmean(ms, dim=0)
                
                    
    results = {
        'trainLoss': trainLoss,
        'testLoss': testLoss,
        'trainPositionError': trainPositionError,
        'trainMaxScore': trainMaxScore,
        'testMaxScore': testMaxScore,
    }
    
    return results

def plotResults(results, args):
    cmap = mpl.colormaps['tab10']
    
    # make plot of loss trajectory
    fig, ax = plt.subplots(1,3,figsize=(9,4), width_ratios=[2.6,1,1],layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax[0].plot(range(args.train_epochs), results['trainLoss'][:,idx], color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Performance')
    ax[0].set_ylim(0, None)
    yMin0, yMax0 = ax[0].get_ylim()

    
    # create inset to show initial train trajectory
    inset = ax[0].inset_axes([0.05, 0.52, 0.45, 0.4])
    for idx, name in enumerate(POINTER_METHODS):
        inset.plot(range(args.train_epochs), results['trainLoss'][:,idx], color=cmap(idx), lw=1.2, label=name)
    inset.set_xlim(-10, 300)
    inset.set_ylim(0, 3)
    inset.set_xticks([0, 150, 300])
    inset.set_xticklabels(inset.get_xticklabels(), fontsize=8)
    inset.set_yticklabels([])
    inset.set_title('Initial Epochs', fontsize=10)
    
    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    minFilt_trainLoss = sp.ndimage.minimum_filter1d(results['trainLoss'].numpy(), size=1500, axis=0)
    startFrom = 1000
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = np.mean(minFilt_trainLoss[startFrom:,idx])
        sdTestReward = np.std(minFilt_trainLoss[startFrom:,idx])
        ax[1].plot(get_x(idx), [mnTestReward, mnTestReward], color=cmap(idx), lw=4, label=name)
        ax[1].plot([idx,idx], [mnTestReward-sdTestReward, mnTestReward+sdTestReward], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Train Perf. (MinimumFilt)')
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc='upper center', fontsize=8)

    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.mean(results['testLoss'][:,idx])
        sdTestReward = torch.std(results['testLoss'][:,idx])
        ax[2].plot(get_x(idx), [mnTestReward, mnTestReward], color=cmap(idx), lw=4, label=name)
        ax[2].plot([idx,idx], [mnTestReward-sdTestReward, mnTestReward+sdTestReward], color=cmap(idx), lw=1.5)
    ax[2].set_xticks(range(len(POINTER_METHODS)))
    ax[2].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[2].set_ylabel('Loss')
    ax[2].set_title('Test Performance')
    ax[2].set_xlim(-1, len(POINTER_METHODS))
    ax[2].set_ylim(0, 1)

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()

    # now plot confidence across positions
    numPos = results['testMaxScore'].size(1)
    fig, ax = plt.subplots(1, 1, figsize=(4,4), layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax.plot(range(numPos), torch.mean(torch.exp(results['testMaxScore'][:,:,idx]), dim=0), color=cmap(idx), lw=1, marker='o', label=name)
    ax.set_xlabel('Output Position')
    ax.set_ylabel('Mean Score')
    ax.set_title('Confidence')
    ax.set_ylim(0, None)
    ax.legend(loc='lower left', fontsize=8)

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName(extra='confidence')))
        
    plt.show()
    

    
if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        # train and test pointerNetwork 
        results = trainTestModel()
        
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
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



