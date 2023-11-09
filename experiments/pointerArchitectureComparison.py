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
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "pointerArchitectureComparison"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer demonstration.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('--train-fraction', type=float, default=0.8, help='the fraction of dominoes in the set to train with')
    parser.add_argument('-hs','--hand-size', type=int, default=8, help='tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=512, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=5000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=100, help='the number of testing epochs')
    parser.add_argument('--gamma', type=float, default=1.0, help='discounting factor')
    parser.add_argument('-nr','--num-runs', type=int, default=5, help='how many networks to train of each type')
    
    parser.add_argument('--embedding-dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=1, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--train-temperature',type=float, default=5.0, help='temperature for training')
    
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    assert args.train_fraction > 0 and args.train_fraction <= 1, "train fraction must be greater than 0 and less than or equal to 1"
    assert args.gamma > 0 and args.gamma <= 1, "gamma must be greater than 0 or less than or equal to 1"
    
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
    handSize = args.hand_size
    batchSize = args.batch_size
    maxOutput = copy(handSize)
    
    # network parameters
    input_dim = 2*(highestDominoe+1)
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    greedy = True 

    # policy parameters
    with_thompson = True
    temperature = args.train_temperature

    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    numRuns = args.num_runs

    # create gamma transform matrix
    gamma = args.gamma
    exponent = torch.arange(maxOutput).view(-1,1) - torch.arange(maxOutput).view(1,-1)
    gamma_transform = (gamma ** exponent * (exponent >= 0)).unsqueeze(0).expand(batchSize, -1, -1).to(device)
    
    # batch inputs for reinforcement learning training
    batch_inputs = {'null_token':False, 
                    'available_token':False, 
                    'ignore_index':ignoreIndex, 
                    'return_full':True,
                    'return_target':False,
                   }
    
    numNets = len(POINTER_METHODS)
    
    # Train network
    print("Training networks...")
    trainReward = torch.zeros((trainEpochs, numNets, numRuns))
    testReward = torch.zeros((testEpochs, numNets, numRuns))
    trainRewardByPos = torch.zeros((trainEpochs, maxOutput, numNets, numRuns))
    testRewardByPos = torch.zeros((testEpochs, maxOutput, numNets, numRuns))
    trainScoreByPos = torch.zeros((trainEpochs, maxOutput, numNets, numRuns))
    testScoreByPos = torch.zeros((testEpochs, maxOutput, numNets, numRuns))
    for run in range(numRuns):
        # reset train set of dominoes
        fullDominoes, trainDominoes, trainIndex = trainTestDominoes(baseDominoes, args.train_fraction)

        # create pointer networks with different pointer methods
        nets = [transformers.PointerNetwork(input_dim, embedding_dim, temperature=temperature, pointer_method=POINTER_METHOD, 
                                             thompson=with_thompson, encoding_layers=encoding_layers, heads=heads, kqnorm=True, 
                                             decoder_method='transformer', greedy=greedy)
                 for POINTER_METHOD in POINTER_METHODS]
        nets = [net.to(device) for net in nets]
        
        # Create an optimizer, Adam with weight decay is pretty good
        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) for net in nets]
        
        print(f"Doing training run {run+1}/{numRuns}...")
        for epoch in tqdm(range(trainEpochs)):
            batch = datasets.generateBatch(highestDominoe, trainDominoes, batchSize, handSize, **batch_inputs)
        
            # unpack batch tuple
            input, _, _, _, _, selection, _ = batch
            input = input.to(device)
            
            # zero gradients, get output of network
            for opt in optimizers: opt.zero_grad()
            log_scores, choices = map(list, zip(*[net(input, max_output=maxOutput) for net in nets]))
        
            # log-probability for each chosen dominoe
            logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
            
            # measure reward
            rewards = [training.measureReward_sortDescend(trainDominoes[selection], choice) for choice in choices]
            G = [torch.bmm(reward.unsqueeze(1), gamma_transform).squeeze(1) for reward in rewards]
        
            # measure J
            J = [-torch.sum(logpol * g) for logpol, g in zip(logprob_policy, G)]
            for j in J: j.backward()

            # update networks
            for opt in optimizers: opt.step()
            
            # save training data
            with torch.no_grad():
                for i in range(numNets):
                    trainReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1)).detach()
                    trainRewardByPos[epoch, :, i, run] = torch.mean(rewards[i], dim=0).detach()

                    # Measure the models confidence -- ignoring the effect of temperature
                    pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                    pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                    trainScoreByPos[epoch, :, i, run] = torch.mean(pretemp_policy, dim=0).detach()
                    
        
        with torch.no_grad():
            # always return temperature to 1 and thompson to False for testing networks
            for net in nets: 
                net.setTemperature(1.0)
                net.setThompson(False)
                
            print('Testing network...')
            for epoch in tqdm(range(testEpochs)):
                batch = datasets.generateBatch(highestDominoe, fullDominoes, batchSize, handSize, **batch_inputs)
        
                # unpack batch tuple
                input, _, _, _, _, selection, _ = batch
            
                # move to correct device
                input = input.to(device)

                # get output of networks
                log_scores, choices = map(list, zip(*[net(input, max_output=maxOutput) for net in nets]))
            
                # log-probability for each chosen dominoe
                logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
                
                # measure reward
                rewards = [training.measureReward_sortDescend(fullDominoes[selection], choice) for choice in choices]
                
                # save training data
                for i in range(numNets):
                    testReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1))
                    testRewardByPos[epoch, :, i, run] = torch.mean(rewards[i], dim=0)

                    # Measure the models confidence -- ignoring the effect of temperature
                    pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                    pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                    testScoreByPos[epoch, :, i, run] = torch.mean(pretemp_policy, dim=0).detach()
                    
    results = {
        'trainReward': trainReward,
        'testReward': testReward,
        'trainRewardByPos': trainRewardByPos,
        'testRewardByPos': testRewardByPos,
        'trainScoreByPos': trainScoreByPos,
        'testScoreByPos': testScoreByPos,
    }
    
    return results, nets


@torch.no_grad()
def eigenAnalyses(nets, args):
    # get a "normal" batch
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)
    
    # do subselection for training
    doubleDominoes = listDominoes[:,0] == listDominoes[:,1]
    nonDoubleReverse = listDominoes[~doubleDominoes][:,[1,0]] # represent each dominoe in both orders
    
    # list of full set of dominoe representations and value of each
    listDominoes = np.concatenate((listDominoes, nonDoubleReverse), axis=0)
    dominoeValue = np.sum(listDominoes, axis=1)
    
    # training inputs
    numDominoes = len(listDominoes)
    dominoeValue = np.sum(listDominoes, axis=1)
    batchSize = args.batch_size * 2 # lots of data!
    handSize = args.hand_size
    batch_inputs = {'null_token' : False, 
                    'available_token' : False, 
                    'ignore_index' :- 1, 
                    'return_full' : True,
                    'return_target' : False,
                   }
    
    selection = np.array([])
    while len(np.unique(selection))!=numDominoes:
        batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, **batch_inputs)
        
        # unpack batch tuple
        input, _, _, _, _, selection, available = batch
        input = input.to(device)
    
    # pre-forward
    batch_size, tokens, _ = input.size()
    mask = torch.ones((batch_size, tokens), dtype=input.dtype).to(device)
    
    # encoding
    embedded = [net.embedding(input) for net in nets]
    encoded = [net.encoding(embed) for net, embed in zip(nets, embedded)]

    # translate to (token x embedding)
    encoded = [encode.view(-1, encode.size(2)).T for encode in encoded]

    # get covariance
    cov = [torch.cov(encode) for encode in encoded]

    # get eigenvalues
    eigvals, eigvecs = map(list, zip(*[torch.linalg.eigh(c) for c in cov]))
    eigvals = [eigval.cpu().numpy() for eigval in eigvals]
    eigvecs = [eigvec.cpu().numpy() for eigvec in eigvecs]

    # sort highest to lowest
    idx_sort = [np.argsort(-eigval) for eigval in eigvals]
    eigvals = [eigval[isort] for eigval, isort in zip(eigvals, idx_sort)]
    eigvecs = [eigvec[:,isort] for eigvec, isort in zip(eigvecs, idx_sort)]

    # negative values are numerical errors
    eigvals = [np.maximum(eigval, 0) for eigval in eigvals] 
    
    return eigvals, eigvecs



def plotResults(results, args, eigvals):
    numNets = results['trainReward'].shape[2]
    numPos = results['testScoreByPos'].shape[1]
    n_string = f" (N={numNets})"
    cmap = mpl.colormaps['tab10']
    trainInspectFrom = [1500, 2000]
    trainInspect = slice(trainInspectFrom[0], trainInspectFrom[1])
    
    fig, ax = plt.subplots(1,2,figsize=(7,3.5), width_ratios=[2, 1], layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax[0].plot(range(args.train_epochs), torch.mean(results['trainReward'][:,idx],dim=1), color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel('Mean Reward'+n_string)
    ax[0].set_title('Training Performance')
    ax[0].set_ylim(None, 8)
    yMin0, yMax0 = ax[0].get_ylim()

    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.mean(results['testReward'][:,idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestReward.mean(), mnTestReward.mean()], color=cmap(idx), lw=4, label=name)
        ax[1].plot([idx,idx], [mnTestReward.min(), mnTestReward.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel('Reward'+n_string)
    ax[1].set_title('Testing Performance')
    ax[1].legend(loc='lower center', fontsize=9)
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(6, 8)
    ax[1].set_yticks([6, 7, 8])

    # create inset to show initial train trajectory
    inset = ax[0].inset_axes([0.5, 0.1, 0.45, 0.45])
    for idx, name in enumerate(POINTER_METHODS):
        inset.plot(range(args.train_epochs), torch.mean(results['trainReward'][:,idx],dim=1), color=cmap(idx), lw=1.2, label=name)
    inset.set_xlim(-20, 400)
    inset.set_ylim(yMin0, 8)
    inset.set_xticks([0, 200, 400])
    inset.set_xticklabels(inset.get_xticklabels(), fontsize=8)
    inset.set_yticklabels([])
    inset.set_title('Initial Epochs', fontsize=10)
    
    width = trainInspectFrom[1]-trainInspectFrom[0]
    height = yMax0 - yMin0
    rect = mpl.patches.Rectangle([trainInspectFrom[0], yMin0], width, height, facecolor='k', edgecolor='none', alpha=0.2)
    ax[0].add_patch(rect)

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()


    # now show confidence by position figure
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        mnTrainScore = torch.mean(results['trainScoreByPos'][trainInspect, :, idx], dim=(0,2))
        ax[0].plot(range(numPos), mnTrainScore, color=cmap(idx), lw=1.2, label=name)
    for idx, name in enumerate(POINTER_METHODS):
        mnTrainScore = torch.mean(results['trainScoreByPos'][trainInspect, :, idx], dim=(0,2))
        ax[0].scatter(range(numPos), mnTrainScore, color=cmap(idx), marker='o', s=24, edgecolor='none')
    ax[0].set_ylim(None, 1)
    ax[0].set_xlabel('Output Position')
    ax[0].set_ylabel('Mean Score'+n_string)
    ax[0].set_title(f"Train Confidence (in grey patch)")
    yMin2, yMax2 = ax[0].get_ylim()
    
    for idx, name in enumerate(POINTER_METHODS):
        mnTestScore = torch.mean(results['testScoreByPos'][:, :, idx], dim=(0,2))
        ax[1].plot(range(numPos), mnTestScore, color=cmap(idx), lw=1.2, marker='o', markersize=np.sqrt(24), label=name)
    ax[1].set_xlabel('Output Position')
    ax[1].set_ylabel('Mean Score'+n_string)
    ax[1].set_title('Test Confidence')
    ax[1].legend(loc='lower right', fontsize=9)
    yMin3, yMax3 = ax[1].get_ylim()
    
    new_ymin = min(yMin2, yMin3)
    ax[0].set_ylim(new_ymin, 1)
    ax[1].set_ylim(new_ymin, 1)
    
    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName('confidence')))
    
    plt.show()


    # Show eigenvalue analysis
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), width_ratios=[2,1], layout='constrained')
    for idx, (name, eigval) in enumerate(zip(POINTER_METHODS, eigvals)):
        ax[0].plot(range(len(eigval)), eigval/np.sum(eigval), color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlim(-1, len(eigval))
    ax[0].set_xlabel('Dimension')
    ax[0].set_ylabel('Fraction of Variance')
    ax[0].set_title('Eigenspectrum')
    ax[0].set_yscale('log')
    ax[0].legend(loc='lower left', fontsize=9)

    for idx, (name, eigval) in enumerate(zip(POINTER_METHODS, eigvals)):
        ax[1].plot(idx, sp.stats.entropy(eigval), color=cmap(idx), marker='o', linestyle='none', label=name)
    ax[1].set_ylabel('Entropy of Eigenspectrum')
    ax[1].set_title('Dimensionality')
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    _, yMax = ax[1].get_ylim()
    ax[1].set_ylim(0, 2.5)
    ax[1].set_yticks(np.linspace(0, 2, 3))
    
    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName('eigenvalues')))
        
    plt.show()
    
    
    
if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        results, nets = trainTestModel()
        nets = [net.to('cpu') for net in nets]
            
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
            for net, method in zip(nets, POINTER_METHODS):
                save_name = f"{method}.pt"
                torch.save(net, savePath / getFileName(extra=save_name))
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
        nets = [torch.load(savePath / getFileName(extra=f"{method}.pt")).to(device) for method in POINTER_METHODS]

    eigval, eigvec = eigenAnalyses(nets, args)
    plotResults(results, args, eigval)
    
    
    






