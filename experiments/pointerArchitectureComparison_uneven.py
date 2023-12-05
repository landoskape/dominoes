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
from dominoes import fileManagement as fm
from dominoes import functions as df
from dominoes import datasets
from dominoes import training
from dominoes import transformers
from dominoes.utils import loadSavedExperiment

device = 'cuda' if torchCuda.is_available() else 'cpu'

# general variables for experiment
POINTER_METHODS = ['PointerStandard', 'PointerDot', 'PointerDotLean', 'PointerDotNoLN', 'PointerAttention', 'PointerTransformer']

# path strings
netPath = fm.netPath()
resPath = fm.resPath()
prmsPath = fm.prmPath()
figsPath = fm.figsPath()

for path in (resPath, prmsPath, figsPath, netPath):
    if not(path.exists()):
        path.mkdir()

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "pointerArchitectureComparison_uneven"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer demonstration.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('--train-fraction', type=float, default=0.8, help='the fraction of dominoes in the set to train with')
    parser.add_argument('-mn','--min-hand-size', type=int, default=4, help='min tokens per sequence')
    parser.add_argument('-mx','--max-hand-size', type=int, default=12, help='max tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=512, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=4000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=100, help='the number of testing epochs')
    parser.add_argument('--gamma', type=float, default=1.0, help='discounting factor')
    parser.add_argument('-nr','--num-runs', type=int, default=8, help='how many networks to train of each type')
    
    parser.add_argument('--embedding-dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=1, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--expansion', type=int, default=4, help='expansion in FF layer of transformer in encoder')
    parser.add_argument('--train-temperature',type=float, default=5.0, help='temperature for training')
    
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--printargs', default=True, action='store_true')
    
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

def makeGammaTransform(gamma, N):
    exponent = torch.arange(N).view(-1,1) - torch.arange(N).view(1,-1)
    gamma_transform = (gamma ** exponent * (exponent >= 0))
    return gamma_transform.to(device)
    
def trainTestModel():
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    baseDominoes = df.listDominoes(highestDominoe)
    
    # other batch parameters
    ignoreIndex = -1 # this is only used when doing uneven batch sizes, which I'm not in this experiment
    minHandSize = args.min_hand_size
    maxHandSize = args.max_hand_size
    batchSize = args.batch_size
    maxOutput = copy(maxHandSize)
    
    # network parameters
    input_dim = 2*(highestDominoe+1)
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    expansion = args.expansion

    # policy parameters
    with_thompson = True
    temperature = args.train_temperature

    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    numRuns = args.num_runs

    # create gamma transform matrix
    gamma = args.gamma
    
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
    trainReward = torch.full((trainEpochs, numNets, numRuns), torch.nan)
    testReward = torch.full((testEpochs, numNets, numRuns), torch.nan)
    trainHandSize = torch.zeros(trainEpochs)
    testHandSize = torch.zeros(testEpochs)
    trainRewardByPos = torch.full((trainEpochs, maxOutput, numNets, numRuns), torch.nan)
    testRewardByPos = torch.full((testEpochs, maxOutput, numNets, numRuns), torch.nan)
    trainScoreByPos = torch.full((trainEpochs, maxOutput, numNets, numRuns), torch.nan)
    testScoreByPos = torch.full((testEpochs, maxOutput, numNets, numRuns), torch.nan)
    for run in range(numRuns):
        # reset train set of dominoes
        fullDominoes, trainDominoes, trainIndex = trainTestDominoes(baseDominoes, args.train_fraction)

        # create pointer networks with different pointer methods
        nets = [transformers.PointerNetwork(input_dim, embedding_dim, temperature=temperature, pointer_method=POINTER_METHOD, 
                                             thompson=with_thompson, encoding_layers=encoding_layers, heads=heads, kqnorm=True, 
                                             decoder_method='transformer', expansion=expansion)
                 for POINTER_METHOD in POINTER_METHODS]
        nets = [net.to(device) for net in nets]
        
        # Create an optimizer, Adam with weight decay is pretty good
        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) for net in nets]
        
        print(f"Doing training run {run+1}/{numRuns}...")
        for epoch in tqdm(range(trainEpochs)):
            c_handsize = np.random.randint(minHandSize, maxHandSize+1)
            trainHandSize[epoch] = c_handsize
            gamma_transform = makeGammaTransform(gamma, c_handsize)
            batch = datasets.generateBatch(highestDominoe, trainDominoes, batchSize, c_handsize, **batch_inputs)
        
            # unpack batch tuple
            input, _, _, _, _, selection, _ = batch
            input = input.to(device)
            
            # zero gradients, get output of network
            for opt in optimizers: opt.zero_grad()
            log_scores, choices = map(list, zip(*[net(input, max_output=c_handsize) for net in nets]))
        
            # log-probability for each chosen dominoe
            logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
            
            # measure reward
            rewards = [training.measureReward_sortDescend(trainDominoes[selection], choice) for choice in choices]
            G = [torch.matmul(reward, gamma_transform) for reward in rewards]
        
            # measure J
            J = [-torch.sum(logpol * g) for logpol, g in zip(logprob_policy, G)]
            for j in J: j.backward()

            # update networks
            for opt in optimizers: opt.step()
            
            # save training data
            with torch.no_grad():
                for i in range(numNets):
                    trainReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1)).detach() * 100 / c_handsize
                    trainRewardByPos[epoch, :c_handsize, i, run] = torch.mean(rewards[i], dim=0).detach()

                    # Measure the models confidence -- ignoring the effect of temperature
                    pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                    pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                    trainScoreByPos[epoch, :c_handsize, i, run] = torch.mean(pretemp_policy, dim=0).detach()
                    
        
        with torch.no_grad():
            # always return temperature to 1 and thompson to False for testing networks
            for net in nets: 
                net.setTemperature(1.0)
                net.setThompson(False)
                
            print('Testing network...')
            for epoch in tqdm(range(testEpochs)):
                c_handsize = np.random.randint(minHandSize, maxHandSize+1)
                testHandSize[epoch] = c_handsize
                gamma_transform = makeGammaTransform(gamma, c_handsize)
                batch = datasets.generateBatch(highestDominoe, fullDominoes, batchSize, c_handsize, **batch_inputs)
        
                # unpack batch tuple
                input, _, _, _, _, selection, _ = batch
            
                # move to correct device
                input = input.to(device)

                # get output of networks
                log_scores, choices = map(list, zip(*[net(input, max_output=c_handsize) for net in nets]))
            
                # log-probability for each chosen dominoe
                logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
                
                # measure reward
                rewards = [training.measureReward_sortDescend(fullDominoes[selection], choice) for choice in choices]
                
                # save training data
                for i in range(numNets):
                    testReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1)) * 100 / c_handsize
                    testRewardByPos[epoch, :c_handsize, i, run] = torch.mean(rewards[i], dim=0)

                    # Measure the models confidence -- ignoring the effect of temperature
                    pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                    pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                    testScoreByPos[epoch, :c_handsize, i, run] = torch.mean(pretemp_policy, dim=0).detach()
                    
    results = {
        'trainReward': trainReward,
        'testReward': testReward,
        'trainHandSize': trainHandSize,
        'testHandSize': testHandSize,
        'trainRewardByPos': trainRewardByPos,
        'testRewardByPos': testRewardByPos,
        'trainScoreByPos': trainScoreByPos,
        'testScoreByPos': testScoreByPos,
    }
    
    return results, nets



def plotResults(results, args):
    numNets = results['trainReward'].shape[2]
    numPos = results['testScoreByPos'].shape[1]
    n_string = f" (N={numNets})"
    cmap = mpl.colormaps['tab10']
    trainInspectFrom = [200, 300]
    trainInspect = slice(trainInspectFrom[0], trainInspectFrom[1])
    savgol_width = 20 if args.train_epochs > 100 else 2
    savgol_order = 1
    smooth_train_trajectory = np.mean(sp.signal.savgol_filter(results['trainReward'].numpy(), 
                                                              savgol_width, 
                                                              savgol_order, 
                                                              axis=0), axis=2)
    
    fig, ax = plt.subplots(1,2,figsize=(7,3.5), width_ratios=[2, 1], layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax[0].plot(range(args.train_epochs), smooth_train_trajectory[:,idx], color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel('Mean Reward'+n_string)
    ax[0].set_title('Training Performance')
    ax[0].set_xlim(-50, 1000)
    ax[0].legend(loc='lower right', fontsize=9)
    yMin0, yMax0 = ax[0].get_ylim()

    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.nanmean(results['testReward'][:,idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestReward.mean(), mnTestReward.mean()], color=cmap(idx), lw=4, label=name)
        for mtr in mnTestReward:
            ax[1].plot(get_x(idx), [mtr, mtr], color=cmap(idx), lw=1.5)
        ax[1].plot([idx,idx], [mnTestReward.min(), mnTestReward.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel('Reward'+n_string)
    ax[1].set_title('Testing Performance')
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(97.95, 100.05)
    ax[1].set_yticks([98, 99, 100])

    idx = {val: idx for idx, val in enumerate(POINTER_METHODS)}['PointerDotLean']
    mnDotLean = torch.min(torch.nanmean(results['testReward'][:,idx], dim=0))
    ax[1].text(idx+0.25, 98.2, f"1 net @ {mnDotLean:.1f}", ha='left', va='center', color=cmap(idx), fontsize=10)
    
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
        mnTrainScore = torch.nanmean(results['trainScoreByPos'][trainInspect, :, idx], dim=(0,2))
        ax[0].plot(range(1,numPos+1), mnTrainScore, color=cmap(idx), lw=1.2, label=name)
    for idx, name in enumerate(POINTER_METHODS):
        mnTrainScore = torch.nanmean(results['trainScoreByPos'][trainInspect, :, idx], dim=(0,2))
        ax[0].scatter(range(1,numPos+1), mnTrainScore, color=cmap(idx), marker='o', s=24, edgecolor='none')
    ax[0].set_xlim(0.5, numPos+0.5)
    ax[0].set_ylim(None, 1.05)
    ax[0].set_xticks(range(1,numPos+1))
    ax[0].set_xlabel('Output Position')
    ax[0].set_ylabel('Mean Score'+n_string)
    ax[0].set_title(f"Train Confidence (in grey patch)")
    yMin2, yMax2 = ax[0].get_ylim()
    
    for idx, name in enumerate(POINTER_METHODS):
        mnTestScore = torch.nanmean(results['testScoreByPos'][:, :, idx], dim=(0,2))
        ax[1].plot(range(1,numPos+1), mnTestScore, color=cmap(idx), lw=1.2, marker='o', markersize=np.sqrt(24), label=name)
    ax[1].set_xlim(0.5, numPos+0.5)
    ax[1].set_ylim(None, 1.05)
    ax[1].set_xticks(range(1,numPos+1))
    ax[1].set_xlabel('Output Position')
    ax[1].set_ylabel('Mean Score'+n_string)
    ax[1].set_title('Test Confidence')
    ax[1].legend(loc='lower right', fontsize=9)
    yMin3, yMax3 = ax[1].get_ylim()
    
    new_ymin = min(yMin2, yMin3)
    ax[0].set_ylim(new_ymin, 1)
    ax[1].set_ylim(0.94, 1)
    
    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName('confidence')))
    
    plt.show()

    
    
if __name__=='__main__':
    args = handleArguments()
    show_results = True

    if args.printargs:
        _, args = loadSavedExperiment(prmsPath, resPath, getFileName(), args)
        for key, val in vars(args).items():
            print(f"{key}={val}")
        show_results = False

    elif not(args.justplot):
        results, nets = trainTestModel()
        nets = [net.to('cpu') for net in nets]
            
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
            for net, method in zip(nets, POINTER_METHODS):
                save_name = f"{method}.pt"
                torch.save(net, netPath / getFileName(extra=save_name))
            np.save(prmsPath / getFileName(), vars(args))
            np.save(resPath / getFileName(), results)
    
    else:
        results, args = loadSavedExperiment(prmsPath, resPath, getFileName(), args)
        
    if show_results:
        plotResults(results, args)
    
    
    






