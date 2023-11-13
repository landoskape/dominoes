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
    baseName = "ptrArchComp_TSP_RL"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer demonstration.')
    parser.add_argument('-nc','--num-cities', type=int, default=8, help='the number of cities')
    parser.add_argument('-bs','--batch-size',type=int, default=128, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=8000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=100, help='the number of testing epochs')
    parser.add_argument('-nr','--num-runs',type=int, default=5, help='how many runs for each network to train')
    parser.add_argument('--gamma', type=float, default=1.0, help='discounting factor')
    parser.add_argument('--temperature',type=float, default=5.0, help='temperature for training')
    
    parser.add_argument('--embedding_dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=1, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()
    
    assert args.gamma > 0 and args.gamma <= 1, "gamma must be greater than 0 or less than or equal to 1"
    
    return args
    
    
def trainTestModel():
    # get values from the argument parser
    num_cities = args.num_cities
    
    # other batch parameters
    batchSize = args.batch_size
    
    # network parameters
    input_dim = 2
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    greedy = True
    temperature = args.temperature
    
    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    numRuns = args.num_runs

    numNets = len(POINTER_METHODS)
    
    # create gamma transform matrix
    gamma = args.gamma
    exponent = torch.arange(num_cities).view(-1,1) - torch.arange(num_cities).view(1,-1)
    gamma_transform = (gamma ** exponent * (exponent >= 0)).unsqueeze(0).expand(batchSize, -1, -1).to(device)

    print(f"Doing training...")
    trainReward = torch.zeros((trainEpochs, numNets, numRuns))
    testReward = torch.zeros((testEpochs, numNets, numRuns))
    trainRewardByPosition = torch.full((trainEpochs, num_cities, numNets, numRuns), torch.nan) # keep track of where there was reward
    testRewardByPosition = torch.full((testEpochs, num_cities, numNets, numRuns), torch.nan) # keep track of where there was reward
    trainScoreByPosition = torch.full((trainEpochs, num_cities, numNets, numRuns), torch.nan) # keep track of confidence of model
    testScoreByPosition = torch.full((testEpochs, num_cities, numNets, numRuns), torch.nan) 
    for run in range(numRuns):
        # create pointer networks with different pointer methods
        nets = [transformers.PointerNetwork(input_dim, embedding_dim, temperature=temperature, pointer_method=POINTER_METHOD, 
                                            thompson=True, encoding_layers=encoding_layers, heads=heads, kqnorm=True, 
                                            decoder_method='transformer', greedy=greedy)
                for POINTER_METHOD in POINTER_METHODS]
        nets = [net.to(device) for net in nets]

        # Create an optimizer, Adam with weight decay is pretty good
        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) for net in nets]

        for epoch in tqdm(range(trainEpochs)):
            # generate batch
            input, _, xy, dists = datasets.tsp_batch(batchSize, num_cities, return_target=False, return_full=True)
            input = input.to(device)

            # zero gradients, get output of network
            for opt in optimizers: opt.zero_grad()
            log_scores, choices = map(list, zip(*[net(input) for net in nets]))
            
            # log-probability for each chosen dominoe
            logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
            
            # measure rewards
            rewards = [training.measureReward_tsp(xy, dists, choice) for choice in choices]
            G = [torch.matmul(reward, gamma_transform) for reward in rewards]

            # measure J
            J = [-torch.sum(logpol * g) for logpol, g in zip(logprob_policy, G)]
            for j in J: j.backward()

            # update networks
            for opt in optimizers: opt.step()
                
            # measure position dependent error 
            with torch.no_grad():
                for i in range(numNets):
                    trainReward[epoch, i] = torch.mean(torch.sum(rewards[i], dim=1))
                    trainRewardByPosition[epoch, :, i, run] = torch.nanmean(rewards, dim=0)
                    
                    # Measure the models confidence -- ignoring the effect of temperature
                    pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                    pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                    trainScoreByPosition[epoch, :, i, run] = torch.mean(pretemp_policy, dim=0).detach()
        
        with torch.no_grad():
            for net in nets: 
                net.setTemperature(1.0)
                net.setThompson(False)

            for epoch in tqdm(range(trainEpochs)):
                # generate batch
                input, _, xy, dists = datasets.tsp_batch(batchSize, num_cities, return_target=False, return_full=True)
                input = input.to(device)

                # zero gradients, get output of network
                for opt in optimizers: opt.zero_grad()
                log_scores, choices = map(list, zip(*[net(input) for net in nets]))
                
                # log-probability for each chosen dominoe
                logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
                
                # measure rewards
                rewards = [training.measureReward_tsp(xy, dists, choice) for choice in choices]
                    
                # measure position dependent error 
                with torch.no_grad():
                    for i in range(numNets):
                        testReward[epoch, i] = torch.mean(torch.sum(rewards[i], dim=1))
                        testRewardByPosition[epoch, :, i, run] = torch.nanmean(rewards, dim=0)
                        
                        # Measure the models confidence -- ignoring the effect of temperature
                        pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                        pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                        testScoreByPosition[epoch, :, i, run] = torch.mean(pretemp_policy, dim=0).detach()

                
                    
    results = {
        'trainReward': trainReward,
        'testReward': testReward,
        'trainRewardByPosition': trainRewardByPosition,
        'testRewardByPosition': testRewardByPosition,
        'trainScoreByPosition': trainScoreByPosition,
        'testScoreByPosition': testScoreByPosition
    }
    
    return results

def plotResults(results, args):
    numRuns = args.num_runs
    cmap = mpl.colormaps['tab10']
    
    # make plot of loss trajectory
    fig, ax = plt.subplots(1,2,figsize=(6,4), width_ratios=[2.6,1],layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax[0].plot(range(args.train_epochs), torch.mean(results['trainLoss'][:,idx], dim=1), color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel(f'Loss N={numRuns}')
    ax[0].set_title('Training Performance')
    ax[0].set_ylim(0, None)
    yMin0, yMax0 = ax[0].get_ylim()

    # create inset to show initial train trajectory
    inset = ax[0].inset_axes([0.05, 0.52, 0.45, 0.4])
    for idx, name in enumerate(POINTER_METHODS):
        inset.plot(range(args.train_epochs), torch.mean(results['trainLoss'][:,idx], dim=1), color=cmap(idx), lw=1.2, label=name)
    inset.set_xlim(-10, 300)
    inset.set_ylim(0, 3)
    inset.set_xticks([0, 150, 300])
    inset.set_xticklabels(inset.get_xticklabels(), fontsize=8)
    inset.set_yticklabels([])
    inset.set_title('Initial Epochs', fontsize=10)
    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.mean(results['testLoss'][:,idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestReward.mean(), mnTestReward.mean()], color=cmap(idx), lw=4, label=name)
        ax[1].plot([idx,idx], [mnTestReward.min(), mnTestReward.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel(f'Loss N={numRuns}')
    ax[1].set_title('Test Performance')
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(0, 1)

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()

    # now plot confidence across positions
    numPos = results['testMaxScore'].size(1)
    fig, ax = plt.subplots(1, 1, figsize=(4,4), layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax.plot(range(numPos), torch.mean(torch.exp(results['testMaxScore'][:,:,idx]), dim=(0,2)), color=cmap(idx), lw=1, marker='o', label=name)
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




