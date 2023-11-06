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
POINTER_METHODS = ['PointerStandard', 'PointerDot', 'PointerAttention', 'PointerTransformer']

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
    parser.add_argument('--train-fraction', type=float, default=0.75, help='the fraction of dominoes in the set to train with')
    parser.add_argument('-hs','--hand-size', type=int, default=8, help='tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=512, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=5000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=100, help='the number of testing epochs')
    parser.add_argument('--gamma', type=float, default=0.5, help='discounting factor')
    parser.add_argument('-nr','--num-runs', type=int, default=5, help='how many networks to train of each type')
    
    parser.add_argument('--embedding-dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=4, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--train-temperature',type=float, default=5.0, help='temperature for training')
    
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    assert args.train_fraction > 0 and args.train_fraction <= 1, "train fraction must be greater than 0 and less than or equal to 1"
    assert args.gamma > 0 and args.gamma <= 1, "gamma must be greater than 0 or less than or equal to 1"
    
    return args
    

def trainTestModel(with_thompson):
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)
    dominoeValue = np.sum(listDominoes, axis=1)

    # create full set of dominoes (representing non-doubles in both ways)
    doubleDominoes = listDominoes[:,0] == listDominoes[:,1]
    nonDoubleReverse = listDominoes[~doubleDominoes][:,[1,0]] # represent each dominoe in both orders

    # list of full set of dominoe representations and value of each
    listDominoes = np.concatenate((listDominoes, nonDoubleReverse), axis=0)
    dominoeValue = np.sum(listDominoes, axis=1)

    # subset dominoes
    keepFraction = args.train_fraction
    keepNumber = int(len(listDominoes)*keepFraction)
    keepIndex = np.sort(np.random.permutation(len(listDominoes))[:keepNumber]) # keep random fraction (but sort in same way)
    keepDominoes = listDominoes[keepIndex]
    keepValue = dominoeValue[keepIndex]

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
    if with_thompson:
        temperature = args.train_temperature
        thompson = True
    else:
        temperature = 1
        thompson = False
    
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
    
    # create pointer networks with different pointer methods (variable defined above)
    pnets = [transformers.PointerNetwork(input_dim, embedding_dim, temperature=temperature, pointer_method=POINTER_METHOD, thompson=thompson,
                                         encoding_layers=encoding_layers, heads=heads, kqnorm=True, decode_with_gru=False, greedy=greedy)
             for POINTER_METHOD in POINTER_METHODS]
    pnets = [pnet.to(device) for pnet in pnets]
    for pnet in pnets: pnet.train()
    
    # Create an optimizer, Adam with weight decay is pretty good
    optimizers = [torch.optim.Adam(pnet.parameters(), lr=1e-3, weight_decay=1e-5) for pnet in pnets]
    
    numNets = len(pnets)
    
    # Train network
    train_type_string = 'with' if thompson else 'without'
    print(f"Training networks {train_type_string} thompson sampling...")
    trainReward = torch.zeros((trainEpochs, numNets, numRuns))
    testReward = torch.zeros((testEpochs, numNets, numRuns))
    trainRewardByPos = torch.zeros((trainEpochs, maxOutput, numNets, numRuns))
    testRewardByPos = torch.zeros((trainEpochs, maxOutput, numNets, numRuns))
    for run in range(numRuns):
        print(f"Doing training run {run+1}/{numRuns}...")
        for epoch in tqdm(range(trainEpochs)):
            batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, **batch_inputs)
        
            # unpack batch tuple
            input, _, _, _, _, selection, available = batch
            input = input.to(device)
            
            # zero gradients, get output of network
            for opt in optimizers: opt.zero_grad()
            log_scores, choices = map(list, zip(*[pnet(input, max_output=maxOutput) for pnet in pnets]))
        
            # log-probability for each chosen dominoe
            logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
            
            # measure reward
            rewards = [training.measureReward_sortDescend(listDominoes[selection], choice) for choice in choices]
            G = [torch.bmm(reward.unsqueeze(1), gamma_transform).squeeze(1) for reward in rewards]
        
            # measure J
            J = [-torch.sum(logpol * g) for logpol, g in zip(logprob_policy, G)]
            for j in J: j.backward()

            # update networks
            for opt in optimizers: opt.step()
            
            # save training data
            for i in range(numNets):
                trainReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1))
                trainRewardByPos[epoch, :, i, run] = torch.mean(rewards[i], dim=0)
        
        with torch.no_grad():
            # always return temperature to 1 and thompson to False for testing networks
            for pnet in pnets: 
                pnet.setTemperature(1)
                pnet.setThompson(False)
                
            print('Testing network...')
            for epoch in tqdm(range(testEpochs)):
                batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, **batch_inputs)
        
                # unpack batch tuple
                input, _, _, _, _, selection, available = batch
            
                # move to correct device
                input = input.to(device)

                # get output of networks
                log_scores, choices = map(list, zip(*[pnet(input, max_output=maxOutput) for pnet in pnets]))
            
                # log-probability for each chosen dominoe
                logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]
                
                # measure reward
                rewards = [training.measureReward_sortDescend(listDominoes[selection], choice) for choice in choices]
                
                # save training data
                for i in range(numNets):
                    testReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1))
                    testRewardByPos[epoch, :, i, run] = torch.mean(rewards[i], dim=0)

    append_name = '_with_thompson' if with_thompson else '_no_thompson'
    results = {
        'trainReward'+append_name: trainReward,
        'testReward'+append_name: testReward,
        'trainRewardByPos'+append_name: trainRewardByPos,
        'testRewardByPos'+append_name: testRewardByPos,
    }

    return results, pnets


def plotResults(results, args):
    fig, ax = plt.subplots(1,4,figsize=(12,3), layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax[0].plot(range(args.train_epochs), torch.mean(results['trainReward_with_thompson'][:,idx],dim=1), lw=1, label=name)
    ax[0].set_ylabel('Reward')
    ax[0].set_title('Training with Thompson')
    yMin0, yMax0 = ax[0].get_ylim()

    for idx, name in enumerate(POINTER_METHODS):
        ax[1].plot(range(args.test_epochs), torch.mean(results['testReward_with_thompson'][:,idx],dim=1), lw=1, label=name)
    ax[1].set_ylabel('Reward')
    ax[1].set_title('Testing with Thompson')
    yMin1, yMax1 = ax[1].get_ylim()
    
    ax[0].set_ylim(min(yMin0,yMin1), max(yMax0, yMax1))
    ax[1].set_ylim(min(yMin0,yMin1), max(yMax0, yMax1))

    for idx, name in enumerate(POINTER_METHODS):
        ax[2].plot(range(args.train_epochs), torch.mean(results['trainReward_no_thompson'][:,idx],dim=1), lw=1, label=name)
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Reward')
    ax[2].set_title('Training w/out Thompson')
    yMin2, yMax2 = ax[2].get_ylim()

    for idx, name in enumerate(POINTER_METHODS):
        ax[3].plot(range(args.test_epochs), torch.mean(results['testReward_no_thompson'][:,idx],dim=1), lw=1, label=name)
    ax[3].set_xlabel('Epoch')
    ax[3].set_ylabel('Reward')
    ax[3].legend(loc='best')
    ax[3].set_title('Testing w/out Thompson')
    yMin3, yMax3 = ax[3].get_ylim()

    ax[2].set_ylim(min(yMin2,yMin3), max(yMax2, yMax3))
    ax[3].set_ylim(min(yMin2,yMin3), max(yMax2, yMax3))

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()
    
if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        # train and test pointerNetwork with thompson sampling
        thompson_setting = [True, False]
        thompson_name = ['with_thompson', 'no_thompson']
        results = {}
        pnets = []
        for with_thompson in thompson_setting:
            c_results, c_pnets = trainTestModel(with_thompson)
            for key, val in c_results.items():
                results[key] = val
            pnets.append(c_pnets)
            
        
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
            for idx, name in enumerate(thompson_name):
                for net, method in zip(pnets[idx], POINTER_METHODS):
                    save_name = f"{method}_{name}"
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
        
    plotResults(results, args)
    
    
    






