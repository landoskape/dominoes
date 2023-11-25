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
from dominoes import transformers

device = 'cuda' if torchCuda.is_available() else 'cpu'

# can edit this for each machine it's being used on
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

for path in (resPath, prmsPath, figsPath):
    if not(path.exists()):
        path.mkdir()

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "pointerDemonstration"
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
    parser.add_argument('-ne','--train-epochs',type=int, default=10000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=1000, help='the number of testing epochs')

    parser.add_argument('--embedding_dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=4, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    assert args.min_seq_length <= args.max_seq_length, "min seq length has to be less than or equal to max seq length"
    assert args.train_fraction > 0 and args.train_fraction <= 1, "train fraction must be greater than 0 and less than or equal to 1"
    
    return args
    

def trainTestModel():
    ignoreIndex = -1
    
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)

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

    # other input and training parameters
    minSeqLength = args.min_seq_length
    maxSeqLength = args.max_seq_length
    batchSize = args.batch_size
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs

    # network parameters
    input_dim = 2*(highestDominoe+1)
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    
    # Create a pointer network
    pnet = transformers.PointerNetwork(input_dim, embedding_dim, encoding_layers=encoding_layers, 
                                       heads=heads, kqnorm=True, decoder_method='transformer')
    pnet = pnet.to(device)
    pnet.train()

    # Create an optimizer, Adam with weight decay is pretty good
    optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train network
    print("Training network...")
    trainLoss = torch.zeros(trainEpochs)
    trainDescending = torch.zeros(trainEpochs)
    trainPositionError = torch.full((trainEpochs, maxSeqLength), torch.nan) # keep track of where there was error
    trainMaxScore = torch.full((trainEpochs, maxSeqLength), torch.nan) # keep track of confidence of model
    for epoch in tqdm(range(trainEpochs)):
        # generate batch
        batch = df.dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, keepDominoes, keepValue, 
                                        highestDominoe, ignoreIndex=ignoreIndex, return_full=True)
        input, target, mask, selection = batch
        input, target, mask = input.to(device), target.to(device), mask.to(device)
        
        selection = [keepDominoes[np.array(sel)] for sel in selection]
        selection = [np.concatenate((sel, np.zeros((maxSeqLength-len(sel),2))), axis=0)
                        for sel in selection]
        selection = torch.stack([torch.tensor(sel) for sel in selection]).to(device)
        value_dominoes = torch.sum(selection, dim=2)

        # zero gradients, get output of network
        optimizer.zero_grad()
        log_scores, choices = pnet(input)

        # measure loss with negative log-likelihood
        unrolled = log_scores.view(-1, log_scores.size(-1))
        loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignoreIndex)
        assert not np.isnan(loss.item()), "model diverged :("

        # update network
        loss.backward()
        optimizer.step()

        # save training data
        trainLoss[epoch] = loss.item()

        # measure whether choices sorted dominoes in descending order
        value_choices = torch.gather(value_dominoes, 1, choices)
        trainDescending[epoch] = torch.mean(1.0*torch.all(torch.diff(value_choices, dim=1)<=0, dim=1))

        # measure position dependent error 
        with torch.no_grad():
            # start by getting score for target at each position 
            target_noignore = target.clone().masked_fill_(target==-1, 0)
            target_score = torch.gather(unrolled, dim=1, index=target_noignore.view(-1,1)).view(batchSize, maxSeqLength)
            # then get max score for each position (which would correspond to the score of the actual choice)
            max_score = torch.max(unrolled, dim=1)[0].view(batchSize, maxSeqLength)
            # then calculate position error
            pos_error = max_score - target_score # high if the chosen score is much bigger than the target score
            # now remove locations where it is masked out
            pos_error.masked_fill_(mask==0, torch.nan)
            # add to accounting
            trainPositionError[epoch] = torch.nansum(pos_error, dim=0)
            trainMaxScore[epoch] = torch.nanmean(max_score, dim=0)
            

    # Test network - same thing as in testing but without updates to model
    with torch.no_grad():
        print("Testing network...")
        pnet.eval()
        
        testLoss = torch.zeros(testEpochs)
        testDescending = torch.zeros(testEpochs)
        for epoch in tqdm(range(testEpochs)):
            # generate batch
            batch = df.dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, listDominoes, dominoeValue, 
                                            highestDominoe, ignoreIndex=ignoreIndex, return_full=True)
            input, target, mask, selection = batch
            input, target, mask = input.to(device), target.to(device), mask.to(device)

            selection = [listDominoes[np.array(sel)] for sel in selection]
            selection = [np.concatenate((sel, np.zeros((maxSeqLength-len(sel),2))), axis=0)
                         for sel in selection]
            selection = torch.stack([torch.tensor(sel) for sel in selection]).to(device)
            value_dominoes = torch.sum(selection, dim=2)
            
            # get network output
            log_scores, choices = pnet(input)

            # measure test loss
            unrolled = log_scores.view(-1, log_scores.size(-1))
            loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignoreIndex)

            # save testing loss
            testLoss[epoch] = loss.item()

            # measure whether choices sorted dominoes in descending order
            value_choices = torch.gather(value_dominoes, 1, choices)
            testDescending[epoch] = torch.mean(1.0*torch.all(torch.diff(value_choices, dim=1)<=0, dim=1))
            
    results = {
        'trainLoss': trainLoss,
        'testLoss': testLoss,
        'trainPositionError': trainPositionError,
        'trainMaxScore': trainMaxScore,
        'trainDescending': trainDescending,
        'testDescending': testDescending,
    }

    return results


def plotResults(results, args):
    # Start by finding loss spikes
    epochs_before = 7
    epochs_after = 20
    pks = sp.signal.find_peaks(results['trainLoss'], height=1.5, threshold=0.2, distance=100)[0]
    pks = [pk for pk in pks if (pk-epochs_before)>0 and (pk+epochs_after)<len(results['trainLoss'])]
    num_pks = len(pks)
    
    # Get spike-triggered loss trajectory and position-dependent error
    epochs_xval = range(-epochs_before, epochs_after)
    total_epochs = epochs_before + epochs_after
    c_loss = torch.full((num_pks, total_epochs), torch.nan)
    c_pos_error = torch.full((num_pks, total_epochs, results['trainPositionError'].shape[1]), torch.nan) 
    c_max_score = torch.full((num_pks, total_epochs, results['trainMaxScore'].shape[1]), torch.nan) 
    for ii, pp in enumerate(pks):
        c_loss[ii] = results['trainLoss'][pp-epochs_before:pp+epochs_after]
        c_pos_error[ii] = results['trainPositionError'][pp-epochs_before:pp+epochs_after,:]
        c_max_score[ii] = results['trainMaxScore'][pp-epochs_before:pp+epochs_after,:]
    
    fig, ax = plt.subplots(1,2,figsize=(8,4), layout='constrained')
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

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()

    
    fig, ax = plt.subplots(1,2,figsize=(8,4), layout='constrained')
    ax[0].plot(range(args.train_epochs), results['trainDescending'], color='k', lw=1)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Fraction')
    ax[0].set_ylim(0, 1)
    ax[0].set_title('Training Fraction Sorted')

    ax[1].plot(range(args.test_epochs), results['testDescending'], color='b', lw=1)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Fraction')
    # ax[1].set_ylim(0, 1)
    ax[1].set_title('Testing Fraction Sorted')

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName('fractionDescending')))
    
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
    
    
    






