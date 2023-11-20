import sys
import os

# add path that contains the dominoes package
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

# standard imports
from copy import copy
import argparse
import time
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

for path in (resPath, prmsPath, figsPath, savePath):
    if not(path.exists()):
        path.mkdir()
        
# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "ptrArchComp_TSP_SL"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer demonstration.')
    parser.add_argument('-nc','--num-cities', type=int, default=10, help='the number of cities')
    parser.add_argument('-bs','--batch-size',type=int, default=128, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=12000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=200, help='the number of testing epochs')
    parser.add_argument('-nr','--num-runs',type=int, default=8, help='how many runs for each network to train')

    parser.add_argument('--embedding_dim', type=int, default=96, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=4, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--greedy', default=False, action='store_true', help='if used, will generate greedy predictions of each step rather than probability-weighted predictions')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    return args
    
    
def trainTestModel():
    # get values from the argument parser
    num_cities = args.num_cities
    num_in_cycle = num_cities

    # other batch parameters
    batchSize = args.batch_size
    
    # For given batch size and number of cities, try out whether using multiple workers is better
    t = time.time()
    _ = datasets.tsp_batch(batchSize, num_cities, return_full=True, return_target=True, threads=12)
    time_parallel = time.time() - t

    t = time.time()
    _ = datasets.tsp_batch(batchSize, num_cities, return_full=True, return_target=True, threads=1)
    time_noparallel = time.time() - t
    
    # Make decision and tell user about it
    threads = 12 if time_parallel < time_noparallel else 1
    print(f"Time checks: parallel={time_parallel:.3f}, noParallel={time_noparallel:.3f}, using threads={threads}")
    
    # network parameters
    input_dim = 2
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    greedy = True # args.greedy
    temperature = 1.0
    
    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    numRuns = args.num_runs

    numNets = len(POINTER_METHODS)
    
    print(f"Doing training...")
    trainLoss = torch.zeros((trainEpochs, numNets, numRuns))
    testLoss = torch.zeros((testEpochs, numNets, numRuns))
    trainTourLength = torch.zeros((trainEpochs, numNets, numRuns))
    testTourLength = torch.zeros((testEpochs, numNets, numRuns))
    trainTourComplete = torch.zeros((trainEpochs, numNets, numRuns))
    testTourComplete = torch.zeros((testEpochs, numNets, numRuns))
    trainTourValidLength = torch.zeros((trainEpochs, numNets, numRuns))
    testTourValidLength = torch.zeros((testEpochs, numNets, numRuns))
    trainPositionError = torch.full((trainEpochs, num_in_cycle, numNets, numRuns), torch.nan) # keep track of where there was error
    trainMaxScore = torch.full((trainEpochs, num_in_cycle, numNets, numRuns), torch.nan) # keep track of confidence of model
    testMaxScore = torch.full((testEpochs, num_in_cycle, numNets, numRuns), torch.nan) 
    for run in range(numRuns):
        print(f"Training round of networks {run+1}/{numRuns}...")
        
        # create pointer networks with different pointer methods
        nets = [transformers.PointerNetwork(input_dim, embedding_dim, temperature=temperature, pointer_method=POINTER_METHOD, 
                                            thompson=False, encoding_layers=encoding_layers, heads=heads, kqnorm=True, 
                                            decoder_method='transformer', greedy=greedy)
                for POINTER_METHOD in POINTER_METHODS]
        nets = [net.to(device) for net in nets]

        # Create an optimizer, Adam with weight decay is pretty good
        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) for net in nets]

        for epoch in tqdm(range(trainEpochs)):
            # generate batch
            input, target, _, dists = datasets.tsp_batch(batchSize, num_cities, return_full=True, threads=threads)
            input, target, dists = input.to(device), target.to(device), dists.to(device)

            # zero gradients, get output of network
            for opt in optimizers: opt.zero_grad()
            log_scores, choices = map(list, zip(*[net(input, max_output=num_in_cycle) for net in nets]))

            # measure loss with negative log-likelihood
            unrolled = [log_score.view(-1, log_score.size(-1)) for log_score in log_scores]
            loss = [torch.nn.functional.nll_loss(unroll, target.view(-1)) for unroll in unrolled]
            for i, l in enumerate(loss):
                assert not np.isnan(l.item()), f"model type {POINTER_METHODS[i]} diverged :("

            # update networks
            for l in loss: l.backward()
            for opt in optimizers: opt.step()
                
            # save training data
            for i, l in enumerate(loss):
                trainLoss[epoch, i] = l.item()

            # measure position dependent error 
            with torch.no_grad():
                # get distance traveled
                start = target[:, -1] # get "start" city, closest to origin (set this way by tsp_batch)
                tour_complete, tour_distance = map(list, zip(*[training.measureReward_tsp(dists, choice, start) for choice in choices]))
                tour_complete = [torch.all(tc==1, dim=1) for tc in tour_complete]
                tour_distance = [torch.sum(td, dim=1) for td in tour_distance]
                
                # start by getting score for target at each position 
                target_score = [torch.gather(unroll, dim=1, index=target.view(-1,1)).view(batchSize, num_in_cycle) for unroll in unrolled]
                # then get max score for each position (which would correspond to the score of the actual choice)
                max_score = [torch.max(unroll, dim=1)[0].view(batchSize, num_in_cycle) for unroll in unrolled]
                # then calculate position error
                pos_error = [ms - ts for ms, ts in zip(max_score, target_score)] # high if the chosen score is much bigger than the target score
                    
                # add to accounting
                for i, (td, tc, pe, ms) in enumerate(zip(tour_distance, tour_complete, pos_error, max_score)):
                    trainTourLength[epoch, i, run] = torch.mean(td)
                    trainTourComplete[epoch, i, run] = torch.mean(1.0*tc)
                    t_check = torch.cat((td[tc], torch.tensor(torch.nan).view(1).to(device)))
                    trainTourValidLength[epoch, i, run] = torch.nanmean(t_check)
                
                    trainPositionError[epoch, :, i, run] = torch.nansum(pe, dim=0)
                    trainMaxScore[epoch, :, i, run] = torch.nanmean(ms, dim=0)
                    
        
        with torch.no_grad():
            print('Testing network...')
            for epoch in tqdm(range(testEpochs)):
                # generate batch
                input, target, _, dists = datasets.tsp_batch(batchSize, num_cities, return_full=True, threads=threads)
                input, target, dists = input.to(device), target.to(device), dists.to(device)

                log_scores, choices = map(list, zip(*[net(input, max_output=num_in_cycle) for net in nets]))
        
                # measure loss with negative log-likelihood
                unrolled = [log_score.view(-1, log_score.size(-1)) for log_score in log_scores]
                loss = [torch.nn.functional.nll_loss(unroll, target.view(-1)) for unroll in unrolled]
                for i, l in enumerate(loss):
                    assert not np.isnan(l.item()), f"model type {POINTER_METHODS[i]} diverged :("

                # get distance traveled
                start = target[:, -1] # get start city (the one closest to origin)
                tour_complete, tour_distance = map(list, zip(*[training.measureReward_tsp(dists, choice, start) for choice in choices]))
                tour_complete = [torch.all(tc==1, dim=1) for tc in tour_complete]
                tour_distance = [torch.sum(td, dim=1) for td in tour_distance]
                
                # get max score 
                max_score = [torch.max(unroll, dim=1)[0].view(batchSize, num_in_cycle) for unroll in unrolled]

                # save training data
                for i, (l, td, tc, ms) in enumerate(zip(loss, tour_distance, tour_complete, max_score)):
                    testLoss[epoch, i, run] = l.item()
                    testTourLength[epoch, i, run] = torch.mean(td)
                    testTourComplete[epoch, i, run] = torch.mean(1.0*tc)
                    t_check = torch.cat((td[tc], torch.tensor(torch.nan).view(1).to(device)))
                    testTourValidLength[epoch, i, run] = torch.nanmean(t_check)
                    testMaxScore[epoch, :, i, run] = torch.nanmean(ms, dim=0)
                
                    
    results = {
        'trainLoss': trainLoss,
        'testLoss': testLoss,
        'trainTourLength': trainTourLength,
        'testTourLength': testTourLength,
        'trainTourComplete': trainTourComplete,
        'testTourComplete': testTourComplete,
        'trainTourValidLength': trainTourValidLength,
        'testTourValidLength': testTourValidLength,
        'trainPositionError': trainPositionError,
        'trainMaxScore': trainMaxScore,
        'testMaxScore': testMaxScore,
    }
    
    return results, nets

def plotResults(results, args):
    numRuns = args.num_runs
    cmap = mpl.colormaps['tab10']
    
    # make plot of loss trajectory
    fig, ax = plt.subplots(1,2,figsize=(6,4), width_ratios=[2.6,1],layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        cdata = sp.ndimage.median_filter(torch.mean(results['trainLoss'][:,idx], dim=1), size=(100,))
        ax[0].plot(range(args.train_epochs), cdata, color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel(f'Loss N={numRuns}')
    ax[0].set_title('Training Performance')
    ax[0].set_ylim(0, None)
    ax[0].legend(loc='best')
    yMin0, yMax0 = ax[0].get_ylim()
    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.mean(results['testLoss'][:,idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestReward.mean(), mnTestReward.mean()], color=cmap(idx), lw=4, label=name)
        for mtr in mnTestReward:
            ax[1].plot(get_x(idx), [mtr, mtr], color=cmap(idx), lw=1.5)
        ax[1].plot([idx,idx], [mnTestReward.min(), mnTestReward.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel(f'Loss N={numRuns}')
    ax[1].set_title('Testing')
    ax[1].set_xlim(-1, len(POINTER_METHODS))

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()

    # make plot of fraction complete tours
    fig, ax = plt.subplots(1,2,figsize=(6,4), width_ratios=[2.6,1],layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        cdata = sp.ndimage.median_filter(torch.mean(results['trainTourComplete'][:,idx], dim=1), size=(100,))
        ax[0].plot(range(args.train_epochs), cdata, color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel(f'fraction N={numRuns}')
    ax[0].set_title('Complete Tours - Training')
    ax[0].set_ylim(0, 1)
    ax[0].legend(loc='best')
    yMin0, yMax0 = ax[0].get_ylim()
    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestComplete = torch.mean(results['testTourComplete'][:,idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestComplete.mean(), mnTestComplete.mean()], color=cmap(idx), lw=4, label=name)
        for mtc in mnTestComplete:
            ax[1].plot(get_x(idx), [mtc, mtc], color=cmap(idx), lw=1.5)
        ax[1].plot([idx,idx], [mnTestComplete.min(), mnTestComplete.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel(f'fraction N={numRuns}')
    ax[1].set_title('Testing')
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(0, 1)

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName('completedTours')))
    
    plt.show()

    # make plot of tour length for valid tours
    fig, ax = plt.subplots(1,2,figsize=(6,4), width_ratios=[2.6,1],layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        cdata = torch.nanmean(results['trainTourValidLength'][:,idx], dim=1)
        idx_nan = torch.isnan(cdata)
        cdata.masked_fill_(idx_nan, 0)
        cdata = sp.signal.savgol_filter(cdata, 50, 1)
        cdata[idx_nan] = torch.nan
        ax[0].plot(range(args.train_epochs), cdata, color=cmap(idx), lw=1.2, label=name)
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_ylabel(f'Tour Length N={numRuns}')
    ax[0].set_title('Training - TourLength (Valid)')
    ax[0].legend(loc='best')
    ax[0].set_ylim(2.85, 2.975)
    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.nanmean(results['testTourValidLength'][:,idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestReward.mean(), mnTestReward.mean()], color=cmap(idx), lw=4, label=name)
        for mtr in mnTestReward:
            ax[1].plot(get_x(idx), [mtr, mtr], color=cmap(idx), lw=1.5)
        ax[1].plot([idx,idx], [mnTestReward.min(), mnTestReward.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_ylabel(f'Tour Length N={numRuns}')
    ax[1].set_title('Testing')
    ax[1].set_xlim(-1, len(POINTER_METHODS))

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName('tourValidLength')))
    
    plt.show()

    
    # now plot confidence across positions
    numPos = results['testMaxScore'].size(1)
    fig, ax = plt.subplots(1, 2, figsize=(6,4), width_ratios=[2.6,1], layout='constrained')
    for idx, name in enumerate(POINTER_METHODS):
        ax[0].plot(range(numPos), torch.mean(torch.exp(results['testMaxScore'][:,:,idx]), dim=(0,2)), color=cmap(idx), lw=1, marker='o', label=name)
    ax[0].set_xlabel('Output Position')
    ax[0].set_ylabel('Mean Score')
    ax[0].set_title('Position-Dependent Confidence')
    ax[0].legend(loc='best', fontsize=8)
    ax[0].set_ylim(0.65, 1)
    
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnScoreByPosition = torch.mean(torch.exp(results['testMaxScore'][:,:,idx]), dim=(0,1))
        ax[1].plot(get_x(idx), [mnScoreByPosition.mean(), mnScoreByPosition.mean()], color=cmap(idx), lw=4, label=name)
        for msbp in mnScoreByPosition:
            ax[1].plot(get_x(idx), [msbp, msbp], color=cmap(idx), lw=1.5)
        ax[1].plot([idx,idx], [mnScoreByPosition.min(), mnScoreByPosition.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha='right', fontsize=8)
    ax[1].set_title('Average')
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(0.65, 1)

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName(extra='confidence')))
        
    plt.show()
    

    
if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        # train and test pointerNetwork 
        results, nets = trainTestModel()
        
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
            for net, method in zip(nets, POINTER_METHODS):
                save_name = f"{method}.pt"
                torch.save(net, savePath / getFileName(extra=save_name))
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




