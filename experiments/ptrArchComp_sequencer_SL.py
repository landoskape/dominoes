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

for path in (resPath, prmsPath, figsPath, savePath):
    if not(path.exists()):
        path.mkdir()
        
# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "ptrArchComp_sequencer_SL"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer dominoe sequencing experiment.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('-hs','--hand-size', type=int, default=12, help='the maximum tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=128, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=12000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=1000, help='the number of testing epochs')
    parser.add_argument('-nr','--num-runs',type=int, default=8, help='how many runs for each network to train')

    parser.add_argument('--embedding_dim', type=int, default=96, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=8, help='the number of heads in transformer layers')
    parser.add_argument('--expansion', type=int, default=2, help='the expansion at the MLP part of the transformer')
    parser.add_argument('--encoding-layers', type=int, default=2, help='the number of stacked transformers in the encoder')
    parser.add_argument('--greedy', default=False, action='store_true', help='if used, will generate greedy predictions of each step rather than probability-weighted predictions')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    return args
    
    
def trainTestModel():
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)

    handSize = args.hand_size
    batchSize = args.batch_size
    null_token = True # using a null token to indicate end of line
    available_token = True # using available token to indicate which value to start on 
    num_output = copy(handSize)+1 # always require end on null token
    ignore_index = -1

    # other batch parameters
    batchSize = args.batch_size
    
    # network parameters
    input_dim = (2 if not(available_token) else 3)*(highestDominoe+1) + (1 if null_token else 0)
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    contextual_encoder = True # don't transform the available token
    greedy = args.greedy
    
    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    numRuns = args.num_runs

    numNets = len(POINTER_METHODS)
    
    print(f"Doing training...")
    trainLoss = torch.zeros((trainEpochs, numNets, numRuns))
    testLoss = torch.zeros((testEpochs, numNets, numRuns))
    for run in range(numRuns):
        print(f"Training round of networks {run+1}/{numRuns}...")
        
        # create pointer networks with different pointer methods
        nets = [transformers.PointerNetwork(input_dim, embedding_dim, pointer_method=POINTER_METHOD, 
                                            contextual_encoder=contextual_encoder, thompson=False, 
                                            encoding_layers=encoding_layers, heads=heads, kqnorm=True, 
                                            decoder_method='transformer', greedy=greedy)
                for POINTER_METHOD in POINTER_METHODS]
        nets = [net.to(device) for net in nets]

        # Create an optimizer, Adam with weight decay is pretty good
        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) for net in nets]

        for epoch in tqdm(range(trainEpochs)):
            # generate input batch
            batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, return_target=True, null_token=null_token,
                                        available_token=available_token, ignore_index=ignore_index, return_full=True)

            # unpack batch tuple
            input, target, _, _, _, selection, available = batch

            # move to correct device
            input, target = input.to(device), target.to(device)

            # divide input into main input and context
            x, context = input[:, :-1].contiguous(), input[:, [-1]] # input [:, [-1]] is context token
            input = (x, context)
            
            # zero gradients, get output of network
            for opt in optimizers: opt.zero_grad()
            log_scores, choices = map(list, zip(*[net(input, max_output=num_output) for net in nets]))

            # measure loss with negative log-likelihood
            unrolled = [log_score.view(-1, log_score.size(-1)) for log_score in log_scores]
            loss = [torch.nn.functional.nll_loss(unroll, target.view(-1), ignore_index=ignore_index)
                    for unroll in unrolled]
            for i, l in enumerate(loss):
                assert not np.isnan(l.item()), f"model type {POINTER_METHODS[i]} diverged :("

            # update networks
            for l in loss: l.backward()
            for opt in optimizers: opt.step()
                
            # save training data
            for i, l in enumerate(loss):
                trainLoss[epoch, i, run] = l.item()
                    
        
        with torch.no_grad():
            print('Testing network...')
            for epoch in tqdm(range(testEpochs)):
                # generate input batch
                batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, return_target=True, null_token=null_token,
                                            available_token=available_token, ignore_index=ignore_index, return_full=True)

                # unpack batch tuple
                input, target, _, _, _, selection, available = batch

                # move to correct device
                input, target = input.to(device), target.to(device)

                # divide input into main input and context
                x, context = input[:, :-1].contiguous(), input[:, [-1]] # input [:, [-1]] is context token
                input = (x, context)
                
                log_scores, choices = map(list, zip(*[net(input, max_output=num_output) for net in nets]))

                # measure loss with negative log-likelihood
                unrolled = [log_score.view(-1, log_score.size(-1)) for log_score in log_scores]
                loss = [torch.nn.functional.nll_loss(unroll, target.view(-1), ignore_index=ignore_index)
                        for unroll in unrolled]
                    
                # save testing data
                for i, l in enumerate(loss):
                    testLoss[epoch, i, run] = l.item()
                
                    
    results = {
        'trainLoss': trainLoss,
        'testLoss': testLoss,
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




