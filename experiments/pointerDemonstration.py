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
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName():
    return "pointerDemonstration"

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer demonstration.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('-mn','--min-seq-length', type=int, default=4, help='the minimum tokens per sequence')
    parser.add_argument('-mx','--max-seq-length', type=int, default=12, help='the maximum tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=512, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=10000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=1000, help='the number of testing epochs')

    parser.add_argument('--embedding_dim', type=int, default=48, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=4, help='the number of heads in transformer layers')
    parser.add_argument('--encoding-layers', type=int, default=1, help='the number of stacked transformers in the encoder')
    parser.add_argument('--greedy', default=False, action='store_true', help='if used, will generate greedy predictions of each step rather than probability-weighted predictions')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    assert args.min_seq_length <= args.max_seq_length, "min seq length has to be less than or equal to max seq length"
    
    return args
    

def trainTestModel():
    ignoreIndex = -1
    
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)
    dominoeValue = np.sum(listDominoes, axis=1)

    minSeqLength = args.min_seq_length
    maxSeqLength = args.max_seq_length
    batchSize = args.batch_size
    
    input_dim = 2*(highestDominoe+1)
    embedding_dim = args.embedding_dim
    heads = args.heads
    encoding_layers = args.encoding_layers
    greedy = args.greedy
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    
    # Create a pointer network
    pnet = transformers.PointerNetwork(input_dim, embedding_dim, encoding_layers=encoding_layers, heads=heads, kqnorm=True, decode_with_gru=False, greedy=greedy)
    pnet = pnet.to(device)
    pnet.train()

    # Create an optimizer, Adam with weight decay is pretty good
    optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train network
    print("Training network...")
    # trainSeqLength = torch.zeros(trainEpochs)
    trainLoss = torch.zeros(trainEpochs)
    # trainPositionError = [None]*trainEpochs
    for epoch in tqdm(range(trainEpochs)):
        
        # right now elements in a batch have to have the same sequence length 
        #cSeqLength = np.random.randint(minSeqLength, maxSeqLength+1)

        # create a data batch and put it on the right device
        #input, target = dominoeBatch(batchSize, cSeqLength, listDominoes, dominoeValue, highestDominoe)
        #input, target = input.to(device), target.to(device)

        input, target, mask = df.dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, listDominoes, dominoeValue, highestDominoe, ignoreIndex=ignoreIndex)
        input, target, mask = input.to(device), target.to(device), mask.to(device)

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

        # # the loss spikes sometimes, and I think it might be due to the autoregressive component of pointer networks.
        # # i.e., if the network gets the first element wrong, it'll get the rest of the elements right given the one it chose... but this will inflate the training loss
        # # anyway, this metric is a way to measure how bad the prediction is as a function of position in sequence to try to identify that issue
        # t = torch.gather(unrolled, dim=1, index=target.view(-1).unsqueeze(-1)).cpu().detach().view(batchSize, cSeqLength)
        # m = torch.max(unrolled, dim=1)[0].cpu().detach().view(batchSize, cSeqLength)
        # trainPositionError[epoch] = torch.mean(m-t,dim=0)

        # save training data
        trainLoss[epoch] = loss.item()
        # trainSeqLength[epoch] = cSeqLength

    # Test network - same thing as in testing but without updates to model
    with torch.no_grad():
        print("Testing network...")
        pnet.eval()
        
        # testSeqLength = torch.zeros(testEpochs)
        testLoss = torch.zeros(testEpochs)
        # testPositionError = [None]*testEpochs
        for epoch in tqdm(range(testEpochs)):
            # input, target = getBatch(batchSize, seqLength, input_dim)
            # cSeqLength = np.random.randint(minSeqLength, maxSeqLength+1)
            
            # input, target = dominoeBatch(batchSize, cSeqLength, listDominoes, dominoeValue, highestDominoe)
            # input, target = input.to(device), target.to(device)

            input, target, mask = df.dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, listDominoes, dominoeValue, highestDominoe, ignoreIndex=ignoreIndex)
            input, target, mask = input.to(device), target.to(device), mask.to(device)

            log_scores, choices = pnet(input)
            
            unrolled = log_scores.view(-1, log_scores.size(-1))
            loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignoreIndex)
            
            # t = torch.gather(unrolled, dim=1, index=target.view(-1).unsqueeze(-1)).cpu().detach().view(batchSize, cSeqLength)
            # m = torch.max(unrolled, dim=1)[0].cpu().detach().view(batchSize, cSeqLength)
            # testPositionError[epoch] = torch.mean(m-t,dim=0)
            testLoss[epoch] = loss.item()
            # testSeqLength[epoch] = cSeqLength

    results = {
        'trainLoss': trainLoss,
        'testLoss': testLoss,
    }

    return results


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
    
    
    






