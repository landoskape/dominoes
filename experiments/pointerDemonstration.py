# mainExperiment at checkpoint 1
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

def dominoeBatch(batchSize, seqLength, listDominoes, dominoeValue, highestDominoe):
    numDominoes = len(listDominoes)
    input_dim = 2*(highestDominoe+1)

    # select list of seqLength dominoes
    selection = np.stack([np.random.choice(numDominoes, seqLength, replace=False).tolist() for _ in range(batchSize)])
    value = dominoeValue[selection]
    sortIdx = [sorted(range(len(val)), key=lambda i : -val[i]) for val in value] # sort dominoes by value in decreasing order
    target = torch.stack([torch.LongTensor(idx) for idx in sortIdx])
    
    firstValue = torch.tensor(listDominoes[selection,0], dtype=torch.int64).unsqueeze(2)
    secondValue = torch.tensor(listDominoes[selection,1], dtype=torch.int64).unsqueeze(2) + highestDominoe+1

    input = torch.zeros((batchSize, seqLength, input_dim), dtype=torch.float)
    input.scatter_(2, firstValue, torch.ones_like(firstValue, dtype=torch.float))
    input.scatter_(2, secondValue, torch.ones_like(secondValue, dtype=torch.float))

    return input, target

def trainTestModel():
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
    
    # Create network
    pnet = transformers.PointerNetwork(input_dim, embedding_dim, encoding_layers=encoding_layers, heads=heads, kqnorm=True, decode_with_gru=False, greedy=greedy)
    pnet = pnet.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train network
    print("Training network...")
    trainSeqLength = torch.zeros(trainEpochs)
    trainLoss = torch.zeros(trainEpochs)
    trainPositionError = [None]*trainEpochs
    for epoch in tqdm(range(trainEpochs)):
        pnet.train()
    
        # input, target = getBatch(batchSize, seqLength, input_dim)
        cSeqLength = np.random.randint(minSeqLength, maxSeqLength+1)
        input, target = dominoeBatch(batchSize, cSeqLength, listDominoes, dominoeValue, highestDominoe)
        input, target = input.to(device), target.to(device)
    
        optimizer.zero_grad()
        log_scores, choices = pnet(input)
        
        unrolled = log_scores.view(-1, log_scores.size(-1))
        loss = torch.nn.functional.nll_loss(unrolled, target.view(-1))
        assert not np.isnan(loss.item()), "model diverged :("
    
        loss.backward()
        optimizer.step()
    
        t = torch.gather(unrolled, dim=1, index=target.view(-1).unsqueeze(-1)).cpu().detach().view(batchSize, cSeqLength)
        m = torch.max(unrolled, dim=1)[0].cpu().detach().view(batchSize, cSeqLength)
        trainPositionError[epoch] = torch.mean(m-t,dim=0)
        trainLoss[epoch] = loss.item()
        trainSeqLength[epoch] = cSeqLength

    # Test network
    print("Testing network...")
    testSeqLength = torch.zeros(testEpochs)
    testLoss = torch.zeros(testEpochs)
    testPositionError = [None]*testEpochs
    for epoch in tqdm(range(testEpochs)):
        pnet.test()
    
        # input, target = getBatch(batchSize, seqLength, input_dim)
        cSeqLength = np.random.randint(minSeqLength, maxSeqLength+1)
        input, target = dominoeBatch(batchSize, cSeqLength, listDominoes, dominoeValue, highestDominoe)
        input, target = input.to(device), target.to(device)
    
        optimizer.zero_grad()
        log_scores, choices = pnet(input)
        
        unrolled = log_scores.view(-1, log_scores.size(-1))
        loss = torch.nn.functional.nll_loss(unrolled, target.view(-1))
        assert not np.isnan(loss.item()), "model diverged :("
    
        loss.backward()
        optimizer.step()
    
        t = torch.gather(unrolled, dim=1, index=target.view(-1).unsqueeze(-1)).cpu().detach().view(batchSize, cSeqLength)
        m = torch.max(unrolled, dim=1)[0].cpu().detach().view(batchSize, cSeqLength)
        testPositionError[epoch] = torch.mean(m-t,dim=0)
        testLoss[epoch] = loss.item()
        testSeqLength[epoch] = cSeqLength

    results = {
        'trainLoss': trainLoss,
        'trainSeqLength': trainSeqLength, 
        'trainPositionError': trainPositionError,
        'testLoss': testLoss,
        'testSeqLength': testSeqLength, 
        'testPositionError': testPositionError
    }

    return results


def plotResults(results, args):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(range(args.train_epochs), results['trainLoss'])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim(0)
    ax[0].set_title('Training Loss')
    yMin, yMax = ax[0].get_ylim()
    
    ax[1].plot(range(args.test_epochs), results['testLoss'])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_ylim(yMin, yMax)
    ax[1].set_title('Testing Loss')

    ax[2].scatter(results['testSeqLength'], results['testLoss'], color='k', alpha=0.5)
    ax[2].set_xlabel('Sequence Length')
    ax[2].set_ylabel('Testing Loss')
    ax[2].set_ylim(yMin, yMax)
    ax[2].set_title('Test Loss vs. Sequence Length')

    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()
    
if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        # estimate ELO with the requested parameters and agents
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
    
    
    






