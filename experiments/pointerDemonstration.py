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

def dominoeUnevenBatch(batchSize, minSeq, maxSeq, listDominoes, dominoeValue, highestDominoe, ignoreIndex=-1):
    """
    retrieve a batch of dominoes and their target order given the value of each dominoe

    dominoes are paired values (combinations with replacement) of integers 
    from 0 to <highestDominoe>. The total value of each dominoe is the sum of
    the two integers associated with that dominoe. For example, the dominoe
    (7|3) has value 10. 

    Each element in the batch contains an input and target. The input is 
    composed of a sequence of dominoes in a random order, transformed into a
    simple representation (explained below). The target is a list of the order
    of dominoes by the one with the highest value to the one with the lowest
    value. Note that many dominoes share the same value, but since the dominoe
    list is always the same, equal value dominoes will always be sorted in the
    same way. 
    
    Each element can have a different sequence length, they will be padded 
    with zeros to whatever the longest sequence is. The ignoreIndex is used to
    determine what to label targets for any padded elements (i.e. any place 
    where no prediction is needed). The nll_loss function then accepts this as
    an input to ignore. This is part of the reason why pointer networks are
    awesome... the input and output can vary in size!!!

    The simple representation is a two-hot vector where the first 
    <highestDominoe+1> elements represent the first value of the dominoe, and
    the second <highestDominoe+1> elements represent the second value of the 
    dominoe. Here are some examples for highest dominoe = 3:

    (0 | 0): [1, 0, 0, 0, 1, 0, 0, 0]
    (0 | 1): [1, 0, 0, 0, 0, 1, 0, 0]
    (0 | 2): [1, 0, 0, 0, 0, 0, 1, 0]
    (0 | 3): [1, 0, 0, 0, 0, 0, 0, 1]
    (1 | 0): [0, 1, 0, 0, 1, 0, 0, 0]
    (2 | 1): [0, 0, 1, 0, 0, 1, 0, 0]

    """
    numDominoes = len(listDominoes)
    input_dim = 2*(highestDominoe+1)
    
    # choose how long each sequence in the batch will be
    seqLength = np.random.randint(minSeq, maxSeq+1, batchSize)
    maxSeqLength = max(seqLength) # max sequence length for padding

    # choose dominoes from the batch, and get their value (in points)
    selection = [np.random.choice(numDominoes, sl, replace=False).tolist() for sl in seqLength]
    value = [dominoeValue[sel] for sel in selection]
    
    # index of first and second value in two-hot representation
    pad = [[0]*(maxSeqLength-sl) for sl in seqLength]
    firstValue = np.stack([listDominoes[sel,0].tolist()+p for p, sel in zip(pad, selection)])
    secondValue = np.stack([(listDominoes[sel,1]+highestDominoe+1).tolist()+p for p, sel in zip(pad, selection)])
    firstValue = torch.tensor(firstValue, dtype=torch.int64).unsqueeze(2)
    secondValue = torch.tensor(secondValue, dtype=torch.int64).unsqueeze(2)

    # create mask (used for scattering and also as an output)
    mask = 1.*(torch.arange(maxSeqLength).view(1,-1).expand(batchSize, -1) < torch.tensor(seqLength).view(-1,1))

    # scatter data into two-hot vectors, except where sequence length is exceed where the mask is 0
    input = torch.zeros((batchSize, maxSeqLength, input_dim), dtype=torch.float)
    input.scatter_(2, firstValue, mask.float().unsqueeze(2))
    input.scatter_(2, secondValue, mask.float().unsqueeze(2))
    
    # sort and pad each list of dominoes by value
    def sortPad(val, padTo, ignoreIndex=-1):
        s = sorted(range(len(val)), key=lambda i: -val[i])
        p = [ignoreIndex]*(padTo-len(val))
        return s+p

    # create a padded sort index, then turn into a torch tensor as the target vector
    sortIdx = [sortPad(val, maxSeqLength, ignoreIndex) for val in value] # pad with ignore index so nll_loss ignores them
    target = torch.stack([torch.LongTensor(idx) for idx in sortIdx])

    return input, target, mask
    
# def dominoeBatch(batchSize, seqLength, listDominoes, dominoeValue, highestDominoe):
#     """
#     retrieve a batch of dominoes and their target order given the value of each dominoe

#     dominoes are paired values (combinations with replacement) of integers 
#     from 0 to <highestDominoe>. The total value of each dominoe is the sum of
#     the two integers associated with that dominoe. For example, the dominoe
#     (7|3) has value 10. 

#     Each element in the batch contains an input and target. The input is 
#     composed of a sequence of dominoes in a random order, transformed into a
#     simple representation (explained below). The target is a list of the order
#     of dominoes by the one with the highest value to the one with the lowest
#     value. Note that many dominoes share the same value, but since the dominoe
#     list is always the same, equal value dominoes will always be sorted in the
#     same way. 

#     The simple representation is a two-hot vector where the first 
#     <highestDominoe+1> elements represent the first value of the dominoe, and
#     the second <highestDominoe+1> elements represent the second value of the 
#     dominoe. Here are some examples for highest dominoe = 3:

#     (0 | 0): [1, 0, 0, 0, 1, 0, 0, 0]
#     (0 | 1): [1, 0, 0, 0, 0, 1, 0, 0]
#     (0 | 2): [1, 0, 0, 0, 0, 0, 1, 0]
#     (0 | 3): [1, 0, 0, 0, 0, 0, 0, 1]
#     (1 | 0): [0, 1, 0, 0, 1, 0, 0, 0]
#     (2 | 1): [0, 0, 1, 0, 0, 1, 0, 0]

#     For simplification, every element in the batch has the same sequence 
#     length, but once I have a mask working, I can vary sequence length within
#     the batch. And that's why pointer networks are awesome. 
#     """
#     numDominoes = len(listDominoes)
#     input_dim = 2*(highestDominoe+1)

#     # select list of seqLength dominoes
#     selection = np.stack([np.random.choice(numDominoes, seqLength, replace=False).tolist() for _ in range(batchSize)])
#     value = dominoeValue[selection]
#     sortIdx = [sorted(range(len(val)), key=lambda i : -val[i]) for val in value] # sort dominoes by value in decreasing order
#     target = torch.stack([torch.LongTensor(idx) for idx in sortIdx])
    
#     firstValue = torch.tensor(listDominoes[selection,0], dtype=torch.int64).unsqueeze(2)
#     secondValue = torch.tensor(listDominoes[selection,1], dtype=torch.int64).unsqueeze(2) + highestDominoe+1

#     input = torch.zeros((batchSize, seqLength, input_dim), dtype=torch.float)
#     input.scatter_(2, firstValue, torch.ones_like(firstValue, dtype=torch.float))
#     input.scatter_(2, secondValue, torch.ones_like(secondValue, dtype=torch.float))

#     return input, target

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

        input, target, mask = dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, listDominoes, dominoeValue, highestDominoe, ignoreIndex=ignoreIndex)
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

            input, target, mask = dominoeUnevenBatch(batchSize, minSeqLength, maxSeqLength, listDominoes, dominoeValue, highestDominoe, ignoreIndex=ignoreIndex)
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
    
    
    






