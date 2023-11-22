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
from dominoes import datasets
from dominoes import training

device = 'cuda' if torchCuda.is_available() else 'cpu'

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

for path in (resPath, prmsPath, figsPath, savePath):
    if not(path.exists()):
        path.mkdir()

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(network=False):
    fname = "pointerSequencer"
    if args.use_rl:
        fname = fname + '_rl'
    if network:
        fname = fname + '.pt'
    return fname

def gamma_type(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a floating-point literal")

    if (x <= 0.0) or (x > 1.0):
        raise argparse.ArgumentTypeError(f"{x} is not in range (0.0, 1.0]")
    return x

def handleArguments():
    parser = argparse.ArgumentParser(description='Run pointer dominoe sequencing experiment.')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('-hs','--hand-size', type=int, default=12, help='the maximum tokens per sequence')
    parser.add_argument('-bs','--batch-size',type=int, default=128, help='number of sequences per batch')
    parser.add_argument('-ne','--train-epochs',type=int, default=20000, help='the number of training epochs')
    parser.add_argument('-te','--test-epochs',type=int, default=1000, help='the number of testing epochs')

    parser.add_argument('--embedding_dim', type=int, default=96, help='the dimensions of the embedding')
    parser.add_argument('--heads', type=int, default=8, help='the number of heads in transformer layers')
    parser.add_argument('--expansion', type=int, default=2, help='the expansion at the MLP part of the transformer')
    parser.add_argument('--encoding-layers', type=int, default=2, help='the number of stacked transformers in the encoder')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature of choice during training')
    parser.add_argument('--use-rl', default=False, action='store_true', help='if used, will learn by reinforcement learning, otherwise supervised')
    parser.add_argument('--gamma', type=gamma_type, default=0.3, help='the gamma value used for discounting in RL')
    parser.add_argument('--justplot', default=False, action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave', default=False, action='store_true')
    
    args = parser.parse_args()

    return args

def get_gamma_transform(gamma, N):
    exponent = torch.arange(N).view(-1,1) - torch.arange(N).view(1,-1)
    gamma_transform = (gamma ** exponent * (exponent >= 0))
    return gamma_transform
    
def trainTestModel():
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = df.listDominoes(highestDominoe)

    handSize = args.hand_size
    batchSize = args.batch_size
    null_token=True
    available_token=True
    num_output = copy(handSize)+1 # always require end on null token
    ignore_index = -1
    
    input_dim = (2 if not(available_token) else 3)*(highestDominoe+1) + (1 if null_token else 0)
    embedding_dim = args.embedding_dim
    heads = args.heads
    expansion = args.expansion
    encoding_layers = args.encoding_layers
    temperature = args.temperature
    contextual_encoder = True # use available token as a context

    use_rl = args.use_rl
    return_target = not(use_rl) # if not using RL, then using supervised learning and we need the target
    thompson = copy(use_rl) # only do thompson sampling if in reinforcement learning
    value_method = '2'

    if use_rl:
        gamma = args.gamma
        gamma_transform = get_gamma_transform(gamma, num_output).to(device)
        
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs

    alpha = 1e-3
    weight_decay = 1e-5
    
    # Create a pointer network
    net = transformers.PointerNetwork(input_dim, embedding_dim, encoding_layers=encoding_layers, heads=heads, expansion=expansion, 
                                      kqnorm=True, contextual_encoder=contextual_encoder, decoder_method='transformer', 
                                      temperature=temperature, pointer_method='PointerStandard', thompson=thompson)
    net = net.to(device)
    
    # Create an optimizer, Adam with weight decay is pretty good
    optimizer = torch.optim.Adam(net.parameters(), lr=alpha, weight_decay=weight_decay)
    
    # Train network
    print("Training network...")
    trainValue = torch.zeros(trainEpochs)
    for epoch in tqdm(range(trainEpochs)):
        # zero gradients
        optimizer.zero_grad()
    
        # generate input batch
        batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, return_target=return_target, null_token=null_token,
                                       available_token=available_token, ignore_index=ignore_index, return_full=True)

        # unpack batch tuple
        input, target, _, _, _, selection, available = batch

        # move to correct device
        input = input.to(device)
        if return_target:
            target = target.to(device)

        # divide input into main input and context
        x, context = input[:, :-1], input[:, [-1]]
        input = (x, context)
        
        # propagate it through the network
        out_scores, out_choices = net(input, max_output=num_output)
        
        # measure loss and do backward pass
        if use_rl:
            # measure rewards for each sequence
            rewards = training.measureReward_sequencer(available, listDominoes[selection], out_choices, value_method=value_method, normalize=False)
            G = torch.matmul(rewards, gamma_transform)
            logprob_policy = torch.gather(out_scores, 2, out_choices.unsqueeze(2)).squeeze(2) # log-probability for each chosen dominoe
            
            # do backward pass on J
            J = -torch.sum(logprob_policy * G)
            J.backward()

            c_train_value = torch.mean(torch.sum(G, dim=1))
            
        else:
            unrolled = out_scores.view(batchSize * num_output, -1)
            loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignore_index)
            loss.backward()

            c_train_value = loss.item()
    
        # update network
        optimizer.step()
    
        trainValue[epoch] = c_train_value
    
    # Test network - same thing as in testing but without updates to model
    with torch.no_grad():
        print("Testing network...")
        net.setTemperature(1.0)
        net.setThompson(False)
        
        testValue = torch.zeros(testEpochs)
        for epoch in tqdm(range(testEpochs)):
            # generate input batch
            batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, return_target=return_target,
                                           null_token=null_token, available_token=available_token, ignore_index=ignore_index, return_full=True)
    
            # unpack batch tuple
            input, target, _, _, _, selection, available = batch
    
            # move to correct device
            input = input.to(device)
            if return_target:
                target = target.to(device)
    
            # divide input into main input and context
            x, context = input[:, :-1], input[:, [-1]]
            input = (x, context)
            
            # propagate it through the network
            out_scores, out_choices = net(input, max_output=num_output)
            
            # measure loss and do backward pass
            if use_rl:
                # measure rewards for each sequence
                rewards = training.measureReward_sequencer(available, listDominoes[selection], out_choices, value_method=value_method, normalize=False)
                G = torch.matmul(rewards, gamma_transform)
                c_test_value = torch.mean(torch.sum(G, dim=1))
            else:
                unrolled = out_scores.view(batchSize * num_output, -1)
                loss = torch.nn.functional.nll_loss(unrolled, target.view(-1), ignore_index=ignore_index)
                c_test_value = loss.item()
        
            testValue[epoch] = c_test_value
            

    results = {
        'trainValue': trainValue,
        'testValue': testValue,
    }

    return results, net


def plotResults(results, args):
    ylabel = 'G' if args.use_rl else 'Loss'
    title = lambda x : x + ('Reward' if args.use_rl else 'Loss')
    fig, ax = plt.subplots(1,2,figsize=(8,4), layout='constrained')
    ax[0].plot(range(args.train_epochs), results['trainValue'], color='k', lw=1)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel(ylabel)
    if not(args.use_rl):
        ax[0].set_ylim(0)
    yMin, yMax = ax[0].get_ylim()
    ax[0].set_title(title('Training'))
    
    ax[1].plot(range(args.test_epochs), results['testValue'], color='b', lw=1)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel(ylabel)
    ax[1].set_ylim(yMin, yMax)
    ax[1].set_title(title('Testing'))

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
        results, pnet = trainTestModel()
        
        # save results if requested
        if not(args.nosave):
            # Save agent parameters
            torch.save(pnet, savePath / getFileName(network=True))
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
    
    
    






