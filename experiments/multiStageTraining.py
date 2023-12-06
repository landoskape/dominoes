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
from scipy.signal import savgol_filter
import torch.cuda as torchCuda
import matplotlib.pyplot as plt

# dominoes package
from dominoes import gameplay as dg
from dominoes import agents as da

parser = argparse.ArgumentParser(description='Run dominoes experiment.')
parser.add_argument('-n','--num-players', type=int, default=4, help='the number of agents in the game of dominoes')
parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
parser.add_argument('-s','--shuffle-agents', type=bool, default=True, help='whether to shuffle the order of the agents each hand')
parser.add_argument('-tg','--train-games',type=int, default=1000, help='the number of training games')
parser.add_argument('-tr','--train-rounds',type=int, default=None, help='the number of training rounds')
parser.add_argument('-va','--value-agent',type=str, default='basicValueAgent', help='which value agent to use')
parser.add_argument('--justplot',default=False,action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
parser.add_argument('--nosave',default=False,action='store_true')

args = parser.parse_args()

valueAgents = {
    'basicValueAgent':da.basicValueAgent,
    'lineValueAgent':da.lineValueAgent,
    'lineValueAgentSmall':da.lineValueAgentSmall
}

assert args.value_agent in valueAgents.keys(), f"requested value agent ({args.value_agent}) is not in the list of possibilities!"

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path('.') / 'experiments' / 'savedResults'
prmsPath = Path('.') / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

for path in (resPath, prmsPath, figsPath, savePath):
    if not(path.exists()):
        path.mkdir()

device = "cuda" if torchCuda.is_available() else "cpu"
print(f"Using device: {device}")

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName():
    return f"multiStageTrainValueAgent_{args.value_agent}"

# method for training agent
def trainValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds):
    # open game with basic value agent playing against three default dominoe agents 
    agents=(valueAgents[args.value_agent], *[None]*(numPlayers-1))
    game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=agents, defaultAgent=da.dominoeAgent, device=device)
    game.getAgent(0).setLearning(True)
    
    # run training rounds in stage 1 (against default dominoeAgent)
    s1WinnerCount = np.zeros(numPlayers)
    s1HandWinnerCount = np.zeros((trainGames, numPlayers))
    s1ScoreTally = np.zeros((trainGames,numPlayers))
    for gameIdx in tqdm(range(trainGames)):
        game.playGame(rounds=trainRounds)
        s1WinnerCount[game.currentWinner] += 1
        s1HandWinnerCount[gameIdx] += np.sum(game.score==0,axis=0)
        s1ScoreTally[gameIdx] += game.currentScore

    trainedAgent = game.getAgent(0)
    agents = (game.getAgent(0), *[None]*(numPlayers-1))
    game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=agents, defaultAgent=da.persistentLineAgent, device=device)

    # run training rounds in stage 1 (against default dominoeAgent)
    s2WinnerCount = np.zeros(numPlayers)
    s2HandWinnerCount = np.zeros((trainGames, numPlayers))
    s2ScoreTally = np.zeros((trainGames,numPlayers))
    for gameIdx in tqdm(range(trainGames)):
        game.playGame(rounds=trainRounds)
        s2WinnerCount[game.currentWinner] += 1
        s2HandWinnerCount[gameIdx] += np.sum(game.score==0,axis=0)
        s2ScoreTally[gameIdx] += game.currentScore
        
    results = {
        's1WinnerCount':s1WinnerCount, 
        's1HandWinnerCount':s1HandWinnerCount,
        's1ScoreTally':s1ScoreTally,
        's2WinnerCount':s2WinnerCount, 
        's2HandWinnerCount':s2HandWinnerCount,
        's2ScoreTally':s2ScoreTally,
    }

    # save results if requested
    if not(args.nosave):
        # Save agent parameters
        description = f"Multistage training of {args.value_agent}, first against dominoeAgent then persistentLineAgent"
        fullSavePath = game.getAgent(0).saveAgentParameters(savePath, modelName=getFileName(), description=description)
        np.save(prmsPath / getFileName(), vars(args))
        np.save(resPath / getFileName(), results)
    
    # return model and results for plotting
    return results

# And a function for plotting results
def plotResults(results):
    filter = lambda x : savgol_filter(x, 20, 1)
    
    trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1
    xs1 = range(args.train_games)
    xs2 = range(int(args.train_games*1.1), int(args.train_games*1.1)+args.train_games)
    
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(xs1, filter(results['s1ScoreTally'][:,0]/trainRounds), c='b', label=f"{args.value_agent}")
    ax[0].plot(xs2, filter(results['s2ScoreTally'][:,0]/trainRounds), c='b')
    ax[0].plot(xs1, filter(np.mean(results['s1ScoreTally'][:,1:],axis=1)/trainRounds), c='k', label='dominoeAgent')
    ax[0].plot(xs2, filter(np.mean(results['s2ScoreTally'][:,1:],axis=1)/trainRounds), c='r', label='persistentLineAgent')
    ax[0].set_ylim(0)
    ax[0].set_xlabel('Training Games')
    ax[0].set_ylabel('Score Per Hand')
    ax[0].legend(loc='best')

    ax[1].plot(xs1, filter(results['s1HandWinnerCount'][:,0]), c='b', label=f"{args.value_agent}")
    ax[1].plot(xs2, filter(results['s2HandWinnerCount'][:,0]), c='b')
    ax[1].plot(xs1, filter(np.mean(results['s1HandWinnerCount'][:,1:],axis=1)), c='k', label='dominoeAgent')
    ax[1].plot(xs2, filter(np.mean(results['s2HandWinnerCount'][:,1:],axis=1)), c='r', label='persistentLineAgent')
    ax[1].set_ylim(0)
    ax[1].set_xlabel('Training Games')
    ax[1].set_ylabel('Number of Won Hands')
    ax[1].legend(loc='best')
    
    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()

# Main script
if __name__=='__main__':
    # if just plotting, load data. Otherwise, run training and testing
    if not(args.justplot):    
        # Sorry for my improper style
        numPlayers = args.num_players
        highestDominoe = args.highest_dominoe
        shuffleAgents = args.shuffle_agents
        trainGames = args.train_games
        trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1
        
        print(f"Performing multistage training of {args.value_agent}")
        results = trainValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds)
    else:
        # Load previous parameters 
        prms = np.load(prmsPath / (getFileName()+'.npy'), allow_pickle=True).item()
        assert prms.keys() <= vars(args).keys(), f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(args).keys())}"
        for (pk,pi), (ak,ai) in zip(prms.items(), vars(args).items()):
            if pk=='justplot': continue
            if pk=='nosave': continue
            if prms[pk] != vars(args)[ak]:
                print(f"Requested argument {ak}={ai} differs from saved, which is: {pk}={pi}. Using saved...")
                setattr(args,pk,pi)

        numPlayers = args.num_players
        highestDominoe = args.highest_dominoe
        shuffleAgents = args.shuffle_agents
        trainGames = args.train_games
        trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1
        
        results = np.load(resPath / (getFileName()+'.npy'), allow_pickle=True).item()
    
    # Print results of experiment
    tenPercent = int(np.ceil(trainGames*0.1))
    print("")
    print(f"Stage 1 Training: {args.value_agent} against dominoeAgent")
    print("Winner count: ", results['s1WinnerCount'])
    print(f"Average score per round in last 10% of training: {np.mean(results['s1ScoreTally'][-tenPercent:]/trainRounds,axis=0)}")

    print("")
    print(f"Stage 2 Training: {args.value_agent} against persistentValueAgent")
    print("Winner count: ", results['s2WinnerCount'])
    print(f"Average score per round in last 10% of training: {np.mean(results['s2ScoreTally'][-tenPercent:]/trainRounds,axis=0)}")

    # Plot results of experiment (and save if requested)
    plotResults(results)
    
    

