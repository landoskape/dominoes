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
from dominoes import leagueManager as lm
from dominoes import gameplay as dg
from dominoes import agents as da
from dominoes import functions as df

valueAgents = {
    'basicValueAgent':da.basicValueAgent,
    'lineValueAgent':da.lineValueAgent,
    'lineValueAgentSmall':da.lineValueAgentSmall
}

opponents = {
    'dominoeAgent':da.dominoeAgent,
    'greedyAgent':da.greedyAgent,
    'stupidAgent':da.stupidAgent,
    'doubleAgent':da.doubleAgent,
    'persistentLineAgent':da.persistentLineAgent
}

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path('.') / 'experiments' / 'savedResults'
prmsPath = Path('.') / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

device = "cuda" if torchCuda.is_available() else "cpu"
print(f"Using device: {device}")

def parseArgs():
    parser = argparse.ArgumentParser(description='Run dominoes experiment.')
    parser.add_argument('-n','--num-players', type=int, default=4, help='the number of agents in the game of dominoes')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('-s','--shuffle-agents', type=bool, default=True, help='whether to shuffle the order of the agents each hand')
    parser.add_argument('-tg','--train-games',type=int, default=3000, help='the number of training games')
    parser.add_argument('-tr','--train-rounds',type=int, default=None, help='the number of training rounds')
    parser.add_argument('-op','--opponent',type=str, default='dominoeAgent', help='which opponent to play the basic value agent against for training and testing')
    parser.add_argument('-va','--value-agent',type=str, default='basicValueAgent', help='which value agent to use')
    parser.add_argument('--noreplay',default=False,action='store_true', help='if used, will turn off replay in the value agents')
    parser.add_argument('--justplot',default=False,action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave',default=False,action='store_true')

    args = parser.parse_args()
    
    assert args.value_agent in valueAgents.keys(), f"requested value agent ({args.value_agent}) is not in the list of possibilities!"
    assert args.opponent in opponents.keys(), f"requested opponent ({args.opponent}) is not in the list of possible opponents!"
     
    return args

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName():
    replayString = '' if args.noreplay else 'withReplay_'
    return f"trainValueAgent_{args.value_agent}_"+replayString+f"against_{args.opponent}"

# method for training agent
def trainValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds):
    # open game with basic value agent playing against three default dominoe agents 
    agents=(valueAgents[args.value_agent], None, None, None)
    game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=agents, defaultAgent=opponents[args.opponent], device=device)
    game.getAgent(0).setLearning(True)
    if args.noreplay: game.getAgent(0).setReplay(False)
    
    # run training rounds
    trainWinnerCount = np.zeros(numPlayers)
    trainHandWinnerCount = np.zeros((trainGames,numPlayers))
    trainScoreTally = np.zeros((trainGames,numPlayers))
    for gameIdx in tqdm(range(trainGames)):
        game.playGame(rounds=trainRounds)
        trainWinnerCount[game.currentWinner] += 1
        trainHandWinnerCount[gameIdx] += np.sum(game.score==0,axis=0)
        trainScoreTally[gameIdx] += game.currentScore
    
    results = {
        'trainWinnerCount':trainWinnerCount, 
        'trainHandWinnerCount':trainHandWinnerCount,
        'trainScoreTally':trainScoreTally,
    }

    # save results if requested
    if not(args.nosave):
        # Save agent parameters
        description = f"{args.value_agent} trained against {args.opponent}"
        fullSavePath = game.getAgent(0).saveAgentParameters(savePath, modelName=getFileName(), description=description)
        np.save(prmsPath / getFileName(), vars(args))
        np.save(resPath / getFileName(), results)
    
    # return model and results for plotting
    return results

# And a function for plotting results
def plotResults(results):
    filter = lambda x : savgol_filter(x, 20, 1)
    trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(range(args.train_games), 
               filter(results['trainScoreTally'][:,0]/trainRounds), 
               c='b', label=args.value_agent)
    ax[0].plot(range(args.train_games), 
               filter(np.mean(results['trainScoreTally'][:,1:],axis=1)/trainRounds),
               c='k', label=f"{args.opponent}")
    ax[0].set_ylim(0)
    ax[0].set_xlabel('Training Games')
    ax[0].set_ylabel('Training Score Per Hand')
    ax[0].legend(loc='best')

    ax[1].plot(range(args.train_games), 
               filter(results['trainHandWinnerCount'][:,0]), 
               c='b', label=args.value_agent)
    ax[1].plot(range(args.train_games), 
               filter(np.mean(results['trainHandWinnerCount'][:,1:],axis=1)), 
               c='k', label=f"{args.opponent}")
    ax[1].set_ylim(0)
    ax[1].set_xlabel('Training Games')
    ax[1].set_ylabel('Training Num Won Hands')
    ax[1].legend(loc='best')
    
    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()

# Main script
if __name__=='__main__': 
    args = parseArgs()
    
    # Sorry for my improper style
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe
    shuffleAgents = args.shuffle_agents
    trainGames = args.train_games
    trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1

    # if just plotting, load data. Otherwise, run training and testing
    if not(args.justplot):
        results = trainValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds)
    else:
        print("Need to check if args match saved args!!!")
        results = np.load(resPath / (getFileName()+'.npy'), allow_pickle=True).item()

    # Print results of experiment
    print("Train winner count: ", results['trainWinnerCount'])
    tenPercent = int(np.ceil(trainGames*0.1))
    avgScore = np.round(np.mean(results['trainScoreTally'][-tenPercent:]/trainRounds,axis=0),1)
    print(f"Average score per round in last 10% of training: {avgScore}")

    # Plot results of experiment (and save if requested)
    plotResults(results)
    
    

