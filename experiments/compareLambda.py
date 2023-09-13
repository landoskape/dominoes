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

parser = argparse.ArgumentParser(description='Run dominoes experiment.')
parser.add_argument('-n','--num-players', type=int, default=4, help='the number of agents in the game of dominoes')
parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
parser.add_argument('-s','--shuffle-agents', type=bool, default=True, help='whether to shuffle the order of the agents each hand')
parser.add_argument('-tg','--train-games',type=int, default=2000, help='the number of training games')
parser.add_argument('-tr','--train-rounds',type=int, default=None, help='the number of training rounds')
parser.add_argument('-op','--opponent',type=str, default='dominoeAgent', help='which opponent to play the basic value agent against for training and testing')
parser.add_argument('--justplot',default=False,action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
parser.add_argument('--nosave',default=False,action='store_true')

args = parser.parse_args()

opponents = {
    'dominoeAgent':da.dominoeAgent,
    'greedyAgent':da.greedyAgent,
    'stupidAgent':da.stupidAgent,
    'doubleAgent':da.doubleAgent,
    'persistentLineAgent':da.persistentLineAgent
}

lambdas = np.linspace(0.05,0.95,4)

assert args.opponent in opponents.keys(), f"requested opponent ({args.opponent}) is not in the list of possible opponents!"

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path('.') / 'experiments' / 'savedResults'
prmsPath = Path('.') / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

device = "cuda" if torchCuda.is_available() else "cpu"
print(f"Using device: {device}")

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName():
    return f"compareLambda_against_{args.opponent}"

# method for training agent
def trainValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds, lambdas):
    numLambdas = len(lambdas)
    winnerCount = np.zeros((numLambdas,2))
    handWinnerCount = np.zeros((numLambdas,2))
    scoreTally = np.zeros((numLambdas,2))
    
    testGames = int(np.ceil(trainGames * 0.1)) # (ten percent of training worth of testing)
    
    for idx, lam in enumerate(lambdas):
        # open game with basic value agent playing against three default dominoe agents 
        agents=(da.lineValueAgent, None, None, None)
        game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=agents, defaultAgent=opponents[args.opponent], device=device)
        game.getAgent(0).setLearning(True)
        game.getAgent(0).lam = lam

        # run training rounds
        for gameIdx in tqdm(range(trainGames)):
            game.playGame(rounds=trainRounds)
        
        # Turn off learning and test agent
        game.getAgent(0).setLearning(False)
        cWinnerCount = np.zeros(numPlayers)
        cHandWinnerCount = np.zeros((testGames,numPlayers))
        cScoreTally = np.zeros((testGames,numPlayers))
        for gameIdx in tqdm(range(testGames)):
            game.playGame(rounds=trainRounds)
            cWinnerCount[game.currentWinner] += 1
            cHandWinnerCount[gameIdx] += np.sum(game.score==0,axis=0)
            cScoreTally[gameIdx] += game.currentScore
        
        # Summarize results for this agent and it's opponents
        winnerCount[idx,0] = cWinnerCount[0]
        winnerCount[idx,1] = np.round(np.mean(cWinnerCount[1:]),1)
        handWinnerCount[idx,0] = np.round(np.mean(cHandWinnerCount[:,0]/trainRounds),1)
        handWinnerCount[idx,1] = np.round(np.mean(cHandWinnerCount[:,1:]/trainRounds),1)
        scoreTally[idx,0] = np.round(np.mean(cScoreTally[:,0]/trainRounds),1)
        scoreTally[idx,1] = np.round(np.mean(cScoreTally[:,1:]/trainRounds),1)
        
    results = {
        'winnerCount': winnerCount,
        'handWinnerCount': handWinnerCount,
        'scoreTally': scoreTally,
        'lambdas':lambdas,
        'opponent':args.opponent
    }
    
    # save results if requested
    if not(args.nosave):
        np.save(prmsPath / getFileName(), vars(args))
        np.save(resPath / getFileName(), results)
    
    # return model and results for plotting
    return results

# And a function for plotting results
def plotResults(results):
    print('hello world')
#     filter = lambda x : savgol_filter(x, 20, 1)
#     trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1
#     fig,ax = plt.subplots(1,2,figsize=(8,4))
#     ax[0].plot(range(args.train_games), 
#                filter(results['trainScoreTally'][:,0]/trainRounds), 
#                c='b', label=args.value_agent)
#     ax[0].plot(range(args.train_games), 
#                filter(np.mean(results['trainScoreTally'][:,1:],axis=1)/trainRounds),
#                c='k', label=f"{args.opponent}")
#     ax[0].set_ylim(0)
#     ax[0].set_xlabel('Training Games')
#     ax[0].set_ylabel('Training Score Per Hand')
#     ax[0].legend(loc='best')

#     ax[1].plot(range(args.train_games), 
#                filter(results['trainHandWinnerCount'][:,0]), 
#                c='b', label=args.value_agent)
#     ax[1].plot(range(args.train_games), 
#                filter(np.mean(results['trainHandWinnerCount'][:,1:],axis=1)), 
#                c='k', label=f"{args.opponent}")
#     ax[1].set_ylim(0)
#     ax[1].set_xlabel('Training Games')
#     ax[1].set_ylabel('Training Num Won Hands')
#     ax[1].legend(loc='best')
    
#     if not(args.nosave):
#         plt.savefig(str(figsPath/getFileName()))
    
#     plt.show()

# Main script
if __name__=='__main__': 
    # Sorry for my improper style
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe
    shuffleAgents = args.shuffle_agents
    trainGames = args.train_games
    trainRounds = args.train_rounds if args.train_rounds is not None else highestDominoe+1

    # if just plotting, load data. Otherwise, run training and testing
    if not(args.justplot):
        results = trainValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds, lambdas)
    else:
        print("Need to check if args match saved args!!!")
        results = np.load(resPath / (getFileName()+'.npy'), allow_pickle=True).item()

    # Print results of experiment
    print("Winner count: \n", results['winnerCount'])
    print("Hand Winner Count: \n", results['handWinnerCount'])
    print("Score Tally: \n", results['scoreTally'])

    # Plot results of experiment (and save if requested)
    #plotResults(results)
    
    

