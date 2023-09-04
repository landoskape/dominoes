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
parser.add_argument('-tg','--train-games',type=int, default=500, help='the number of training games')
parser.add_argument('-tr','--train-rounds',type=int, default=50, help='the number of training rounds')
parser.add_argument('-pg','--test-games',type=int, default=100, help='the number of testing games')
parser.add_argument('-pr','--test-rounds',type=int, default=None, help='the number of testing rounds')
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

assert args.opponent in opponents.keys(), f"requested opponent ({args.opponent}) is not in the list of possible opponents!"

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'
resPath = Path('.') / 'experiments' / 'savedResults'
prmsPath = Path('.') / 'experiments' / 'savedParameters'

device = "cuda" if torchCuda.is_available() else "cpu"
print(f"Using device: {device}")

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName():
    return f"trainBasicValueAgent_withOpponent_{args.opponent}"

# method for training agent
def trainAndTestBasicValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds, testGames, testRounds):
    # open game with basic value agent playing against three default dominoe agents 
    agents=(da.basicValueAgent, None, None, None)
    game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=agents, defaultAgent=opponents[args.opponent], device=device)
    game.getAgent(0).setLearning(True)
    
    # run training rounds
    trainWinnerCount = np.zeros(numPlayers)
    trainScoreTally = np.zeros((trainGames,numPlayers))
    for gameIdx in tqdm(range(trainGames)):
        game.playGame(rounds=trainRounds)
        trainWinnerCount[game.currentWinner] += 1
        trainScoreTally[gameIdx] += game.currentScore
    
    # create league for measuring ELO
    league = lm.leagueManager(highestDominoe, numPlayers, shuffleAgents=True)

    # Add agents
    league.addAgent(game.getAgent(0))
    league.getAgent(0).setLearning(False) # I don't want these agents to update their parameters anymore
    league.addAgentType(opponents[args.opponent], num2add=numPlayers-1)

    # Do a bunch of games, return scores and ELOs
    game, leagueIndex = league.createGame() # only need to create a game once because there's numPlayers agents in the league
    testWinnerCount = np.zeros(numPlayers)
    testScoreTally = np.zeros(numPlayers)
    for gameIdx in range(testGames):
        game.playGame()
        testWinnerCount[leagueIndex[game.currentWinner]] += 1
        testScoreTally += game.currentScore[leagueIndex]
        league.updateElo(leagueIndex, game.currentScore) # update ELOs

    results = {
        'trainWinnerCount':trainWinnerCount, 
        'trainScoreTally':trainScoreTally,
        'testWinnerCount':testWinnerCount,
        'testScoreTally':testScoreTally,
        'elo':league.elo
    }

    # save results if requested
    if not(args.nosave):
        # Save agent parameters
        fullSavePath = league.getAgent(0).saveAgentParameters(savePath, modelName=getFileName(), description=f"basicValueAgent trained against {args.opponent}")
        np.save(prmsPath / getFileName(), vars(args))
        np.save(resPath / getFileName(), results)
    
    # return model and results for plotting
    return results

# And a function for plotting results
def plotResults(results):
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(range(args.train_games), results['trainScoreTally'][:,0], c='b', label='basicValueAgent')
    ax[0].plot(range(args.train_games), np.mean(results['trainScoreTally'][:,1:],axis=1), c='k', label=f"{args.opponent}")
    ax[0].set_xlabel('Training Games')
    ax[0].set_ylabel('Training Score')
    ax[0].legend(loc='lower left')
    plt.show()

# Main script
if __name__=='__main__': 
    # Sorry for my improper style
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe
    shuffleAgents = args.shuffle_agents
    trainGames = args.train_games
    trainRounds = args.train_rounds
    testGames = args.test_games
    testRounds = args.test_rounds if args.test_rounds is not None else highestDominoe+1

    # if just plotting, load data. Otherwise, run training and testing
    if not(args.justplot):
        results = trainAndTestBasicValueAgent(numPlayers, highestDominoe, shuffleAgents, trainGames, trainRounds, testGames, testRounds)
    else:
        print("Need to check if args match saved args!!!")
        results = np.load(resPath / (getFileName()+'.npy'), allow_pickle=True).item()

    # Print results of experiment
    print("Train winner count: ", results['trainWinnerCount'])
    print("Test winner count: ", results['testWinnerCount'])
    print("ELO in testing: ", results['elo'])
    print(f"Average score per round in testing: {results['testScoreTally']/testGames/testRounds}")

    # Plot results of experiment
    plotResults(results)
    
    

