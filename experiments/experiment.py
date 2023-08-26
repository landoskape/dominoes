# mainExperiment at checkpoint 1
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch.cuda as torchCuda

from dominoes import gameplay as dg
from dominoes import agents as da
from dominoes import functions as df

parser = argparse.ArgumentParser(description='Run dominoes experiment.')
parser.add_argument('-e','--experiment',type=int, default=0, help='the experiment ID, see file for list of experiments')
parser.add_argument('-n','--num-players', type=int, default=4, help='the number of agents in the game of dominoes')
parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
parser.add_argument('-s','--shuffle-agents', type=bool, default=True, help='whether to shuffle the order of the agents each hand')
parser.add_argument('-tg','--training-games',type=int, default=500, help='the number of training games')
parser.add_argument('-tr','--training-rounds',type=int, default=50, help='the number of training rounds')
parser.add_argument('-pg','--performance-games',type=int, default=10, help='the number of performance games')
parser.add_argument('-pr','--performance-rounds',type=int, default=50, help='the number of performance rounds')
# parser.add_argument('-sv','--save-networks',type=bool, default=True, help='whether or not to save intermediate agents')

args = parser.parse_args()

# can edit this for each machine it's being used on
savePath = Path('.') / 'experiments' / 'savedNetworks'

device = "cuda" if torchCuda.is_available() else "cpu"
print(f"Using device: {device}")

def experiment0(numPlayers, highestDominoe, shuffleAgents, trainingGames, trainingRounds, performanceGames, performanceRounds):
    game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=(da.lineValueAgent, da.greedyAgent, da.greedyAgent, da.greedyAgent), device=device)
    game.getAgent(0).setLearning(True)
    
    # run training rounds
    trainingWinnerCount0 = np.zeros(numPlayers)
    trainingScoreTally0 = np.zeros((trainingGames,numPlayers))
    for gameIdx in tqdm(range(trainingGames)):
        game.playGame(rounds=trainingRounds)
        trainingWinnerCount0[game.currentWinner] += 1
        trainingScoreTally0[gameIdx] += game.currentScore
    
    # measure performance
    game.getAgent(0).setLearning(False)
    performanceWinnerCount0 = np.zeros(numPlayers)
    performanceScoreTally0 = np.zeros(numPlayers)
    for _ in tqdm(range(performanceGames)):
        game.playGame(rounds=performanceRounds)
        performanceWinnerCount0[game.currentWinner] += 1 
        performanceScoreTally0 += game.currentScore

    savePath0 = game.getAgent(0).saveAgentParameters(savePath, description="lineValueAgent trained from initialization against greedyAgents")
    
    # create a new game with bestLineAgents
    game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, shuffleAgents=shuffleAgents, agents=(da.lineValueAgent, da.bestLineAgent, da.bestLineAgent, da.bestLineAgent), device=device)
    game.getAgent(0).loadAgentParameters(savePath0)
    
    # run training rounds
    trainingWinnerCount1 = np.zeros(numPlayers)
    trainingScoreTally1 = np.zeros((trainingGames,numPlayers))
    for gameIdx in tqdm(range(trainingGames)):
        game.playGame(rounds=trainingRounds)
        trainingWinnerCount1[game.currentWinner] += 1
        trainingScoreTally1[gameIdx] += game.currentScore
    
    # measure performance
    game.getAgent(0).setLearning(False)
    performanceWinnerCount1 = np.zeros(numPlayers)
    performanceScoreTally1 = np.zeros(numPlayers)
    for _ in tqdm(range(performanceGames)):
        game.playGame(rounds=performanceRounds)
        performanceWinnerCount1[game.currentWinner] += 1 
        performanceScoreTally1 += game.currentScore

    # store the agent
    savePath1 = game.getAgent(0).saveAgentParameters(savePath, description="lineValueAgent pretrained against greedyAgents then trained against bestLineAgents")

if __name__=='__main__':
    experiment0(args.num_players, args.highest_dominoe, args.shuffle_agents, args.training_games, args.training_rounds, args.performance_games, args.performance_rounds)
    
    
    

