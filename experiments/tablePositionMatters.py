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
import torch.cuda as torchCuda

# dominoes package
from dominoes import gameplay as dg
from dominoes import agents as da
from dominoes import functions as df

# input arguments
parser = argparse.ArgumentParser(description='Run dominoes experiment.')
parser.add_argument('-np','--num-players',type=int, default=4, help='the number of players for each game')
parser.add_argument('-hd','--highest-dominoe',type=int, default=9, help='highest dominoe value in the set')
parser.add_argument('-nr','--num-rounds',type=int, default=1000, help='how many rounds to play to estimate average score')
parser.add_argument('--nosave',default=False,action='store_true')

args = parser.parse_args()

networkPath = Path(mainPath) / 'experiments' / 'savedNetworks' / 'lineValueAgentParameters_230817_1.npy' # path for loading saved network
savePath = Path(mainPath) / 'docs' / 'media' # path for saving png result
device = 'cuda' if torchCuda.is_available() else 'cpu'

if __name__=='__main__':
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe
    dominoes = df.listDominoes(highestDominoe)
    numDominoes = len(dominoes)

    # Instantiate agents
    agentsForComparison = [
        da.doubleAgent(numPlayers, highestDominoe, dominoes, numDominoes, device=device),
        da.persistentLineAgent(numPlayers, highestDominoe, dominoes, numDominoes, device=device),
        da.lineValueAgent(numPlayers, highestDominoe, dominoes, numDominoes, device=device)
    ]
    # Set parameters for persistent line agent
    agentsForComparison[1].maxLineLength = 12

    # Set parameters for linevalue agent
    agentsForComparison[2].loadAgentParameters(networkPath)
    agentsForComparison[2].setLearning(False)

    # Loop through agents to compare list and 
    numAgentsToCompare = len(agentsForComparison)
    scorePerHand = np.zeros((numAgentsToCompare, numPlayers-1))
    for idx, agent in enumerate(agentsForComparison):
        agents = (agent, *[None]*(numPlayers-1))
        # create game with agents specified and shuffleAgents=False so the order never changes
        game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers, agents=agents, defaultAgent=da.dominoeAgent, shuffleAgents=False, device=device)
        game.playGame(rounds=args.num_rounds, withUpdates=True)

        # Average score per hand of default dominoeAgent (in order of agents sitting after the double agent)
        scorePerHand[idx] = game.currentScore[1:] / args.num_rounds

    colors = 'krb'
    
    # Plot ELO trajectory and final ELO estimates for each agent type
    fig = plt.figure()
    for idx in range(numAgentsToCompare):
        plt.plot(range(numPlayers-1), scorePerHand[idx], color=colors[idx], marker='.', markersize=10, linewidth=2.5, label=f"Against {agentsForComparison[idx].agentName}")
    plt.xlabel('Distance from better agent')
    plt.ylabel('Average Score Per Hand')
    plt.ylim(0)
    plt.legend(fontsize=12, loc='lower left')
    
    if not(args.nosave):
        plt.savefig(str(savePath/'tablePositionMatters.png')) # save figure and show to user
    plt.show()

    
    
    

