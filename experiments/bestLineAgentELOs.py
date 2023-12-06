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

# dominoes package
from dominoes import leagueManager as lm
from dominoes import agents as da

# input arguments
parser = argparse.ArgumentParser(description='Run dominoes experiment.')
parser.add_argument('-np','--num-players',type=int, default=4, help='the number of players for each game')
parser.add_argument('-hd','--highest-dominoe',type=int, default=9, help='highest dominoe value in the set')
parser.add_argument('-ng','--num-games',type=int, default=10000, help='how many games to play to estimate ELO')
# note: ELO is probability based, so increasing the number of rounds will usually exaggerate differences in ELO
parser.add_argument('-nr','--num-rounds',type=int, default=None, help='how many rounds to play for each game')
# adding copies of agents allows ELO to include estimates based on agents playing with themselves in the league - 
# this is important because the game dynamics change depending on what agents are present
parser.add_argument('-ne','--num-each',type=int, default=2, help='how many copies of each agent type to include in the league')
parser.add_argument('-fe','--fraction-estimate',type=float, default=0.05, help='final fraction of elo estimates to use')
parser.add_argument('--nosave',default=False,action='store_true')

args = parser.parse_args()
assert 0 < args.fraction_estimate < 1, "fraction-estimate needs to be a float between 0 and 1"

# path for saving .png 
savePath = Path(mainPath) / 'docs' / 'media'

if __name__=='__main__':
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe
    maxLineLengths = [6, 8, 10, 12]
    
    # create a league manager with the requested parameters
    league = lm.leagueManager(highestDominoe, numPlayers, shuffleAgents=True)

    # add copies of each agent type
    numEach = args.num_each
    for idx, maxLine in enumerate(maxLineLengths):
        league.addAgentType(da.bestLineAgent, num2add=numEach)
        for agent in league.agents[idx*numEach:]:
            agent.maxLineLength = maxLine
    for idx, maxLine in enumerate(maxLineLengths):
        league.addAgentType(da.persistentLineAgent, num2add=numEach)
        for agent in league.agents[len(maxLineLengths)*numEach+idx*numEach:]:
            agent.maxLineLength = maxLine

    assert numPlayers <= league.numAgents, "the number of players must be less than the number of agents in the league!"
    
    numGames = args.num_games
    num2EstimateWith = int(numGames * args.fraction_estimate)

    # Run lots of games, update and track ELO scores 
    trackElo = np.zeros((numGames, league.numAgents))
    for gameIdx in tqdm(range(numGames)):
        game, leagueIndex = league.createGame()
        game.playGame()
        league.updateElo(leagueIndex, game.currentScore) # update ELO
        trackElo[gameIdx] = copy(league.elo)

    # average ELO scores across agent type, and get agentType names
    numAgentTypes = 2*len(maxLineLengths)
    avgEloPerAgentType = np.mean(trackElo.T.reshape(numAgentTypes,numEach,numGames),axis=1)
    agentTypeNames = [f"{agent.agentName}({agent.maxLineLength})" for agent in league.agents[::numEach]]

    # use last fraction of ELO estimates to get an averaged estimate of ELO
    eloEstimate = np.mean(avgEloPerAgentType[:,-num2EstimateWith:],axis=1)
    for name, elo in zip(agentTypeNames, eloEstimate):
        print(f"Agent {name} has a final ELO of {elo:.1f}")

    # Get these colors to coordinate across subplots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[i] for i in range(numAgentTypes)]

    # Plot ELO trajectory and final ELO estimates for each agent type
    fig,ax = plt.subplots(1,2, figsize=(12,5))
    for idx, (name, elo) in enumerate(zip(agentTypeNames, avgEloPerAgentType)):
        ax[0].plot(range(numGames), elo, label=name, linewidth=2, c=colors[idx])
    ax[0].set_xlabel('Number of games')
    ax[0].set_ylabel('Average ELO')
    ax[0].set_ylim(0, 2000)
    ax[0].legend(fontsize=12, loc='lower left')
    
    ax[1].bar(range(numAgentTypes), eloEstimate, color=colors, tick_label=agentTypeNames)
    plt.xticks(rotation=15)
    ax[1].set_ylabel('ELO')
    ax[1].set_ylim(0, 2000)
    if not(args.nosave):
        plt.savefig(str(savePath/'bestLineAgentELOs.png')) # save figure and show to user
    plt.show()

    
    
    

