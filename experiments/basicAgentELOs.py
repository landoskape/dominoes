# mainExperiment at checkpoint 1
import sys
import os

# add path that contains the dominoes package
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from copy import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dominoes import leagueManager as lm
from dominoes import gameplay as dg
from dominoes import agents as da
from dominoes import functions as df

parser = argparse.ArgumentParser(description='Run dominoes experiment.')
parser.add_argument('-np','--num-players',type=int, default=4, help='the number of players for each game')
parser.add_argument('-hd','--highest-dominoe',type=int, default=9, help='highest dominoe value in the set')
parser.add_argument('-ng','--num-games',type=int, default=10000, help='how many games to play to estimate ELO')
parser.add_argument('-nr','--num-rounds',type=int, default=None, help='how many rounds to play for each game')
parser.add_argument('-ne','--num-each',type=int, default=4, help='how many copies of each agent type to include in the league')
parser.add_argument('-fe','--fraction-estimate',type=float, default=0.05, help='final fraction of elo estimates to use')

args = parser.parse_args()
assert 0 < args.fraction_estimate < 1, "fraction-estimate needs to be a float between 0 and 1"

savePath = Path(mainPath) / 'docs' / 'media'

if __name__=='__main__':
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe
    
    league = lm.leagueManager(highestDominoe, numPlayers, shuffleAgents=True)
    
    numEach = args.num_each
    league.addAgentType(da.bestLineAgent, num2add=numEach)
    league.addAgentType(da.doubleAgent, num2add=numEach)
    league.addAgentType(da.greedyAgent, num2add=numEach)
    league.addAgentType(da.dominoeAgent, num2add=numEach)
    league.addAgentType(da.stupidAgent, num2add=numEach)

    assert numPlayers <= league.numAgents, "the number of players must be less than the number of agents in the league!"
    
    numGames = args.num_games
    num2EstimateWith = int(numGames * args.fraction_estimate)
    
    trackElo = np.zeros((numGames, league.numAgents))
    for gameIdx in tqdm(range(numGames)):
        game, leagueIndex = league.createGame()
        game.playGame()
        league.updateElo(leagueIndex, game.currentScore) # update ELO
        trackElo[gameIdx] = copy(league.elo)
    
    avgEloPerAgentType = np.mean(trackElo.T.reshape(5,numEach,numGames),axis=1)
    agentTypeNames = [agent.agentName for agent in league.agents[::numEach]]

    eloEstimate = np.mean(avgEloPerAgentType[:,-num2EstimateWith:],axis=1)
    for name, elo in zip(agentTypeNames, eloEstimate):
        print(f"Agent {name} has a final ELO of {elo}")
    
    avgEloPerAgentType = np.mean(trackElo.T.reshape(5,numEach,numGames),axis=1)
    agentTypeNames = [agent.agentName for agent in league.agents[::numEach]]
    
    fig,ax = plt.subplots(1,2, figsize=(12,4))
    for name, elo in zip(agentTypeNames, avgEloPerAgentType):
        ax[0].plot(range(numGames), elo, label=name, linewidth=2)
    ax[0].set_xlabel('Number of games')
    ax[0].set_ylabel('Average ELO')
    ax[0].set_ylim(0, 2000)
    ax[0].legend(fontsize=12, loc='lower left')
    
    ax[1].bar(range(5), eloEstimate, color='k', tick_label=agentTypeNames)
    plt.xticks(rotation=45)
    ax[1].set_ylabel('ELO')
    ax[1].set_ylim(0, 2000)
    plt.savefig(str(savePath/'basicAgentELOs.png'))
    plt.show()

    
    
    

