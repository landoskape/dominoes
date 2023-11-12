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
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.cuda as torchCuda

# dominoes package
from dominoes import leagueManager as lm
from dominoes import gameplay as dg
from dominoes import agents as da
from dominoes import functions as df

device = 'cuda' if torchCuda.is_available() else 'cpu'

# can edit this for each machine it's being used on
resPath = Path(mainPath) / 'experiments' / 'savedResults'
prmsPath = Path(mainPath) / 'experiments' / 'savedParameters'
figsPath = Path(mainPath) / 'docs' / 'media'

for path in (resPath, prmsPath, figsPath):
    if not(path.exists()):
        path.mkdir()
        
# paths for loading previously trained agents
networkPath = Path(mainPath) / 'experiments' / 'savedNetworks' 

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

# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName():
    return f"evaluateReplay_against_{args.opponent}"

def getNetworkPath(replay=True):
    replayString = 'withReplay_' if replay else ''
    name = f"trainValueAgent_{args.value_agent}_"+replayString+f"against_{args.opponent}"
    path = networkPath / (name+'.npy')
    return path

def handleArguments():
    parser = argparse.ArgumentParser(description='Run dominoes experiment.')
    parser.add_argument('-n','--num-players', type=int, default=4, help='the number of agents in the game of dominoes')
    parser.add_argument('-hd','--highest-dominoe', type=int, default=9, help='the highest dominoe in the board')
    parser.add_argument('-s','--shuffle-agents', type=bool, default=True, help='whether to shuffle the order of the agents each hand')
    parser.add_argument('-tg','--num-games',type=int, default=400, help='the number of training games')

    # ELO is probabilistic, so ELO will be exagerrated with more data (e.g. if there are more rounds per game)
    parser.add_argument('-tr','--num-rounds',type=int, default=50, help='the number of training rounds') 
    parser.add_argument('-op','--opponent',type=str, default='dominoeAgent', help='which opponent to play the basic value agent against for training and testing')
    parser.add_argument('-va','--value-agent',type=str, default='basicValueAgent', help='which value agent to use')
    parser.add_argument('-fe','--fraction-estimate',type=float, default=0.25, help='final fraction of elo estimates to use')
    parser.add_argument('--justplot',default=False,action='store_true', help='if used, will only plot the saved results (results have to already have been run and saved)')
    parser.add_argument('--nosave',default=False,action='store_true')
    
    args = parser.parse_args()
    
    assert args.value_agent in valueAgents.keys(), f"requested value agent ({args.value_agent}) is not in the list of possibilities!"
    assert args.opponent in opponents.keys(), f"requested opponent ({args.opponent}) is not in the list of possible opponents!"
     
    return args


def estimateELO(numGames, numRounds):
    # create a league manager with the requested parameters
    league = lm.leagueManager(args.highest_dominoe, args.num_players, shuffleAgents=True, device=device)
    
    names = []
    # add valueAgent with replay
    league.addAgentType(valueAgents[args.value_agent])
    league.getAgent(0).loadAgentParameters(getNetworkPath(replay=True))
    league.getAgent(0).setLearning(False)
    names.append(f"{args.value_agent}_withReplay")
    
    # add valueAgent without replay
    league.addAgentType(valueAgents[args.value_agent])
    league.getAgent(1).loadAgentParameters(getNetworkPath(replay=False))
    league.getAgent(1).setLearning(False)
    names.append(f"{args.value_agent}_noReplay")
    
    # add opponents necessary for full table
    league.addAgentType(opponents[args.opponent], args.num_players-2)
    names += [args.opponent]*(args.num_players-2)
    
    # Run lots of games, update and track ELO scores 
    trackElo = np.zeros((numGames, league.numAgents))
    trackScore = np.full((numGames, league.numAgents), np.nan)
    trackHandWins = np.full((numGames, league.numAgents), np.nan)
    numRounds = numRounds if numRounds is not None else league.highestDominoe+1
    for gameIdx in tqdm(range(numGames)):
        game, leagueIndex = league.createGame()
        game.playGame(rounds=numRounds)
        league.updateElo(leagueIndex, game.currentScore) # update ELO
        trackElo[gameIdx] = copy(league.elo)
        trackScore[gameIdx, leagueIndex] = game.currentScore/numRounds # track agent score (average per hand)
        trackHandWins[gameIdx, leagueIndex] = np.sum(game.score==0,axis=0)/numRounds # track how many times each agent won a hand

    # Estimate final ELO
    num2EstimateWith = int(numGames * args.fraction_estimate)
    eloEstimate = np.mean(trackElo[-num2EstimateWith:],axis=0)
    averageScore = np.nanmean(trackScore,axis=0)
    averageHandWins = np.nanmean(trackHandWins,axis=0)

    # Create results array
    results = {
        'elo':eloEstimate,
        'averageScore':averageScore,
        'averageHandWins':averageHandWins,
        'trackedElo':trackElo,
        'names':names
    }
 
    return results


# And a function for plotting results
def plotResults(results, args):
    elo = results['elo']
    averageScore = results['averageScore']
    averageHandWins = results['averageHandWins']
    trackedElo = results['trackedElo']
    
    names = results['names']
    numAgents = len(names)
    
    # Show plot of tracked ELO trajectories to make sure it reached asymptotic ELO ratings
    f1 = plt.figure(1)
    for name, telo in zip(names, trackedElo.T):
        plt.plot(range(args.num_games), telo, label=name)
    plt.ylim(0)
    plt.legend(loc='best')
    plt.show()

    # Create discrete colormap
    colors = ['b','k',*(['0.3']*(numAgents-2))]
    f2,ax = plt.subplots(1,3,figsize=(14,4), layout='constrained')

    tick_labels = ['replay','noReplay',*names[2:]]
    ax[0].bar(x=np.arange(numAgents), height=elo, color=colors, tick_label=tick_labels)
    ax[0].tick_params(labelrotation=25)
    ax[0].set_ylim(0)
    ax[0].set_ylabel('ELO')
    
    ax[1].bar(x=np.arange(numAgents), height=averageScore, color=colors, tick_label=tick_labels)
    ax[1].tick_params(labelrotation=25)
    ax[1].set_ylim(0)
    ax[1].set_ylabel('avg score/hand')
    
    ax[2].bar(x=np.arange(numAgents), height=averageHandWins, color=colors, tick_label=tick_labels)
    ax[2].tick_params(labelrotation=25)
    ax[2].set_ylim(0)
    ax[2].set_ylabel('avg fraction of hand wins')
    
    if not(args.nosave):
        plt.savefig(str(figsPath/getFileName()))
    
    plt.show()
    

if __name__=='__main__':
    args = handleArguments()
    
    if not(args.justplot):
        # estimate ELO with the requested parameters and agents
        results = estimateELO(args.num_games, args.num_rounds)
        
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
        
    print(f"ELO: {results['elo']}")
    print(f"AvgScore: {results['averageScore']}")
    print(f"AvgHandWins: {results['averageHandWins']}")
    
    
    
    

