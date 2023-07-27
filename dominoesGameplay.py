import time
from functools import partial
import numpy as np
from tqdm import tqdm
from copy import copy

import dominoesFunctions as df
import dominoesAgents as da

# Gameplay object
class dominoeGame:
    def __init__(self, highestDominoe, infiniteGame=True, numPlayers=None, agents=None, defaultAgent=da.dominoeAgent, device=None):
        # store inputs
        assert (numPlayers is not None) or (agents is not None), "either numPlayers or agents need to be specified"
        if (numPlayers is not None) and (agents is not None): 
            assert numPlayers == len(agents), "the number of players specified does not equal the number of agents provided..."
        if numPlayers is None: numPlayers = len(agents)
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
        self.infiniteGame = infiniteGame
        # create list of dominoes and number of dominoes for convenience
        self.dominoes = df.listDominoes(self.highestDominoe)
        self.numDominoes = df.numberDominoes(self.highestDominoe)
        self.numDominoeDistribution()
        self.extraDominoeOffset = np.random.randint(self.numPlayers+1) # which player(s) start with extra dominoes (if there are any)
        self.handNumber = self.highestDominoe # which hand are we at? (initialize at highest dominoe always...)
        self.playNumber = 0
        self.handActive = True # boolean determining whether a hand is completed (once it clicks to false, some new actions happen and a new hand begins)
        self.gameActive = True # boolean determining whether the game is still in progress (i.e. if the 0/0 hand hasn't happened yet)
        self.terminateGameCounter = 0 # once everyone cant play, start counting this up to terminate game and not be stuck in a loop
        self.score = np.zeros((0,self.numPlayers), dtype=int)
        self.nextPlayerShift = np.mod(np.arange(self.numPlayers)-1,self.numPlayers) # create index to shift to next players turn (faster than np.roll()...)
        
        # create agents (let's be smarter about this and provide dictionaries with parameters etc etc, but for now it's okay)
        if agents is None: agents = [defaultAgent]*self.numPlayers
        if agents is not None: assert len(agents)==self.numPlayers, "number of agents provided is not equal to number of players"
        if agents is not None: agents = [agent if hasattr(agent,'className') and agent.className=='dominoeAgent' else defaultAgent for agent in agents]
        # if agents is not None: assert np.all([agent.name=='dominoeAgent' for agent in agents])
        self.agents = [None]*self.numPlayers
        for agentIdx,agent in enumerate(agents):
            if isinstance(agent, da.dominoeAgent):
                assert (agent.numPlayers==numPlayers) and (agent.highestDominoe==highestDominoe), f"provided agent (agentIdx:{agentIdx}) did not have the correct number of players or dominoes"
                self.agents[agentIdx] = agent
                self.agents[agentIdx].agentIndex = agentIdx
                self.agents[agentIdx].device = device
            else:
                self.agents[agentIdx] = agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIdx, device=device)
        
        # # create agents (let's be smarter about this and provide dictionaries with parameters etc etc, but for now it's okay)
        # if agents is None: agents = [defaultAgent]*self.numPlayers
        # if agents is not None: assert len(agents)==self.numPlayers, "number of agents provided is not equal to number of players"
        # if agents is not None: agents = [agent if hasattr(agent,'className') and agent.className=='dominoeAgent' else defaultAgent for agent in agents]
        # # if agents is not None: assert np.all([agent.name=='dominoeAgent' for agent in agents])
        # self.agents = [agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIndex, device=device) for (agentIndex,agent) in enumerate(agents)]
        
        # these are unnecessary because the math is correct, but might as well keep it as a low-cost sanity check
        assert len(self.dominoes)==self.numDominoes, "the number of dominoes isn't what is expected!"
        assert np.sum(self.dominoePerTurn)==self.numDominoes, "the distribution of dominoes per turn doesn't add up correctly!"
        
        # performance monitoring
        self.initHandTime = [0, 0]
        self.presentGameStateTime = [0, 0]
        self.performPrestateValueTime = [0, 0]
        self.agentPlayTime = [0, 0]
        self.processPlayTime = [0, 0]
        
    def numDominoeDistribution(self):
        # return list of the number of dominoes each player gets per turn, either randomly at initial or shifted each time
        numberEach = int(np.floor(self.numDominoes/self.numPlayers))
        self.playersWithExtra = int(self.numDominoes - numberEach*self.numPlayers)
        self.dominoePerTurn = numberEach * np.ones(self.numPlayers,dtype=int)
        self.dominoePerTurn[:self.playersWithExtra] += 1
    
    def distribute(self):
        # randomly distribute dominoes at beginning of hand, then shift the extraDominoeOffset for fair distribution across an entire game 
        # produces a binary list of arrays of the dominoes in hand for each player
        startStopIndex = [0, *np.cumsum(np.roll(self.dominoePerTurn, self.extraDominoeOffset + (self.playersWithExtra*self.handNumber)))]
        idx = np.random.permutation(self.numDominoes) # randomized dominoe order to be distributed to each player
        assignments = [idx[startStopIndex[i]:startStopIndex[i+1]] for i in range(self.numPlayers)]
        assert np.array_equal(np.arange(self.numDominoes), np.sort(np.concatenate(assignments))), "dominoe assignments are not complete" # sanity check
        return assignments
    
    def assignDominoes(self,assignments):
        # serve dominoes to each agent
        for agent,assignment in zip(self.agents,assignments): 
            agent.serve(assignment)
            
    def presentGameState(self):
        for idx, agent in enumerate(self.agents):
            agent.gameState(self.played, self.available, self.handsize, self.cantplay, self.didntplay, self.turncounter, self.dummyAvailable, self.dummyPlayable, turnIdx=self.turncounter[idx])
            
    def performPrestateValueEstimate(self):
        for idx, agent in enumerate(self.agents):
            agent.estimatePrestateValue(turnIdx=self.turncounter[idx])
            
    def performPoststateValueUpdates(self):
        for idx, agent in enumerate(self.agents):
            agent.updatePoststateValue(finalScore=None, turnIdx=self.turncounter[idx])
            
    def performFinalScoreUpdates(self):
        finalScore = np.array([np.sum(agent.handValues) for agent in self.agents]) if not self.handActive else None
        for agent in self.agents:
            agent.updatePoststateValue(finalScore=finalScore, turnIdx=0) # force update by setting turnIdx to 0...
            
    def initializeHand(self):
        if not self.gameActive:
            print(f"Game has already finished")
            return
        # reset values
        self.playNumber = 0
        self.terminateGameCounter = 0
        self.handActive = True
        # identify which dominoe is the first double
        idxFirstDouble = np.where(np.all(self.dominoes==self.handNumber,axis=1))[0]
        assert len(idxFirstDouble)==1, "more or less than 1 double identified as first..."
        idxFirstDouble = idxFirstDouble[0]
        assignments = self.distribute() # distribute dominoes randomly
        idxFirstPlayer = np.where([idxFirstDouble in assignment for assignment in assignments])[0][0] # find out which player has the first double
        assignments[idxFirstPlayer] = np.delete(assignments[idxFirstPlayer], assignments[idxFirstPlayer]==idxFirstDouble) # remove it from their hand
        self.assignDominoes(assignments) # serve dominoes to agents
        self.nextPlayer = idxFirstPlayer # keep track of whos turn it is
        # prepare initial gameState arrays
        self.played = [idxFirstDouble] # at the beginning, only the double/double of the current hand has been played
        self.available = self.handNumber * np.ones(self.numPlayers, dtype=int) # at the beginning, everyone can only play on the double/double of the handNumber
        self.handsize = np.array([len(assignment) for assignment in assignments], dtype=int) # how many dominoes in each hand
        self.cantplay = np.full(self.numPlayers, False) # whether or not each player has a penny up
        self.didntplay = np.full(self.numPlayers, False) # whether or not each player played last time
        self.turncounter = np.mod(np.arange(self.numPlayers)-idxFirstPlayer, self.numPlayers).astype(int) # which turn it is for each player
        self.lineStarted = np.full(self.numPlayers, False) # flips to True once anyone has played on the line
        self.dummyAvailable = int(self.handNumber) # dummy also starts with #handNumber
        self.dummyPlayable = False # dummy is only playable when everyone has started their line
        # prepare gameplay tracking arrays
        self.lineSequence = [[]]*self.numPlayers # list of dominoes played by each player
        self.linePlayDirection = [[]]*self.numPlayers # boolean for whether dominoe was played forward or backward
        self.linePlayer = [[]]*self.numPlayers # which player played each dominoe
        self.linePlayNumber = [[]]*self.numPlayers # which play (in the game) it was
        self.dummySequence = [] # same as above for the dummy line
        self.dummyPlayDirection = []
        self.dummyPlayer = []
        self.dummyPlayNumber = []
        
    def doTurn(self, updates=False):
        # state change trackers to determine what game states to change at end of turn
        moveToNextPlayer = False
        
        # 1. feed gameState to next agent
        self.presentGameState() # present game state to every agent
        
        # 2. tell agent to perform prestate value estimation
        self.performPrestateValueEstimate()
        
        # 3. request "play"
        dominoe, location = self.agents[self.nextPlayer].play()
        
        if dominoe is None:
            # if no play is available, penny up and move to next player
            self.cantplay[self.nextPlayer]=True
            self.didntplay[self.nextPlayer]=True
            self.turncounter = np.roll(self.turncounter,1)
            moveToNextPlayer = True
        else:          
            self.didntplay[self.nextPlayer]=False
            self.handsize[self.nextPlayer] -= 1 # remove 1 from nextPlayers handsize
            # if they are out, end game
            if self.handsize[self.nextPlayer]==0: 
                self.handActive=False
            self.played.append(dominoe) # update played dominoe list
            isDouble = self.dominoes[dominoe][0]==self.dominoes[dominoe][1] # is double played? 
            playOnDummy = (location == -1)
            if playOnDummy:
                playDirection, nextAvailable = df.playDirection(self.dummyAvailable, self.dominoes[dominoe]) # returns which direction and next available value
                self.dummySequence.append(dominoe)
                self.dummyPlayDirection.append(playDirection)
                self.dummyPlayer.append(self.nextPlayer)
                self.dummyPlayNumber.append(self.playNumber)
                self.dummyAvailable = nextAvailable
            else:
                lineIdx = np.mod(self.nextPlayer + location, self.numPlayers) # shift to allocentric location 
                playDirection, nextAvailable = df.playDirection(self.available[lineIdx], self.dominoes[dominoe])
                self.lineSequence[lineIdx].append(dominoe)
                self.linePlayDirection[lineIdx].append(playDirection)
                self.linePlayer[lineIdx].append(self.nextPlayer)
                self.linePlayNumber[lineIdx].append(self.playNumber)
                self.available[lineIdx] = nextAvailable
                # if player played a non-double on their own line, take their penny off 
                if not isDouble and lineIdx==self.nextPlayer: 
                    self.cantplay[self.nextPlayer]=False
            if not isDouble:
                self.turncounter = np.roll(self.turncounter,1) # move turn counter to next player
                moveToNextPlayer=True
        
        # 4. update general game state 
        self.playNumber += 1    
        if moveToNextPlayer:
            self.nextPlayer = np.mod(self.nextPlayer+1, self.numPlayers)
        
        # check if everyone's line has started
        if not self.dummyPlayable: 
            if all([len(line) for line in self.lineSequence]): 
                self.dummyPlayable = True
        
        # if everyone cant play, start iterating terminateGameCounter
        if np.all(self.didntplay):
            self.terminateGameCounter += 1
        else:
            self.terminateGameCounter = 0
        
        # if everyone hasn't played while they all have pennies up, end game
        if self.terminateGameCounter > self.numPlayers:
            self.handActive = False
        
        # 5. tell agents to perform poststateValueUpdate
        if self.handActive:
            # if hand isn't over, present new game state and do poststate value updates
            self.presentGameState() # present game state to every agent
            self.performPoststateValueUpdates()
            
    def playHand(self): 
        if not self.gameActive:
            print(f"Game has already finished")
            return
        self.initializeHand()
        # request plays from each agent until someone goes out or no more plays available
        while self.handActive:
            self.doTurn()        
        self.performFinalScoreUpdates() # when game is over, do a parameter update with the final score for each network
        self.handNumber = np.mod(self.handNumber - 1, self.highestDominoe+1)
        return np.array([df.handValue(self.dominoes, agent.myHand) for agent in self.agents], dtype=int) # return score of hand
    
    def playGame(self, rounds=None, withUpdates=False):
        rounds = self.highestDominoe+1 if rounds is None else rounds
        self.score = np.zeros((rounds,self.numPlayers),dtype=int)
        if withUpdates:
            roundCounter = tqdm(range(rounds))
        else:
            roundCounter = range(rounds)
        for idxRound in roundCounter:
            handScore = self.playHand()
            self.score[idxRound] = handScore
        self.currentScore = np.sum(self.score,axis=0)
        self.currentWinner = np.argmin(self.currentScore)
    

    def printResults(self):
        if hasattr(self, 'currentScore'):
            print(self.score)
            print(self.currentScore)
            print(f"The current winner is agent: {self.currentWinner} with a score of {self.currentScore[self.currentWinner]}!")
        else:
            print("Game has not begun!")
            
            
            
            
            
            
            
            
            
class dominoeGameValueAgents:
    def __init__(self, highestDominoe, infiniteGame=True, numPlayers=None, agents=None, defaultAgent=da.dominoeAgent, device=None):
        # store inputs
        assert (numPlayers is not None) or (agents is not None), "either numPlayers or agents need to be specified"
        if (numPlayers is not None) and (agents is not None): 
            assert numPlayers == len(agents), "the number of players specified does not equal the number of agents provided..."
        if numPlayers is None: numPlayers = len(agents)
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
        self.infiniteGame = infiniteGame
        # create list of dominoes and number of dominoes for convenience
        self.dominoes = df.listDominoes(self.highestDominoe)
        self.numDominoes = df.numberDominoes(self.highestDominoe)
        self.numDominoeDistribution()
        self.extraDominoeOffset = np.random.randint(self.numPlayers+1) # which player(s) start with extra dominoes (if there are any)
        self.handNumber = self.highestDominoe # which hand are we at? (initialize at highest dominoe always...)
        self.playNumber = 0
        self.handActive = True # boolean determining whether a hand is completed (once it clicks to false, some new actions happen and a new hand begins)
        self.gameActive = True # boolean determining whether the game is still in progress (i.e. if the 0/0 hand hasn't happened yet)
        self.terminateGameCounter = 0 # once everyone cant play, start counting this up to terminate game and not be stuck in a loop
        self.score = np.zeros((0,self.numPlayers), dtype=int)
        self.nextPlayerShift = np.mod(np.arange(self.numPlayers)-1,self.numPlayers) # create index to shift to next players turn (faster than np.roll()...)
        
        # create agents (let's be smarter about this and provide dictionaries with parameters etc etc, but for now it's okay)
        if agents is None: agents = [defaultAgent]*self.numPlayers
        if agents is not None: assert len(agents)==self.numPlayers, "number of agents provided is not equal to number of players"
        if agents is not None: agents = [agent if hasattr(agent,'className') and agent.className=='dominoeAgent' else defaultAgent for agent in agents]
        # if agents is not None: assert np.all([agent.name=='dominoeAgent' for agent in agents])
        self.agents = [None]*self.numPlayers
        for agentIdx,agent in enumerate(agents):
            if isinstance(agent, da.dominoeAgent):
                assert (agent.numPlayers==numPlayers) and (agent.highestDominoe==highestDominoe), f"provided agent (agentIdx:{agentIdx}) did not have the correct number of players or dominoes"
                self.agents[agentIdx] = agent
                self.agents[agentIdx].agentIndex = agentIdx
                self.agents[agentIdx].device = device
            else:
                self.agents[agentIdx] = agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIdx, device=device)
        # self.agents = [agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIndex, device=device) for (agentIndex,agent) in enumerate(agents)]
        
        # these are unnecessary because the math is correct, but might as well keep it as a low-cost sanity check
        assert len(self.dominoes)==self.numDominoes, "the number of dominoes isn't what is expected!"
        assert np.sum(self.dominoePerTurn)==self.numDominoes, "the distribution of dominoes per turn doesn't add up correctly!"
        
        # performance monitoring
        self.initHandTime = [0, 0]
        self.presentGameStateTime = [0, 0]
        self.performPrestateValueTime = [0, 0]
        self.performPoststateValueTime = [0, 0]
        self.agentPlayTime = [0, 0]
        self.updateGameStateTime = [0, 0]
        self.documentGameplayTime = [0, 0]
        
    def numDominoeDistribution(self):
        # return list of the number of dominoes each player gets per turn, either randomly at initial or shifted each time
        numberEach = int(np.floor(self.numDominoes/self.numPlayers))
        self.playersWithExtra = int(self.numDominoes - numberEach*self.numPlayers)
        self.dominoePerTurn = numberEach * np.ones(self.numPlayers,dtype=int)
        self.dominoePerTurn[:self.playersWithExtra] += 1
    
    def distribute(self):
        # randomly distribute dominoes at beginning of hand, then shift the extraDominoeOffset for fair distribution across an entire game 
        # produces a binary list of arrays of the dominoes in hand for each player
        startStopIndex = [0, *np.cumsum(np.roll(self.dominoePerTurn, self.extraDominoeOffset + (self.playersWithExtra*self.handNumber)))]
        idx = np.random.permutation(self.numDominoes) # randomized dominoe order to be distributed to each player
        assignments = [idx[startStopIndex[i]:startStopIndex[i+1]] for i in range(self.numPlayers)]
        assert np.array_equal(np.arange(self.numDominoes), np.sort(np.concatenate(assignments))), "dominoe assignments are not complete" # sanity check
        return assignments
    
    def assignDominoes(self,assignments):
        # serve dominoes to each agent
        for agent,assignment in zip(self.agents,assignments): 
            agent.serve(assignment)
            
    def presentGameState(self, currentPlayer):
        for agent in self.agents:
            agent.gameState(self.played, self.available, self.handSize, self.cantPlay, self.didntPlay, self.turnCounter, self.dummyAvailable, self.dummyPlayable, currentPlayer=currentPlayer)
            
    def performPrestateValueEstimate(self, currentPlayer):
        for agent in self.agents:
            agent.estimatePrestateValue(currentPlayer=currentPlayer)
            
    def performPoststateValueUpdates(self, currentPlayer):
        for agent in self.agents:
            agent.updatePoststateValue(finalScore=None, currentPlayer=currentPlayer)
            
    def performFinalScoreUpdates(self):
        finalScore = np.array([np.sum(agent.handValues) for agent in self.agents]) if not self.handActive else None
        for agent in self.agents:
            agent.updatePoststateValue(finalScore=finalScore, currentPlayer=agent.agentIndex) # force update by setting currentPlayer to agentIndex
            
    def initializeHand(self):
        if not self.gameActive:
            print(f"Game has already finished")
            return
        # reset values
        self.playNumber = 0
        self.terminateGameCounter = 0
        self.handActive = True
        # identify which dominoe is the first double
        idxFirstDouble = np.where(np.all(self.dominoes==self.handNumber,axis=1))[0]
        assert len(idxFirstDouble)==1, "more or less than 1 double identified as first..."
        idxFirstDouble = idxFirstDouble[0]
        assignments = self.distribute() # distribute dominoes randomly
        idxFirstPlayer = np.where([idxFirstDouble in assignment for assignment in assignments])[0][0] # find out which player has the first double
        assignments[idxFirstPlayer] = np.delete(assignments[idxFirstPlayer], assignments[idxFirstPlayer]==idxFirstDouble) # remove it from their hand
        self.assignDominoes(assignments) # serve dominoes to agents
        self.nextPlayer = idxFirstPlayer # keep track of whos turn it is
        # prepare initial gameState arrays
        self.played = [idxFirstDouble] # at the beginning, only the double/double of the current hand has been played
        self.available = self.handNumber * np.ones(self.numPlayers, dtype=int) # at the beginning, everyone can only play on the double/double of the handNumber
        self.handSize = np.array([len(assignment) for assignment in assignments], dtype=int) # how many dominoes in each hand
        self.cantPlay = np.full(self.numPlayers, False) # whether or not each player has a penny up
        self.didntPlay = np.full(self.numPlayers, False) # whether or not each player played last time
        self.turnCounter = np.mod(np.arange(self.numPlayers)-idxFirstPlayer, self.numPlayers).astype(int) # which turn it is for each player
        self.lineStarted = np.full(self.numPlayers, False) # flips to True once anyone has played on the line
        self.dummyAvailable = int(self.handNumber) # dummy also starts with #handNumber
        self.dummyPlayable = False # dummy is only playable when everyone has started their line
        # prepare gameplay tracking arrays
        self.lineSequence = [[] for _ in range(self.numPlayers)] # list of dominoes played by each player
        self.linePlayDirection = [[] for _ in range(self.numPlayers)] # boolean for whether dominoe was played forward or backward
        self.linePlayer = [[] for _ in range(self.numPlayers)] # which player played each dominoe
        self.linePlayNumber = [[] for _ in range(self.numPlayers)] # which play (in the game) it was
        self.dummySequence = [] # same as above for the dummy line
        self.dummyPlayDirection = []
        self.dummyPlayer = []
        self.dummyPlayNumber = []
        
        
    def updateGameState(self, dominoe, location, playerIndex, gameState, copyData=True, playInfo=False):
        # function for updating game state given a dominoe index, a location, and the playerIndex playing the dominoe
        # this is the gameplay simulation engine, which can be used to update self.(--), or to sample a possible future state from the agents...
        
        # do this to prevent simulation from overwriting game variables
        if copyData: gameState = [copy(gs) for gs in gameState]
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = gameState # unfold gameState input 
        
        if dominoe is None:
            # if no play is available, penny up and move to next player
            cantPlay[playerIndex] = True
            didntPlay[playerIndex] = True
            turnCounter = turnCounter[self.nextPlayerShift]
            playDirection, nextAvailable = None, None # required outputs for 
            moveToNextPlayer = True
        else:
            didntPlay[playerIndex] = False
            handSize[playerIndex] -= 1
            played.append(dominoe)
            isDouble = self.dominoes[dominoe][0]==self.dominoes[dominoe][1] # is double played? 
            playOnDummy = (location == -1)
            if playOnDummy:
                playDirection, nextAvailable = df.playDirection(dummyAvailable, self.dominoes[dominoe]) # returns which direction and next available value
                dummyAvailable = nextAvailable
            else:
                lineIdx = np.mod(playerIndex + location, self.numPlayers)
                playDirection, nextAvailable = df.playDirection(available[lineIdx], self.dominoes[dominoe]) 
                if not isDouble and lineIdx==playerIndex:
                    cantPlay[playerIndex] = False
                lineStarted[lineIdx] = True
                available[lineIdx] = nextAvailable
            if not isDouble:
                turnCounter = turnCounter[self.nextPlayerShift]    
            if not dummyPlayable:
                dummyPlayable = np.all(lineStarted) # if everyone has played, make the dummy playable
            moveToNextPlayer = not(isDouble)
        
        if playInfo:
            return played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable, playDirection, nextAvailable, moveToNextPlayer
        else:
            return played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable
            
    
    def documentGameplay(self, dominoe, location, playerIndex, playDirection, nextAvailable, moveToNextPlayer):
        # after updating gamestate, document gameplay
        if dominoe is not None:
            if (location == -1):
                self.dummySequence.append(dominoe)
                self.dummyPlayDirection.append(playDirection)
                self.dummyPlayer.append(playerIndex)
                self.dummyPlayNumber.append(self.playNumber)
            else:
                lineIdx = np.mod(playerIndex + location, self.numPlayers)
                self.lineSequence[lineIdx].append(dominoe)
                self.linePlayDirection[lineIdx].append(playDirection)
                self.linePlayer[lineIdx].append(playerIndex)
                self.linePlayNumber[lineIdx].append(self.playNumber)
        
        # if everyone cant play, start iterating terminateGameCounter
        if np.all(self.didntPlay): 
            self.terminateGameCounter+=1
        else: 
            self.terminateGameCounter=0 #otherwise reset
        
        # if everyone hasn't played while they all have pennies up, end game
        if self.terminateGameCounter > self.numPlayers:
            self.handActive = False
            
        # if it wasn't a double, move to next player
        if moveToNextPlayer:
            self.nextPlayer = np.mod(self.nextPlayer+1, self.numPlayers)
        
        # iterate playCounter
        self.playNumber += 1

        # if anyone is out, end game
        if np.any(self.handSize==0): 
            self.handActive=False
        
    def doTurn(self):
        
        # 0. Store index of agent who's turn it is
        currentPlayer = copy(self.nextPlayer)
        
        # 1. Present game state and gameplay simulation engine to every agent
        self.presentGameState(currentPlayer)
        
        # 2. tell agent to perform prestate value estimation
        self.performPrestateValueEstimate(currentPlayer)
        
        # 3. request "play"
        gameState = self.played, self.available, self.handSize, self.cantPlay, self.didntPlay, self.turnCounter, self.lineStarted, self.dummyAvailable, self.dummyPlayable
        gameEngine = partial(self.updateGameState, playerIndex=self.nextPlayer, gameState=gameState)
        dominoe, location = self.agents[self.nextPlayer].play(gameEngine, self)
        
        # 4. given play, update game state
        gameState = self.updateGameState(dominoe, location, self.nextPlayer, gameState, copyData=False, playInfo=True)
        self.played, self.available, self.handSize, self.cantPlay, self.didntPlay, self.turnCounter, self.lineStarted, self.dummyAvailable, self.dummyPlayable = gameState[:-3]
        playDirection, nextAvailable, moveToNextPlayer = gameState[-3:]
        
        # 5. document play
        self.documentGameplay(dominoe, location, self.nextPlayer, playDirection, nextAvailable, moveToNextPlayer)
    
        # 6. implement poststateValueUpdates
        if self.handActive:
            # if hand is still active, do poststate value updates
            self.presentGameState(currentPlayer) # present game state to every agent
            self.performPoststateValueUpdates(currentPlayer)
        
        
    def playHand(self):
        if not self.gameActive: 
            print(f"Game has already finished.")
            return 
        self.initializeHand()
        while self.handActive:
            self.doTurn()
        self.performFinalScoreUpdates() # once hand is over, do final score parameter updates for each agent
            
        self.handNumber = np.mod(self.handNumber - 1, self.highestDominoe+1)
        return np.array([df.handValue(self.dominoes, agent.myHand) for agent in self.agents], dtype=int) # return score of hand
    
            
    def playGame(self, rounds=None, withUpdates=False):
        rounds = self.highestDominoe+1 if rounds is None else rounds
        self.score = np.zeros((rounds,self.numPlayers),dtype=int)
        if withUpdates:
            roundCounter = tqdm(range(rounds))
        else:
            roundCounter = range(rounds)
        for idxRound in roundCounter:
            handScore = self.playHand()
            self.score[idxRound] = handScore
        self.currentScore = np.sum(self.score,axis=0)
        self.currentWinner = np.argmin(self.currentScore)

        
    def printResults(self):
        if hasattr(self, 'currentScore'):
            print(self.score)
            print(self.currentScore)
            print(f"The current winner is agent: {self.currentWinner} with a score of {self.currentScore[self.currentWinner]}!")
        else:
            print("Game has not begun!")
            
