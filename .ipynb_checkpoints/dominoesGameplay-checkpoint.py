import numpy as np
import itertools
import dominoesFunctions as df
import dominoesAgents as da

# Gameplay object
class dominoeGame:
    def __init__(self, numPlayers, highestDominoe, defaultAgent=da.dominoeAgent, agents=None):
        # store inputs
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
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
        self.score = np.full((self.highestDominoe+1, self.numPlayers), np.nan)
        
        # create agents
        if agents is None: agents = [defaultAgent]*self.numPlayers
        if agents is not None: assert len(agents)==self.numPlayers, "number of agents provided is not equal to number of players"
        if agents is not None: assert np.all([agent.name=='dominoeAgent' for agent in agents])
        self.agents = [agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIndex) for (agentIndex,agent) in enumerate(agents)]
        
        # these are unnecessary because the math is correct, but might as well keep it as a low-cost sanity check
        assert len(self.dominoes)==self.numDominoes, "the number of dominoes isn't what is expected!"
        assert np.sum(self.dominoePerTurn)==self.numDominoes, "the distribution of dominoes per turn doesn't add up correctly!"
    
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
        for agent in self.agents:
            agent.gameState(self.played, self.available, self.handsize, self.cantplay, self.turncounter, self.dummyAvailable, self.dummyPlayable)
            
    def initializeHand(self):
        # identify which dominoe is the first double
        idxFirstDouble = np.where(np.all(self.dominoes==self.handNumber,axis=1))[0]
        assert len(idxFirstDouble)==1, "more or less than 1 double identified as first..."
        idxFirstDouble = idxFirstDouble[0]
        assignments = self.distribute() # distribute dominoes randomly
        idxFirstPlayer = np.where([idxFirstDouble in assignment for assignment in assignments])[0][0] # find out which player has the first double
        assignments[idxFirstPlayer] = np.delete(assignments[idxFirstPlayer], assignments[idxFirstPlayer]==idxFirstDouble) # remove it from their hand
        turncounter = np.roll(np.arange(self.numPlayers), idxFirstPlayer) # assign turns
        self.assignDominoes(assignments) # serve dominoes to agents
        self.nextPlayer = idxFirstPlayer # keep track of whos turn it is
        # prepare initial gameState arrays
        self.played = [idxFirstDouble] # at the beginning, only the double/double of the current hand has been played
        self.available = self.handNumber * np.ones(self.numPlayers) # at the beginning, everyone can only play on the double/double of the handNumber
        self.handsize = np.array([len(assignment) for assignment in assignments]) # how many dominoes in each hand
        self.cantplay = np.full(self.numPlayers, False) # whether or not each player has a penny up
        self.turncounter = np.roll(np.arange(self.numPlayers), -idxFirstPlayer) # which turn it is for each player
        self.dummyAvailable = self.handNumber # dummy also starts with #handNumber
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
        
    def doTurn(self, updates=False):
        # 1. feed gameState to next agent
        self.agents[self.nextPlayer].gameState(self.played, self.available, self.handsize, self.cantplay, self.turncounter, self.dummyAvailable, self.dummyPlayable)
        # 2. request "play"
        dominoe, location = self.agents[self.nextPlayer].play()
        if dominoe is None:
            # if no play is available, penny up and move to next player
            self.cantplay[self.nextPlayer]=True
            self.turncounter = np.roll(self.turncounter,1)
            self.nextPlayer = np.mod(self.nextPlayer+1, self.numPlayers)
        else:            
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
                self.nextPlayer = np.mod(self.nextPlayer+1, self.numPlayers) # move to next player
        
        # update general game state
        self.playNumber += 1    
        
        # check if everyone's line has started
        if not self.dummyPlayable: 
            if all([len(line) for line in self.lineSequence]): 
                self.dummyPlayable = True
        
        # if everyone cant play, start iterating terminateGameCounter
        if np.all(self.cantplay):
            self.terminateGameCounter+=1
        else:
            self.terminateGameCounter==0
        
        # if everyone hasn't played while they all have pennies up, end game
        if self.terminateGameCounter > self.numPlayers:
            self.handActive = False
            
        return None
        
        
    def playHand(self): 
        # request plays from each agent until someone goes out or no more plays available
        while self.handActive:
            self.doTurn()
        handScore = np.array([df.handValue(self.dominoes, agent.myHand) for agent in self.agents])
        self.score[self.highestDominoe - self.handNumber] = handScore
        return None
    
    def playGame(self):
        # -- initialize hand --
        # -- play hand --
        # -- move to next handNumber, repeat --
        # -- once hand number 0 finished, print scores --
        return None
        
