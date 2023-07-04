import numpy as np
import itertools
import dominoesFunctions as df
import dominoesAgents as da
import time
from tqdm import tqdm

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
        if agents is not None: agents = [agent if hasattr(agent,'name') and agent.name=='dominoeAgent' else defaultAgent for agent in agents]
        # if agents is not None: assert np.all([agent.name=='dominoeAgent' for agent in agents])
        self.agents = [agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIndex, device=device) for (agentIndex,agent) in enumerate(agents)]
        
        # these are unnecessary because the math is correct, but might as well keep it as a low-cost sanity check
        assert len(self.dominoes)==self.numDominoes, "the number of dominoes isn't what is expected!"
        assert np.sum(self.dominoePerTurn)==self.numDominoes, "the distribution of dominoes per turn doesn't add up correctly!"
        
        # performance monitoring
        self.prepareScoreTime = [0, 0]
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
        for agent in self.agents:
            agent.gameState(self.played, self.available, self.handsize, self.cantplay, self.didntplay, self.turncounter, self.dummyAvailable, self.dummyPlayable)
            
    def performPrestateValueEstimate(self):
        trueHandValue = np.array([np.sum(agent.handValues) for agent in self.agents])
        for agent in self.agents:
            agent.estimatePrestateValue(trueHandValue)
            
    def performPoststateValueUpdates(self):
        finalScore = np.array([np.sum(agent.handValues) for agent in self.agents]) if not self.handActive else None
        for agent in self.agents:
            agent.updatePoststateValue(finalScore=finalScore)
            
    def initializeHand(self):
        t = time.time()
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
        # performance monitoring
        self.initHandTime[0] += time.time()-t
        self.initHandTime[1] += 1
        
    def doTurn(self, updates=False):
        # notes on what information is ~currently~ encoded in gameState
        # - played: list of indices of dominoes that have already been played
        # - available: list of values on each line available for next play
        # - handsize: list of values of numbers of dominoes in each hand
        # - cantplay: boolean list of whether each line has a "penny up" meaning available for anyone to play on it
        # - didntplay: boolean list of whether each player played last turn
        # - turncounter: list of turns until each players turn
        # - dummyAvailable: value available on dummy
        # - dummyPlayable: whether dummy can be played on
        # - handActive: boolean indicating whether current hand is happening or if it's over
        
        # notes on updates:
        # -- convert gamestate values into things that are already prepared for the neural networks (except for reshaping)
        # -- add secondary function to accept the gameplay and track data (lineSequence etc) --
        
        
        # state change trackers to determine what game states to change at end of turn
        moveToNextPlayer = False
        
        # 1. feed gameState to next agent
        #print("Feed game state to every agent at beginning of doTurn()")
        #print("This will probably require smarter handling of gamestate data to always be prepared for the networks...")
        #print("Compute gradV(S,w)/w, compute V(S,w), compute eligibility (lambda * z(t-1) + gradV(S,w)/w")
        #print("At end of doTurn(), compute V(S_t+1,w), compute tdError (switch V(S_t+1,w) to R_t+1 if end, compute w_t+1")
        t = time.time()
        self.presentGameState() # present game state to every agent
        #self.agents[self.nextPlayer].gameState(self.played, self.available, self.handsize, self.cantplay, self.didntplay, self.turncounter, self.dummyAvailable, self.dummyPlayable)    
        self.presentGameStateTime[0] += time.time()-t
        self.presentGameStateTime[1] += 1
        
        # 2. tell agent to perform prestate value estimation
        t = time.time()
        self.performPrestateValueEstimate()
        self.performPrestateValueTime[0] += time.time()-t
        self.performPrestateValueTime[1] += 1
        
        # 3. request "play"
        t = time.time()
        dominoe, location = self.agents[self.nextPlayer].play()
        self.agentPlayTime[0] += time.time()-t
        self.agentPlayTime[1] += 1
        
        t = time.time()
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
        self.processPlayTime[0] += time.time()-t
        self.processPlayTime[1] += 1
        
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
            self.terminateGameCounter+=1
        else:
            self.terminateGameCounter==0
        
        # if everyone hasn't played while they all have pennies up, end game
        if self.terminateGameCounter > self.numPlayers:
            self.handActive = False
        
        # 5. tell agents to perform poststateValueUpdate
        self.presentGameState() # present game state to every agent
        t = time.time()
        self.performPoststateValueUpdates()
        self.performPrestateValueTime[0] += time.time()-t
        self.performPrestateValueTime[1] += 1
        
        
    def playHand(self): 
        if not self.gameActive:
            print(f"Game has already finished")
            return
        self.initializeHand()
        # request plays from each agent until someone goes out or no more plays available
        while self.handActive:
            self.doTurn()
        self.handNumber = np.mod(self.handNumber - 1, self.highestDominoe)
        return np.array([df.handValue(self.dominoes, agent.myHand) for agent in self.agents], dtype=int) # return score of hand
    
    def playGame(self, rounds=None, withUpdates=False):
        t = time.time()
        rounds = self.highestDominoe+1 if rounds is None else rounds
        self.score = np.zeros((rounds,self.numPlayers),dtype=int)
        self.prepareScoreTime[0] += time.time()-t
        self.prepareScoreTime[1] += 1
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
        if self.gameActive:
            if self.handNumber == self.highestDominoe:
                print(f"Game has not begun!")
                return
            print(self.currentScore)
            print(f"After playing {self.handNumber+1}'s, agent {self.currentWinner} is in the lead with a score of {self.currentScore[self.currentWinner]}!")
        else:
            print(self.score)
            print(self.currentScore)
            print(f"The winner is agent: {self.currentWinner} with a score of {self.currentScore[self.currentWinner]}!")

        
