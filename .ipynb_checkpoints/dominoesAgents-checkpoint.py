import numpy as np
import random
import itertools
import dominoesFunctions as df

# 1. one-hot of dominoes in hand
# 2. one-hot of dominoes already played
# 3. multi-length vector, one input for each other player, containing: 1) how many dominoes in their hand, 2) whether they have a penny up, 3) how many turns until they play
# 4. number of turns until the agent plays
# 5. one-hot of dominoes available to play on
# ---- instead of 5) above, a list for each player + the dummy of index of available dominoes? 
# 6. dominoe played on last turn? 

class dominoeAgent:
    """
    Top-level dominoe agent. 
    Contains all the standard initialization functions and gameplay methods that every agent requires. 
    Specific instances of dominoeAgent will be created for training and comparison of different strategies. 
    """
    
    # give this class a name so I can identify the class constructors
    name='dominoeAgent'
    
    # initialization function
    def __init__(self, numPlayers, highestDominoe, dominoes, numDominoes, agentIndex):
        # meta-variables (includes redundant information, but if I train 1000s of networks I want it to be as fast as possible)
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
        self.dominoes = dominoes
        self.numDominoes = numDominoes
        self.agentIndex = agentIndex
        self.dominoeValue = np.sum(self.dominoes, axis=1).astype(float)
        self.dominoeDouble = self.dominoes[:,0]==self.dominoes[:,1]
        
        # game-play related variables
        self.handNumber = highestDominoe # which hand are we playing? (always starts with highestDominoe number)
        self.myHand = [] # list of dominoes in hand
        self.handValues = [] # np arrays of dominoe values in hand (Nx2)
        self.played = [] # list of dominoes that have already been played
        self.available = np.zeros(self.numPlayers,dtype=int) # dominoes that are available to be played on
        self.handsize = np.zeros(self.numPlayers,dtype=int)
        self.cantplay = np.full(self.numPlayers,False) # 
        self.turncounter = np.zeros(self.numPlayers,dtype=int) # how many turns before each opponent plays
        self.dummyAvailable = [] # index of dominoe available on dummy
        self.dummyPlayable = False # boolean indicating whether the dummyline has been started
        
    # -- input from game manager --
    def serve(self, assignment):
        # serve receives an assignment (indices) of dominoes that make up my hand
        self.myHand = assignment
        self.dominoesInHand()
        
    def gameState(self, played, available, handsize, cantplay, turncounter, dummyAvailable, dummyPlayable):
        # gamestate input, served to the agent each time it requires action (either at it's turn, or each turn for an RNN)
        # each agent will transform these inputs into a "perspective" which converts them to agent-centric information about the game-state
        self.played = played # list of dominoes that have already been played
        self.available = self.egocentric(available) # list of value available on each players line (centered on agent)
        self.handsize = self.egocentric(handsize) # list of handsize for each player (centered on agent)
        self.cantplay = self.egocentric(cantplay) # list of whether each player can/can't play (centered on agent)
        self.turncounter = self.egocentric(turncounter) # list of how many turns until each player is up (meaningless unless agent receives all-turn updates)
        self.dummyAvailable = dummyAvailable # index of dominoe available on dummy line
        self.dummyPlayable = dummyPlayable # bool determining whether the dummy line is playable
    
    # -- functions to process gamestate --
    def egocentric(self, variable):
        return np.roll(variable, -self.agentIndex) # shift perspective from allocentric to egocentric
    
    def dominoesInHand(self):
        self.handValues = self.dominoes[self.myHand]
    
    def playOptions(self):
        # generate list of playable options (starts as boolean, becomes value)
        lineOptions = np.full((self.numPlayers, self.numDominoes), False)
        for idx,value in enumerate(self.available):
            if idx==0 or self.cantplay[idx]: 
                idxPlayable = np.where(np.any(self.handValues==value,axis=1))[0]
                lineOptions[idx,self.myHand[idxPlayable]]=True
        dummyOptions = np.full(self.numDominoes, False)
        idxPlayable = np.where(np.any(self.handValues==self.dummyAvailable,axis=1))[0]
        dummyOptions[self.myHand[idxPlayable]]=True*self.dummyPlayable
        return lineOptions,dummyOptions
           
    def play(self):
        dominoe, location = self.selectPlay()
        if dominoe is not None:
            assert dominoe in self.myHand, "dominoe selected to be played is not in hand"
            self.myHand = np.delete(self.myHand, self.myHand==dominoe)
            self.dominoesInHand()
        return dominoe, location
    
    def selectPlay(self):
        # select dominoe to play, for the default class, the selection is random based on available plays
        lineOptions, dummyOptions = self.playOptions() # get options that are available
        idxPlayer, idxDominoe = np.where(lineOptions) # find where options are available
        idxDummyDominoe = np.where(dummyOptions)[0]
        idxDummy = -1 * np.ones(len(idxDummyDominoe), dtype=int)
        # if no valid options available, return None
        if len(idxPlayer)==0 and len(idxDummy)==0: 
            return None,None
        # otherwise, process options to make choice
        lineOptionValue = self.optionValue(lineOptions) # measure value of each option
        dummyOptionValue = self.optionValue(dummyOptions)  
        valuePlayers = lineOptionValue[idxPlayer, idxDominoe] # retrieve value of valid options
        valueDummy = dummyOptionValue[idxDummyDominoe]
        # concatenate lineIdx, dominoeIdx, optionValue
        lineIdx = np.concatenate((idxPlayer, idxDummy))
        dominoeIdx = np.concatenate((idxDominoe, idxDummyDominoe))
        optionValue = np.concatenate((valuePlayers, valueDummy))
        # make and return choice
        idxChoice = self.makeChoice(optionValue)
        return dominoeIdx[idxChoice], lineIdx[idxChoice] 
        
    def optionValue(self, options):
        # convert option to play value using simplest method possible - value is 1 if option available
        return 1*options
    
    def makeChoice(self, optionValue):
        # default behavior is to use thompson sampling (picking a dominoe to play randomly, weighted by value of dominoe)
        return random.choices(range(len(optionValue)), k=1, weights=optionValue)[0]
        
    def printHand(self):
        print(self.myHand)
        print(self.handValues)
        

        
class greedyAgent(dominoeAgent):
    def optionValue(self, options):
        # convert option to play value using simplest method possible - the number of points on each dominoe
        return self.dominoeValue * options
    
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
    

class stupidAgent(dominoeAgent):
    def optionValue(self, options):
        # convert option to play value using simplest method possible - the number of points on each dominoe
        return self.dominoeValue * options
    
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
    
    
class doubleAgent(dominoeAgent):
    def optionValue(self, options):
        # double agent treats any double opportunity as infinitely valuable (and greedily plays it when it can!)
        optionValue = self.dominoeValue * options
        if np.any(self.dominoeDouble*options):
            optionValue[self.dominoeDouble*options]=np.inf
        return optionValue
    
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
        
        
        
        
        
        
        
        
        