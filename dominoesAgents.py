import numpy as np
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
        self.available = self.egocentric(available) # list of dominoe available on each players line (centered on agent)
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
        lineoptions = np.full((self.numDominoes, self.numPlayers), False)
        dummyoptions = np.full(self.numDominoes, False)
        # for dominoe in self.available:
        #     dvals = 
        return None
           
    def play(self, dominoe):
        assert dominoe in self.myHand, "dominoe selected to be played is not in hand"
        self.myHand.remove(dominoe)
        self.dominoesInHand()
        return dominoe
        
        