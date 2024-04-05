import random
import numpy as np


class dominoeAgent:
    """
    Top-level dominoe agent.
    Contains all the standard initialization functions and gameplay methods that every agent requires.
    Specific instances of dominoeAgent will be created for training and comparison of different strategies.
    """

    # give this class a name so I can identify the class constructors
    className = "dominoeAgent"
    agentName = "default"

    # initialization function
    def __init__(self, numPlayers, highestDominoe, dominoes, numDominoes, device=None, **kwargs):
        # meta-variables (includes redundant information, but if I train 1000s of networks I want it to be as fast as possible)
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
        self.dominoes = dominoes
        self.numDominoes = numDominoes
        self.dominoeValue = np.sum(self.dominoes, axis=1).astype(float)
        self.dominoeDouble = self.dominoes[:, 0] == self.dominoes[:, 1]
        self.device = device if device is not None else "cpu"

        # game-play related variables
        self.handNumber = highestDominoe  # which hand are we playing? (always starts with highestDominoe number)
        self.myHand = []  # list of dominoes in hand
        self.handValues = []  # np arrays of dominoe values in hand (Nx2)
        self.played = []  # list of dominoes that have already been played
        self.available = np.zeros(self.numPlayers, dtype=int)  # dominoes that are available to be played on
        self.handsize = np.zeros(self.numPlayers, dtype=int)
        self.cantplay = np.full(self.numPlayers, False)  # whether or not there is a penny up (if they didn't play on their line)
        self.didntplay = np.full(self.numPlayers, False)  # whether or not the player played
        self.turncounter = np.zeros(self.numPlayers, dtype=int)  # how many turns before each opponent plays
        self.dummyAvailable = []  # index of dominoe available on dummy
        self.dummyPlayable = False  # boolean indicating whether the dummyline has been started

        # specialized initialization functions
        self.specializedInit(**kwargs)

    def specializedInit(self, **kwargs):
        # can be edited for each agent
        return None

    # ------------------
    # -- top level functions for managing metaparameters and saving of the agent --
    # ------------------
    def agentParameters(self):
        prmNames = ["numPlayers", "highestDominoe", "dominoes", "numDominoes"]
        prm = {}
        for key in prmNames:
            prm[key] = getattr(self, key)
        specialPrms = self.specialParameters()
        for key, val in specialPrms.items():
            prm[key] = val
        return prm

    def specialParameters(self):
        # can be edited for each agent
        return {}

    def dominoesInHand(self, updateObject=True):
        # simple function to return real values of dominoes from index of dominoes in hand
        handValues = self.dominoes[self.myHand]
        if updateObject:
            self.handValues = handValues
            return
        return handValues

    def updateAgentIndex(self, newIndex):
        self.agentIndex = newIndex
        self.egoShiftIdx = np.mod(np.arange(self.numPlayers) + newIndex, self.numPlayers)

    def printHand(self):
        print(self.myHand)
        print(self.handValues)

    # ------------------
    # -- functions to process gamestate --
    # ------------------
    def serve(self, assignment):
        # serve receives an assignment (indices) of dominoes that make up my hand
        self.myHand = assignment
        self.dominoesInHand()

    def initHand(self):
        # edited on an agent by agent basis. Not needed for default agents
        return None

    def egocentric(self, variable):
        return variable[self.egoShiftIdx]

    def linePlayedOn(self):
        # edited on an agent by agent basis, usually not needed unless agents use the utils.constructLineRecursive function
        return None

    def checkTurnUpdate(self, currentPlayer, postState=False):
        relevantTurn = (currentPlayer is not None) and (
            currentPlayer == self.agentIndex
        )  # only update gamestate when it's this agents turn by default
        relevantState = not (postState)  # only update gamestate for pre-state gamestates by default
        return relevantTurn and relevantState

    def gameState(
        self, played, available, handsize, cantplay, didntplay, turncounter, dummyAvailable, dummyPlayable, currentPlayer=None, postState=False
    ):
        # gamestate input, served to the agent each time it requires action (either at it's turn, or each turn for an RNN)
        # agents convert these variables to agent-centric information about the game-state
        if not (self.checkTurnUpdate(currentPlayer, postState=postState)):
            return None

        self.played = played  # list of dominoes that have already been played
        self.available = self.egocentric(available)  # list of value available on each players line (centered on agent)
        self.handsize = self.egocentric(handsize)  # list of handsize for each player (centered on agent)
        self.cantplay = self.egocentric(cantplay)  # list of whether each player can/can't play (centered on agent)
        self.didntplay = self.egocentric(didntplay)  # list of whether each player didn't play (centered on agent)
        self.turncounter = self.egocentric(
            turncounter
        )  # list of how many turns until each player is up (meaningless unless agent receives all-turn updates)
        self.dummyAvailable = dummyAvailable  # index of dominoe available on dummy line
        self.dummyPlayable = dummyPlayable  # bool determining whether the dummy line is playable
        self.processGameState()

    def processGameState(self, *args, **kwargs):
        # edited on an agent by agent basis, nothing needed for the default agent
        return None

    def estimatePrestateValue(self, *args, **kwargs):
        # edited on an agent by agent basis, nothing needed for the default agent
        return None

    def updatePoststateValue(self, *args, **kwargs):
        # edited on an agent by agent basis, nothing needed for the default agent
        return None

    # ------------------
    # -- functions to choose and play a dominoe --
    # ------------------
    def play(self, gameEngine=None):
        # this function is called by the gameplay object
        # it is what's used to play a dominoe when it's this agents turn
        dominoe, location = self.selectPlay(gameEngine=gameEngine)
        if dominoe is not None:
            assert dominoe in self.myHand, "dominoe selected to be played is not in hand"
            self.myHand = np.delete(self.myHand, self.myHand == dominoe)
            self.dominoesInHand()
        return dominoe, location

    def selectPlay(self, gameEngine=None):
        locations, dominoes = self.playOptions()
        if len(locations) == 0:
            return None, None
        optionValue = self.optionValue(locations, dominoes)
        idxChoice = self.makeChoice(optionValue)  # make and return choice
        return dominoes[idxChoice], locations[idxChoice]

    def playOptions(self):
        # generates list of playable options given game state
        # it produces a (numPlayers x numDominoes) array with True's indicating viable dominoe-location pairings
        # (and also a (numDominoes,) array for the dummy line
        lineOptions = np.full((self.numPlayers, self.numDominoes), False)
        for idx, value in enumerate(self.available):
            if idx == 0 or self.cantplay[idx]:
                idxPlayable = np.where(np.any(self.handValues == value, axis=1))[0]
                lineOptions[idx, self.myHand[idxPlayable]] = True
        dummyOptions = np.full(self.numDominoes, False)
        idxPlayable = np.where(np.any(self.handValues == self.dummyAvailable, axis=1))[0]
        dummyOptions[self.myHand[idxPlayable]] = True * self.dummyPlayable

        idxPlayer, idxDominoe = np.where(lineOptions)  # find where options are available
        idxDummyDominoe = np.where(dummyOptions)[0]
        idxDummy = -1 * np.ones(len(idxDummyDominoe), dtype=int)

        # concatenate locations, dominoes
        locations = np.concatenate((idxPlayer, idxDummy))
        dominoes = np.concatenate((idxDominoe, idxDummyDominoe))
        return locations, dominoes

    def optionValue(self, locations, dominoes):
        # convert option to play value using simplest method possible - value is 1 if option available
        return np.ones_like(dominoes)

    def makeChoice(self, optionValue):
        # default behavior is to use thompson sampling (picking a dominoe to play randomly, weighted by value of dominoe)
        return random.choices(range(len(optionValue)), k=1, weights=optionValue)[0]
