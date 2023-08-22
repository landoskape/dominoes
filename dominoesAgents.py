import re
from glob import glob
from datetime import datetime
from copy import copy
import random
import itertools
import numpy as np
import torch
import dominoesFunctions as df
import dominoesNetworks as dnn


class dominoeAgent:
    """
    Top-level dominoe agent. 
    Contains all the standard initialization functions and gameplay methods that every agent requires. 
    Specific instances of dominoeAgent will be created for training and comparison of different strategies. 
    """
    
    # give this class a name so I can identify the class constructors
    className = 'dominoeAgent'
    agentName = 'default'
    
    # initialization function
    def __init__(self, numPlayers, highestDominoe, dominoes, numDominoes, agentIndex, device=None, **kwargs):
        # meta-variables (includes redundant information, but if I train 1000s of networks I want it to be as fast as possible)
        self.numPlayers = numPlayers
        self.numPlayerRange = [self.numPlayers]
        self.highestDominoe = highestDominoe
        self.highestDominoeRange = range(1, 1000)
        self.dominoes = dominoes
        self.numDominoes = numDominoes
        self.agentIndex = agentIndex
        self.dominoeValue = np.sum(self.dominoes, axis=1).astype(float)
        self.dominoeDouble = self.dominoes[:,0]==self.dominoes[:,1]
        self.egoShiftIdx = np.mod(np.arange(numPlayers)+agentIndex, numPlayers)
        self.device = device if device is not None else "cpu"
        
        # game-play related variables
        self.handNumber = highestDominoe # which hand are we playing? (always starts with highestDominoe number)
        self.myHand = [] # list of dominoes in hand
        self.handValues = [] # np arrays of dominoe values in hand (Nx2)
        self.played = [] # list of dominoes that have already been played
        self.available = np.zeros(self.numPlayers,dtype=int) # dominoes that are available to be played on
        self.handsize = np.zeros(self.numPlayers,dtype=int)
        self.cantplay = np.full(self.numPlayers,False) # whether or not there is a penny up (if they didn't play on their line)
        self.didntplay = np.full(self.numPlayers, False) # whether or not the player played
        self.turncounter = np.zeros(self.numPlayers,dtype=int) # how many turns before each opponent plays
        self.dummyAvailable = [] # index of dominoe available on dummy
        self.dummyPlayable = False # boolean indicating whether the dummyline has been started
        
        self.requireUpdates = False
        
        # specialized initialization functions 
        self.specializedInit(**kwargs)
    
    def specializedInit(self,**kwargs):
        # can be edited for each agent
        return None
    
    # ------------------
    # -- top level functions for managing metaparameters and saving of the agent --
    # ------------------
    def agentParameters(self):
        prmNames = ['numPlayers', 'highestDominoe', 'dominoes', 'numDominoes']
        prm = {}
        for key in prmNames: prm[key]=getattr(self, key)
        specialPrms = self.specialParameters()
        for key,val in specialPrms.items(): prm[key]=val
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
        self.egoShiftIdx = np.mod(np.arange(self.numPlayers)+newIndex, self.numPlayers)
        
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
        # edited on an agent by agent basis, usually not needed unless agents use the df.constructLineRecursive function
        return None
    
    def checkTurnUpdate(self, currentPlayer, postState=False):
        relevantTurn = (currentPlayer is not None) and (currentPlayer == self.agentIndex) # only update gamestate when it's this agents turn by default
        relevantState = not(postState) # only update gamestate for pre-state gamestates by default
        return relevantTurn and relevantState
    
    def gameState(self, played, available, handsize, cantplay, didntplay, turncounter, dummyAvailable, dummyPlayable, currentPlayer=None, postState=False):
        # gamestate input, served to the agent each time it requires action (either at it's turn, or each turn for an RNN)
        # agents convert these variables to agent-centric information about the game-state
        if not(self.checkTurnUpdate(currentPlayer, postState=postState)): return None

        self.played = played # list of dominoes that have already been played
        self.available = self.egocentric(available) # list of value available on each players line (centered on agent)
        self.handsize = self.egocentric(handsize) # list of handsize for each player (centered on agent)
        self.cantplay = self.egocentric(cantplay) # list of whether each player can/can't play (centered on agent)
        self.didntplay = self.egocentric(didntplay) # list of whether each player didn't play (centered on agent)
        self.turncounter = self.egocentric(turncounter) # list of how many turns until each player is up (meaningless unless agent receives all-turn updates)
        self.dummyAvailable = dummyAvailable # index of dominoe available on dummy line
        self.dummyPlayable = dummyPlayable # bool determining whether the dummy line is playable
        self.processGameState()
    
    def processGameState(self,*args,**kwargs):
        # edited on an agent by agent basis, nothing needed for the default agent 
        return None 
    
    def estimatePrestateValue(self,*args,**kwargs):
        # edited on an agent by agent basis, nothing needed for the default agent 
        return None
    
    def updatePoststateValue(self,*args,**kwargs):
        # edited on an agent by agent basis, nothing needed for the default agent 
        return None
    
    # ------------------
    # -- functions to choose and play a dominoe --
    # ------------------
    def makeChoice(self, optionValue):
        # default behavior is to use thompson sampling (picking a dominoe to play randomly, weighted by value of dominoe)
        return random.choices(range(len(optionValue)), k=1, weights=optionValue)[0]
    
    def optionValue(self, locations, dominoes):
        # convert option to play value using simplest method possible - value is 1 if option available
        return np.ones_like(dominoes)
    
    def playOptions(self):
        # generates list of playable options given game state 
        # it produces a (numPlayers x numDominoes) array with True's indicating viable dominoe-location pairings
        # (and also a (numDominoes,) array for the dummy line
        lineOptions = np.full((self.numPlayers, self.numDominoes), False)
        for idx,value in enumerate(self.available):
            if idx==0 or self.cantplay[idx]:
                idxPlayable = np.where(np.any(self.handValues==value,axis=1))[0]
                lineOptions[idx,self.myHand[idxPlayable]]=True
        dummyOptions = np.full(self.numDominoes, False)
        idxPlayable = np.where(np.any(self.handValues==self.dummyAvailable,axis=1))[0]
        dummyOptions[self.myHand[idxPlayable]]=True*self.dummyPlayable
        
        idxPlayer, idxDominoe = np.where(lineOptions) # find where options are available
        idxDummyDominoe = np.where(dummyOptions)[0]
        idxDummy = -1 * np.ones(len(idxDummyDominoe), dtype=int)
        
        # concatenate locations, dominoes
        locations = np.concatenate((idxPlayer, idxDummy))
        dominoes = np.concatenate((idxDominoe, idxDummyDominoe))
        return locations, dominoes
    
    def selectPlay(self, gameEngine=None):
        locations, dominoes = self.playOptions()
        if len(locations)==0: return None, None
        optionValue = self.optionValue(locations, dominoes)
        idxChoice = self.makeChoice(optionValue) # make and return choice
        return dominoes[idxChoice], locations[idxChoice]
    
    def play(self, gameEngine=None):
        # this function is called by the gameplay object
        # it is what's used to play a dominoe when it's this agents turn
        dominoe, location = self.selectPlay(gameEngine=gameEngine)
        if dominoe is not None:
            assert dominoe in self.myHand, "dominoe selected to be played is not in hand"
            self.myHand = np.delete(self.myHand, self.myHand==dominoe)
            self.dominoesInHand()
        return dominoe, location
    
# ----------------------------------------------------------------------------
# --------------------------- simple rule agents -----------------------------
# ----------------------------------------------------------------------------
class greedyAgent(dominoeAgent):
    # greedy agent plays whatever dominoe has the highest number of points
    agentName = 'greedyAgent'
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
    def optionValue(self, locations, dominoes):
        return self.dominoeValue[dominoes]
    
class stupidAgent(dominoeAgent):
    # stupid agent plays whatever dominoe has the least number of points
    agentName = 'stupidAgent' 
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
    def optionValue(self, locations, dominoes):
        return self.dominoeValue[dominoes]
    
class doubleAgent(dominoeAgent):
    # double agent plays any double it can play immediately, then plays the dominoe with the highest number of points
    agentName = 'doubleAgent'
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
    def optionValue(self, locations, dominoes):
        optionValue = self.dominoeValue[dominoes]
        optionValue[self.dominoeDouble[dominoes]]=np.inf
        return optionValue
    
    
# ----------------------------------------------------------------------------
# -------------- agents that care about possible sequential lines ------------
# ----------------------------------------------------------------------------
class bestLineAgent(dominoeAgent):
    agentName = 'bestLineAgent'
    
    def specializedInit(self,**kwargs):
        self.inLineDiscount = 0.9
        self.offLineDiscount = 0.7
        self.lineTemperature = 1
        self.maxLineLength = 10
        
        self.needsLineUpdate = True
        self.useSmartUpdate = True
        
        self.playValue = np.sum(self.dominoes,axis=1)
        self.nonDouble = self.dominoes[:,0]!=self.dominoes[:,1]
    
    def initHand(self):
        self.needsLineUpdate = True
        
    def linePlayedOn(self):
        # if my line was played on, then recompute sequences if it's my turn
        self.needsLineUpdate = True
    
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
    
    def dominoeLineValue(self):
        # if we need a line update, then run constructLineRecursive
        # (this should only ever happen if it's the first turn or if the line was played on by another agent)
        if self.needsLineUpdate: 
            self.lineSequence,self.lineDirection = df.constructLineRecursive(self.dominoes, self.myHand, self.available[0], maxLineLength=self.maxLineLength)
            self.needsLineUpdate = False if self.useSmartUpdate else True
        
        # if no line is possible, return None
        if self.lineSequence==[[]]: return None, None, None
        
        # Otherwise, compute line value for each line and return best play 
        numInHand = len(self.handValues)
        numLines = len(self.lineSequence)
        inLineValue = np.zeros(numLines)
        offLineValue = np.zeros(numLines)
        lineDiscountFactors = [None]*numLines
        notInSequence = [None]*numLines
        
        for line in range(numLines):
            linePlayNumber = np.cumsum(self.nonDouble[self.lineSequence[line]])-1 # turns to play each dominoe if playing this line continuously
            lineDiscountFactors[line] = self.inLineDiscount**linePlayNumber # discount factor (gamma**timeStepsInFuture)
            inLineValue[line] = lineDiscountFactors[line] @ self.playValue[self.lineSequence[line]] # total value of line, discounted for future plays
            offDiscount = self.offLineDiscount**(linePlayNumber[-1] if len(self.lineSequence[line])>0 else 1)
            # total value of remaining dominoes in hand after playing line, multiplied by a discount factor
            notInSequence[line] = list(set(self.myHand).difference(self.lineSequence[line]))
            offLineValue[line] = offDiscount*np.sum(self.playValue[notInSequence[line]]) 
            
        lineValue = inLineValue - offLineValue
        lineProbability = df.softmax(lineValue/self.lineTemperature)
        bestLine = np.argmax(lineProbability)
        return self.lineSequence[bestLine], notInSequence[bestLine], lineValue[bestLine]
    
    def optionValue(self, locations, dominoes):
        optionValue = self.dominoeValue[dominoes] # start with just dominoe value
        optionValue[self.dominoeDouble[dominoes]]=np.inf # always play a double
        
        # get best line etc. 
        bestLine,notInLine,bestLineValue = self.dominoeLineValue()
            
        # if there is a best line, inflate that plays value to the full line value
        if bestLine is not None:
            idxBestPlay = np.where((locations==0) & (dominoes==bestLine[0]))[0]
            assert len(idxBestPlay)==1, "this should always be 1 if a best line was found..."
            optionValue[idxBestPlay[0]] = bestLineValue
        return optionValue
    
    def selectPlay(self, gameEngine=None):
        # select dominoe to play, for the default class, the selection is random based on available plays
        locations, dominoes = self.playOptions() # get options that are available
        # if there are no options, return None
        if len(locations)==0: return None, None
        # if there are options, then measure their value
        optionValue = self.optionValue(locations, dominoes)
        # make choice of which dominoe to play
        idxChoice = self.makeChoice(optionValue)
        # update possible line sequences based on choice
        self.lineSequence,self.lineDirection = df.updateLine(self.lineSequence, self.lineDirection, dominoes[idxChoice], locations[idxChoice]==0)
        self.needsLineUpdate = False if self.useSmartUpdate else True
        # return choice to game play object
        return dominoes[idxChoice], locations[idxChoice]
        

# ----------------------------------------------------------------------------
# -------------------------- RL Agents that estimate value -------------------
# ----------------------------------------------------------------------------
class valueAgent(dominoeAgent):
    # valueAgent is a mid-level class for all agents that perform TD-Lambda like value computations.
    agentName = 'valueAgent'
    def specializedInit(self,**kwargs):
        # meta parameters
        self.highestDominoeRange = [self.highestDominoe]
        self.learning = True
        self.lam = 0.8
        self.alpha = 1e-5
        self.trackFinalScoreError = []
        self.finalScoreOutputDimension = 1
        
        # create binary arrays for presenting gamestate information to the RL networks
        self.binaryPlayed = np.zeros(self.numDominoes)
        self.binaryLineAvailable = np.zeros((self.numPlayers,self.highestDominoe+1))
        self.binaryDummyAvailable = np.zeros(self.highestDominoe+1)
        self.binaryHand = np.zeros(self.numDominoes)
        
        # prepare value network
        self.prepareNetwork()
        
        # Also prepare special parameters eligibility traces
        self.extraParameters = []
        self.extraEligibility = []
    
    def initHand(self):
        self.zeroEligibility()
        
    def prepareNetwork(self):
        raise ValueError("It looks like you instantiated an object of the valueAgent class directly. This class is only used to provide a scaffold for complete valueAgents, see possible agents in this script!")
    
    def zeroEligibility(self):
        raise ValueError("It looks like you instantiated an object of the valueAgent class directly. This class is only used to provide a scaffold for complete valueAgents, see possible agents in this script!")    
        
    def setLearning(self, learningState):
        self.learning = learningState

    def resetBinaries(self):
        self.binaryPlayed[:]=0 
        self.binaryLineAvailable[:]=0 
        self.binaryDummyAvailable[:]=0 
        self.binaryHand[:]=0 
    
    def checkTurnUpdate(self, currentPlayer, postState=False):
        relevantTurn = (currentPlayer is not None) and (currentPlayer == self.agentIndex)
        relevantState = True # always update gameState even at postState
        return relevantTurn and relevantState
    
    def processGameState(self):
        # vectorize game state data
        self.resetBinaries() # set binaries back to 0
        self.binaryPlayed[self.played]=1 # indicate which dominoes have been played
        for lineIdx, dominoeIdx in enumerate(self.available): 
            self.binaryLineAvailable[lineIdx,dominoeIdx]=1 # indicate which value is available on each line
        self.binaryDummyAvailable[self.dummyAvailable]=1 # indicate which value is available on the dummy line
        self.binaryHand[self.myHand]=1 # indicate which dominoes are present in hand
        self.prepareValueInputs() # this can be specified for individual kinds of valueAgents 
    
    def prepareValueInputs(self):
        raise ValueError("It looks like you instantiated an object of the valueAgent class directly. This class is only used to provide a scaffold for complete valueAgents, see possible agents in this script!")
    
    def sampleFutureGamestate(self, nextState, newHand, **kwargs):
        # this is for creating a new game state and value inputs based on a simulated play without overwriting any parameters relating to the true game state
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = nextState
        
        # -- what usually happens in self.gameState() --
        available = self.egocentric(available) # list of value available on each players line (centered on agent)
        handSize = self.egocentric(handSize) # list of handsize for each player (centered on agent)
        cantPlay = self.egocentric(cantPlay) # list of whether each player can/can't play (centered on agent)
        didntPlay = self.egocentric(didntPlay) # list of whether each player didn't play (centered on agent)
        turnCounter = self.egocentric(turnCounter) # list of how many turns until each player is up (meaningless unless agent receives all-turn updates)
        
        # -- what usually happens in self.processGameState() --
        binaryPlayed = np.zeros(self.numDominoes)
        binaryLineAvailable = np.zeros((self.numPlayers,self.highestDominoe+1))
        binaryDummyAvailable = np.zeros(self.highestDominoe+1)
        binaryHand = np.zeros(self.numDominoes)
        binaryPlayed[played]=1 # indicate which dominoes have been played
        for lineIdx, dominoeIdx in enumerate(available): 
            binaryLineAvailable[lineIdx,dominoeIdx]=1 # indicate which value is available on each line
        binaryDummyAvailable[dummyAvailable]=1 # indicate which value is available on the dummy line
        binaryHand[newHand]=1 # indicate which dominoes are present in hand
        
        # This needs to be here for the lineValueAgents
        kwargs['myHand']=newHand
        
        # -- prepare value input in simulated future gamestate --
        return self.simulateValueInputs(binaryHand, binaryPlayed, binaryLineAvailable, binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, dummyPlayable, **kwargs)
        
    def estimatePrestateValue(self, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer, postState=False)): return None
        if not(self.learning): return None

        # at the beginning of each turn, zero out the gradients 
        self.finalScoreNetwork.zero_grad()
        
        # predict V(S,w) of prestate using current weights
        self.finalScoreOutput = self.finalScoreNetwork(self.valueNetworkInput)
        
        # compute gradient of V(S,w) with respect to weights and add it to eligibility traces
        for idx,hvOutput in enumerate(self.finalScoreOutput):
            hvOutput.backward(retain_graph=True) # measure gradient of weights with respect to this output value
            for trace,prms in zip(self.finalScoreEligibility[idx], self.finalScoreNetwork.parameters()):
                trace *= self.lam # discount past eligibility traces by lambda
                trace += prms.grad # add new gradient to eligibility trace
                prms.grad.zero_() # reset gradients for parameters of finalScoreNetwork in between each backward call from the output
            for trace,prms in zip(self.extraEligibility, self.extraParameters):
                trace *= self.lam # discount past eligibility traces by lambda
                trace += prms.grad # add new gradient to eligibility trace
                prms.grad.zero_()
                    
    @torch.no_grad() # don't need to estimate any gradients here, that was done in estimatePrestateValue()!
    def updatePoststateValue(self, finalScore=None, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer, postState=True)): return None
        if not(self.learning): return None

        # otherwise, do post-state value update
        if finalScore is None: 
            # if final score is none, then the game hasn't ended and we should learn from the poststate value estimate
            tdError = self.finalScoreNetwork(self.valueNetworkInput) - self.finalScoreOutput
        else:
            # if the final score is an array, then we should shift its perspective and learn from the true difference in our penultimate estimate and the actual final score
            finalScore = self.egocentric(finalScore)[:self.finalScoreOutputDimension]
            finalScore = torch.tensor(finalScore).to(self.device)
            tdError = finalScore - self.finalScoreOutput
        
        for idx,td in enumerate(tdError):
            for prmIdx,prms in enumerate(self.finalScoreNetwork.parameters()):
                assert self.finalScoreEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.finalScoreEligibility[idx][prmIdx] * td # TD(lambda) update rule
            for prmIdx,prms in enumerate(self.extraParameters):
                assert self.extraEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.extraEligibility[idx][prmIdx] * td # TD(lambda) update rule
        if finalScore is not None:
            # when the final score is provided (i.e. the game ended), then add the error between the penultimate estimate and the true final score to a list for performance monitoring
            self.trackFinalScoreError.append(torch.mean(torch.abs(finalScore-self.finalScoreOutput)).to('cpu'))
        return None   
    
    def selectPlay(self, gameEngine):      
        # first, identify valid play options
        locations, dominoes = self.playOptions() # get options that are available
        if len(locations)==0: return None, None
        # for each play option, simulate future gamestate and estimate value from it (without gradients)
        optionValue = [self.optionValue(dominoe, location, gameEngine) for (dominoe,location) in zip(dominoes, locations)]
        idxChoice = self.makeChoice(optionValue) # make choice and return
        return dominoes[idxChoice], locations[idxChoice] 
    
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
    
    def saveAgentParameters(self, path, description=True):
        # generate parameters to save
        networkParameters = self.finalScoreNetwork.state_dict()
        agentParameters = self.agentParameters()
        if description:
            modelDescription = input("Describe this model briefly: ")
        else:
            modelDescription = "lineValueNetwork parameters"
        parameters = np.array([agentParameters, networkParameters, modelDescription])
        
        saveDate = datetime.now().strftime('%y%m%d')
        modelName = f"lineValueAgentParameters_{saveDate}"
        
        # Handle model save number
        existingSaves = glob(str(path / modelName)+'_*.npy')
        saveNumbers = [int(re.search(r'^.*(\d+)\.npy', esave).group(1)) for esave in existingSaves]
        saveNumber = max(saveNumbers)+1 if len(saveNumbers)>0 else 0
        modelName += f'_{saveNumber}'
        np.save(path / modelName, parameters) 
        
    def loadAgentParameters(self, path):
        parameters = np.load(path,allow_pickle=True)
        unmutableAttributes = ['numPlayers', 'highestDominoe', 'dominoes', 'numDominoes']
        for uatt in unmutableAttributes:
            if not(np.array_equal(parameters[0][uatt],getattr(self, uatt))):
                raise ValueError(f"Mismatched attribute: {uatt}")
        for key in parameters[0].keys():
            setattr(self, key, parameters[0][key])
        self.finalScoreNetwork.load_state_dict(parameters[1])
    
    def specialParameters(self):
        specialPrmNames = ['lam', 'alpha']
        specialPrms = {}
        for key in specialPrmNames:
            specialPrms[key] = getattr(self, key)
        return specialPrms
    
    
    
    
class basicValueAgent(valueAgent):
    agentName = 'basicValueAgent'    
    def prepareNetwork(self):
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.valueNetwork(self.numPlayers,self.numDominoes,self.highestDominoe,self.finalScoreOutputDimension)
        self.finalScoreNetwork.to(self.device)
        
        # Prepare Training Functions & Optimizers
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)]
    
    def zeroEligibility(self):
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)]
        
    def prepareValueInputs(self, updateObject=True):
        self.valueNetworkInput = self.generateValueInput().to(self.device)
        
    def generateValueInput(self):
        valueNetworkInput = torch.tensor(np.concatenate((self.binaryHand, self.binaryPlayed, self.binaryLineAvailable.flatten(), self.binaryDummyAvailable, 
                                            self.handsize, self.cantplay, self.didntplay, self.turncounter, np.array(self.dummyPlayable).reshape(-1)))).float()
        return valueNetworkInput    
    
    def simulateValueInputs(self, binaryHand, binaryPlayed, binaryLineAvailable, binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, dummyPlayable, **kwargs):
        return torch.tensor(np.concatenate((binaryHand, binaryPlayed, binaryLineAvailable.flatten(), binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, np.array(dummyPlayable).reshape(-1)))).float()
        
    def optionValue(self, dominoe, location, gameEngine):
        # enter dominoe and location into gameEngine, return new gamestate
        # with new gamestate, estimate value 
        # return final score estimate for ~self~ only
        # make choice will return the argmin of option value, attempting to bring about the lowest final score possible
        nextState = gameEngine(dominoe, location)
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = nextState
        # remove dominoe from hand
        newHand = np.delete(self.myHand, self.myHand==dominoe)
        simulatedValueInput = self.sampleFutureGamestate(nextState, newHand)
        with torch.no_grad(): finalScoreOutput = self.finalScoreNetwork(simulatedValueInput)
        # return optionValue
        return finalScoreOutput[0].detach().cpu().numpy()
    
    
class lineValueAgent(valueAgent):
    agentName = 'lineValueAgent'
    def specializedInit(self,**kwargs):
        # do general valueAgent initialization
        super().specializedInit(**kwargs)
        
        # add parameters for measuring line value (will be learned eventually?)
        self.inLineDiscount = 0.9
        self.offLineDiscount = 0.7
        self.lineTemperature = 1.0
        self.maxLineLength = 10
        
        self.needsLineUpdate = True
        self.useSmartUpdate = True
        
        self.nonDouble = self.dominoes[:,0]!=self.dominoes[:,1]
        self.playValue = np.sum(self.dominoes, axis=1)
        
    def initHand(self):
        super().initHand()
        self.needsLineUpdate = True
        
    def linePlayedOn(self):
        # if my line was played on, then recompute sequences if it's my turn
        self.needsLineUpdate = True
        
    def checkTurnUpdate(self, currentPlayer, postState=False):
        relevantTurn = True # update every turn -- #(currentPlayer is not None) and (currentPlayer == self.agentIndex)
        relevantState = True # update gameState for pre and post states
        return relevantTurn and relevantState
    
    def specialParameters(self):
        # add a few special parameters to store and reload for this type of agent
        specialPrms = super().specialParameters()
        extraPrms = ['inLineDiscount', 'offLineDiscount', 'lineTemperature', 'maxLineLength']
        for key in extraPrms:
            specialPrms[key] = getattr(self, key)
        return specialPrms

    def prepareNetwork(self):
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.lineRepresentationNetwork(self.numPlayers,self.numDominoes,self.highestDominoe,self.finalScoreOutputDimension)
        self.finalScoreNetwork.to(self.device)

        # Prepare Training Functions & Optimizers
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)]
    
    def zeroEligibility(self):
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)]
    
    def prepareValueInputs(self):
        # First, get all possible lines that can be made on agent's own line.
        if self.needsLineUpdate: self.lineSequence, self.lineDirection = df.constructLineRecursive(self.dominoes, self.myHand, self.available[0], maxLineLength=self.maxLineLength)
        lineValue = self.estimateLineValue(self.lineSequence) # estimate line value given current game state
        self.lineValueInput, self.gameStateInput = self.generateValueInput(lineValue)
        self.valueNetworkInput = (self.lineValueInput, self.gameStateInput)
        
    def estimateLineValue(self, lineSequence, myHand=None):
        if myHand is None: myHand = self.myHand
        
        numInHand = len(myHand)
        expInLineValue = np.zeros(numInHand) # average line value for lines that this dominoe is in 
        expOffLineValue = np.zeros(numInHand) # average remaining dominoe value for lines that this dominoe is in 
        expLossValue = np.zeros(numInHand) # probability of dominoe being out of a line * value of this dominoe (e.g. expected cost if not played) 
        expDominoeValue = np.zeros(numInHand) # expected discounted value of this dominoe (dominoeValue * discount factor) for all lines in which it can be played 
        expInLineDominoes = np.zeros(numInHand) # average number of dominoes in line for lines that this dominoe is in
        expOffLineDominoes = np.zeros(numInHand) # average number of dominoes out of line for lines that this dominoe is in
        
        # measure line value
        numLines = len(lineSequence)
        inLineValue = np.zeros(numLines)
        offLineValue = np.zeros(numLines)
        inLineDominoes = np.zeros(numLines)
        offLineDominoes = np.zeros(numLines)
        lineDiscountFactors = [None]*numLines
        for line in range(numLines):
            linePlayNumber = np.cumsum(self.nonDouble[lineSequence[line]],axis=0)-1 # turns to play each dominoe if playing this line continuously
            lineDiscountFactors[line] = self.inLineDiscount**linePlayNumber # discount factor (gamma**timeStepsInFuture)
            inLineValue[line] = lineDiscountFactors[line] @ self.playValue[lineSequence[line]] # total value of line, discounted for future plays
            offDiscount = self.offLineDiscount**(linePlayNumber[-1] if len(lineSequence[line])>0 else 1)
            offLineValue[line] = offDiscount*np.sum(self.playValue[list(set(myHand).difference(lineSequence[line]))]) # total value of remaining dominoes in hand after playing line, multiplied by a discount factor
            inLineDominoes[line] = len(lineSequence[line]) # number of dominoes in line
            offLineDominoes[line] = numInHand - len(lineSequence[line]) # number of dominoes remaining after playing this line
        
        lineValue = inLineValue - offLineValue
        lineProbability = df.softmax(lineValue / self.lineTemperature)
        
        # make grid of where all dominoes are -- then use this to get E[V] for dominoe if played in line and -E[V] for dominoe if not played
        dominoeInLine = np.zeros((numInHand,numLines))
        for domIdx, dominoe in enumerate(myHand):
            inLineProbability = 0
            offLineProbability = 0
            for line in range(numLines):
                # store total probabilities, so the expected values can be scaled correctly
                inLineProbability += lineProbability[line]*(dominoe in lineSequence[line])
                offLineProbability += lineProbability[line]*(not(dominoe in lineSequence[line]))

                # add up elements in expected value expressions
                expInLineValue[domIdx] += inLineValue[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add lineValue*lineProbability if dominoe is in line
                expOffLineValue[domIdx] += offLineValue[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add offLineValue*lineProbability if dominoe is in line
                expInLineDominoes[domIdx] += inLineDominoes[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add lineDominoes*lineProbability if dominoe is in line
                expOffLineDominoes[domIdx] += offLineDominoes[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add lineDominoes*lineProbability if dominoe is in line
                expLossValue[domIdx] += self.playValue[dominoe]*lineProbability[line]*(not(dominoe in lineSequence[line])) # add dominoeValue*lineProbability if dominoe isn't in line
                expDominoeValue[domIdx] += lineProbability[line]*sum([self.playValue[dom]*lineDiscountFactors[line][idx] for idx,dom in enumerate(lineSequence[line]) if dom==dominoe]) 

            # scale to total probability
            if inLineProbability>0:
                expInLineValue[domIdx] /= inLineProbability
                expOffLineValue[domIdx] /= inLineProbability
                expInLineDominoes[domIdx] /= inLineProbability
                expOffLineDominoes[domIdx] /= inLineProbability
            if inLineProbability<1:
                expLossValue[domIdx] /= (1-inLineProbability)
        
        handLineValue = np.stack((expInLineValue, expOffLineValue, expLossValue, expDominoeValue, expInLineDominoes, expOffLineDominoes))
        lineValue = np.zeros((handLineValue.shape[0],self.numDominoes))
        lineValue[:, myHand] = handLineValue
        return lineValue
        
    def generateValueInput(self, lineValue):
        lineValueInput = torch.tensor(lineValue).float().to(self.device)
        gameStateInput = torch.tensor(np.concatenate((self.binaryHand, self.binaryPlayed, self.binaryLineAvailable.flatten(), self.binaryDummyAvailable, 
                                            self.handsize, self.cantplay, self.didntplay, self.turncounter, np.array(self.dummyPlayable).reshape(-1)))).float().to(self.device)
        return lineValueInput, gameStateInput
   
    def simulateValueInputs(self, binaryHand, binaryPlayed, binaryLineAvailable, binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, dummyPlayable, **kwargs):
        lineValueInput = torch.tensor(self.estimateLineValue(kwargs['lineSequence'], myHand=kwargs['myHand'])).float().to(self.device)
        gameStateInput = torch.tensor(np.concatenate((binaryHand, binaryPlayed, binaryLineAvailable.flatten(), binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, np.array(dummyPlayable).reshape(-1)))).float().to(self.device)
        return lineValueInput, gameStateInput
    
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
    
    def optionValue(self, dominoe, location, gameEngine):
        # enter dominoe and location into gameEngine, return new gamestate
        # with new gamestate, estimate value 
        # return final score estimate for ~self~ only
        # make choice will return the argmin of option value, attempting to bring about the lowest final score possible
        nextState = gameEngine(dominoe, location) # enter play option (dominoe-location pair) into the gameEngine, and return simulated new gameState
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = nextState # name the outputs 
        lineSequence, lineDirection = df.updateLine(self.lineSequence, self.lineDirection, dominoe, location==0)
        # remove dominoe from hand
        newHand = np.delete(self.myHand, self.myHand==dominoe)
        handValues = self.dominoesInHand(updateObject=False)
        simulatedValueInput = self.sampleFutureGamestate(nextState, newHand, lineSequence=lineSequence)
        with torch.no_grad(): finalScoreOutput = self.finalScoreNetwork(simulatedValueInput)
        # return optionValue
        return finalScoreOutput.detach().cpu().numpy()
    
    def selectPlay(self, gameEngine):
        # first, identify valid play options
        locations, dominoes = self.playOptions() # get options that are available
        if len(locations)==0: return None, None
        # for each play option, simulate future gamestate and estimate value from it (without gradients)
        optionValue = [self.optionValue(dominoe, location, gameEngine) for (dominoe,location) in zip(dominoes, locations)]
        # make choice of which dominoe to play
        idxChoice = self.makeChoice(optionValue) 
        # update possible line sequences based on choice
        self.lineSequence,self.lineDirection = df.updateLine(self.lineSequence, self.lineDirection, dominoes[idxChoice], locations[idxChoice]==0)
        self.needsLineUpdate = False if self.useSmartUpdate else True
        # return choice to game play object
        return dominoes[idxChoice], locations[idxChoice] 
        
    
    
class lineValueAgentSmall(lineValueAgent):
    agentName = 'lineValueAgentSmall'
    def prepareNetwork(self):
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.lineRepresentationNetworkSmall(self.numPlayers,self.numDominoes,self.highestDominoe,self.finalScoreOutputDimension)
        self.finalScoreNetwork.to(self.device)

        # Prepare Training Functions & Optimizers
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)]
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    