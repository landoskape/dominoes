import re
from glob import glob
from datetime import datetime
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
    def __init__(self, numPlayers, highestDominoe, dominoes, numDominoes, agentIndex, device=None):
        # meta-variables (includes redundant information, but if I train 1000s of networks I want it to be as fast as possible)
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
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
        self.specializedInit()
    
    def specializedInit(self):
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
    
    def dominoesInHand(self):
        # simple function to return real values of dominoes from index of dominoes in hand
        self.handValues = self.dominoes[self.myHand]
    
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
    
    def egocentric(self, variable):
        return variable[self.egoShiftIdx]
        
    def checkTurnUpdate(self, currentPlayer):
        # if turn idx isn't zero, then don't update game state
        return (currentPlayer is not None) and (currentPlayer == self.agentIndex)
    
    def gameState(self, played, available, handsize, cantplay, didntplay, turncounter, dummyAvailable, dummyPlayable, currentPlayer=None, postState=None):
        # gamestate input, served to the agent each time it requires action (either at it's turn, or each turn for an RNN)
        # agents convert these variables to agent-centric information about the game-state
        if not(self.checkTurnUpdate(currentPlayer)): return None
        if postState==True: return None
        
        self.played = played # list of dominoes that have already been played
        self.available = self.egocentric(available) # list of value available on each players line (centered on agent)
        self.handsize = self.egocentric(handsize) # list of handsize for each player (centered on agent)
        self.cantplay = self.egocentric(cantplay) # list of whether each player can/can't play (centered on agent)
        self.didntplay = self.egocentric(didntplay) # list of whether each player didn't play (centered on agent)
        self.turncounter = self.egocentric(turncounter) # list of how many turns until each player is up (meaningless unless agent receives all-turn updates)
        self.dummyAvailable = dummyAvailable # index of dominoe available on dummy line
        self.dummyPlayable = dummyPlayable # bool determining whether the dummy line is playable
        self.processGameState(postState=postState)
    
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
    
    def optionValue(self, options):
        # convert option to play value using simplest method possible - value is 1 if option available
        return 1*options
    
    def playOptions(self):
        # generates list of playable options given game state 
        # it produces a (numPlayers x numDominoes) array with True's indicating viable dominoe-location pairings
        # (and also a (numDominoes,) array for the dummy line
        print('in play options, I think I can speed this up by using an outerproduct of boolean vectors')
        lineOptions = np.full((self.numPlayers, self.numDominoes), False)
        for idx,value in enumerate(self.available):
            if idx==0 or self.cantplay[idx]:
                idxPlayable = np.where(np.any(self.handValues==value,axis=1))[0]
                lineOptions[idx,self.myHand[idxPlayable]]=True
        dummyOptions = np.full(self.numDominoes, False)
        idxPlayable = np.where(np.any(self.handValues==self.dummyAvailable,axis=1))[0]
        dummyOptions[self.myHand[idxPlayable]]=True*self.dummyPlayable
        return lineOptions,dummyOptions
    
    def selectPlay(self, gameEngine=None, game=None):
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
    
    def play(self, gameEngine=None, game=None):
        # this function is called by the gameplay object
        # it is what's used to play a dominoe when it's this agents turn
        dominoe, location = self.selectPlay(gameEngine=gameEngine, game=game)
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
    def optionValue(self, options):
        return self.dominoeValue * options
    
class stupidAgent(dominoeAgent):
    # stupid agent plays whatever dominoe has the least number of points
    agentName = 'stupidAgent' 
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
    def optionValue(self, options):
        return self.dominoeValue * options
    
class doubleAgent(dominoeAgent):
    # double agent plays any double it can play immediately, then plays the dominoe with the highest number of points
    agentName = 'doubleAgent'
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
    def optionValue(self, options):
        optionValue = self.dominoeValue * options
        if np.any(self.dominoeDouble*options):
            optionValue[self.dominoeDouble*options]=np.inf
        return optionValue
    
    
# ----------------------------------------------------------------------------
# -------------- agents that care about possible sequential lines ------------
# ----------------------------------------------------------------------------
class bestLineAgent(dominoeAgent):
    agentName = 'bestLineAgent'
    
    def specializedInit(self):
        self.inLineDiscount = 0.9
        self.offLineDiscount = 0.7
        self.lineTemperature = 1
        self.maxLineLength = 10
    
    def selectPlay(self, gameEngine=None, game=None):
        # select dominoe to play, for the default class, the selection is random based on available plays
        lineOptions, dummyOptions = self.playOptions() # get options that are available
        idxPlayer, idxDominoe = np.where(lineOptions) # find where options are available
        idxDummyDominoe = np.where(dummyOptions)[0]
        idxDummy = -1 * np.ones(len(idxDummyDominoe), dtype=int)
        # if no valid options available, return None
        if len(idxPlayer)==0 and len(idxDummy)==0: 
            return None,None
        # concatenate lineIdx, dominoeIdx, optionValue
        lineIdx = np.concatenate((idxPlayer, idxDummy))
        dominoeIdx = np.concatenate((idxDominoe, idxDummyDominoe))
        optionValue = self.optionValue(lineIdx, dominoeIdx)
        # make and return choice
        idxChoice = self.makeChoice(optionValue)
        return dominoeIdx[idxChoice], lineIdx[idxChoice] 
    
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)
    
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
    
    def dominoeLineValue(self):
        # create a dictionary that goes from the index within hand to the absolute index
        hand2absolute = {}
        for idx,val in enumerate(self.myHand): hand2absolute[idx]=val
            
        lineSequence,lineDirection = df.constructLineRecursive(self.handValues, self.available[0], maxLineLength=self.maxLineLength)
        playValue = np.sum(self.handValues,axis=1)
        nonDouble = self.handValues[:,0]!=self.handValues[:,1]
        
        # if no line is possible, return None
        if lineSequence==[[]]: return None, None, None
        
        # Otherwise, compute line value for each line and return best play 
        numInHand = len(self.handValues)
        numLines = len(lineSequence)
        inLineValue = np.zeros(numLines)
        offLineValue = np.zeros(numLines)
        lineDiscountFactors = [None]*numLines
        notInSequence = [None]*numLines
        
        for line in range(numLines):
            linePlayNumber = np.cumsum(nonDouble[lineSequence[line]])-1 # turns to play each dominoe if playing this line continuously
            try:
                lineDiscountFactors[line] = self.inLineDiscount**linePlayNumber # discount factor (gamma**timeStepsInFuture)
            except:
                print('hi')
            inLineValue[line] = lineDiscountFactors[line] @ playValue[lineSequence[line]] # total value of line, discounted for future plays
            offDiscount = self.offLineDiscount**(linePlayNumber[-1] if len(lineSequence[line])>0 else 1)
            # total value of remaining dominoes in hand after playing line, multiplied by a discount factor
            notInSequence[line] = list(set(range(numInHand)).difference(lineSequence[line]))
            offLineValue[line] = offDiscount*np.sum(playValue[notInSequence[line]]) 
            
        lineValue = inLineValue - offLineValue
        lineProbability = df.softmax(lineValue/self.lineTemperature)
        bestLine = np.argmax(lineProbability)
        bestPlay = lineSequence[bestLine][0]
        return self.myHand[lineSequence[bestLine]], self.myHand[notInSequence[bestLine]], lineValue[bestLine]
        
        

# ----------------------------------------------------------------------------
# -------------------------- RL Agents that estimate value -------------------
# ----------------------------------------------------------------------------
class valueAgent0(dominoeAgent):
    agentName = 'valueAgent0'
    def specializedInit(self):
        # create binary arrays for presenting gamestate information to the RL networks
        self.binaryPlayed = np.zeros(self.numDominoes)
        self.binaryLineAvailable = np.zeros((self.numPlayers,self.highestDominoe+1))
        self.binaryDummyAvailable = np.zeros(self.highestDominoe+1)
        self.binaryHand = np.zeros(self.numDominoes)
        
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.valueNetwork(self.numPlayers,self.numDominoes,self.highestDominoe)
        self.finalScoreNetwork.to(self.device)
        
        # Prepare Training Functions & Optimizers
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreNetwork.outputDimension)]
        
        # meta parameters
        self.learning = True
        self.lam = 0.9
        self.alpha = 1e-5
        self.trackFinalScoreError = []
    
    def activateLearning(self, learningState):
        self.learning = learningState
        
    def selectPlay(self, gameEngine, game):
        # first, identify valid play options
        lineOptions, dummyOptions = self.playOptions() # get options that are available
        idxPlayer, idxDominoe = np.where(lineOptions) # find where options are available
        idxDummyDominoe = np.where(dummyOptions)[0]
        idxDummy = -1 * np.ones(len(idxDummyDominoe), dtype=int)
        # if no valid options available, return None
        if len(idxPlayer)==0 and len(idxDummy)==0: 
            return None,None
        # concatenate lineIdx, dominoeIdx, optionValue
        lineIdx = np.concatenate((idxPlayer, idxDummy))
        dominoeIdx = np.concatenate((idxDominoe, idxDummyDominoe))
        # for each play option, simulate future gamestate and estimate value from it (without gradients)
        optionValue = np.zeros(len(lineIdx))
        for idx in range(len(lineIdx)):
            optionValue[idx] = self.optionValue(dominoeIdx[idx], lineIdx[idx], gameEngine) # for (dominoe,location) in zip(dominoeIdx,lineIdx)])
        # make choice and return
        idxChoice = self.makeChoice(optionValue)
        return dominoeIdx[idxChoice], lineIdx[idxChoice] 
    
    def optionValue(self, dominoe, location, gameEngine):
        # enter dominoe and location into gameEngine, return new gamestate
        # with new gamestate, estimate value 
        # return final score estimate for ~self~ only
        # make choice will return the argmin of option value, attempting to bring about the lowest final score possible
        nextState = gameEngine(dominoe, location)
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = nextState
        # remove dominoe from hand
        self.myHand = np.delete(self.myHand, self.myHand==dominoe)
        self.dominoesInHand()
        self.gameState(played, available, handSize, cantPlay, didntPlay, turnCounter, dummyAvailable, dummyPlayable)
        with torch.no_grad():
            valueNetworkInput = self.generateValueInput().to(self.device)
            finalScoreOutput = self.finalScoreNetwork(valueNetworkInput)
        # put dominoe back in hand
        self.myHand = np.append(self.myHand, dominoe)
        self.dominoesInHand()
        # return optionValue
        return finalScoreOutput[0]
    
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
        
    def resetBinaries(self):
        self.binaryPlayed[:]=0 
        self.binaryLineAvailable[:]=0 
        self.binaryDummyAvailable[:]=0 
        self.binaryHand[:]=0 
     
    def checkTurnUpdate(self, currentPlayer):
        # if turn idx isn't zero, then don't update game state
        return (currentPlayer is not None) and (currentPlayer == self.agentIndex)
    
    # valueAgent0 uses a network model to predict: 1) the final score from the gameState at each turn (learning from the end), 2) and the current hand value of each player (using omniscient info)
    def processGameState(self,postState=None):
        if (postState is not None) and (postState==True) and not(self.learning): return None
        # vectorize game state data
        self.resetBinaries() # set binaries back to 0
        self.binaryPlayed[self.played]=1 # indicate which dominoes have been played
        for lineIdx, dominoeIdx in enumerate(self.available): 
            self.binaryLineAvailable[lineIdx,dominoeIdx]=1 # indicate which value is available on each line
        self.binaryDummyAvailable[self.dummyAvailable]=1 # indicate which value is available on the dummy line
        self.binaryHand[self.myHand]=1 # indicate which dominoes are present in hand
        self.valueNetworkInput = self.generateValueInput().to(self.device)
    
    def generateValueInput(self):
        valueNetworkInput = torch.tensor(np.concatenate((self.binaryHand, self.binaryPlayed, self.binaryLineAvailable.flatten(), self.binaryDummyAvailable, 
                                            self.handsize, self.cantplay, self.didntplay, self.turncounter, np.array(self.dummyPlayable).reshape(-1)))).float()
        return valueNetworkInput      
    
    def estimatePrestateValue(self, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer)): return None
    
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
                prms.grad[:] = 0 # reset gradients for parameters of finalScoreNetwork in between each backward call from the output
    
    
    @torch.no_grad() # don't need to estimate any gradients here, that was done in estimatePrestateValue()!
    def updatePoststateValue(self, finalScore=None, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer)): return None
    
        # otherwise, do post-state value update
        if finalScore is None: 
            # if final score is none, then the game hasn't ended and we should learn from the poststate value estimate
            tdError = self.finalScoreNetwork(self.valueNetworkInput) - self.finalScoreOutput
        else:
            # if the final score is an array, then we should shift its perspective and learn from the true difference in our penultimate estimate and the actual final score
            finalScore = torch.tensor(self.egocentric(finalScore)).to(self.device)
            tdError = finalScore - self.finalScoreOutput
        for idx,td in enumerate(tdError):
            for prmIdx,prms in enumerate(self.finalScoreNetwork.parameters()):
                assert self.finalScoreEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.finalScoreEligibility[idx][prmIdx] * td # TD(lambda) update rule
        if finalScore is not None:
            # when the final score is provided (i.e. the game ended), then add the error between the penultimate estimate and the true final score to a list for performance monitoring
            self.trackFinalScoreError.append(torch.mean(torch.abs(finalScore-self.finalScoreOutput)).to('cpu'))
        return None   
    
    
class lineValueAgent(dominoeAgent):
    agentName = 'lineValueAgent'
    
    def specializedInit(self):
        # parameters for measuring line value (will be learned eventually?)
        self.inLineDiscount = 0.9
        self.offLineDiscount = 0.7
        self.lineTemperature = 1
        self.maxLineLength = 10
        
        # create binary arrays for presenting gamestate information to the RL networks
        self.binaryPlayed = np.zeros(self.numDominoes)
        self.binaryLineAvailable = np.zeros((self.numPlayers,self.highestDominoe+1))
        self.binaryDummyAvailable = np.zeros(self.highestDominoe+1)
        self.binaryHand = np.zeros(self.numDominoes)
        
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.lineRepresentationNetwork(self.numPlayers,self.numDominoes,self.highestDominoe)
        self.finalScoreNetwork.to(self.device)
        
        # Prepare Training Functions & Optimizers
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreNetwork.outputDimension)]
        
        # meta parameters
        self.learning = True
        self.lam = 0.8
        self.alpha = 3e-6
        self.trackFinalScoreError = []
        
    
    def activateLearning(self, learningState):
        self.learning = learningState
        
    def selectPlay(self, gameEngine, game):        
        # first, identify valid play options
        lineOptions, dummyOptions = self.playOptions() # get options that are available
        idxPlayer, idxDominoe = np.where(lineOptions) # find where options are available
        idxDummyDominoe = np.where(dummyOptions)[0]
        idxDummy = -1 * np.ones(len(idxDummyDominoe), dtype=int)
        # if no valid options available, return None
        if len(idxPlayer)==0 and len(idxDummy)==0: 
            return None,None
        # concatenate lineIdx, dominoeIdx, optionValue
        lineIdx = np.concatenate((idxPlayer, idxDummy))
        dominoeIdx = np.concatenate((idxDominoe, idxDummyDominoe))
        # for each play option, simulate future gamestate and estimate value from it (without gradients)
        optionValue = np.zeros(len(lineIdx))
        for idx in range(len(lineIdx)):
            optionValue[idx] = self.optionValue(dominoeIdx[idx], lineIdx[idx], gameEngine) # for (dominoe,location) in zip(dominoeIdx,lineIdx)])
        # make choice and return
        idxChoice = self.makeChoice(optionValue)
        return dominoeIdx[idxChoice], lineIdx[idxChoice] 
    
    def optionValue(self, dominoe, location, gameEngine):
        # enter dominoe and location into gameEngine, return new gamestate
        # with new gamestate, estimate value 
        # return final score estimate for ~self~ only
        # make choice will return the argmin of option value, attempting to bring about the lowest final score possible
        nextState = gameEngine(dominoe, location) # enter play option (dominoe-location pair) into the gameEngine, and return simulated new gameState
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = nextState # name the outputs 
        # remove dominoe from hand
        self.myHand = np.delete(self.myHand, self.myHand==dominoe)
        self.dominoesInHand()
        self.gameState(played, available, handSize, cantPlay, didntPlay, turnCounter, dummyAvailable, dummyPlayable) # load simulated gameState into object attributes
        with torch.no_grad():
            finalScoreOutput = self.finalScoreNetwork(self.lineValueInput, self.gameStateInput)
        # put dominoe back in hand
        self.myHand = np.append(self.myHand, dominoe)
        self.dominoesInHand()
        # return optionValue
        return finalScoreOutput
    
    def makeChoice(self, optionValue):
        return np.argmin(optionValue)
        
    def resetBinaries(self):
        self.binaryPlayed[:]=0
        self.binaryLineAvailable[:]=0
        self.binaryDummyAvailable[:]=0
        self.binaryHand[:]=0
        
    # valueAgent0 uses a network model to predict: 1) the final score from the gameState at each turn (learning from the end), 2) and the current hand value of each player (using omniscient info)
    def processGameState(self, postState=None):
        if (postState is not None) and (postState==True) and not(self.learning): return None
        # vectorize game state data
        self.resetBinaries() # set binaries back to 0
        self.binaryPlayed[self.played]=1 # indicate which dominoes have been played
        for lineIdx, dominoeIdx in enumerate(self.available): 
            self.binaryLineAvailable[lineIdx,dominoeIdx]=1 # indicate which value is available on each line
        self.binaryDummyAvailable[self.dummyAvailable]=1 # indicate which value is available on the dummy line
        self.binaryHand[self.myHand]=1 # indicate which dominoes are present in hand
        self.estimateLineValue() # estimate line value given current game state
        self.lineValueInput, self.gameStateInput = self.generateValueInput()
        
    def estimateLineValue(self):
        # First, get all possible lines that can be made on agent's own line.
        lineSequence = df.constructLineRecursive(self.handValues, self.available[0], maxLineLength=self.maxLineLength)[0] # don't need line direction, only need to know elements and values of each line
        nonDouble = self.handValues[:,0]!=self.handValues[:,1]
        playValue = np.sum(self.handValues,axis=1)
        
        numInHand = len(self.myHand)
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
            linePlayNumber = np.cumsum(nonDouble[lineSequence[line]])-1 # turns to play each dominoe if playing this line continuously
            lineDiscountFactors[line] = self.inLineDiscount**linePlayNumber # discount factor (gamma**timeStepsInFuture)
            inLineValue[line] = lineDiscountFactors[line] @ playValue[lineSequence[line]] # total value of line, discounted for future plays
            offDiscount = self.offLineDiscount**(linePlayNumber[-1] if len(lineSequence[line])>0 else 1)
            offLineValue[line] = offDiscount*np.sum(playValue[list(set(range(numInHand)).difference(lineSequence[line]))]) # total value of remaining dominoes in hand after playing line, multiplied by a discount factor
            inLineDominoes[line] = len(lineSequence[line]) # number of dominoes in line
            offLineDominoes[line] = numInHand - len(lineSequence[line]) # number of dominoes remaining after playing this line

        lineValue = inLineValue - offLineValue
        lineProbability = df.softmax(lineValue/self.lineTemperature)

        # make grid of where all dominoes are -- then use this to get E[V] for dominoe if played in line and -E[V] for dominoe if not played
        dominoeInLine = np.zeros((numInHand,numLines))
        for dominoe in range(numInHand):
            inLineProbability = 0
            offLineProbability = 0
            for line in range(numLines):
                # store total probabilities, so the expected values can be scaled correctly
                inLineProbability += lineProbability[line]*(dominoe in lineSequence[line])
                offLineProbability += lineProbability[line]*(not(dominoe in lineSequence[line]))

                # add up elements in expected value expressions
                expInLineValue[dominoe] += inLineValue[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add lineValue*lineProbability if dominoe is in line
                expOffLineValue[dominoe] += offLineValue[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add offLineValue*lineProbability if dominoe is in line
                expInLineDominoes[dominoe] += inLineDominoes[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add lineDominoes*lineProbability if dominoe is in line
                expOffLineDominoes[dominoe] += offLineDominoes[line]*lineProbability[line]*(dominoe in lineSequence[line]) # add lineDominoes*lineProbability if dominoe is in line
                expLossValue[dominoe] += playValue[dominoe]*lineProbability[line]*(not(dominoe in lineSequence[line])) # add dominoeValue*lineProbability if dominoe isn't in line
                expDominoeValue[dominoe] += lineProbability[line]*sum([playValue[dom]*lineDiscountFactors[line][idx] for idx,dom in enumerate(lineSequence[line]) if dom==dominoe]) 

            # scale to total probability
            if inLineProbability>0:
                expInLineValue[dominoe] /= inLineProbability
                expOffLineValue[dominoe] /= inLineProbability
                expInLineDominoes[dominoe] /= inLineProbability
                expOffLineDominoes[dominoe] /= inLineProbability
            if inLineProbability<1:
                expLossValue[dominoe] /= (1-inLineProbability)

        lineValue = np.stack((expInLineValue, expOffLineValue, expLossValue, expDominoeValue, expInLineDominoes, expOffLineDominoes))
        self.lineValue = np.zeros((6,self.numDominoes))
        self.lineValue[:,self.myHand] = lineValue
        
    def generateValueInput(self):
        lineValueInput = torch.tensor(self.lineValue).float().to(self.device)
        gameStateInput = torch.tensor(np.concatenate((self.binaryHand, self.binaryPlayed, self.binaryLineAvailable.flatten(), self.binaryDummyAvailable, 
                                            self.handsize, self.cantplay, self.didntplay, self.turncounter, np.array(self.dummyPlayable).reshape(-1)))).float().to(self.device)
        return lineValueInput, gameStateInput
    
    def checkTurnUpdate(self, currentPlayer):
        # if turn idx isn't zero, then don't update game state
        return True # (currentPlayer is not None) and (currentPlayer == self.agentIndex)
    
    def estimatePrestateValue(self, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer)): return None
        if not(self.learning): return None
        
        # at the beginning of each turn, zero out the gradients 
        self.finalScoreNetwork.zero_grad()
        
        # predict V(S,w) of prestate using current weights
        self.finalScoreOutput = self.finalScoreNetwork(self.lineValueInput, self.gameStateInput)
        
        # compute gradient of V(S,w) with respect to weights and add it to eligibility traces
        for idx,hvOutput in enumerate(self.finalScoreOutput):
            hvOutput.backward(retain_graph=True) # measure gradient of weights with respect to this output value
            for trace,prms in zip(self.finalScoreEligibility[idx], self.finalScoreNetwork.parameters()):
                trace *= self.lam # discount past eligibility traces by lambda
                trace += prms.grad # add new gradient to eligibility trace
                prms.grad[:] = 0 # reset gradients for parameters of finalScoreNetwork in between each backward call from the output
    
    
    @torch.no_grad() # don't need to estimate any gradients here, that was done in estimatePrestateValue()!
    def updatePoststateValue(self,finalScore=None, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer)): return None
        if not(self.learning): return None
        
        if finalScore is None: 
            # if final score is none, then the game hasn't ended and we should learn from the poststate value estimate
            tdError = self.finalScoreNetwork(self.lineValueInput, self.gameStateInput) - self.finalScoreOutput
        else:
            # if the final score is an array, then we should shift its perspective and learn from the true difference in our penultimate estimate and the actual final score
            finalScore = torch.tensor(self.egocentric(finalScore))[0].to(self.device)
            tdError = finalScore - self.finalScoreOutput
        for idx,td in enumerate(tdError):
            for prmIdx,prms in enumerate(self.finalScoreNetwork.parameters()):
                assert self.finalScoreEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.finalScoreEligibility[idx][prmIdx] * td # TD(lambda) update rule
        if finalScore is not None:
            # when the final score is provided (i.e. the game ended), then add the error between the penultimate estimate and the true final score to a list for performance monitoring
            self.trackFinalScoreError.append(torch.mean(torch.abs(finalScore-self.finalScoreOutput)).to('cpu'))
        return None   
    
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
        modelName += f'_{max(saveNumbers)+1}'
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
        specialPrmNames = ['inLineDiscount', 'offLineDiscount', 'lineTemperature', 'maxLineLength', 'lam', 'alpha']
        specialPrms = {}
        for key in specialPrmNames:
            specialPrms[key] = getattr(self, key)
        return specialPrms
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    