import numpy as np
import random
import itertools
import dominoesFunctions as df
import dominoesNetworks as dnn
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


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
    
    def updateModel(self):
        # default is no learning -- this can be overwritten in special RL agents
        return None
    
    # -- input from game manager --
    def serve(self, assignment):
        # serve receives an assignment (indices) of dominoes that make up my hand
        self.myHand = assignment
        self.dominoesInHand()
        
    def gameState(self, played, available, handsize, cantplay, didntplay, turncounter, dummyAvailable, dummyPlayable):
        # gamestate input, served to the agent each time it requires action (either at it's turn, or each turn for an RNN)
        # each agent will transform these inputs into a "perspective" which converts them to agent-centric information about the game-state
        self.played = played # list of dominoes that have already been played
        self.available = self.egocentric(available) # list of value available on each players line (centered on agent)
        self.handsize = self.egocentric(handsize) # list of handsize for each player (centered on agent)
        self.cantplay = self.egocentric(cantplay) # list of whether each player can/can't play (centered on agent)
        self.didntplay = self.egocentric(didntplay) # list of whether each player didn't play (centered on agent)
        self.turncounter = self.egocentric(turncounter) # list of how many turns until each player is up (meaningless unless agent receives all-turn updates)
        self.dummyAvailable = dummyAvailable # index of dominoe available on dummy line
        self.dummyPlayable = dummyPlayable # bool determining whether the dummy line is playable
        self.processGameState()
        
    def estimatePrestateValue(self):
        return None
        
    def processGameState(self):
        # processGameState method is always called, but either does nothing in this default case or transforms the input for the RL-Agent cases\ 
        return None 
    
    # -- functions to process gamestate --
    def egocentric(self, variable):
        return variable[self.egoShiftIdx] 
    
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
        
        
class valueAgent0(dominoeAgent):
    def specializedInit(self):
        # create binary arrays for presenting gamestate information to the RL networks
        self.binaryPlayed = np.zeros(self.numDominoes)
        self.binaryLineAvailable = np.zeros((self.numPlayers,self.highestDominoe+1))
        self.binaryDummyAvailable = np.zeros(self.highestDominoe+1)
        self.binaryHand = np.zeros(self.numDominoes)
        
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.handValueNetwork = dnn.valueNetwork(self.numPlayers,self.numDominoes,self.highestDominoe)
        self.finalScoreNetwork = dnn.valueNetwork(self.numPlayers,self.numDominoes,self.highestDominoe)
        self.handValueNetwork.to(self.device)
        self.finalScoreNetwork.to(self.device)
        
        # Prepare Training Functions & Optimizers
        # self.handValueEligibility = [[torch.zeros(prms.shape) for prms in self.handValueNetwork.parameters()] for _ in range(self.handValueNetwork.outputDimension)]
        self.handValueLoss = nn.SmoothL1Loss()
        self.handValueOptimizer = torch.optim.SGD(self.handValueNetwork.parameters(), lr=1e-3)
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreNetwork.outputDimension)]
        # self.finalScoreLoss = nn.SmoothL1Loss()
        # self.finalScoreOptimizer = torch.optim.SGD(self.finalScoreNetwork.parameters(), lr=1e-3)
        
        # meta parameters
        self.lam = 0.5
        self.alpha = 1e-5
        self.storeSelfHandvalueError = []
        self.storeOtherHandvalueError = []
        self.trackFinalScoreError = []
        self.requireUpdates = True
        
    def resetBinaries(self):
        self.binaryPlayed[:]=0
        self.binaryLineAvailable[:]=0
        self.binaryDummyAvailable[:]=0
        self.binaryHand[:]=0
        
    # valueAgent0 uses a network model to predict: 1) the final score from the gameState at each turn (learning from the end), 2) and the current hand value of each player (using omniscient info)
    def processGameState(self):
        # vectorize game state data
        self.resetBinaries() # set binaries back to 0
        self.binaryPlayed[self.played]=1 # indicate which dominoes have been played
        for lineIdx, dominoeIdx in enumerate(self.available): 
            self.binaryLineAvailable[lineIdx,dominoeIdx]=1 # indicate which value is available on each line
        self.binaryDummyAvailable[self.dummyAvailable]=1 # indicate which value is available on the dummy line
        self.binaryHand[self.myHand]=1 # indicate which dominoes are present in hand
        
    def estimatePrestateValue(self,trueHandValue):
        # at the beginning of each turn, zero out the gradients 
        self.handValueNetwork.zero_grad()
        self.finalScoreNetwork.zero_grad()
        
        # predict prestate value and update eligibility trace
        self.valueNetworkInput = self.generateValueInput().to(self.device) # convert gameState info into proper input to network
        
        # predict V(S,w) of prestate using current weights
        self.handValueOutput = self.handValueNetwork(self.valueNetworkInput)
        self.finalScoreOutput = self.finalScoreNetwork(self.valueNetworkInput)
        
        # for handValue network, learn with vanilla L1 loss between estimate and trueHandValues
        trueHandValue = self.egocentric(trueHandValue)
        loss = self.handValueLoss(self.handValueOutput, torch.tensor(trueHandValue).float().to(self.device))
        loss.backward()
        self.handValueOptimizer.step()
        self.handValueOptimizer.zero_grad() # reset gradients
        self.storeSelfHandvalueError.append(self.handValueOutput.detach().cpu().numpy()[0] - trueHandValue[0]) # store difference in own hand value estimate and true hand value
        self.storeOtherHandvalueError.append(np.mean(self.handValueOutput.detach().cpu().numpy()[1:].sum() - trueHandValue[1:].sum())) # store average difference between total other hand value est. and true
        
        # for finalScore network, compute gradient of V(S,w) with respect to weights and add it to eligibility traces
        for idx,hvOutput in enumerate(self.finalScoreOutput):
            hvOutput.backward(retain_graph=True) # measure gradient of weights with respect to this output value
            for trace,prms in zip(self.finalScoreEligibility[idx], self.finalScoreNetwork.parameters()):
                trace *= self.lam
                trace += prms.grad
                prms.grad[:] = 0 # reset gradients for each output of finalScoreNetwork
                
        # end of estimatePrestateValue()
        return None
    
    @torch.no_grad() # don't need to estimate any gradients here, that was done in estimatePrestateValue()!
    def updatePoststateValue(self,finalScore=None):
        if finalScore is None: 
            # if final score is none, then the game hasn't ended and we should learn from the poststate value estimate
            tdError = self.finalScoreNetwork(self.valueNetworkInput) - self.finalScoreOutput
        else:
            finalScore = torch.tensor(self.egocentric(finalScore)).to(self.device)
            # if the final score is an array, then we should shift perspective and learn from the true difference in our last estimate and the actual final score
            tdError = finalScore - self.finalScoreOutput
            #print(finalScore)
            #print(self.finalScoreOutput)
        for idx,td in enumerate(tdError):
            for prmIdx,prms in enumerate(self.finalScoreNetwork.parameters()):
                assert self.finalScoreEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.finalScoreEligibility[idx][prmIdx] * td # TD(lambda) update rule
        if finalScore is not None:
            self.trackFinalScoreError.append(torch.mean(torch.abs(finalScore-self.finalScoreOutput)).to('cpu'))
        return None
    
    def generateValueInput(self):
        valueNetworkInput = torch.tensor(np.concatenate((self.binaryHand, self.binaryPlayed, self.binaryLineAvailable.flatten(), self.binaryDummyAvailable, 
                                            self.handsize, self.cantplay, self.didntplay, self.turncounter, np.array(self.dummyPlayable).reshape(-1)))).float()
        return valueNetworkInput
    
    def updateModel(self, trueHandValue):#, currentScoreEstimate, previousScoreEstimate):
        # Vanilla L1 loss for hand value network
        loss = self.handValueLoss(self.handValueOutput, torch.tensor(self.egocentric(trueHandValue)).float().to(self.device))
        loss.backward()
        self.handValueOptimizer.step()
        self.handValueOptimizer.zero_grad() # reset gradients
        self.storeError.append(np.mean(np.abs(self.handValueOutput.detach().cpu().numpy() - trueHandValue))) 
        
        # TD-Lambda for final score network (with only 1 update at end of model)
#         self.finalScoreOutput.backward() # add gradient of current prediction to weights
#         for p in self.finalScoreNetwork.parameters():
#             p.grad *= self.lam
        
        # # TD-Lambda for final score network (I think this is a bit different from TD-Lambda, but it's close?)
        # for p in self.finalScoreNetwork.parameters():
        #     p.grad *= self.lam
        # loss = self.finalScoreLoss(self.valueNetworkOutput, target)
        # loss.backward()
        # self.optimizer.step()
        # TD-Lambda for final score network
        # t=0: 
        #         measure jacobian of scoreEstimate with respect to weights
        # at t=1: 
        #         measure newJacobian of scoreEstimate with respect to weights
        #         measure newScoreEstimate (if t=end, use true score)
        #         multiply alpha * dot(scoreEstimateDifference, last jacobian)
        #         update weights
        #         
        return None
    
                
    
        
        
        
        
        
        
        