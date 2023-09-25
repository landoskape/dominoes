import numpy as np
from .. import functions as df
from .dominoeAgent import dominoeAgent
import re
from glob import glob
from datetime import datetime
from copy import copy
import random
import itertools
import torch
from .. import networks as dnn

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
        self.lam = 0.5
        self.alpha = 3e-5
        self.replay = True
        self.probSaveForReplay = 0.05
        self.sizeReplayBuffer = 2000
        self.replayAlpha = 3*copy(self.alpha)
        self.replayRepetitions = 1
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
        self.doReplayUpdate()
            
    def prepareNetwork(self):
        raise ValueError("It looks like you instantiated an object of the valueAgent class directly. This class is only used to provide a scaffold for complete valueAgents, see possible agents in this script!")
    
    def zeroEligibility(self):
        raise ValueError("It looks like you instantiated an object of the valueAgent class directly. This class is only used to provide a scaffold for complete valueAgents, see possible agents in this script!")    
        
    def setLearning(self, learningState):
        self.learning = learningState
    
    def setReplay(self, replayState):
        self.replay = replayState
    
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
        
    def zeroEligibility(self):
        for fseOutput in self.finalScoreEligibility:
            for fse in fseOutput:
                fse *= 0
        
    def estimatePrestateValue(self, currentPlayer=None):
        if not(self.checkTurnUpdate(currentPlayer, postState=False)): return None
        if not(self.learning): return None
        
        # at the beginning of each turn, zero out the gradients 
        self.finalScoreNetwork.zero_grad()
        
        # predict V(S,w) of prestate using current weights
        self.finalScoreOutput = self.finalScoreNetwork(self.valueNetworkInput)
        
        # compute gradient of V(S,w) with respect to weights and add it to eligibility traces
        for idx,fsOutput in enumerate(self.finalScoreOutput):
            fsOutput.backward(retain_graph=True) # measure gradient of weights with respect to this output value
            for trace,prms in zip(self.finalScoreEligibility[idx], self.finalScoreNetwork.parameters()):
                trace *= self.lam # discount past eligibility traces by lambda
                trace += prms.grad # add new gradient to eligibility trace
                prms.grad.zero_() # reset gradients for parameters of finalScoreNetwork in between each backward call from the output
            for trace,prms in zip(self.extraEligibility, self.extraParameters):
                trace *= self.lam # discount past eligibility traces by lambda
                trace += prms.grad # add new gradient to eligibility trace
                prms.grad.zero_()
        
        self.updateReplayBuffer()
                    
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
            finalScore = torch.tensor(finalScore).float().to(self.device)
            tdError = finalScore - self.finalScoreOutput
            self.updateReplayTarget(finalScore)

        for idx,td in enumerate(tdError):
            for prmIdx,prms in enumerate(self.finalScoreNetwork.parameters()):
                assert self.finalScoreEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.finalScoreEligibility[idx][prmIdx] * td # TD(lambda) update rule
            for prmIdx,prms in enumerate(self.extraParameters):
                assert self.extraEligibility[idx][prmIdx].shape == prms.shape, "oops!"
                prms += self.alpha * self.extraEligibility[idx][prmIdx] * td # TD(lambda) update rule
                    
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
    
    def saveAgentParameters(self, path, modelName=None, description=None):
        # generate parameters to save
        networkParameters = self.finalScoreNetwork.state_dict()
        agentParameters = self.agentParameters()
        if description is None: 
            modelDescription = input("Describe this model briefly: ")
        elif description==False:
            modelDescription = "lineValueNetwork parameters"
        else: 
            assert isinstance(description, str)
            modelDescription = description
        parameters = np.array([agentParameters, networkParameters, modelDescription])

        if modelName is None:
            saveDate = datetime.now().strftime('%y%m%d')
            modelName = f"lineValueAgentParameters_{saveDate}"
            
            # Handle model save number
            existingSaves = glob(str(path / modelName)+'_*.npy')
            saveNumbers = [int(re.search(r'^.*(\d+)\.npy', esave).group(1)) for esave in existingSaves]
            saveNumber = max(saveNumbers)+1 if len(saveNumbers)>0 else 0
            modelName += f'_{saveNumber}'
        else:
            assert isinstance(modelName, str), "requested model name is not a string"
            
        np.save(path / modelName, parameters) 
        return str(path / modelName)+'.npy'
        
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
    def specializedInit(self,**kwargs):
        super().specializedInit(**kwargs)
        # prepare replay 
        self.replayBufferIndex = []
        self.replayBuffer = [torch.zeros((0, self.finalScoreNetwork.inputDimension)).to(self.device), ]
        self.replayTarget = torch.zeros((0,1)).to(self.device)
        self.loss = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.finalScoreNetwork.parameters(), lr=self.replayAlpha)
        
    def checkTurnUpdate(self, currentPlayer, postState=False):
        relevantTurn = True # update every turn -- #(currentPlayer is not None) and (currentPlayer == self.agentIndex)
        relevantState = True # update gameState for pre and post states
        return relevantTurn and relevantState
    
    @torch.no_grad()
    def updateReplayBuffer(self):
        # Add to replay buffer probabilistically (and skip if not doing replay)
        if not(self.replay) or (np.random.rand() > self.probSaveForReplay): 
            return None
        
        if self.replayBuffer[0].size(0) < self.sizeReplayBuffer:
            # we haven't filled the replay buffer up, concatenate to end
            self.replayBufferIndex.append(self.replayBuffer[0].size(0))
            self.replayBuffer[0] = torch.cat((self.replayBuffer[0], self.valueNetworkInput.clone().unsqueeze(0)), dim=0)
        else:
            # pick a random replay to replace
            idx = np.random.randint(self.replayBuffer[0].size(0), size=1)[0]
            self.replayBufferIndex.append(idx)
            self.replayBuffer[0][idx] = self.valueNetworkInput.clone()
    
    @torch.no_grad()
    def updateReplayTarget(self, finalScore):
        # Nothing to do if we're not doing replay or if there weren't any probabilistic replays saved this hand
        if not(self.replay) or len(self.replayBufferIndex)==0: 
            return None
        
        if any(rbi>(self.replayTarget.size(0)-1) for rbi in self.replayBufferIndex):
            num2add = max(self.replayBufferIndex) + 1 - self.replayTarget.size(0)
            self.replayTarget = torch.cat((self.replayTarget, torch.zeros((num2add,1)).to(self.device)))
        
        self.replayTarget[self.replayBufferIndex]=finalScore.clone()
    
    def doReplayUpdate(self):
        if self.learning and self.replay and self.replayBuffer[0].size(0)>0:
            self.replayBufferIndex = [] # reset so any replays to add from this hand get appended here
            for _ in range(self.replayRepetitions):
                self.optimizer.zero_grad()
                finalScoreOutput = self.finalScoreNetwork(self.replayBuffer[0], withBatch=True)
                finalScoreError = self.loss(finalScoreOutput, self.replayTarget)
                finalScoreError.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
        
    def prepareNetwork(self):
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.valueNetwork(self.numPlayers,self.numDominoes,self.highestDominoe,self.finalScoreOutputDimension)
        self.finalScoreNetwork.to(self.device)
        
        # Prepare Training Functions & Optimizers
        self.finalScoreEligibility = [[torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)]
    
    def prepareValueInputs(self, updateObject=True):
        self.valueNetworkInput = self.generateValueInput().to(self.device)
        
    def generateValueInput(self):
        valueNetworkInput = torch.tensor(np.concatenate((self.binaryHand, self.binaryPlayed, self.binaryLineAvailable.flatten(), self.binaryDummyAvailable, 
                                            self.handsize, self.cantplay, self.didntplay, self.turncounter, np.array(self.dummyPlayable).reshape(-1)))).float()
        return valueNetworkInput    
    
    def simulateValueInputs(self, binaryHand, binaryPlayed, binaryLineAvailable, binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, dummyPlayable, **kwargs):
        return torch.tensor(np.concatenate((binaryHand, binaryPlayed, binaryLineAvailable.flatten(), binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, np.array(dummyPlayable).reshape(-1)))).float().to(self.device)
        
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
        # if torch.any(torch.isnan(finalScoreOutput)):
        #     pdb.set_trace()
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
        
        # prepare replay 
        replayDims0 = (self.finalScoreNetwork.numLineFeatures, self.numDominoes)
        replayDims1 = (self.finalScoreNetwork.inputDimension-self.finalScoreNetwork.numOutputCNN, )
        self.replayBufferIndex = []
        self.replayBuffer = [torch.zeros((0, *replayDims0)).to(self.device), torch.zeros((0, *replayDims1)).to(self.device)]
        self.replayTarget = torch.zeros((0,1)).to(self.device)
        self.loss = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.finalScoreNetwork.parameters(), lr=self.replayAlpha)
        
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

    @torch.no_grad()
    def updateReplayBuffer(self):
        # Add to replay buffer probabilistically (and skip if not doing replay)
        if not(self.replay) or (np.random.rand() > self.probSaveForReplay): 
            return None
        
        if self.replayBuffer[0].size(0) < self.sizeReplayBuffer:
            # we haven't filled the replay buffer up, concatenate to end
            self.replayBufferIndex.append(self.replayBuffer[0].size(0))
            self.replayBuffer[0] = torch.cat((self.replayBuffer[0], self.valueNetworkInput[0].clone().unsqueeze(0)), dim=0)
            self.replayBuffer[1] = torch.cat((self.replayBuffer[1], self.valueNetworkInput[1].clone().unsqueeze(0)), dim=0)
        else:
            # pick a random replay to replace
            idx = np.random.randint(self.replayBuffer[0].size(0), size=1)[0]
            self.replayBufferIndex.append(idx)
            self.replayBuffer[0][idx] = self.valueNetworkInput[0].clone()
            self.replayBuffer[1][idx] = self.valueNetworkInput[1].clone()
    
    @torch.no_grad()
    def updateReplayTarget(self, finalScore):
        # Nothing to do if we're not doing replay or if there weren't any probabilistic replays saved this hand
        if not(self.replay) or len(self.replayBufferIndex)==0: 
            return None
        
        if any(rbi>(self.replayTarget.size(0)-1) for rbi in self.replayBufferIndex):
            num2add = max(self.replayBufferIndex) + 1 - self.replayTarget.size(0)
            self.replayTarget = torch.cat((self.replayTarget, torch.zeros((num2add,1)).to(self.device)))
            
        self.replayTarget[self.replayBufferIndex]=finalScore.clone()
        
    def doReplayUpdate(self):
        if self.learning and self.replay and self.replayBuffer[0].size(0)>0:
            self.replayBufferIndex = [] # reset so any replays to add from this hand get appended here
            for _ in range(self.replayRepetitions):
                self.optimizer.zero_grad()
                finalScoreOutput = self.finalScoreNetwork(self.replayBuffer, withBatch=True)
                finalScoreError = self.loss(finalScoreOutput, self.replayTarget)
                finalScoreError.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
            
    def prepareNetwork(self):
        # initialize valueNetwork -- predicts next hand value of each player along with final score for each player (using omniscient information to begin with...) 
        self.finalScoreNetwork = dnn.lineRepresentationNetwork(self.numPlayers,self.numDominoes,self.highestDominoe,self.finalScoreOutputDimension)
        self.finalScoreNetwork.to(self.device)

        # Prepare Training Functions & Optimizers
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
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    