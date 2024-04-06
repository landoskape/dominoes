import numpy as np
from .. import utils
import torch
from .tdAgents import valueAgent
from ..networks import dominoe_networks as dnn


class transformerAgent(valueAgent):
    """transformer agent uses a transformer network to process their hand"""

    agentName = "transformerAgent"

    def specializedInit(self, **kwargs):
        # transformer parameters
        self.embedding_dim = 128
        self.heads = 8
        self.expansion = 2
        self.kqnorm = True
        self.bias = False
        self.encoding_layers = 3
        self.include_hand_index = False

        # do general valueAgent initialization
        super().specializedInit(**kwargs)

        self.replay = False
        print("setting replay to false for tformer agent and not setting any of the replay variables in special init")

        # # prepare replay
        # replayDims0 = (self.finalScoreNetwork.numLineFeatures, self.numDominoes)
        # replayDims1 = (self.finalScoreNetwork.inputDimension-self.finalScoreNetwork.numOutputCNN, )
        # self.replayBufferIndex = []
        # self.replayBuffer = [torch.zeros((0, *replayDims0)).to(self.device), torch.zeros((0, *replayDims1)).to(self.device)]
        # self.replayTarget = torch.zeros((0,1)).to(self.device)
        # self.replayWeight = torch.zeros((0,1)).to(self.device)
        # self.loss = torch.nn.L1Loss(reduction='none')
        # self.optimizer = torch.optim.SGD(self.finalScoreNetwork.parameters(), lr=self.replayAlpha)

    def checkTurnUpdate(self, currentPlayer, postState=False):
        relevantTurn = True  # update every turn -- #(currentPlayer is not None) and (currentPlayer == self.agentIndex)
        relevantState = True  # update gameState for pre and post states
        return relevantTurn and relevantState

    @torch.no_grad()
    def updateReplayBuffer(self):
        if True:
            if self.replay:
                print("from updateReplayBuffer: Transformer agenet doesn't have replay coded yet")
            return None

        # Add to replay buffer probabilistically (and skip if not doing replay)
        if not (self.replay) or (np.random.rand() > self.probSaveForReplay):
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
        if True:
            if self.replay:
                print("from updateReplayTarget: Transformer agent doesn't have replay coded yet")
            return None

        # Nothing to do if we're not doing replay or if there weren't any probabilistic replays saved this hand
        if not (self.replay) or len(self.replayBufferIndex) == 0:
            return None

        if any(rbi > (self.replayTarget.size(0) - 1) for rbi in self.replayBufferIndex):
            num2add = max(self.replayBufferIndex) + 1 - self.replayTarget.size(0)
            self.replayTarget = torch.cat((self.replayTarget, torch.zeros((num2add, 1)).to(self.device)))
            self.replayWeight = torch.cat((self.replayWeight, torch.zeros((num2add, 1)).to(self.device)))

        self.replayTarget[self.replayBufferIndex] = finalScore.clone()
        self.replayWeight[self.replayBufferIndex] = 1

    def doReplayUpdate(self):
        if True:
            if self.replay:
                print("from doReplayUpdate: Transformer Agent doesn't have replay coded yet")
            return None

        if self.learning and self.replay and self.replayBuffer[0].size(0) > 0:
            self.replayBufferIndex = []  # reset so any replays to add from this hand get appended here
            for _ in range(self.replayRepetitions):
                self.optimizer.zero_grad()
                finalScoreOutput = self.finalScoreNetwork(self.replayBuffer, withBatch=True)
                finalScoreError = self.loss(finalScoreOutput, self.replayTarget)
                replayLoss = (finalScoreError * self.replayWeight).mean()
                replayLoss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.replayWeight *= self.replayLambda

    def prepareNetwork(self):
        # initialize valueNetwork
        self.finalScoreNetwork = dnn.handRepresentationNetwork(
            self.numPlayers,
            self.numDominoes,
            self.highestDominoe,
            self.finalScoreOutputDimension,
            embedding_dim=self.embedding_dim,
            heads=self.heads,
            expansion=self.expansion,
            kqnorm=self.kqnorm,
            bias=self.bias,
            encoding_layers=self.encoding_layers,
            include_hand_index=self.include_hand_index,
        )
        self.finalScoreNetwork.to(self.device)

        # Prepare Elibility Traces
        self.finalScoreEligibility = [
            [torch.zeros(prms.shape).to(self.device) for prms in self.finalScoreNetwork.parameters()] for _ in range(self.finalScoreOutputDimension)
        ]

    def prepareValueInputs(self):
        self.handRepresentationInput, self.gameStateInput = self.generateValueInput()
        self.valueNetworkInput = (self.handRepresentationInput, self.gameStateInput)

    def generateValueInput(self):
        handRepresentationInput = utils.twohotDominoe(self.myHand, self.dominoes, self.highestDominoe, withBatch=True).to(self.device)
        gameStateInput = (
            torch.tensor(
                np.concatenate(
                    (
                        self.binaryHand if self.include_hand_index else np.empty(0),
                        self.binaryPlayed,
                        self.binaryLineAvailable.flatten(),
                        self.binaryDummyAvailable,
                        self.handsize,
                        self.cantplay,
                        self.didntplay,
                        self.turncounter,
                        np.array(self.dummyPlayable).reshape(-1),
                    )
                )
            )
            .float()
            .to(self.device)
        )
        return handRepresentationInput, gameStateInput

    def simulateValueInputs(
        self, binaryHand, binaryPlayed, binaryLineAvailable, binaryDummyAvailable, handSize, cantPlay, didntPlay, turnCounter, dummyPlayable, **kwargs
    ):
        handRepresentationInput = utils.twohotDominoe(kwargs["myHand"], kwargs["dominoes"], kwargs["highestDominoe"], withBatch=True).to(self.device)
        gameStateInput = (
            torch.tensor(
                np.concatenate(
                    (
                        binaryHand if self.include_hand_index else np.empty(0),
                        binaryPlayed,
                        binaryLineAvailable.flatten(),
                        binaryDummyAvailable,
                        handSize,
                        cantPlay,
                        didntPlay,
                        turnCounter,
                        np.array(dummyPlayable).reshape(-1),
                    )
                )
            )
            .float()
            .to(self.device)
        )
        return handRepresentationInput, gameStateInput

    def makeChoice(self, optionValue):
        return np.argmin(optionValue)

    def optionValue(self, dominoe, location, gameEngine):
        # enter dominoe and location into gameEngine, return new gamestate
        # with new gamestate, estimate value
        # return final score estimate for ~self~ only
        # make choice will return the argmin of option value, attempting to bring about the lowest final score possible
        nextState = gameEngine(dominoe, location)  # enter play option (dominoe-location pair) into the gameEngine, and return simulated new gameState
        played, available, handSize, cantPlay, didntPlay, turnCounter, lineStarted, dummyAvailable, dummyPlayable = nextState  # name the outputs
        # remove dominoe from hand
        newHand = np.delete(self.myHand, self.myHand == dominoe)
        handValues = self.dominoesInHand(updateObject=False)
        simulatedValueInput = self.sampleFutureGamestate(nextState, newHand)
        with torch.no_grad():
            finalScoreOutput = self.finalScoreNetwork(simulatedValueInput)
        # return optionValue
        return finalScoreOutput.detach().cpu().numpy()

    def selectPlay(self, gameEngine):
        # first, identify valid play options
        locations, dominoes = self.playOptions()  # get options that are available
        if len(locations) == 0:
            return None, None
        # for each play option, simulate future gamestate and estimate value from it (without gradients)
        optionValue = [self.optionValue(dominoe, location, gameEngine) for (dominoe, location) in zip(dominoes, locations)]
        # make choice of which dominoe to play
        idxChoice = self.makeChoice(optionValue)
        # return choice to game play object
        return dominoes[idxChoice], locations[idxChoice]
