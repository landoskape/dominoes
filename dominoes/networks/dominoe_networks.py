import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from . import transformers_modules as transformers


def get_device(tensor):
    """simple method to get device of input tensor"""
    return "cuda" if tensor.is_cuda else "cpu"


class handRepresentationNetwork(nn.Module):
    def __init__(
        self,
        numPlayers,
        numDominoes,
        highestDominoe,
        finalScoreOutputDimension,
        embedding_dim=128,
        heads=8,
        expansion=2,
        kqnorm=True,
        bias=False,
        encoding_layers=1,
        include_hand_index=True,
        weightPrms=(0.0, 0.1),
        biasPrms=0.0,
    ):
        super().__init__()
        assert finalScoreOutputDimension <= numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.input_dim = 2 * (highestDominoe + 1)
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.expansion = expansion
        self.kqnorm = kqnorm
        self.bias = bias
        self.encoding_layers = encoding_layers
        self.include_hand_index = include_hand_index

        # the hand representation network uses a transformer to process the dominoes in the agents hand
        self.embedding = nn.Linear(in_features=self.input_dim, out_features=embedding_dim, bias=bias)
        self.encodingLayers = nn.ModuleList(
            [
                transformers.TransformerLayer(embedding_dim, heads=heads, expansion=expansion, contextual=False, kqnorm=kqnorm)
                for _ in range(encoding_layers)
            ]
        )

        # then combines the transformer output with a simple feedforward network to estimate the final score
        # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.ff_input_dim = (
            embedding_dim
            + numDominoes
            + (numDominoes if self.include_hand_index else 0)
            + (highestDominoe + 1) * (numPlayers + 1)
            + 4 * numPlayers
            + 1
        )
        self.outputDimension = finalScoreOutputDimension

        # create layers (all linear fully connected)
        self.fc1 = nn.Linear(self.ff_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.outputDimension)
        torch.nn.init.normal_(self.fc1.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc2.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc3.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc4.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.constant_(self.fc1.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc2.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)

        self.ffLayer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.LayerNorm((self.fc1.out_features)),
            self.fc2,
            nn.ReLU(),
            nn.LayerNorm((self.fc2.out_features)),
            self.fc3,
            nn.ReLU(),
            nn.LayerNorm((self.fc3.out_features)),
            self.fc4,
        )

    def forwardEncoder(self, x, mask=None):
        for elayer in self.encodingLayers:
            x = elayer(x, mask=mask)
        return x

    def forward(self, x, mask=None, withBatch=False):
        # transformer requires batch dimension, so it should be =1 if withBatch=False
        if not (withBatch):
            assert x[0].size(0) == 1, "hand input has >1 batch dimension but withBatch set to false"

        # get output of transformer and condense it across tokens
        embedded = self.embedding(x[0])
        handRepresentation = self.forwardEncoder(embedded, mask=mask)  # process hand through transformer
        if mask is None:
            mask = torch.ones((embedded.size(0), embedded.size(1)), dtype=embedded.dtype).to(get_device(embedded))
        handRepresentationCondensed = torch.sum(handRepresentation * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)

        # if not using the batch dimension, squeeze it out of the transformer output
        if not (withBatch):
            handRepresentationCondensed = handRepresentationCondensed.squeeze(0)

        # then prepare feed forward input
        ffInput = torch.cat((handRepresentationCondensed, x[1]), dim=1 if withBatch else 0)
        netOutput = self.ffLayer(ffInput)
        return netOutput


class lineRepresentationNetwork(nn.Module):
    def __init__(
        self,
        numPlayers,
        numDominoes,
        highestDominoe,
        finalScoreOutputDimension,
        numOutputCNN=1000,
        weightPrms=(0.0, 0.1),
        biasPrms=0.0,
        predict_score=True,
    ):
        super().__init__()
        assert finalScoreOutputDimension <= numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.numOutputCNN = numOutputCNN
        self.inputDimension = (
            2 * numDominoes + (highestDominoe + 1) * (numPlayers + 1) + 4 * numPlayers + 1 + self.numOutputCNN
        )  # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = finalScoreOutputDimension
        self.predict_score = predict_score
        if not (self.predict_score):
            message = f"Creating lineRepresentationNetwork that predicts win probability, but output dimension is {finalScoreOutputDimension} and numPlayers is {numPlayers}"
            assert finalScoreOutputDimension == numPlayers - 1, message

        # the lineRepresentationValue gets passed through a 1d convolutional network
        # this will transform the (numDominoe, numLineFeatures) input representation into an (numOutputChannels, numLineFeatures) output representation
        # then, this can be passed as an extra input into a FF network
        # the point is to use the same weights on the representations of every single dominoe, then process these transformed representations into the rest of the network
        self.numLineFeatures = 6
        numOutputChannels = 10
        numOutputValues = numOutputChannels * numDominoes
        self.cnn_c1 = nn.Conv1d(self.numLineFeatures, numOutputChannels, 1)
        self.cnn_f1 = nn.Linear(numOutputValues, self.numOutputCNN)
        self.cnn_ln = nn.LayerNorm((self.numOutputCNN))  # do layer normalization on cnn outputs -- which will change in scale depending on number

        # create ff network that integrates the standard network input with the convolutional output
        self.fc1 = nn.Linear(self.inputDimension, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, self.outputDimension)
        torch.nn.init.normal_(self.fc1.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc2.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc3.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc4.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.constant_(self.fc1.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc2.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)

        self.ffLayer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.LayerNorm((self.fc1.out_features)),
            self.fc2,
            nn.ReLU(),
            nn.LayerNorm((self.fc2.out_features)),
            self.fc3,
            nn.ReLU(),
            nn.LayerNorm((self.fc3.out_features)),
            self.fc4,
        )

    def cnnForward(self, x, withBatch=False):
        x = F.relu(self.cnn_c1(x))
        x = x.view(x.size(0), -1) if withBatch else x.view(-1)
        x = self.cnn_ln(F.relu(self.cnn_f1(x)))
        return x

    def forward(self, x, withBatch=False):
        cnnOutput = self.cnnForward(x[0], withBatch=withBatch)
        ffInput = torch.cat((cnnOutput, x[1]), dim=1 if withBatch else 0)
        netOutput = self.ffLayer(ffInput)
        if self.predict_score:
            # then just use network output
            return netOutput
        else:
            # if not predicting score, convert to win probability
            return torch.sigmoid(netOutput)


class lineRepresentationNetworkSmall(nn.Module):
    def __init__(
        self,
        numPlayers,
        numDominoes,
        highestDominoe,
        finalScoreOutputDimension,
        numOutputCNN=10,
        weightPrms=(0.0, 0.1),
        biasPrms=0.0,
        predict_score=True,
    ):
        super().__init__()
        assert finalScoreOutputDimension < numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.numOutputCNN = numOutputCNN
        self.inputDimension = (
            2 * numDominoes + (highestDominoe + 1) * (numPlayers + 1) + 4 * numPlayers + 1 + self.numOutputCNN
        )  # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = finalScoreOutputDimension
        self.predict_score = predict_score
        if not (self.predict_score):
            assert (
                finalScoreOutputDimension == numPlayers - 1
            ), f"Creating lineRepresentationNetwork that predicts win probability, but output dimension is {finalScoreOutputDimension} and numPlayers is {numPlayers}"

        # the lineRepresentationValue gets passed through a 1d convolutional network
        # this will transform the (numDominoe, numLineFeatures) input representation into an (numOutputChannels, numLineFeatures) output representation
        # then, this can be passed as an extra input into a FF network
        # the point is to use the same weights on the representations of every single dominoe, then process these transformed representations into the rest of the network
        self.numLineFeatures = 6
        numOutputChannels = 10
        numOutputValues = numOutputChannels * numDominoes
        self.cnn_c1 = nn.Conv1d(self.numLineFeatures, numOutputChannels, 1)
        self.cnn_f1 = nn.Linear(numOutputValues, self.numOutputCNN)
        self.cnn_ln = nn.LayerNorm((self.numOutputCNN))  # do layer normalization on cnn outputs -- which will change in scale depending on number

        # create ff network that integrates the standard network input with the convolutional output
        self.fc1 = nn.Linear(self.inputDimension, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, self.outputDimension)
        torch.nn.init.normal_(self.fc1.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc2.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc3.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc4.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.constant_(self.fc1.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc2.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)

        self.ffLayer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.LayerNorm((self.fc1.out_features)),
            self.fc2,
            nn.ReLU(),
            nn.LayerNorm((self.fc2.out_features)),
            self.fc3,
            nn.ReLU(),
            nn.LayerNorm((self.fc3.out_features)),
            self.fc4,
        )

    def cnnForward(self, x, withBatch=False):
        x = F.relu(self.cnn_c1(x))
        x = x.view(x.size(0), -1) if withBatch else x.view(-1)
        x = self.cnn_ln(F.relu(self.cnn_f1(x)))
        return x

    def forward(self, x, withBatch=False):
        cnnOutput = self.cnnForward(x[0], withBatch=withBatch)
        ffInput = torch.cat((cnnOutput, x[1]), dim=1 if withBatch else 0)
        netOutput = self.ffLayer(ffInput)
        if self.predict_score:
            return netOutput
        else:
            return torch.sigmoid(netOutput)


class valueNetwork(nn.Module):
    """
    MLP that predicts hand value or end score from gameState on each turn in the dominoesGame
    Number of players and number of dominoes can vary, and it uses dominoesFunctions to figure out what the dimensions are
    (but I haven't come up with a smart way to construct the network yet for variable players and dominoes, so any learning is specific to that combination...)
    # --inherited-- Activation function is Relu by default (but can be chosen with hiddenactivation).
    # --inherited-- Output activation function is identity, because we're using CrossEntropyLoss
    """

    def __init__(self, numPlayers, numDominoes, highestDominoe, finalScoreOutputDimension, weightPrms=(0.0, 0.1), biasPrms=0.0):
        super().__init__()
        assert finalScoreOutputDimension < numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.inputDimension = (
            2 * numDominoes + (highestDominoe + 1) * (numPlayers + 1) + 4 * numPlayers + 1
        )  # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = finalScoreOutputDimension

        # create layers (all linear fully connected)
        self.fc1 = nn.Linear(self.inputDimension, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, self.outputDimension)
        torch.nn.init.normal_(self.fc1.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc2.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc3.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc4.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.constant_(self.fc1.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc2.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)

        self.ffLayer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.LayerNorm((self.fc1.out_features)),
            self.fc2,
            nn.ReLU(),
            nn.LayerNorm((self.fc2.out_features)),
            self.fc3,
            nn.ReLU(),
            nn.LayerNorm((self.fc3.out_features)),
            self.fc4,
        )

    def forward(self, x, withBatch=None):
        return self.ffLayer(x)
