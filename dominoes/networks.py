import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from . import functions as df


class lineRepresentationNetwork(nn.Module):
    def __init__(self, numPlayers, numDominoes, highestDominoe, finalScoreOutputDimension, numOutputCNN=1000, weightPrms=(0.,0.1),biasPrms=0.,actFunc=F.relu,pDropout=0):
        super().__init__()
        assert finalScoreOutputDimension<numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.numOutputCNN = numOutputCNN
        self.inputDimension = 2*numDominoes + (highestDominoe+1)*(numPlayers+1) + 4*numPlayers + 1 + self.numOutputCNN # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = finalScoreOutputDimension
        self.actFunc = actFunc
        
        # the lineRepresentationValue gets passed through a 1d convolutional network
        # this will transform the (numDominoe, numLineFeatures) input representation into an (numOutputChannels, numLineFeatures) output representation
        # then, this can be passed as an extra input into a FF network
        # the point is to use the same weights on the representations of every single dominoe, then process these transformed representations into the rest of the network
        numLineFeatures = 6
        numOutputChannels = 10
        numOutputValues = numOutputChannels * numDominoes
        self.cnn_c1 = nn.Conv1d(numLineFeatures, numOutputChannels, 1)
        self.cnn_f1 = nn.Linear(numOutputValues, self.numOutputCNN)
        self.cnn_ln = nn.LayerNorm((self.numOutputCNN)) # do layer normalization on cnn outputs -- which will change in scale depending on number 
        
        
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
            self.fc4
        )
    
    def cnnForward(self, x, withBatch=False):
        x = F.relu(self.cnn_c1(x))
        x = x.view(x.size(0), -1) if withBatch else x.view(-1)
        x = self.cnn_ln(F.relu(self.cnn_f1(x)))
        return x
    
    def forward(self, x, withBatch=False):
        cnnOutput = self.cnnForward(x[0], withBatch=withBatch)
        ffInput = torch.cat((cnnOutput, x[1]), dim=1 if withBatch else 0)
        netOutput = self.ffLayer(torch.cat((cnnOutput,x[1])))
        return netOutput
    
class lineRepresentationNetworkSmall(nn.Module):
    def __init__(self, numPlayers, numDominoes, highestDominoe, finalScoreOutputDimension, numOutputCNN=10, weightPrms=(0.,0.1),biasPrms=0.,actFunc=F.relu,pDropout=0):
        super().__init__()
        assert finalScoreOutputDimension<numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.numOutputCNN = numOutputCNN
        self.inputDimension = 2*numDominoes + (highestDominoe+1)*(numPlayers+1) + 4*numPlayers + 1 + self.numOutputCNN # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = finalScoreOutputDimension
        self.actFunc = actFunc
        
        # the lineRepresentationValue gets passed through a 1d convolutional network
        # this will transform the (numDominoe, numLineFeatures) input representation into an (numOutputChannels, numLineFeatures) output representation
        # then, this can be passed as an extra input into a FF network
        # the point is to use the same weights on the representations of every single dominoe, then process these transformed representations into the rest of the network
        numLineFeatures = 6
        numOutputChannels = 10
        numOutputValues = numOutputChannels * numDominoes
        self.cnn_c1 = nn.Conv1d(numLineFeatures, numOutputChannels, 1)
        self.cnn_f1 = nn.Linear(numOutputValues, self.numOutputCNN)
        self.cnn_ln = nn.LayerNorm((self.numOutputCNN)) # do layer normalization on cnn outputs -- which will change in scale depending on number 
        
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
            self.fc4
        )

    def cnnForward(self, x, withBatch=False):
        x = F.relu(self.cnn_c1(x))
        x = x.view(x.size(0), -1) if withBatch else x.view(-1)
        x = self.cnn_ln(F.relu(self.cnn_f1(x)))
        return x
    
    def forward(self, x, withBatch=False):
        cnnOutput = self.cnnForward(x[0], withBatch=withBatch)
        ffInput = torch.cat((cnnOutput, x[1]), dim=1 if withBatch else 0)
        netOutput = self.ffLayer(torch.cat((cnnOutput,x[1])))
        return netOutput
    
    
        
class valueNetwork(nn.Module):
    """
    MLP that predicts hand value or end score from gameState on each turn in the dominoesGame
    Number of players and number of dominoes can vary, and it uses dominoesFunctions to figure out what the dimensions are
    (but I haven't come up with a smart way to construct the network yet for variable players and dominoes, so any learning is specific to that combination...)
    # --inherited-- Activation function is Relu by default (but can be chosen with hiddenactivation). 
    # --inherited-- Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,numPlayers,numDominoes,highestDominoe,finalScoreOutputDimension,weightPrms=(0.,0.1),biasPrms=0.):
        super().__init__()
        assert finalScoreOutputDimension<numPlayers, "finalScoreOutputDimension can't be greater than the number of players"
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.inputDimension = 2*numDominoes + (highestDominoe+1)*(numPlayers+1) + 4*numPlayers + 1 # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
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
            self.fc4
        )
        
    def forward(self, x):
        return self.ffLayer(x)
    