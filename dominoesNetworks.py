import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
import dominoesFunctions as df

class lineRepresentationNetwork(nn.Module):
    def __init__(self, numPlayers, numDominoes, highestDominoe, weightPrms=(0.,0.1),biasPrms=0.,actFunc=F.relu,pDropout=0):
        super().__init__()
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.numOutputCNN = 1000
        self.inputDimension = 2*numDominoes + (highestDominoe+1)*(numPlayers+1) + 4*numPlayers + 1 + self.numOutputCNN # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = 1
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
        
        self.cnnLayer = nn.Sequential(self.cnn_c1, nn.ReLU(), nn.Flatten(start_dim=0), self.cnn_f1, nn.ReLU(), self.cnn_ln)
        
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
        
        self.ffLayer = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3, nn.ReLU(), self.fc4)

    def forward(self, x):
        cnnOutput = self.cnnLayer(x[0])
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
    def __init__(self,numPlayers,numDominoes,highestDominoe,weightPrms=(0.,0.1),biasPrms=0.,actFunc=F.relu,pDropout=0):
        super().__init__()
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.inputDimension = 2*numDominoes + (highestDominoe+1)*(numPlayers+1) + 4*numPlayers + 1 # see dominoesAgents>generateValueInput() for explanation of why this dimensionality
        self.outputDimension = numPlayers
        
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
        
        # create special layers
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=pDropout)
        
    def forward(self, x):
        self.hidden1 = self.actFunc(self.fc1(x))
        self.hidden2 = self.actFunc(self.fc2(self.dropout(self.hidden1)))
        self.hidden3 = self.actFunc(self.fc3(self.dropout(self.hidden2)))
        self.output = self.fc4(self.dropout(self.hidden3))
        return self.output 
    
    def setDropout(self,pDropout):
        self.dropout.p = pDropout
    
    def getDropout(self):
        return self.dropout.p
    
    def getActivations(self,x):
        out = self.forward(x)
        activations = []
        activations.append(self.hidden1)
        activations.append(self.hidden2)
        activations.append(self.hidden3)
        activations.append(self.output)
        return activations
    
    def getNetworkWeights(self):
        netWeights = []
        netWeights.append(self.fc1.weight.data.clone().detach())
        netWeights.append(self.fc2.weight.data.clone().detach())
        netWeights.append(self.fc3.weight.data.clone().detach())
        netWeights.append(self.fc4.weight.data.clone().detach())
        return netWeights
    
    def compareNetworkWeights(self, initWeights):
        currWeights = self.getNetworkWeights()
        deltaWeights = []
        for iw,cw in zip(initWeights,currWeights):
            iw = torch.flatten(iw,1)
            cw = torch.flatten(cw,1)
            deltaWeights.append(torch.norm(cw-iw,dim=1))
        return deltaWeights
    
    
    
class handValueNetwork(nn.Module):
    """
    MLP that predicts hand value or end score from gameState on each turn in the dominoesGame
    Number of players and number of dominoes can vary, and it uses dominoesFunctions to figure out what the dimensions are
    (but I haven't come up with a smart way to construct the network yet for variable players and dominoes, so any learning is specific to that combination...)
    # --inherited-- Activation function is Relu by default (but can be chosen with hiddenactivation). 
    # --inherited-- Output activation function is identity, because we're using CrossEntropyLoss
    """
    def __init__(self,numPlayers,numDominoes,highestDominoe,weightPrms=(0.,0.1),biasPrms=0.,actFunc=F.relu):
        super().__init__()
        self.numPlayers = numPlayers
        self.numDominoes = numDominoes
        self.highestDominoe = highestDominoe
        self.inputDimension = 2*numDominoes
        self.outputDimension = 2 # my hand value, other hand value
        
        # create layers (all linear fully connected)
        numHidden = 800
        self.fc1 = nn.Linear(self.inputDimension, numHidden)
        self.fc2 = nn.Linear(numHidden, numHidden)
        self.fc3 = nn.Linear(numHidden, numHidden)
        self.fc4 = nn.Linear(numHidden, self.outputDimension)
        torch.nn.init.normal_(self.fc1.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc2.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc3.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.normal_(self.fc4.weight, mean=weightPrms[0], std=weightPrms[1])
        torch.nn.init.constant_(self.fc1.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc2.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc3.bias, val=biasPrms)
        torch.nn.init.constant_(self.fc4.bias, val=biasPrms)
        
        # create special layers
        self.actFunc = actFunc
        
    def forward(self, x):
        self.hidden1 = self.actFunc(self.fc1(x))
        self.hidden2 = self.actFunc(self.fc2(self.hidden1))
        self.hidden3 = self.actFunc(self.fc3(self.hidden2))
        self.output = self.fc4(self.hidden3)
        return self.output 
    