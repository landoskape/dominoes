from copy import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import dominoesGameplay as dg
import dominoesAgents as da
import dominoesNetworks as dn
import dominoesFunctions as df
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Sub Task 1: Train a network to predict self-hand value and other hand value based on the tokens in the hand and what's on the board
class handValueNetwork(nn.Module):
    def __init__(self, highestDominoe, doubleLayer=True, withBias=True):
        super().__init__()
        self.highestDominoe = highestDominoe
        self.dominoes = df.listDominoes(highestDominoe)
        self.numDominoes = len(self.dominoes)
        self.doubleLayer = doubleLayer

        self.inputDimension = 2 * self.numDominoes
        self.outputDimension = 3

        # create layers (all linear fully connected)
        self.fc1 = nn.Linear(self.inputDimension, self.outputDimension, bias=withBias)
        if self.doubleLayer:
            self.fc2 = nn.Linear(self.outputDimension, self.outputDimension, bias=withBias)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        if self.doubleLayer:
            out = F.relu(self.fc2(out))
        return out


# batch generation function
def generateBatch(numDominoes, dominoes, batchSize):
    numPlayed = np.random.randint(0, numDominoes + 1, batchSize)
    numInHand = np.random.randint(0, numDominoes + 1 - numPlayed, batchSize)
    playedIdx = [np.random.choice(numDominoes, numplay, replace=False) for numplay in numPlayed]
    inHandIdx = [
        np.random.choice(list(set(range(numDominoes)).difference(playidx)), numhand, replace=False)
        for (playidx, numhand) in zip(playedIdx, numInHand)
    ]
    played = torch.zeros((batchSize, numDominoes))
    inHand = torch.zeros((batchSize, numDominoes))
    for p, pi in zip(played, playedIdx):
        p[pi] = 1
    for h, hi in zip(inHand, inHandIdx):
        h[hi] = 1
    playedValue = torch.tensor([np.sum(dominoes[play == 1]) for play in played])
    inHandValue = torch.tensor([np.sum(dominoes[hand == 1]) for hand in inHand])
    outHandValue = np.sum(dominoes) - playedValue
    return played, inHand, playedValue, inHandValue, outHandValue


def subTask1(highestDominoe=12, doubleLayer=False, withBias=False, initBias=False):
    # create dominoes
    dominoes = df.listDominoes(highestDominoe)
    numDominoes = len(dominoes)

    # create network
    net = handValueNetwork(highestDominoe, doubleLayer=doubleLayer, withBias=withBias)
    net.to(device)
    if withBias and initBias:
        if doubleLayer:
            net.fc2.bias.data[1] = np.sum(dominoes)
        else:
            net.fc1.bias.data[1] = np.sum(dominoes)

    # training parameters and preallocation
    batchSize = 1000
    numIterations = 1000
    trainingLoss = torch.zeros(numIterations)
    lossFunction = nn.L1Loss()
    optimizer = torch.optim.Adadelta(net.parameters())

    for it in tqdm(range(numIterations)):
        # start by zeroing the gradients prior to the batch
        optimizer.zero_grad()

        # generate a batch input, process the values into inputs & targets
        played, inHand, playedValue, inHandValue, outHandValue = generateBatch(numDominoes, dominoes, batchSize)
        networkInput = torch.cat((played, inHand), dim=1).to(device)
        networkTarget = torch.stack((inHandValue, playedValue, outHandValue), dim=1).to(device)

        # measure network output
        networkOutput = net(networkInput)

        # measure loss and update netweork
        loss = lossFunction(networkOutput, networkTarget)
        loss.backward()
        optimizer.step()

        # store loss
        trainingLoss[it] = loss.item()

    # measure performance
    with torch.no_grad():
        testBatch = 1000
        played, inHand, playedValue, inHandValue, outHandValue = generateBatch(numDominoes, dominoes, testBatch)
        testInput = torch.cat((played, inHand), dim=1).to(device)
        testTarget = torch.stack((inHandValue, playedValue, outHandValue), dim=1).to(device)
        testOutput = net(testInput)
        loss = lossFunction(testOutput, testTarget)
        testLoss = loss.item()

    def printResults(dominoes, trainingLoss, testOutput, testTarget, net):
        print("Note that printResults in subTask1 isn't perfect right now...")
        dominoeValue = np.sum(dominoes, axis=1)
        fig, ax = plt.subplots(1, 7, figsize=(16, 4))
        ax[0].plot(range(len(trainingLoss)), trainingLoss)
        ax[1].scatter(dominoeValue, net.fc1.weight.data[0][: len(dominoes)].clone().cpu().numpy(), s=10)
        ax[2].scatter(dominoeValue, net.fc1.weight.data[1][: len(dominoes)].clone().cpu().numpy(), s=10)
        ax[3].scatter(dominoeValue, net.fc1.weight.data[2][: len(dominoes)].clone().cpu().numpy(), s=10)
        ax[4].scatter(testTarget[:, 0].clone().cpu().numpy(), testOutput[:, 0].clone().cpu().numpy())
        ax[5].scatter(testTarget[:, 1].clone().cpu().numpy(), testOutput[:, 1].clone().cpu().numpy())
        ax[6].scatter(testTarget[:, 2].clone().cpu().numpy(), testOutput[:, 2].clone().cpu().numpy())
        return fig, ax

    return net, testOutput, testTarget, testLoss, trainingLoss, printResults
