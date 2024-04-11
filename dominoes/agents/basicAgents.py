import numpy as np
from .. import utils
from ..datasets.support import construct_line_recursive
from .dominoeAgent import dominoeAgent


# ----------------------------------------------------------------------------
# --------------------------- simple rule agents -----------------------------
# ----------------------------------------------------------------------------
class greedyAgent(dominoeAgent):
    # greedy agent plays whatever dominoe has the highest number of points
    agentName = "greedyAgent"

    def makeChoice(self, optionValue):
        return np.argmax(optionValue)

    def optionValue(self, locations, dominoes):
        return self.dominoeValue[dominoes]


class stupidAgent(dominoeAgent):
    # stupid agent plays whatever dominoe has the least number of points
    agentName = "stupidAgent"

    def makeChoice(self, optionValue):
        return np.argmin(optionValue)

    def optionValue(self, locations, dominoes):
        return self.dominoeValue[dominoes]


class doubleAgent(dominoeAgent):
    # double agent plays any double it can play immediately, then plays the dominoe with the highest number of points
    agentName = "doubleAgent"

    def makeChoice(self, optionValue):
        return np.argmax(optionValue)

    def optionValue(self, locations, dominoes):
        optionValue = self.dominoeValue[dominoes]
        optionValue[self.dominoeDouble[dominoes]] = np.inf
        return optionValue


# ----------------------------------------------------------------------------
# -------------- agents that care about possible sequential lines ------------
# ----------------------------------------------------------------------------
class bestLineAgent(dominoeAgent):
    agentName = "bestLineAgent"

    def specializedInit(self, **kwargs):
        self.inLineDiscount = 0.9
        self.offLineDiscount = 0.7
        self.lineTemperature = 1
        self.maxLineLength = 12

        self.needsLineUpdate = True
        self.useSmartUpdate = True

        self.playValue = np.sum(self.dominoes, axis=1)
        self.nonDouble = self.dominoes[:, 0] != self.dominoes[:, 1]

    def initHand(self):
        self.needsLineUpdate = True

    def linePlayedOn(self):
        # if my line was played on, then recompute sequences if it's my turn
        self.needsLineUpdate = True

    def selectPlay(self, gameEngine=None):
        # select dominoe to play, for the default class, the selection is random based on available plays
        locations, dominoes = self.playOptions()  # get options that are available
        # if there are no options, return None
        if len(locations) == 0:
            return None, None
        # if there are options, then measure their value
        optionValue = self.optionValue(locations, dominoes)
        # make choice of which dominoe to play
        idxChoice = self.makeChoice(optionValue)
        # update possible line sequences based on choice
        self.lineSequence, self.lineDirection = utils.updateLine(
            self.lineSequence, self.lineDirection, dominoes[idxChoice], locations[idxChoice] == 0
        )
        self.needsLineUpdate = False if self.useSmartUpdate else True
        # return choice to game play object
        return dominoes[idxChoice], locations[idxChoice]

    def optionValue(self, locations, dominoes):
        optionValue = self.dominoeValue[dominoes]  # start with just dominoe value
        optionValue[self.dominoeDouble[dominoes]] = np.inf  # always play a double

        # get best line etc.
        bestLine, bestLineValue = self.getBestLine()

        # if there is a best line, inflate that plays value to the full line value
        if bestLine is not None:
            idxBestPlay = np.where((locations == 0) & (dominoes == bestLine[0]))[0]
            assert len(idxBestPlay) == 1, "this should always be 1 if a best line was found..."
            optionValue[idxBestPlay[0]] = bestLineValue

        # and return list of option values
        return optionValue

    def getBestLine(self):
        if self.needsLineUpdate:
            self.lineSequence, self.lineDirection = construct_line_recursive(
                self.dominoes, self.available[0], hand_index=self.myHand, maxLineLength=self.maxLineLength
            )
            self.needsLineUpdate = False if self.useSmartUpdate else True

        # if no line is possible, return Nones
        if self.lineSequence == [[]]:
            return None, None

        # Otherwise, compute line value for each line and return best line
        numLines = len(self.lineSequence)
        lineValue = np.zeros(numLines)
        for line in range(numLines):
            lineValue[line] = self.getLineValue(self.lineSequence[line])

        # choose best line and return it (and it's line value)
        lineProbability = utils.softmax(lineValue / self.lineTemperature)
        bestLineIdx = np.argmax(lineProbability)
        return self.lineSequence[bestLineIdx], lineValue[bestLineIdx]

    def getLineValue(self, line):
        linePlayNumber = np.cumsum(self.nonDouble[line]) - 1
        lineDiscountFactor = self.inLineDiscount**linePlayNumber  # discount factor (gamma**timeStepsInFuture)
        inLineValue = lineDiscountFactor @ self.playValue[line]  # total value of line, discounted for future plays
        offDiscount = self.offLineDiscount ** (linePlayNumber[-1] if len(line) > 0 else 1)
        # total value of remaining dominoes in hand after playing line, multiplied by a discount factor
        notInSequence = list(set(self.myHand).difference(line))
        offLineValue = offDiscount * np.sum(self.playValue[notInSequence])
        return inLineValue - offLineValue

    def makeChoice(self, optionValue):
        return np.argmax(optionValue)


class persistentLineAgent(bestLineAgent):
    agentName = "persistentLineAgent"

    def specializedInit(self, **kwargs):
        super().specializedInit()
        self.hasBestLine = False  # true if there is a valid bestLine chosen, otherwise false
        self.maxLineLength = 12  # set this larger because the agent will keep a line for longer

    def initHand(self):
        super().initHand()
        self.hasBestLine = False

    def linePlayedOn(self):
        super().linePlayedOn()
        self.hasBestLine = False

    def selectPlay(self, gameEngine=None):
        # select dominoe to play, for the default class, the selection is random based on available plays
        locations, dominoes = self.playOptions()  # get options that are available
        # if there are no options, return None
        if len(locations) == 0:
            return None, None
        # if there are options, then measure their value
        optionValue = self.optionValue(locations, dominoes)
        # make choice of which dominoe to play
        idxChoice = self.makeChoice(optionValue)

        # persistent line agent only cares about whether or not it played on it's predefined "best line"
        if locations[idxChoice] == 0:
            if dominoes[idxChoice] == self.bestLine[0]:
                # if choice is on own line and dominoe matches line, update line to start on next dominoe
                if len(self.bestLine) > 1:
                    self.bestLine = self.bestLine[1:]
                else:
                    # (if line is over, require new update)
                    self.hasBestLine = False
                    self.needsLineUpdate = True
            else:
                # if choice is on own line and it's a different dominoe, require new update
                self.hasBestLine = False
                self.needsLineUpdate = True
        else:
            # if choice is on different line, and the dominoe was in the best line, do update
            if self.bestLine is not None and dominoes[idxChoice] in self.bestLine:
                self.hasBestLine = False
                self.needsLineUpdate = True

        # return choice to game play object
        return dominoes[idxChoice], locations[idxChoice]

    def optionValue(self, locations, dominoes):
        optionValue = self.dominoeValue[dominoes]  # start with just dominoe value
        optionValue[self.dominoeDouble[dominoes]] = np.inf  # always play a double

        if self.hasBestLine:
            bestLineValue = self.getLineValue(self.bestLine)
        else:
            # get best line and it's value
            self.bestLine, bestLineValue = self.getBestLine()
            self.hasBestLine = self.bestLine is not None

        # if there is a best line, inflate that plays value to the full line value
        if self.bestLine is not None:
            idxBestPlay = np.where((locations == 0) & (dominoes == self.bestLine[0]))[0]
            if len(idxBestPlay) != 1:
                print(f"idxBestPlay: {idxBestPlay}")
                print(f"Locations: {locations}")
                print(f"Dominoes: {dominoes}, bestLine: {self.bestLine}")
            assert len(idxBestPlay) == 1, "this should always be 1 if a best line was found..."
            optionValue[idxBestPlay[0]] = bestLineValue

        return optionValue
