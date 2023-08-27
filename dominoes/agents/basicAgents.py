import numpy as np
from .. import functions as df
from .dominoeAgent import dominoeAgent

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
        
    def makeChoice(self, optionValue):
        return np.argmax(optionValue)

    
        

