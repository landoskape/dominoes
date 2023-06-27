import numpy as np
import itertools
import dominoesFunctions as df
import dominoesAgents as da

# Gameplay object
class dominoeGame:
    def __init__(self, numPlayers, highestDominoe, defaultAgent=da.dominoeAgent, agents=None):
        # store inputs
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
        # create list of dominoes and number of dominoes for convenience
        self.dominoes = df.listDominoes(self.highestDominoe)
        self.numDominoes = df.numberDominoes(self.highestDominoe)
        self.numDominoeDistribution()
        self.extraDominoeOffset = np.random.randint(self.numPlayers+1) # which player(s) start with extra dominoes (if there are any)
        self.handNumber = self.highestDominoe # which hand are we at? (initialize at highest dominoe always...)
        
        # create agents
        if agents is None: agents = [defaultAgent]*self.numPlayers
        if agents is not None: assert len(agents)==self.numPlayers, "number of agents provided is not equal to number of players"
        if agents is not None: assert np.all([agent.name=='dominoeAgent' for agent in agents])
        self.agents = [agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIndex) for (agentIndex,agent) in enumerate(agents)]
        
        # these are unnecessary because the math is correct, but might as well keep it as a low-cost sanity check
        assert len(self.dominoes)==self.numDominoes, "the number of dominoes isn't what is expected!"
        assert np.sum(self.dominoePerTurn)==self.numDominoes, "the distribution of dominoes per turn doesn't add up correctly!"
    
    def numDominoeDistribution(self):
        # return list of the number of dominoes each player gets per turn, either randomly at initial or shifted each time
        numberEach = int(np.floor(self.numDominoes/self.numPlayers))
        self.playersWithExtra = int(self.numDominoes - numberEach*self.numPlayers)
        self.dominoePerTurn = numberEach * np.ones(self.numPlayers,dtype=int)
        self.dominoePerTurn[:self.playersWithExtra] += 1
    
    def distribute(self):
        # randomly distribute dominoes at beginning of hand, then shift the extraDominoeOffset for fair distribution across an entire game 
        # produces a binary list of arrays of the dominoes in hand for each player
        startStopIndex = [0, *np.cumsum(np.roll(self.dominoePerTurn, self.extraDominoeOffset + (self.playersWithExtra*self.handNumber)))]
        idx = np.random.permutation(self.numDominoes) # randomized dominoe order to be distributed to each player
        assignments = [idx[startStopIndex[i]:startStopIndex[i+1]] for i in range(self.numPlayers)]
        assert np.array_equal(np.arange(self.numDominoes), np.sort(np.concatenate(assignments))), "dominoe assignments are not complete" # sanity check
        return assignments
    
    def assignDominoes(self,assignments):
        # serve dominoes to each agent
        for agent,assignment in zip(self.agents,assignments): 
            agent.serve(assignment)
            
    def initializeHand(self):
        # identify which dominoe is the first double
        idxFirstDouble = np.where(np.all(self.dominoes==self.handNumber,axis=1))[0]
        assert len(idxFirstDouble)==1, "more or less than 1 double identified as first..."
        idxFirstDouble = idxFirstDouble[0]
        assignments = self.distribute() # distribute dominoes randomly
        idxFirstPlayer = np.where([idxFirstDouble in assignment for assignment in assignments])[0][0] # find out which player has the first double
        assignments[idxFirstPlayer] = np.delete(assignments[idxFirstPlayer], assignments[idxFirstPlayer]==idxFirstDouble) # remove it from their hand
        turncounter = np.roll(np.arange(self.numPlayers), idxFirstPlayer) # assign turns
        self.assignDominoes(assignments) # serve dominoes to agents 
        
    def playHand(self): 
        # request plays from each agent until someone goes out or no more plays available
        return None
        
