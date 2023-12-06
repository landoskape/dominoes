import random
import torch.cuda as torchCuda
from . import gameplay as dg
from .agents import dominoeAgent
from . import utils

# league manager is an object that can contain a mutable list of dominoes agents
# the agents should be instantiated objects that are all instances of the dominoeAgent class
# there must be >=numPlayers (a meta parameter)
# hopefully I can use pointers so the gameplay object can "point" to the agents in the agent manager
# the league manager will create an agentManager object for every game
# then the gameplay object will load the agentManager and play a game
# the gameplay object will return the score and winner back to the league manager
# then the league manager will update ELOs and 
# I want to equip the agent manager with some methods for measuring the ELO of each agent 
class leagueManager:
    def __init__(self, highestDominoe, numPlayers, shuffleAgents=True, elo_k=32, elo_base=1500, device=None):
        self.highestDominoe = highestDominoe
        self.numPlayers = numPlayers
        self.dominoes = utils.listDominoes(highestDominoe)
        self.numDominoes = len(self.dominoes)
        self.shuffleAgents = shuffleAgents
        self.elo_k = elo_k
        self.elo_base = elo_base
        self.agents = []
        self.elo = []
        self.numAgents = 0
        self.device = device if device is not None else "cuda" if torchCuda.is_available() else "cpu"
        
    def addAgents(self, agentList):
        # Wrapper for "addAgent" when there are multiple pre-instantiated agents to add
        for idx, agent in enumerate(agentList): 
            # Check if all agents in list are valid first
            assert self.checkAgent(agent), f"Agent #{idx} is not a dominoe agent"
            assert isinstance(agent, dominoeAgent), f"Agent #{idx} must be an instantiated object of a dominoe agent"
            assert self.checkParameters(agent), f"Agent #{idx} has the wrong game parameters (either numPlayers or highestDominoe)"
        agentIndex = []
        for agent in agentList:
            # Then once you know they are valid, add all of them (double assertions are worth it for expected behavior)
            cIndex = self.addAgent(agent)
            agentIndex.append(cIndex[0])
        
    def addAgent(self, agent):
        # This method adds a single instantiated agent to the league
        assert self.checkAgent(agent), "agent is not a dominoe agent"
        assert isinstance(agent,dominoeAgent), "agent must be an instantiated object of a dominoe agent"
        assert self.checkParameters(agent), "agent has the wrong game parameters (either numPlayers or highestDominoe"
        agent.device = self.device # update device of agent
        self.agents.append(agent)
        self.elo.append(self.elo_base)
        agentIndex = [self.numAgents]
        self.numAgents += 1
        return agentIndex
        
    def addAgentType(self, agentType, num2add=1):
        # This method adds a new agent to the league
        # "agentType" is a class that is a subtype of "dominoeAgent" (not an instantiated agent)
        # Can add multiple instantiations of this agent by changing the num2add parameter
        assert isinstance(num2add, int) and num2add>0, "num2add must be a positive integer"
        assert self.checkAgent(agentType), "agentType is not a dominoe agent"
        assert not(isinstance(agentType,dominoeAgent)), "agentType must be a class definition of a dominoeAgent, not an instantiated object"
        agentIndex = [i for i in range(self.numAgents,self.numAgents+num2add)]
        for _ in range(num2add):
            self.agents.append(agentType(self.numPlayers, self.highestDominoe, self.dominoes, self.numDominoes, device=self.device))
            self.elo.append(self.elo_base)
            self.numAgents += 1
        return agentIndex

    def checkAgent(self, agent):
        # Supporting function to make sure that "agent" is either a instance of a dominoe agent or a class definition
        return hasattr(agent, 'className') and agent.className=='dominoeAgent'

    def checkParameters(self, agent):
        # for instantiated agents, check that their metaparameters are consistent with the league parameters
        return (self.highestDominoe in agent.highestDominoeRange) and (self.numPlayers in agent.numPlayerRange)
    
    def createGame(self):
        leagueIndex = random.sample(range(self.numAgents), k=self.numPlayers)
        agentList = [self.agents[idx] for idx in leagueIndex]
        gameTable = dg.dominoeGame(self.highestDominoe, agents=agentList, shuffleAgents=self.shuffleAgents, device=self.device)
        return gameTable, leagueIndex

    def updateElo(self, leagueIndex, gameResults):
        # This method updates the ELO ratings of the agents in the game based on the gameResults
        updates = [0]*len(leagueIndex)
        idxScoreOrder = utils.argsort(gameResults) # scores in order
        sLeagueIndex = [leagueIndex[i] for i in idxScoreOrder]
        sGameResults = [gameResults[i] for i in idxScoreOrder]
        for idx in range(len(sLeagueIndex)-1):
            # Retrieve ELOs
            Ra, Rb = self.elo[sLeagueIndex[idx]], self.elo[sLeagueIndex[idx+1]]
            Ea = utils.eloExpected(Ra, Rb) # eloExpected score (probability of winning)
            Eb = 1 - Ea 
            rUpdate = round(utils.eloUpdate(1, Ea, self.elo_k),2) # Calculate ELO update
            updates[idx] += rUpdate # add update
            updates[idx+1] -= rUpdate
        # Go through elo updates and update elo for each agent that played
        for idx, sli in enumerate(sLeagueIndex):
            self.elo[sli] += updates[idx]

    def getAgent(self, agentIndex):
        return self.agents[agentIndex]



