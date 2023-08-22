import random
import torch.cuda as torchCuda
import dominoesAgents as da
import dominoesFunctions as df

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
    def __init__(self, highestDominoe, numPlayers, shuffleAgents=True, replace=False, device=None):
        self.highestDominoe = highestDominoe
        self.numPlayers = numPlayers
        self.dominoes = df.listDominoes(highestDominoe)
        self.numDominoes = len(self.dominoes)
        self.replace = replace
        self.shuffleAgents = shuffleAgents
        self.agents = []
        self.numAgents = 0
        self.device = device if device is not None else "cuda" if torchCuda.is_available() else "cpu"
        
    def addAgents(self, agentList):
        # Wrapper for "addAgent" when there are multiple pre-instantiated agents to add
        for idx, agent in enumerate(agentList): 
            # Check if all agents in list are valid first
            assert self.checkAgent(agent), f"Agent #{idx} is not a dominoe agent"
            assert isinstance(agent, da.dominoeAgent), f"Agent #{idx} must be an instantiated object of a dominoe agent"
            assert self.checkParameters(agent), f"Agent #{idx} has the wrong game parameters (either numPlayers or highestDominoe)"
        for agent in agentList:
            # Then once you know they are valid, add all of them (double assertions are worth it for expected behavior)
            self.addAgent(agent)
        
    def addAgent(self, agent):
        # This method adds a single instantiated agent to the league
        assert self.checkAgent(agent), "agent is not a dominoe agent"
        assert isinstance(agent,da.dominoeAgent), "agent must be an instantiated object of a dominoe agent"
        assert self.checkParameters(agent), "agent has the wrong game parameters (either numPlayers or highestDominoe"
        agent.device = self.device # update device of agent
        self.agents.append(agent)
        self.numAgents += 1
        
    def addAgentType(self, agentType, num2add=1):
        # This method adds a new agent to the league
        # "agentType" is a class that is a subtype of "dominoeAgent" (not an instantiated agent)
        # Can add multiple instantiations of this agent by changing the num2add parameter
        assert isinstance(num2add, int) and num2add>0, "num2add must be a positive integer"
        assert self.checkAgent(agentType), "agentType is not a dominoe agent"
        assert not(isinstance(agentType,da.dominoeAgent)), "agentType must be a class definition of a dominoeAgent, not an instantiated object"
        for _ in range(num2add):
            self.agents.append(agentType(self.numPlayers, self.highestDominoe, self.dominoes, self.numDominoes, device=self.device))
            self.numAgents += 1

    def checkAgent(self, agent):
        # Supporting function to make sure that "agent" is either a instance of a dominoe agent or a class definition
        return hasattr(agent, 'className') and agent.className=='dominoeAgent'

    def checkParameters(self, agent):
        # for instantiated agents, check that their metaparameters are consistent with the league parameters
        return (self.highestDominoe in agent.highestDominoeRange) and (self.numPlayers in agent.numPlayerRange)
        
    def createGameTable(self, numPlayers=None):
        # This method creates a game table using the agents in the league
        numPlayers = numPlayers if numPlayers is not None else self.numPlayers
        if self.replace:
            leagueIndex = random.choices(range(self.numAgents), k=numPlayers)
        else:
            leagueIndex = random.sample(range(self.numAgents), k=numPlayers)
        agentList = [self.agents[idx] for idx in leagueIndex]
        return gameTable(self.highestDominoe, agentList, self.shuffleAgents), leagueIndex

    def updateElo(self, leagueIndex, gameResults):
        # This method updates the ELO ratings of the agents in the game based on the gameResults
        None

    def getAgent(self, agentIndex):
        assert agentIndex in self.originalAgentIndex, "requested agent index does not exist"
        idxAgent = self.originalAgentIndex.index(agentIndex)
        return self.agents[idxAgent]


class gameTable:
    def __init__(self, highestDominoe, agentList, shuffleAgents=False, device=None):
        self.numPlayers = len(agentList)
        self.highestDominoe = highestDominoe
        self.dominoes = df.listDominoes(highestDominoe)
        self.numDominoes = len(self.dominoes)
        self.shuffleAgents = shuffleAgents
        self.device = device if device is not None else "cuda" if torchCuda.is_available() else "cpu"
        self.agents = agentList
        
        # create an index for managing shuffling of agents
        self.originalAgentIndex = [idx for idx in range(self.numPlayers)]
        
        for idx, agent in enumerate(self.agents):
            assert isinstance(agent, da.dominoeAgent), f"Agent{idx} is not a dominoeAgent"
            assert (self.numPlayers in agent.numPlayerRange) and (self.highestDominoe in agent.highestDominoeRange), f"Agent{idx} has wrong number of parameters"
            agent.updateAgentIndex(idx)
    
    # ----------------
    # -- functions for managing agents --
    # ----------------
    def getAgent(self, agentIndex):
        assert agentIndex in self.originalAgentIndex, "requested agent index does not exist"
        idxAgent = self.originalAgentIndex.index(agentIndex)
        return self.agents[idxAgent]



