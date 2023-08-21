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
    def __init__(self, highestDominoe, numPlayers, replace=False, device=None, **metaParameters):
        self.highestDominoe = highestDominoe
        self.numPlayers = numPlayers
        self.dominoes = df.listDominoes(highestDominoe)
        self.numDominoes = len(self.dominoes)
        self.replace = replace
        self.agents = []
        self.numAgents = 0
        self.device = device if device is not None else "cuda" if torchCuda.is_available() else "cpu"
        
    def addAgents(self, agents):
        # Wrapper for "addAgent" when there are multiple pre-instantiated agents to add
        None
        
    def addAgent(self, agent):
        # This method adds a single instantiated agent to the league
        None
        
    def addAgentType(self, agentType, num2add=1):
        # This method adds a new agent to the league
        # "agentType" is a class that is a subtype of "dominoeAgent" (not an instantiated agent)
        # Can add multiple instantiations of this agent by changing the num2add parameter
        assert isinstance(num2add, int) and num2add>0, "num2add must be a positive integer"
        assert self.checkAgent(agentType) and not(isinstance(agentType,da.dominoeAgent)), "agentType must be a class definition of a dominoeAgent"
        for _ in range(num2add):
            self.agents.append(agentType(self.numPlayers, self.highestDominoe, self.dominoes, self.numDominoes, device=self.device))
            self.numAgents += 1

    def checkAgent(self, agent):
        # Supporting function to make sure that "agent" is either a instance of a dominoe agent or a class definition
        return hasattr(agent, 'className') and agent.className=='dominoeAgent'
        
    def createTable(self, numPlayers):
        # This method creates a game table using the agents in the league
        None

    def updateElo(self, gameTable, gameResults):
        # This method updates the ELO ratings of the agents in the game based on the gameResults
        None


assert (numPlayers is not None) or (agents is not None), "either numPlayers or agents need to be specified"
        if (numPlayers is not None) and (agents is not None): 
            assert numPlayers == len(agents), "the number of players specified does not equal the number of agents provided..."
        if numPlayers is None: 
            numPlayers = len(agents)
            
        self.numPlayers = numPlayers
        self.highestDominoe = highestDominoe
        self.dominoes = df.listDominoes(highestDominoe)
        self.numDominoes = len(self.numDominoes)
        self.shuffleAgents = shuffleAgents
        self.device = device if device is not None else "cuda" if torchCuda.is_available() else "cpu"
        
        # create an index for managing shuffling of agents
        self.originalAgentIndex = [idx for idx in range(self.numPlayers)]
        
        # create agents 
        if agents is None: agents = [defaultAgent]*self.numPlayers # if agents is None, set them to the default agent
        # Check that each agent is either None or a valid instance/class of a dominoeAgent
        for idx, agent in enumerate(agents): 
            if agent is not None: 
                assert hasattr(agent, 'className') and agent.className=='dominoeAgent', f"Agent {idx} is of type: {type(agent)}. You must only provide dominoeAgents!"
        agents = [agent if agent is not None else defaultAgent]
        self.agents = [None]*self.numPlayers
        for agentIdx,agent in enumerate(agents):
            if isinstance(agent, da.dominoeAgent):
                # if the agent is an instance of the dominoeAgent, then that means this agent is already instantiated
                # we need to check that they have the right parameters (i.e. numPlayers and highestDominoe), then add them to the agent manager
                assert (agent.numPlayers==numPlayers) and (agent.highestDominoe==highestDominoe), f"provided agent (agentIdx:{agentIdx}) did not have the correct number of players or dominoes"
                self.agents[agentIdx] = agent
                self.agents[agentIdx].agentIndex = agentIdx
                self.agents[agentIdx].device = device
            else:
                self.agents[agentIdx] = agent(numPlayers, highestDominoe, self.dominoes, self.numDominoes, agentIdx, device=device)
    




























