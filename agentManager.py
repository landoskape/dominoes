import torch.cuda as torchCuda

class agentManager:
    def __init__(self, highestDominoe, numPlayers=None, agents=None, shuffleAgents=False, defaultAgent=da.dominoeAgent, device=None):
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
    
    # ----------------
    # -- functions for managing agents --
    # ----------------
    def getAgent(self, agentIndex):
        assert agentIndex in self.originalAgentIndex, "requested agent index does not exist"
        idxAgent = self.originalAgentIndex.index(agentIndex)
        return self.agents[idxAgent]


