

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
    def __init__(self):
        None
        