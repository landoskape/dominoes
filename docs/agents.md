# Documentation: Agent Functionality

This file explains how to manage agents and discusses the strategies of some
agents available in the agent library. This file focuses on managing agents, 
including how to create an agent, using an agent in a game, and how standard 
agents function. (Standard agents are the hand-crafted agents that don't
require trained RL networks). 

## Features of an Agent
Every dominoe agent is a subclass of the `dominoeAgent` class found in the 
[`dominoeAgent.py`](../dominoes/agents/dominoeAgent.py) file. Dominoes agents 
contain all the methods and parameters required for playing legal moves with a
particular strategy in a dominoes game. 

### Initialization
Dominoes agents are initialized to play dominoes in matches with a pre-
specified number of players and domiones set (i.e. the highest dominoe to
play, usually 9s or 12s). Some agents can play with a range of players or 
different dominoes sets, but the RL agents are not usually coded this way due
to constraints on the input array to their networks. Dominoes agents keep
lists and arrays that represent information about the game state (all 
allocated in the `__init__` function). These are updated each time it's an 
agents turn to play, or for value agents, they can be updated each for each 
players turn. 

### Top-level methods
1. `agentParameters`: returns a dictionary of key parameters for the agent,
   primarily the number of players and the dominoes set (highest dominoe). If
   an agent has any special parameters (see below), these are added.
2. `specialParameters`: returns a dictionary of special parameters for the
   agent, coded in each subclass definition.
3. `dominoesInHand`: each agent has a list of dominoes in their hand
   (`self.myHand`), which is an index of the dominoes in the set.
   This method converts this list of indices to the values of each dominoe.
4. `updateAgentIndex`: for each hand within a game, each agent has an agent
   index indicating which "chair" the agent is sitting at a game table. Agents
   play in order (counter-clockwise in real life). The game object feeds each
   agent lists indicating the game state starting at agentIndex 0, so this
   also creates an "`egoShiftIdx`", which converts the game state to the
   agent's perspective.
5. `printHand`: convenience function for printing the dominoes in an agent's
   hand to the command prompt.

### Processing the game state
1. `serve`: receives a list of indices corresponding to the dominoes in the
   agent's hand at the beginning of a round. The game object randomly assigns
   dominoes to each agent and then "serves" it to the agent.
2. `initHand`: called whenever a new hand is initialized. This is unnecessary
   for some basic hand-crafted agents, but is overwritten and useful for more
   complex agents that have to perform strategy-specific functions at the
   beginning of each new hand.
3. `egocentric`: helper method for shifting the order of a game state array to
   to the agents egocentric perspective (with `egoShiftIdx`).
4. `linePlayedOn`: like `initHand`, this is usually not needed but is
   overwritten and called whenever an agents line is played on by a different
   agent.
5. `checkTurnUpdate`: returns `True` or `False` depending on whether an agent
   should update its representation of the game state as a function of the
   who's turn it is and whether it is a pre- or post- state update (see
   below). The simple hand-crafted agents only need to update their game state
   representation for pre-state updates when it's their turn. Value agents can
   also update on every turn (to train their value networks on more game
   states), and also on the post-state update as a way to perform intermediary
   updates to their value function (with the TD-lambda algorithm, more below).
6. `gameState`: the agent receives a list of variables describing the current
   game state from the game object. The agent loads these into its own
   representation of the game state (with appropriate shifts to egocentric
   perspective), and then performs additional processing using the
   `processGameState` method.
7. `processGameState`: this is not required for simple agents, but is
   overwritten and called for more complex agents that need to perform further
   processing on the game state.
8. `estimatePrestateValue`: specifically used for TD-lambda agents that
   perform various functions at the pre- and post- state stage to update their
   value function. This is present in the top-level agent class to simplify
   the `doTurn` function in the gameplay code.
9. `updatePoststateValue`: same as `estimatePrestateValue`, but calls
    post-state value updates for TD-lambda agents.

#### Choosing an option to play
Choosing a play is divided into several modular methods that make it easy to 
explore different strategies for agents. The `play` method should not be 
changed for most agents, as it is contains the basic requirements for playing
a dominoe. However, each of the following methods are useful for exploring 
new strategies. 
1. `play`: this method is called by the game object to request a play. The
   agent selects a dominoe to play, removes it from its own hand, and then
   returns the dominoe index and location to the game object.
2. `selectPlay`: selects which dominoe to play given the options that are
   available. Optionally accepts a "gameEngine" method which is primarily used
   by the TD-lambda agents.
3. `playOptions`: returns two paired lists indicating all legal play options.
   Locations 0-n indicate plays that are on an agent's line, with this agent
   being location 0. The dummy line is represented as location -1.
4. `optionValue`: returns the option value for each possible option. For
   simple agents, this is usually a straightforward function (for example),
   the default "`dominoeAgent`" just assigns a value of `1` to each dominoe.
5. `makeChoice`: chooses which dominoe to play given the optionValues.


## Developing Strategies
The central focus of this repository is developing strategies (i.e. a policy)
and analyzing how well they perform. This section of the documentation shows
the basic steps to develop an agent strategy by explaining the strategies used
by the hand-crafted agents in this repository. 

### Dominoe Agent
The default dominoe agent (e.g. the top-level class found in the 
[`dominoeAgent.py`](../dominoes/agents/dominoeAgent.py) file) uses the 
simplest strategy: it plays a random option out of all legal plays. To 
accomplish this, it assigns an optionValue of 1 to each legal move (see 
above), then chooses an option in proportion to the optionValue (i.e. 
randomly with a uniform distribution). 

### Greedy Agent 
Greedy agents set the option value of each legal option to the total number of
points on each legal dominoe. For example, the dominoe (3|4) will be assigned 
a value of 7. To accomplish this: it overwrites the `optionValue` function as 
follows: 
```
def optionValue(self, locations, dominoes):
    return self.dominoeValue[dominoes]
```
Then, the greedy agent simply plays whichever option has the highest value by 
overwriting the `makeChoice` function: 
```
def makeChoice(self, optionValue):
    return np.argmax(optionValue)
```

### Stupid Agent
Stupid agents work just like the greedy agent, except they play the dominoe 
with the least value by using `np.argmin` rather than `np.argmax`. 

#### Double Agent
Double agents assign an infinite value to any double option (because it 
allows) the agent to play again. Then, to any other option that isn't a 
double, the value is set to the number of points on the option, just like
for greedy agents. 
    

