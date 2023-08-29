# Documentation: Gameplay

This file explains how to manage the gameplay of dominoes. Essentially, it
describes the `dominoeGame` class, found in 
[`gameplay`](../dominoes/gameplay.py), which is the primary and exclusive 
(atm) method for running a game of domiones. 

## Initialization
The dominoes gameplay object is initialized with the meta-parameters for a 
game and the agents in the game. The meta-parameters include which dominoes 
set is to be used (i.e. the highest dominoe value, usually 9s or 12s) and 
whether or not to shuffle the order of agents each hand within a game. (Score
depends on order, so unbiased estimates of agent score require shuffling - 
TODO: add link to experiment demonstrating this phenomenon). 

Assign agents to a gameplay object works in two ways: 
1. The simplest is to set a default agent with `defaultAgent=<agent_type>`
   then specify the number of players with `numPlayers=<int>`. Then, upon
   initialization, the game object will create instances of the default agent.
2. Alternatively, you can provide a list of agents with `agents=<list>`. The
   elements of the list can either contain instantiated agents or the type of
   an agent to use. Additionally, adding `None` to the list uses the default
   agent. If a list of agents and numPlayers are both provided, they need to
   agree or an exception will be raised.

Usage for a game with one greedyAgent and three dominoeAgents:
```
highestDominoe = 9
agents = (da.greedyAgent, None, None, None)
game = dg.dominoeGame(highestDominoe, agents=agents, defaultAgent=da.dominoeAgent, shuffleAgents=True, device=None)
```

## Top-level methods
1. `getAgent`: this is a convenience function for returning a particular agent
   based on their index at initialization (e.g. the greedyAgent above is idx=0
   and the dominoeAgents have indices 1,2,3). This is critical to use when
   `shuffleAgents=True`, because the gameplay object will shuffle the order of
   the agents in the `agents` list and keep track of the original agent index.
2. `numDominoeDistribution`: at the beginning of each hand, dominoes are
   assigned to each player evenly and randomly. However, if the number of
   dominoes is not divisible by the number of players, then the remainder is
   handed out to as many players as needed such that no player has 2 more
   dominoes than any other player. This method determines how many dominoes to
   assign to each player.
3. `distribute`: this method creates a list of list of indices corresponding
   to the dominoes assigned to each player for a particular hand. 


## Methods to operate a game
1. `playGame`: play game is the primary method of the gameplay object. It
   loops around hands until a game is finished. The optional input `rounds`
   determines how many hands to play within a game (the default is the highest
   dominoe+1). This method resets and keeps track of scores throughout the
   game.
2. `playHand`: play hand is used to operate a single hand. It initializes the
   hand, requests turns until the hand is no longer active, then informs the
   agents to perform final score updates (only relevant for TD-lambda agents).
3. `doTurn`: this method is the belly of the gameplay, it manages every
   process that needs to happen throughout each turn of a hand. In order:
   - It stores a copy of the current player (i.e. the index of the
     agent who's turn it is). This is necessary because the "next player"
     attribute will be updated but the current player needs to be stored.
   - Presents game state to each agent, indicating who's turn it is and that
     this is a "pre-state" game state update. Agents decide whether to process
     the game state based on whether it's their turn and if it's a pre/post
     game state update.
   - Tells agents to perform prestate value estimates. Agents decide whether
     to do so based on whether it is their turn. This is where TD-lambda
     agents estimate the current value and populate their eligibility traces
     for the TD-lambda update rule.
   - Requests a play from whatever agent is up next. For TD-lambda agents, it
     creates a "gameEngine" that is passed to each agent, allowing them to
     simulate the next game state for each possible move. Basic agents ignore
     the gameEngine input.
   - Updates the game state based on what the agent's next play was.
   - Documents the gameplay so the user can assess what happened after the
     hand is completed (these variables are reset whenever a new hand starts).
   - Inform agents if their hand was played on (important for agents that
     process line sequences because it tells them they need to update their
     list of possible sequences).
   - Tells agents to perform poststate value updates. Agents decide whether to
     do so based on whether it is their turn and if they are set to "learning"
     mode (only TD-lambda agents do this).
4. `initializeHand`: prepares all variables required for initializing a hand,
   including resetting of the meta parameters (e.g. `handActive`), shuffles
   agents if requested, and informs agents that a new hand has started.
5. `updateGameState`: update game state takes in the next play by whichever
   agents turn it was, along with a representation of the current game state,
   and updates the game state according to the rules of dominoes.
6. `documentGameplay`: given the most recent play, document gameplay stores
   the move in a set of vectors used for keeping track of what happened
   throughout a game.
7. `printResults`: print the results of a game, including the score on each
   hand, the final score at the end, and details about the winner's
   performance. 

## Methods used to communicate with agents
1. `assignDominoes`: uses the agents `serve` method to assign a list of
   dominoes the agent gets at the start of the hand.
2. `agentInitHand`: uses the agents `initHand` method to tell them to do any
   necessary methods to start a new hand.
3. `presentGameState`: uses the agents `gameState` method to inform the agent
   of the current game state. Additionally, it tells the agent who the current
   player is and whether this is a pre-state game state or a post-state
   game state so the agents can decide whether they need to process the game
   state at this time.
4. `performPrestateValueEstimate`: uses the agents `estimatePrestateValue`
   method to tell agents to perform pre-state value updates.
5. `performPoststateValueEstimate`: uses the agents `updatePoststateValue`
   method to tell agents to perform post-state value updates.
6. `performFinalScoreUpdates`: calls `updatePoststateValue` again, but forces
   an update by setting `currentPlayer=agent.agentIndex`, which makes them do
   the update (TD-lambda agents should perform a final score update regardless
   of whether they played the final move!).

   