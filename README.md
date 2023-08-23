# Dominoes ML Repository

This repository contains a package for running the game of dominoes with 
python code. It has several key components, all explained in detail below.
Overall, it contains a gameplay engine that can manage a game, a library of
agents that play the game with different strategies, and a league manager, 
which is used to manage a group of agents that play games with each other. 

I developed the repository to accomplish two main goals: 
1. Create a dominoes agent that plays the game better than me, and hopefully
   better than most humans!
2. Teach myself about deep reinforcement learning tools and standard coding
   practices. 

## Requirements

This repository requires several packages that are available for download via
the standard methods, including conda or pip. First, clone this repository to 
your computer. Then, in a terminal, change directory to wherever you cloned
the repository and use the `environment.yml` file to create a new conda 
environment. 

```
cd /path/to/cloned/repository
conda env create -f environment.yml
```

Note: I have tested and developed this code on a Windows 10 machine so cannot 
guarantee that it works on other operating systems. I think most compatibility
issues will relate to pytorch and nvidia tools, so if the environment creation 
fails, I would recommend creating an environment called "dominoes", then adding 
each package manually. For everything above pytorch in the `environment.yml` 
file, just type `pip install <package_name>`. Then, for the pytorch/torch 
packages, use the recommended command from the 
[pytorch website](https://pytorch.org/get-started/locally/).

```
conda create -n dominoes
conda activate dominoes
pip install <package_name> # go in order through the environment.yml file, ignore the pytorch packages

# use whatever line of code is suggested from the pytorch website:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Standard usage

### Imports
The code depends on several modules written in this repository. To try all the
code examples below, first run the following import statements: 
```
import leagueManager as lm
import dominoesGameplay as dg
import dominoesAgents as da
import dominoesNetworks as dn
import dominoesFunctions as df
```

### Creating a league, running a game, updating ELO scores
Start by creating a league with a pre-specified highest dominoe (e.g. which
set to use), and the number of players per game:
```
# Start by creating a league
highestDominoe = 9 # Choose what the highest dominoe value is (usually 9 or 12)
numPlayers = 4 # Choose how many players per game
league = lm.leagueManager(highestDominoe, numPlayers, shuffleAgents=True, replace=False)
```

Add agents by class type (this is not the only way, see leagueManager for an
explanation of how to add agents to a league).:
```
league.addAgentType(da.bestLineAgent)
league.addAgentType(da.doubleAgent)
league.addAgentType(da.greedyAgent)
league.addAgentType(da.dominoeAgent)
```

Create a game table from the league which specifies which players in 
the league will play a game against each other. Create a game object from the 
table to operate the gameplay. Play the game and print the results.
```
gameTable, leagueIndex = league.createGameTable()
game = dg.dominoeGameFromTable(gameTable)
game.playGame()
game.printResults()
```

Finally, return the results to the league manager to update ELO scores.
```
league.updateElo(leagueIndex, game.currentScore)
```

### Running a game and showing the results: 
Start by creating a game object from a gameTable. Then, play game with a 
specified number of rounds. Usually, the number of rounds is equal to the 
highest dominoe plus 1 (e.g. for 9s, play from 0-9). But for training or 
statistics purposes, it is useful to set rounds to a high number.
```
game = dg.dominoeGameFromTable(gameTable) 
game.playGame(rounds=3) # Play the game 
```

Show the scores for each round: 
```
game.printResults()

# output: 
Scores for each round:
[[14 35  0 19]
 [ 8 17  0  7]
 [ 0  9  7  1]]

Final score:
[22 61  7 27]

The winner is agent: 2 with a score of 7, they went out in 2/3 rounds.
```

Then, you can display a record of the events in the gameplay with the
following lines: 
```
df.gameSequenceToString(game.dominoes, game.lineSequence, game.linePlayDirection, player=None, playNumber=None, labelLines=True)
df.gameSequenceToString(game.dominoes, game.dummySequence, game.dummyPlayDirection, player=None, playNumber=None, labelLines=True) 

output:
player 0:  [' 4|8 ', ' 8|2 ', ' 2|9 ', ' 9|9 ', ' 9|5 ', ' 5|5 ', ' 5|0 ', ' 0|4 ', ' 4|1 ', ' 1|2 ', ' 2|0 ', ' 0|1 ']
player 1:  [' 4|7 ', ' 7|7 ', ' 7|5 ', ' 5|6 ', ' 6|1 ', ' 1|9 ', ' 9|8 ', ' 8|8 ', ' 8|1 ', ' 1|3 ', ' 3|4 ']
player 2:  [' 4|6 ', ' 6|3 ', ' 3|5 ', ' 5|1 ']
player 3:  [' 4|9 ', ' 9|6 ', ' 6|6 ', ' 6|8 ', ' 8|5 ', ' 5|2 ', ' 2|6 ', ' 6|7 ', ' 7|0 ', ' 0|0 ', ' 0|6 ']
dummy:  [' 4|2 ', ' 2|3 ', ' 3|8 ', ' 8|0 ', ' 0|9 ', ' 9|3 ', ' 3|7 ', ' 7|9 ']
```

Or, for a more verbose output, set `player` and `playNumber` as follows.
```
df.gameSequenceToString(game.dominoes, game.lineSequence, game.linePlayDirection, player=game.linePlayer, playNumber=game.linePlayNumber, labelLines=True)
df.gameSequenceToString(game.dominoes, game.dummySequence, game.dummyPlayDirection, player=game.dummyPlayer, playNumber=game.dummyPlayNumber, labelLines=True) 
```


## Description

This experiment is based off a custom toolbox for running dominoes games in python. It has a "gameplay" object, which
runs a game of dominoes and manages the agents that are in the game, and "agent" objects which follow certain rules for 
playing the game. In this experiment, the focus is on the lineValueAgent, which is a TD-lambda based reinforcement 
learning agent. The experiment has two stages: first the lineValueAgent is initialized randomly and learns to play against
greedyAgents (explanation following). Then, the performance is monitored and recorded, and the lineValueAgent's network 
parameters are saved. After this, a new game is created where the same lineValueAgent now has to play against bestLine 
agents (explanation following). It trains against bestLine agents and is tested a while later. 

The idea is to demonstrate that this agent can learn to play dominoes well, and to demonstrate that it learns much better
when starting it's play against an easy opponent and then honing it's value function while playing against a much stronger
opponent. 

### greedyAgent
The greedy agent plays whatever dominoe has the highest value (e.g. the 7/9 dominoe has a higher value than the 6/9 dominoe), 
independent of which location that play is available on. 

### bestLineAgent
The bestLineAgent uses a brute-force algorithm to construct all possible legal sequences of dominoes that it can play 
starting on it's own line. This way, it knows how the dominoes "fit together", so to speak, so that it can decide what to play
based on this information. It picks a "bestLine" based on the discounted value of each dominoe in each possible sequence. 
Then, it assigns the full (discounted) value of the sequence to the first dominoe of that sequence. For every other dominoe, 
it simply measures how many points are on that dominoe (e.g., the 7/9 dominoe has 16 points). For double dominoes, which allow
you to play again, it assigns an infinite value. Then, the bestLine agent plays whatever dominoe has the highest value, 
therefore playing a double if it can, then usually it's own best line, unless a different play has more value than the entire
best sequence playing on it's own line. 

### lineValueAgent
The lineValueAgent learns a value function based on the current observable game state as well as some hand-crafted features
that help it to decide which move to play based on the way the dominoes in it's hand sequence together. In short, all possible
sequences of dominoes are computed (same as in the bestLineAgent). Then, the probability of each sequence is computed by taking
a softmax over the discounted value in the sequence minus all dominoes not in that sequence. (This way, the probability of each
line is based on the positive minus the negative value of the line). Then, each dominoe in the agents hand is associated with 
"lineFeatures" which include information about the value of each sequence it's a part of (weighted by sequence probability), 
and a few other line features that you can inspect in the agents.py file. 

To choose moves, the lineValueAgent simulates the future game state after all possible legal moves it can make (dominoes is 
deterministic) and measures the "post-state" value function. Then, it picks whichever move has the highest post-state value, which
in this case means the lowest expected number of points of the lineValue agent at the end of the hand. 

## Examples
To test the code before devoting lots of computation time: 
`python experiment.py -tg 2 -pg 2 -tr 2 -pr 2`

Then to train the model fully with the default parameters: 
`python experiment.py`

On my computer, which has a NVIDIA GeForce RTX 3060, running the code with default parameters takes about 8-9 hours. 

## Contributing
Feel free to contribute to this project by creating issues or pull requests. Suggestions and improvements are welcome!

## License
This project is licensed under the MIT License.
