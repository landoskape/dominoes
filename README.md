# Dominoes ML Repository

This repository contains a package for running the game of dominoes with 
python code. It contains a gameplay engine that can manage a game, a library 
of agents that play the game with different strategies, and a league manager, 
which is used to manage a group of agents that play games with each other. 

I developed the repository to accomplish two main goals: 
1. Create a dominoes agent that plays the game better than me, and hopefully
   better than most humans!
2. Teach myself about deep reinforcement learning tools and standard coding
   practices. 

## Requirements

This repository requires several packages that are available for download via
the standard methods, including conda or pip. First, clone this repository to 
your computer. Then, in a command window, change directory to wherever you 
cloned the repository and use the `environment.yml` file to create a new conda 
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

## Tutorials and Basic Usage

### Standard Imports
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
Start by creating a league. Specify which set of dominoes to use (e.g. the
highest dominoe and the number of players per game). 
```
# Start by creating a league
highestDominoe = 9 # Choose what the highest dominoe value is (usually 9 or 12)
numPlayers = 4 # Choose how many players per game
league = lm.leagueManager(highestDominoe, numPlayers, shuffleAgents=True, replace=False)
```

Add agents by class type (see leagueManager documentation for a full
explanation of how to add agents to a league):
```
league.addAgentType(da.bestLineAgent)
league.addAgentType(da.doubleAgent)
league.addAgentType(da.greedyAgent)
league.addAgentType(da.dominoeAgent)
league.addAgentType(da.stupidAgent)
```

Create a gameplay object from the league to specify which players will play
against each other and to operate the gameplay. Play the game and print the 
results.
```
game, leagueIndex = league.createGame()
game.playGame()
game.printResults()
```

Finally, return the results to the league manager to update ELO scores. Note: 
thank you to [Tom Kerrigan](http://www.tckerrigan.com/Misc/Multiplayer_Elo/) 
for an efficient method for multiplayer ELO. 
```
league.updateElo(leagueIndex, game.currentScore)
```

### Running a game and showing the results: 
Start by creating a game object using the default agent type. Then, play game
with a specified number of rounds. Usually, the number of rounds is equal to 
the highest dominoe plus 1 (e.g. for 9s, play from 0-9). But for training or 
statistics purposes, it is useful to set rounds to a high number.
```
highestDominoe = 9
numPlayers = 4
game = dg.dominoeGame(highestDominoe, numPlayers=numPlayers) 
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

Or, for a more verbose output, set `player` and `playNumber` as follows. This 
appends the player index and the play number to each dominoe listed, which is 
a lot of text to look at, but contains all the information needed to 
understand what happened each game. 
```
df.gameSequenceToString(game.dominoes, game.lineSequence, game.linePlayDirection, player=game.linePlayer, playNumber=game.linePlayNumber, labelLines=True)
df.gameSequenceToString(game.dominoes, game.dummySequence, game.dummyPlayDirection, player=game.dummyPlayer, playNumber=game.dummyPlayNumber, labelLines=True) 
```


## Contributing
Feel free to contribute to this project by opening issues or submitting pull 
requests. I'm doing this to learn about RL and ML so suggestions, 
improvements, and collaborations are more than welcome!

## License
This project is licensed under the MIT License.
