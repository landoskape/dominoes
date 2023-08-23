# Dominoes ML Repository

This repository contains a package for running the game of dominoes with 
python code. It has several key components, all explained in detail below.
Overall, it contains a gameplay engine that can manage a game, a library of
agents that play the game with different strategies, and a league manager, 
which is used to manage a group of agents that play games with each other. 

I developed the repository to accomplish two main goals: 
1. 

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

## Usage

`python experiment.py [-options]`

The full list of options can be found in the experiment.py file as arguments of the ArgumentParser. 

Key arguments: 

 - num-players: how many players to create in the game (currently not active!)
 - highest-dominoe: which is the highest dominoe value to use (default is 9, typically either 6, 9, or 12)
 - training/performance games: how many games of dominoes to play for each block of training or performance testing.
     - note that training and performance are each performed twice
 - training/performance rounds: how many hands to play per game for each block of training or performance testing.
     - note that the more rounds you use, the more accurate the estimate of performance is.
     - For training, games and rounds are interchangeable and generally serve as the "trainingEpochs" parameter.

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