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

## Documentation
In lieu of a wiki (which I found out about after writing all these md files),
this table of contents links to documentation that explains how to use this 
repository and presents analyses of the agents I have developed. 
1. [Rules of the game](docs/dominoeRules.md)
2. [Gameplay object](docs/gameplay.md)
3. Dominoe Agents (Code structure and hand-crafted policies) 
   - [Anatomy of a dominoe agent](docs/agents.md)
   - [Basic policies](docs/basicPolicies.md)
4. [Multiplayer ELO System](docs/multiplayerElo.md)
5. [Tutorials and basic usage](docs/tutorials.md)
6. [Experiments](experiments)
7. [Basic Analysis](docs/basicAnalysis.md)
8. Reinforcement Learning Agents
   - [The TD-Lambda algorithm implemented in dominoes](docs/TDLambdaAgents.md)


## Contributing
Feel free to contribute to this project by opening issues or submitting pull 
requests. I'm doing this to learn about RL and ML so suggestions, 
improvements, and collaborations are more than welcome!

## License
This project is licensed under the MIT License.
