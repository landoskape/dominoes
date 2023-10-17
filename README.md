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
environment. Note: I highly recommend using mamba instead of conda. The best 
way to do that is with 
[miniforge](https://github.com/conda-forge/miniforge#mambaforge) but if you 
want to use an existing conda setup then instructions are 
[here](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install).
If you are using conda instead of mamba, replace `mamba` with `conda`, they 
work identically (except mamba is faster!).

```
cd /path/to/cloned/repository
mamba env create -f environment.yml
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
mamba create -n dominoes
mamba activate dominoes
pip install <package_name> # go in order through the environment.yml file, ignore the pytorch packages

# use whatever line of code is suggested from the pytorch website:
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Documentation
In lieu of a wiki (which I found out about after writing all these md files),
this table of contents links to documentation that explains how to use this 
repository and presents analyses of the agents I have developed. 
1. Groundwork for Dominoes Package in Python
    - [Rules of the game](docs/dominoeRules.md) -- (not written yet, sorry!)
    - [Gameplay object](docs/gameplay.md)
    - Dominoe Agents (Code structure and hand-crafted policies)
        - [Anatomy of a dominoe agent](docs/agents.md)
        - [Basic policies](docs/basicPolicies.md)
    - [Multiplayer ELO System](docs/multiplayerElo.md)
    - [Basic Analysis](docs/basicAnalysis.md)
    - [Tutorials and basic usage](docs/tutorials.md)
2. [Experiments](experiments)
    - Transformer and Pointer Network Code -- (not written yet, sorry)
    - [Pointer Network Toy Problem](docs/pointerDemonstration.md)
4. Reinforcement Learning Agents
   - [The TD-Lambda Algorithm](docs/TDLambdaAgents.md)
   - TD-Lambda Agents -- (not written yet, sorry)

## Contributing
Feel free to contribute to this project by opening issues or submitting pull 
requests. I'm doing this to learn about RL and ML so suggestions, 
improvements, and collaborations are more than welcome!

## License
This project is licensed under the MIT License.
