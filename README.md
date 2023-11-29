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
fails, I would recommend commenting out the lines in the `environment.yml` 
file related to pytorch, (`pytorch`, `torchvision`, `torchaudio`, 
`pytorch-cuda=12.1`), creating the environment as above, then installing the 
torch packages as recommended from the  
[pytorch website](https://pytorch.org/get-started/locally/). Note that for you
to use your GPU (if it's installed), the pytorch-cuda version needs to be the 
same as whatever is installed on your computer. To figure this out, open a 
command prompt and type `nvidia-smi`. It'll show you the CUDA Version in the 
top right if it's installed. 

```
mamba create -n dominoes
mamba activate dominoes
pip install <package_name> # go in order through the environment.yml file, ignore the pytorch packages

# use whatever line of code is suggested from the pytorch website:
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Documentation
Until I learn how to build a project page, the presentation and documentation
of this repository is going to live on markdown files in the [docs](docs) 
folder. These files explain how to use this repository and present analyses of
the agents I have developed. This is a list of them with links to the file.

### Documentation for Dominoes Package and Experiments
1. Groundwork for Dominoes in Python
    - [Rules of the game](docs/dominoeRules.md) -- (not written yet, sorry!)
    - [Gameplay object](docs/gameplay.md)
    - Dominoe Agents (Code structure and hand-crafted policies)
        - [Anatomy of a dominoe agent](docs/agents.md)
        - [Basic policies](docs/basicPolicies.md)
    - [Multiplayer ELO System](docs/multiplayerElo.md)
    - [Tutorials and basic usage](docs/tutorials.md)
2. Reinforcement Learning Agents
   - [The TD-Lambda Algorithm](docs/TDLambdaAgents.md)

### Documentation for Pointer Network Experiments
1. [Toy Problem (& intro to pointer networks)](docs/pointerDemonstration.md)
2. [Novel Architecture Comparison on Toy Problem](docs/pointerArchitectureComparison.md)
3. [Tests on the Traveling Salesman Problem](docs/pointerArchComp_TSP.md)
4. [A Novel Complex Sequencing Problem](docs/pointerSequencer.md)

## Contributing
Feel free to contribute to this project by opening issues or submitting pull 
requests. I'm doing this to learn about RL and ML so suggestions, 
improvements, and collaborations are more than welcome!

## License
This project is licensed under the MIT License.
