# Dominoes ML Experiments

This folder contains python scripts for running experiments with the dominoes 
package. Here, you can find a brief table of contents that explains the 
purpose of each experiment.

## Standard usage
To run these experiments, you should already have cloned the dominoes 
repository and created an environment to run it (see instructions in the top
level [README](../README.md)). Next, navigate to the wherever your repository
is cloned (e.g. where you can see the dominoes, docs, & experiments folders). 

Finally, you can run each of these experiments with the following command, 
where `<experiment_name>` is replaced with the name of the file, and optional
arguments are defined in the `ArgumentParser` for each file. 
```
python experiments/<experiment_name> --optional --arguments
```

## Experiments
This is a very brief table of contents explaining the goal of each experiment.
The point of this section is just to give a rough idea for what each file 
does. More detail on the results is explained in the documentation files (you 
can find these files listed in the "Documentation" section of the repositories 
[README](../README.md) file). And of course, you can look through the files to 
see what their code does (I think they're pretty well commented...). If you
have a question about an experiment or an idea for a new one, let me know!

### Constructing a Dominoes RL Agent
- [Basic Agent ELOs](basicAgentELOs.py)
- [Training a TD-Lambda Agent](trainValueAgent.py)
- [Performance of TD-Lambda Agents](valueAgentELOs.py)

### Pointer Network Analysis
- [Pointer Network Demonstration](pointerDemonstration.py)
- Architecture Comparisons
  - [With Reinforcement Learning](pointerArchitectureComparison.py)
  - [With RL and Uneven Batch Size](pointerArchitectureComparison_uneven.py)
  - [With Supervised Learning](pointerArchitectureComparison_SL.py)

### Additional Curiosities
- [Best Line Agent ELOs](bestLineAgentELOs.py)
- [Table Position Matters](tablePositionMatters.py)



