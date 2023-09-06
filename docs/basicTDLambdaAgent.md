# Documentation: Basic TD-Lambda Agent Training

The primary focus of this repository is to train deep RL models to excel at 
the game of dominoes. For now, the repository exclusively uses agents trained
with the TD-lambda algorithm. I have explored how variations in TD-lambda 
agents perform, including analyses related to different input information, 
different architectures, and different training programs. 

This documentation file explores the first few architectures and training 
programs. It starts by explaining the core of how TD-lambda agents work, then
shows some initial results of TD-lambda agent performance. 

## Features of TD-Lambda Agents
Here, I explain the code (and algorithm) underlying how TD-lambda agents
learn to play dominoes. This section provides an overview with a few key
details; for futher information see the [code](../dominoes/agents/tdAgents.py)
defining TD-lambda agents and the 
[network architectures](../dominoes/networks.py) that are used by them. 

### The TD-Lambda Algorithm
The foundation of the TD-Lambda algorithm is 
[Temporal Difference Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning).
In temporal-difference learning, a value function $V(S)$ is used to predict 
the value ($V$) of the current state ($S$). The value function is adjusted by
any rewards (or punishments) that are received, denoted $r$, and the predicted
value of the next observed state. 
$$V(s_t) &larr; V(s_t) + \alpha (r_t + V(S_{t+1}) - V(s_t))$$. 
