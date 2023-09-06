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

$$V(S_t) &larr; V(S_t) + \alpha (r_{t+1} + V(S_{t+1}) - V(S_t))$$

The term in the parentheses is called the temporal difference error because it
reflects the error in predicting the next states reward from the previous 
state ($TD_t = (r_{t+1} + V(S_{t+1}) - V(S_t))$). 

Suppose the value function is defined as a neural network $f$ with parameters 
$\theta$: $f_V(S, \theta)$. To implement an update of the value function, we 
need to determine how the parameters of the network affect the estimate of the
value. For this, we need the gradient of the value with respect to $\theta$, 
which we call the "eligibility trace", denoted $E$:

$$E = \frac{\partial}{\partial \theta} f_V(S, \theta)$$

We can't just add the eligibility trace to the networks parameters, we have to
make sure that we update the parameters such that the value function will 
become progressively more accurate over time. To do that, the eligibility 
trace needs to be scaled by the temporal difference error ($TD$), which makes
sure that the sign of the update is right, and also ensures that the scale of 
the update is proportional to how much error there was in the estimate. And of
course, everything is scaled by a learning rate $\alpha$. So, here we have it,
looking at the update to a specific parameter $\theta_i$:

$$\theta_i &larr; \theta_i + \alpha TD_t E_i$$
