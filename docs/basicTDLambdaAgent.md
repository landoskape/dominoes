# Documentation: Basic TD-Lambda Agent Training

The primary focus of this repository is to train deep RL models to excel at 
the game of dominoes. For now, the repository exclusively uses agents trained
with the TD-lambda algorithm. I have explored how variations in TD-lambda 
agents perform, including analyses related to different input information, 
different architectures, and different training programs. 

This documentation file explores the first few architectures and training 
programs. It starts by explaining the core of how TD-lambda agents work, then
shows some initial results of TD-lambda agent performance. 

## The TD-Lambda Algorithm
The foundation of the TD-Lambda algorithm is 
[Temporal Difference Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning).
In temporal-difference learning, a value function $V(S)$ is used to predict 
the value ($V$) of the current state ($S$). The value function is adjusted by
any rewards (or punishments) that are received ($r$), and the predicted
value of the next observed state. 

$$\large V(S_t) &larr; V(S_t) + \alpha (r_{t+1} + V(S_{t+1}) - V(S_t))$$

The term in the parentheses is called the temporal difference error because it
reflects the error in predicting the next states reward from the previous 
state. It is denoted as $\delta$:

$$\large \delta_t = (r_{t+1} + V(S_{t+1}) - V(S_t))$$ 

Suppose the value function is defined as a neural network $f$ with parameters 
$\theta$: 

$$\large V(S) = f_V(S, \theta)$$ 

How does one implement an update of the value function? To do so, we need to
determine how the parameters of the network affect the estimate of the value.
For this, we need the gradient of the value network with respect to the 
parameters $\theta$:

$$\large \nabla_{\theta}f_V(S, \theta)$$

However, in autocorrelated games like dominoes, it makes sense to keep track
of how the parameters have been influencing the estimate of the final score
throughout each hand. This value is a temporally discounted accumulation of
gradients that is referred to as the eligibility trace, because it represents
the "eligibility" of each parameter to be updated by temporal difference 
errors. The eligibility trace is denoted $Z$ and is measured as follows:

$$\large Z_t = \sum_{k=1}^{T}\lambda^{t-k}\nabla_{\theta}f_V(S_k, \theta)$$

Fortunately, this equation is recursive so can be updated each time step 
without recomputing the gradients of all past time steps as follows:

$$\large Z_{t+1} = \lambda Z_t + \nabla_{\theta}f_V(S_t, \theta)$$

We can't just add the eligibility trace to the networks parameters, we have to
make sure that we update the parameters such that the value function will 
become progressively more accurate over time. To do that, the eligibility 
trace needs to be scaled by the temporal difference error ($\delta$), which 
makes sure that the sign of the update is right, and also ensures that the 
scale of the update is proportional to how much error there was in the 
estimate. And of course, everything is scaled by a learning rate $\alpha$. So,
here we have it, looking at the update to a specific parameter $\theta_i$, 
associated with its own elgibility trace $Z_i$:

$$\large \theta_i &larr; \theta_i + \alpha \delta Z_i$$

## Application of TD-Lambda Learning to Dominoes
In a game of dominoes, the goal of the game is to end each hand with as few
points as possible. Therefore, an agent's value function is defined as its
estimate of its final score at the end of the game (the sum of the points in
its hand when a player goes out - see the [rules](dominoeRules.md)). Final 
score is denoted by $R_{final}$. 

Following the convention of the influential 
[TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon) model of TD-Lambda 
learning, the temporal difference is defined in two different ways depending 
on the game state. 
- If the hand is not over, then the temporal difference is 
  defined as the difference in the models prediction of the final score before
  and after a turn occurs. The model prediction before and after a turn occurs 
  are referred to, respectively, as the pre-state and the post-state model 
  prediction. For mathematical notation, I will refer to these as $f_V(S_t)$ for
  pre-state and $f_V(S_{t+})$ for post-state.

$$\large \text{if hand is not over:} \hspace{30pt}
\delta_t = f_V(S_{t+}, \theta) - f_V(S_t, \theta)$$

- If the hand is over, then the temporal difference is defined as the
  difference between the true final score ($R_{final}$) and the model
  prediction from the previous game state.
  
$$\large \text{if hand is over:} \hspace{30pt}
\delta_t = R_{final} - f_V(S_t, \theta)$$



## Features of TD-Lambda Agents
Here, I explain the code (and algorithm) underlying how TD-lambda agents
learn to play dominoes. This section provides an overview with a few key
details; for futher information see the [code](../dominoes/agents/tdAgents.py)
defining TD-lambda agents and the 
[network architectures](../dominoes/networks.py) that are used by them. 








