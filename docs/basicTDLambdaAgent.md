# Documentation: Basic TD-Lambda Agent Training

The primary focus of this repository is to train deep RL models to excel at 
the game of dominoes. For now, the repository exclusively uses agents trained
with the TD-lambda algorithm. I have explored how various TD-lambda agents 
perform, including analyses related to differences in input structure, 
network architectures, and training programs. 

This documentation file starts by explaining the TD-lambda algorithm and how 
it is applied to dominoes, then shows how it is implemented by TD-lambda 
agents.

## The TD-Lambda Algorithm
The foundation of the TD-Lambda algorithm is 
[Temporal Difference Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning).
In temporal-difference learning, a value function $V(S)$ is used to predict 
the value ($V$) of the current state ($S$). The value function is adjusted by
any rewards (or punishments) that are received ($r$), and the predicted
value of the next observed state, with some learning rate $\alpha$ and a 
discounting of future reward $\gamma$. 

$$\large V(S) &larr; V(S) + \alpha (r + \gamma V(S') - V(S))$$

The term in the parentheses is called the temporal difference error because it
reflects the error in predicting the next states reward from the previous 
state. It is denoted as $\delta$:

$$\large \delta = (r + \gamma V(S') - V(S))$$ 

Suppose the value function is defined as a neural network $f$ with parameters 
$\theta$: 

$$\large V(S) = f_V(S, \theta)$$ 

How does one implement an update of the value function? To do so, we need to
determine how the parameters of the network affect the estimate of the value.
For this, we need the gradient of the value network with respect to the 
parameters $\theta$:

$$\large \nabla_{\theta}f_V(S, \theta)$$

In an autocorrelated games like dominoes, it makes sense to keep track of how
the parameters have been influencing the estimate of the final score 
throughout each hand. This value is a temporally discounted accumulation of
gradients that is referred to as the eligibility trace, because it represents
the "eligibility" of each parameter to be updated by temporal difference 
errors. The eligibility trace is denoted $Z$ and is measured as follows:

$$\large Z_t = \sum_{k=1}^{t}\lambda^{t-k}\nabla_{\theta}f_V(S_k, \theta)$$

Fortunately, this equation is recursive, so it can be updated each time step 
without recomputing the gradients of all past time steps as follows:

$$\large Z_{t+1} = \lambda Z_t + \nabla_{\theta}f_V(S_t, \theta)$$

Here, $\lambda$ represents the temporal discounting of past eligibility, and
is the reason that TD-Lambda gets its name. Lambda scales between zero and
one, i.e. $0\le\lambda\le1$. Higher values of lambda lead to longer-lasting
memory of how past states lead to future rewards. 

Now, we can't just add the eligibility trace to the networks parameters, we 
have to make sure that we update the parameters such that the value function 
will become progressively more accurate over time. To do that, the eligibility
trace needs to be scaled by the temporal difference error ($\delta$), which 
makes sure that the sign of the update is right, and also ensures that the 
scale of the update is proportional to how much error there was in the 
estimate. And of course, everything is scaled by a learning rate $\alpha$. So,
here we have it, looking at the update to a specific parameter $\theta_i$, 
associated with its own component of the elgibility trace $Z_i$:

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
  prediction. For mathematical notation, I will refer to these as $f_V(S_t)$
  for pre-state and $f_V(S_{t+})$ for post-state.

To choose a move, TD-lambda agents simply simulate the future game state for
each possible legal move and estimate the final score given that simulated 
future state, then pick whichever move leads to the lowest estimate of their
final score. 

$$\large \text{if hand is not over:} \hspace{30pt}
\delta_t = f_V(S_{t+}, \theta) - f_V(S_t, \theta)$$

- If the hand is over, then the temporal difference is defined as the
  difference between the true final score ($R_{final}$) and the model
  prediction from the previous game state.
  
$$\large \text{if hand is over:} \hspace{30pt}
\delta_t = R_{final} - f_V(S_t, \theta)$$



## Features of TD-Lambda Agents
Here, I explain the code that implements TD-lambda in value agents as they 
learn to play dominoes. This section provides an overview with a few key 
details; for further information see the 
[code](../dominoes/agents/tdAgents.py) defining TD-lambda agents and the 
[network architectures](../dominoes/networks.py) that are used as value 
functions.

### Overall Architecture
Following the style of the hand-crafted agents in this repository, there is a
parent class called `valueAgent` that defines the core code of any agent 
learning with the TD-lambda algorithm. This parent class is not meant to be 
used on its own. Child classes defined within the same file inherit from 
`valueAgent` and add their own rules for measuring value. I'll also mention 
that if you add a new agent, then you should make sure to add it to the 
imports [here](../dominoes/agents/__init__.py). 

### Creating a deep network to represent the value function
This repository contains several network architectures designed to represent 
the value function (all coded in pytorch). These architectures are located in
the [networks](../dominoes/networks.py) file and are created by the 
`prepareNetwork()` method of each `valueAgent`. At the time of the network 
creation, TD-lambda agents also initialize their eligibility traces with torch
tensors that have the same shape as the network parameters. 

### Preparing input to the value function
Since all network architectures are coded in pytorch, the first step of 
measuring the value of a game state is converting the game state into a torch
tensor. Like every [`dominoeAgent`](../dominoes/agents/dominoeAgent.py), 
TD-lambda agents have a method called `processGameState()` that is called by 
the `gameState()` method when the [`dominoeGame`](../dominoes/gameplay.py) 
object feeds the game state to each agent. For TD-lambda agents, this takes 
each component of the game state and converts it to a binarized tensor 
representation, then uses the additional method `prepareValueInputs()` to 
concatenate each tensor into a vector(s) that represents the input to the 
value function. 

### Measuring pre-state value estimates
Every turn, the `dominoeGame` object tells each agent to estimate the 
pre-state value ( $f_V(S)$ ) with the method `performPrestateValueEstimate()`. 
Each agent decides whether or not to estimate the pre-state value based on a
agent-method called `checkTurnUpdate()` based on who's turn it is (and also if
they are in the "learning" state, which can be updated by `setLearning()`). 

Note: I've found that TD-Lambda agents learn most effectively when they 
estimate and update their value function on every turn, this might be overkill
but it works so that's how the code is setup right now. 

Estimating pre-state value is exclusively used to measure the eligibility 
trace of the value functions's parameters ($Z$) with the gradient of the value
function ( $\nabla_{\theta}f_V(S, \theta)$ ). Therefore, agents first zero their
gradients, then estimate the final score given the current game state, and 
finally compute the gradients to update the eligibility. 

Let an agent be known as `self`, and let `finalScoreOutput` be the output of 
the value function, let `finalScoreEligibility` be the eligibility traces for
each parameter, and let `finalScoreNetwork` be the value function of each 
agent. Here is the code:

```
for idx,fsOutput in enumerate(self.finalScoreOutput):
    fsOutput.backward(retain_graph=True) # measure gradient of weights with respect to this output value
    for trace,prms in zip(self.finalScoreEligibility[idx], self.finalScoreNetwork.parameters()):
        trace *= self.lam # discount past eligibility traces by lambda
        trace += prms.grad # add new gradient to eligibility trace
        prms.grad.zero_() # reset gradients for parameters of finalScoreNetwork in between each backward call from the output
```

We enumerate through the finalScoreOutput (at the moment, this is just a 
single value representing the own agent's final score estimate, but the code 
is written so that this can be extended to represent every players final 
score). Then, we compute the gradient with respect to each output using 
`fsOutput.backward(retain_graph=True)`, making sure to retain the graph so 
that it is presevered across the for loop. Then, for each set of parameters 
(corresponding to the weights and biases of each layer of the 
`finalScoreNetwork`), we scale the previous eligibility trace by $\lambda$, 
add the new gradient, then zero the gradients so each backward call is 
independent. 












