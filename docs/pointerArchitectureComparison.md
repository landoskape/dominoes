# Documentation: Pointer Network Architecture Comparison

This documentation file shows the results of an experiment in which I compare
different architectures for the pointer attention layer. I use the same toy 
problem as in [pointer demonstration](pointerDemonstration.md), but train the
networks with the REINFORCE algorithm rather than supervised learning. 

I'll start by explaining the reinforcement learning setup, then explain the 
different architectures of the pointer attention layer. After that, I'll show
the results and analysis of the networks performance and mechanisms of solving
the task. 

You can run this experiment yourself with the following command. The default
parameters will train 5 networks for each architecture, so this takes about an
hour on my computer. For a faster test, use the argument ``--num-runs 1``.
```
python experiments/pointerArchitectureComparison.py
```

## REINFORCE
The task is identical to the previous experiment in 
[pointer demonstration](pointerDemonstration.md), apart from one small 
detail. The network receives a representation of a set of dominoes, and has to
sort them with a pointer network by generating a sequence of indices to the 
dominoes in its "hand" from highest dominoe value to lowest. See the pointer
demonstration documentation for more explanation. 

In the demonstration, I used supervised learning to train the network. Here, I
use the REINFORCE algorithm. Briefly, the training process performs gradient 
ascent on a term called $J(\theta)$:

$$\large J(\theta) = \mathbb{E}[\sum_{t=0}^{T-1} r_{t+1} | \pi_\theta]$$

Where $J(\theta)$ represents the expected value of reward over the course of a
"rollout" from timesteps $t=0$ to $t=T-1$ given the policy $\pi_\theta$. The
gradient of $J(\theta)$ with respect to the policy is:

$$\large \nabla_{\theta}J(\theta) = \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t)G_t $$

Where $G_t$ represents the net discounted reward from time $t$ into the 
future:

$$\large G_t = \sum_{t'=t+1}^{T-1} \gamma^{t'-t-1}r_{t'}$$

Note: thanks to the excellent Medium 
[article](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63) 
written by [Chris Yoon](https://medium.com/@thechrisyoon) for helping me learn
this. 

### Rewards
As in any reinforcement learning problem, the reward needs to be well-defined
and chosen well. For this task, I assign a reward of $1$ if the agent chooses 
a dominoe that is less than or equal to the value of the previous dominoe and 
if that dominoe has not been chosen yet. Otherwise, the reward is $-1$. This
way, the agent maximizes total reward in a rollout by playing the dominoes in 
decreasing order of value. Note: unlike the supervised learning method, in 
which dominoes of equal value _have_ to be played in the same order each time,
this RL setup affords flexibility and the order no longer matters for equal 
value dominoes. 

### Training
To train the network, I used the Adam optimizer with $lr=1e^{-3}$ and L2 
regularization with $\lambda=1e^{-5}$. The $J(\theta)$ term is flipped in sign
so the PyTorch gradient descent algorithm effectively performs gradient ascent
on this problem. 

### Testing
Just like in the "demonstration" toy problem, training is done with held out
dominoes, which are replaced for testing. This checks generalization 
performance and confirms that the networks are really solving the intended 
problem and not just memorizing the training data. 

## New Pointer Attention Architectures
As far as I can tell, the only pointer attention layer that is ever used in 
the literature is the one introduced in 
[this paper](https://arxiv.org/pdf/1409.0473.pdf) and used in the original
[paper](https://papers.nips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)
on pointer networks. Here, I introduce four new architectures (one of which
has three variants). I'll refer to them as the "pointer layer" throughout.

#### Inputs to Pointer Layer
Before the pass through the pointer layer, the full network generates three
tensors. First is the `encoded` tensor, containing the encoded representation
of each token in a batch. For example, if the batch size is 512, the maxmimum
number of tokens is 12, and the embedding dimension is 256, then the `encoded`
tensor will have shape `(512, 12, 256)`. Second, is the `context` tensor, 
which characterizes the entire set of tokens per batch element. There is a 
single `context` tensor per batch element with the same embedding dimension.
Finally, there is an `output` tensor, which represents the last token chosen
by the network. This can either be a weighted average of tokens or a greedy 
representation of whatever token was chosen. 

Let $e = \text{encoded}$, $c = \text{context}$, and $o = \text{output}$.

### Standard Pointer Layer
The standard pointer layer projects the `encoded` and `context` tensors to a
new space, adds them together (with broadcasting), then projects them onto an
"attention" vector aftering passing them through a hyperbolic tangent 
nonlinearity. That looks like this:

$$\large u_i = v^T \tanh (W_1 e_i + W_2 c)$$

### Pointer "Dot" Layer
The pointer dot layer also projects the `encoded` and `context` tensors to a 
new space, but then takes the dot product between each projected `encoded` 
vector and the projected `context` vector. This skips the tanh and projection
onto $v^T$. Because the nonlinearity is dropped, a `LayerNorm` is used on the
`encoded` and `context` representation before the dot product is taken. 

$$\large u_i = LN(W_1 e_i) \cdot LN(W_2 c) $$














## Results

The main result of the problem is shown here: 
![pointer toy figure](media/pointerDemonstration.png)

As you can see, the network quickly learns to sort dominoes by their value 
effectively. Pretty cool! There are two key observations to make that I'd like
to discuss: the lack of an increase in loss for the testing phase and the loss
spikes during the training phase. 

### Test loss is similar to final training loss
The testing loss is essentially identical to the final training loss. This is 
despite the fact that the network is all of a sudden seeing new dominoes 
during the testing phase (see above for explanation of the hold out 
procedure). What this means is that the network doesn't simply learn a look up
table between input and output. Instead it really learns exactly what the 
task is: sort dominoes based on the sum of their value.

### Loss spikes during training
During the training phase, there are some pretty large spikes in the loss. 
What is going on? To understand why those loss spikes occur, take a look at 
the following figure. It shows a few loss-spikes in the top panel (as 
identified by scipy's find-peaks method), with the loss-spike triggered 
average of the error as a function of output sequence position in the bottom
panel.

![pointer loss spike](media/pointerDemonstration_lossSpike.png)

The error is defined as the average difference in the max score and the target
score for each position. (The score is the log-probability for each option, 
and the max score is the one that the model "chose" for each step based on its
greedy policy).

Right before the loss spike, there is a small increase in error for the middle
positions (purple to brown on the colormap). These same positions are the ones
that spike in error when the loss gets very high. Interestingly, the early and
late positions are somewhat immune from this spike in error. Why would this
be?

In this next figure, I'm plotting the "confidence" (e.g. the max probability) 
of each choice as a function of sequence position. As before, it's a
loss-spike triggered average of the same epochs. On the right is the baseline
confidence for each position. 

![pointer loss spike confidence](media/pointerDemonstration_lossSpike_confidence.png)

What this indicates is that the network is less confident about later output
positions, and least confident about the middle positions. I conclude two
things from this:

- The pointer network architecture may accumulate uncertainty if it is trained
  on problems that require it to produce long output sequences. Alternatively,
  the pointer network might _specifically be bad_ at producing a high
  confidence "point" towards intermediary positions in a sequence.
- Additionally, in this particular problem, the middle choices are harder,
  because some dominoes have the same value but have to be sorted in a certain
  order. See the [target](###Target) section above for more explanation.

These two points probably explain how the loss spikes occur -- sometimes the
network is tasked with sorting a particularly challenging set of dominoes. 
When that happens, the networks architectural challenges lead it to get the 
output very wrong, such that a huge misleading error backpropagates through
and confuses it for the next few epochs. Interesting! Maybe there's a way to 
teach a pointer network how to filter error based on whether it's likely to be
due to these shift-based failure modes. 

Also interesting: the return to good performance is much faster after one of 
these spikes than it would be at a similar training loss after initialization.
Curious. I wonder why that's happening. 












