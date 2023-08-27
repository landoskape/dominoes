# Documentation: Creating a new policy

The central focus of this repository is developing strategies (i.e. a policy)
and analyzing how well they perform. This section of the documentation shows
the basic steps to develop an agent strategy by explaining the strategies used
by the hand-crafted agents in this repository.

For an analysis of how well these hand-crafted agents perform, see the 
[section](multiplayerElo.md) discussing multi-player ELO. 

## Coding a policy
Coding a new policy requires overwriting the default dominoes agent functions.
For simple agents (examples below), this only requires overwriting the methods
documented under "Choosing an option to play" in the [agents](agents.md) 
documentation. For more advanced agents (such as the TD-lambda agents), some
methods related to processing the game state will need to be overwritten too. 

In general, the only methods that need overwriting are the `optionValue` 
method, which assigns a value to each legal play, and the `makeChoice` method,
which chooses a dominoe based on the option values of each legal play.
However, `selectPlay` sometimes needs overwriting, in particular for the 
TD-lambda agents, which have to measure option value in sequence. 

## Basic policies employed by hand-crafted agents
This repository contains several hand-crafted agents that play the game with 
simple policies. This section of the documentation explains these policies and
in doing so, demonstrates how to develop a policy. 

### Dominoe Agent
The default dominoe agent (e.g. the top-level class found in the 
[`dominoeAgent.py`](../dominoes/agents/dominoeAgent.py) file) uses the 
simplest strategy: it plays a random option out of all legal plays. To 
accomplish this, it assigns an optionValue of 1 to each legal move, then 
chooses an option with Thompson sampling. (Since all options have the same
value, this means choosing randomly with a uniform distribution).

### Greedy Agent 
Greedy agents set the option value of each legal option to the total number of
points on each legal dominoe. For example, the dominoe (3|4) will be assigned 
a value of 7. To accomplish this: it overwrites the `optionValue` function as 
follows: 
```
def optionValue(self, locations, dominoes):
    return self.dominoeValue[dominoes]
```
Then, the greedy agent simply plays whichever option has the highest value by 
overwriting the `makeChoice` function: 
```
def makeChoice(self, optionValue):
    return np.argmax(optionValue)
```

### Stupid Agent
Stupid agents work just like the greedy agent, except they play the dominoe 
with the least value by using `np.argmin` rather than `np.argmax`. 

#### Double Agent
Double agents assign an infinite value to any double option (because it 
allows) the agent to play again. Then, to any other option that isn't a 
double, the value is set to the number of points on the option, just like
for greedy agents. 

## Advanced Strategy: Best-line Agent
This repository also contains a slightly more advanced agent that chooses
plays in a way that maximizes it's ability to play all of it's dominoes during
a hand based on the way they connect to each other - i.e. where each dominoe
represents the edge of a graph. It requires additional functions beyond those
that are used in the default dominoe agent. 

### Best-line Agent strategy
Identifying a set of dominoes that form a sequence of legal plays, starting at
whatever number is available to play on an agent's own line is fundamental to 
success in dominoes. This is a challenging computation, related to the 
traveling salesman problem. The best-line agent solves this with an 
abbreviated brute-force search, in which it stops sampling unique sequences 
after they reach a certain length, and in which it efficiently updates a 
precomputed list of all possible sequences rather than recomputing it every 
time. 

Once all sequences are computed, it picks the "best one" based on several 
meta-parameters. To do this, it measures the total discounted value of each
sequence based on how many turns it will take to play each dominoe in the 
sequence, then subtracts the discounted value of each dominoe in the agent's
hand that isn't in a particular sequence. 

Finally, the agent assigns the full sequence value to the first dominoe of the
"best" sequence, assigns an infinite value to any double, and assigns the 
standard dominoe value to all other dominoes (e.g. the total number of points 
on the dominoe). Then, it chooses the play with the highest value. This means 
that the bestLine agent will always play a double, then will greedily hold on 
to any dominoe in it's best "line", and only play a different dominoe if it 
has a higher value than all dominoes in it's best line. 

This strategy is good (it's the best out of all the hand-crafted agents, see
the [section](multiplayerElo.md) on ELO), but is missing several key ideas for
strong dominoes play. For example, the best sequence isn't necessarily the one 
with the highest overall value, it's more important to get rid of as many 
points as possible before the current hand is over. Suppose the best line 
agent had the dominoes (0|1)-(1|2)-(2|0)-(0|9)-(9|9) left in it's hand. 
Playing these in order would get all of it's dominoes out (assuming it has a 
0 available on it's line). However, if a different agent is likely to go out
in two turns, it should probably skip to (0|9)-(9|9) to get those points out
before the turn is over. This is something I hope to analyze in deep RL 
agents. However, letting the line discount factor be dependent on the expected
number of turns remaining might be a way to dramatically improve the agents 
performance (I might get to coding this later). 

### Meta-Parameters:
- inLineDiscount: temporal discounting for dominoes in each line
- offLineDiscount: temporal discounting for dominoes not in each line
- lineTemperature: varies how noisy the choice is for the "best line". In
  practice, this isn't really used because it only picks the line with the
  highest value rather than Thompson sampling or some other strategy...
- maxLineLength: this is the maximum number of dominoes to sequence together
  into a possible line. If this is set to a high number, then the computation
  time can become prohibitively long, especially for numPlayer/highestDominoe
  combinations that lead to each agent having many dominoes in their line.
- needsLineUpdate: boolean variable indicating whether the agent needs to
  fully recompute all possible lines it can play (set to True at the beginning
  of a new hand or if a different agent plays on this agents line).
- useSmartUpdate: boolean variable indicating whether to use the efficient
  update of all possible lines (see below).

### Measuring Line Value
The best line agent adds an intermediary step to computing option value for 
it's policy: the `dominoeLineValue` method. Line value is composed of two 
components: the inLineValue and the offLineValue. 
- inLineValue: the total number of discounted points in a sequence of dominoes
  where the discount factor is applied once for every turn required to get to
  that dominoe in the sequence (e.g. first dominoe in a sequence has value =
  1\*dominoeValue, second dominoe has value = inLineDiscount\*dominoeValue,
  etc, skipping turns if they use a double).
- offLineValue: thte total number of discounted points of dominoes not in a
  particular sequence (where the offLineDiscount is applied as many times as
  there are turns in each sequence). 



