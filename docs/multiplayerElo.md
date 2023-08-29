# Documentation: Multiplayer ELO

Evaluating an agent's policy requires a method of comparing the skill level
and win probability of each agent in comparison with the other agents. A good
choice for this problem is the ELO system, which is used to determine the 
probability one agent will win against another agent given their ELO ratings. 

The ELO rating system was developed for head-to-head zero-sum matchups (like 
in Chess); however, I am using a multiplayer variant suggested by
[Tom Kerrigan](http://www.tckerrigan.com/Misc/Multiplayer_Elo/). In short,
multiplayer ELO works by treating a match of N agents as N-1 head-to-head
matches, in which ELO is updated for each pair of agents with neighboring 
scores as if the game was only between those two agents. For example, suppose
three agents were playing dominoes and had scores of 0, 10, and 15. Then, 
we assume that agent 0 beat agent 1, and assume that agent 1 beat agent 2. 
Therefore, agents 0 and 2 receive one ELO update, and agent 1 receives two 
updates.

## ELO Updates
ELO updates are coded in the [`leagueManager`](../dominoes/leagueManager.py)
module. The league manager has a "league" of N agents, of which M of them play
games against each other (M<=N). After each game, the game results and the 
league index of each agent are fed back to the league manager to update the 
ELO scores of those agents. 

First, the expected score for each agent pair in the game is measured, where 
expected score is equal to the probability of winning (not the dominoe score). 
Then, ELO scores are updated based on the true winner and a "k" parameter, 
which is set by the league manager and determines how much an agents ELO can 
change for any particular game. The baseline ELO is set to 1500, which is an 
arbitrary choice, and can be changed to whatever range of values you are most
comfortable with. The specific equations used for ELO are coded in the 
[`functions`](../dominoes/functions.py) module, named `eloExpected` and 
`eloUpdate`. 

## Measuring ELO of basic agents
To test the multiplayer ELO system and evaluate the policies of basic 
hand-crafted agents, I wrote a script called 
[`basicAgentELOs`](../experiments/basicAgentELOs.py) that creates a league of
basic agents and plays many games between agents, updating their ELO ratings 
until they stabilize. For details on the parameters of the experiment, see the
ArgumentParser help statements and read the comments in the main function. 

From the top-level directory of this repository, you can run this experiment
on your own computer with the following command:
```
python experiments/basicAgentELOs.py
```
The script has several optional input arguments that determine the parameters
of the league and meta-parameters for the experiment (e.g. how many games to 
play for estimating ELO). 

The main result of the experiment is shown here:
![elo figure](media/basicAgentELOs.png)

To summarize, ELO ratings stablize quickly (after about 1000 games, which 
takes about 5 minutes of wall time on my computer), and indicate that the best
hand-crafted policy is that of the persistent-line agent, followed closely by 
the double agent. As expected, agents that play randomly or play dominoes with 
the lowest point value perform worse than other agents. 

## Measuring ELO of best line agents with different maxLineLength parameters
To measure how important the maxLineLength parameter is for the best line 
agents, as well as comparing the policy differences between the best line
agent and the persistent line agent (see [Basic policies](basicPolicies.md)),
I ran an additional experiment that is very similar to the basic agent ELO 
experiment, but using a league with exclusively best/persistent line agents. 
This code is found in 
[`bestLineAgentELOs`](../experiments/bestLineAgentELOs.py). The choice of max
line lengths is hard coded in the file, and is set to `[6,8,10,12]`. The 
league contains multiple copies of best line agents and persistent line agents
with each possible max line length. 

The main result of the experiment is shown here: 
![bestline elo figure](media/bestLineAgentELOs.png)

Best line agents perform a little better when they perform a deeper recursive
search for their line: note that the agents that looks for maximum lines of 10
or 12 perform better than those that cutoff at 6 or 8. Interestingly, the
persistent line agent performs a little better than the best line agent. I'm 
not exactly sure why this is true, but it probably relates to changes in the 
valuation of different lines due to discounting factors. 
