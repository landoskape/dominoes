# Documentation: Transformer and Pointer Network Demonstration

The most sophisticated agents use transformers to encode the set of dominoes
in their hand. I plan on using pointer networks as a method of pretraining
these dominoes agents (not coded yet). This documentation file shows how the
transformers and pointer networks work, and displays the results of a simple
toy problem solved by pointer networks. 

## Toy Problem
The toy problem is written as an experiment called 
[pointerDemonstration](../experiments/pointerDemonstration.md). It trains a
pointer network to sort dominoes by the value on each dominoe given a random
set of dominoes in random order, with a variable set size, due to the magic of
pointer networks. Since it's a simple problem, the pointer network learns the
task in just a few minutes. 

You can run this toy problem yourself with the following command:
```
python experiments/pointerDemonstration.md
```

The main result of the problem is shown here: 
![pointer toy figure](media/pointerDemonstration.png)

As you can see, the network quickly learns to sort dominoes by their value 
effectively. Pretty cool! The third panel of the figure shows that the lost 
depends on the sequence size, which makes sense because there is more room 
for error and uncertainty in a longer list of dominoes to sort, and the 
negative log-likelihood loss function penalizes the network for uncertainty. 


