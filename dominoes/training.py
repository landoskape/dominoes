from copy import copy
import torch
from torch import nn
import torch.nn.functional as F
from . import transformers

@torch.no_grad()
def measureReward_sortDescend(hands, choices):
    assert choices.ndim==2, "choices should be a (batch_size, max_output) tensor of indices"
    batch_size, max_output = choices.shape
    num_in_hand = hands.shape[1]
    device = transformers.get_device(choices)
    
    # initialize these tracker variables
    havent_played = torch.ones((batch_size, num_in_hand), dtype=torch.bool).to(device) # True until dominoe has been played (include null for easier coding b/c out_choices includes idx to null
    hands = torch.tensor(hands, dtype=torch.float).to(device)
    
    rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
    last_value = 10000*torch.ones((batch_size, ), dtype=torch.float).to(device) # initialize last value high
    
    # then for each output:
    for idx in range(max_output):
        # for next choice, get bool of whether choice has already been played
        idx_not_played = torch.gather(havent_played, 1, choices[:, idx].view(-1,1)).squeeze(1)

        # update which dominoes have been played
        havent_played.scatter_(1, choices[:,idx].view(-1,1), torch.zeros((batch_size,1), dtype=torch.bool).to(device))

        # for dominoes that haven't been played, add their value to rewards
        next_play = torch.gather(hands, 1, choices[:, idx].view(-1,1,1).expand(-1,1,2)).squeeze(1)
        value_play = torch.sum(next_play, dim=1)

        # check if it's lower in value
        idx_smaller = (value_play <= last_value) & idx_not_played
        last_value[idx_smaller] = value_play[idx_smaller]
        
        # add reward for valid plays, subtract for invalid
        rewards[idx_smaller, idx] += 1 
        rewards[~idx_smaller, idx] -= 1 

    return rewards


@torch.no_grad()
def measureReward_sequencer(available, hands, choices, value_method='dominoe', normalize=True, return_direction=False, verbose=None):
    """
    reward function for sequencer
    
    there is a positive reward in two conditions:
    1. Valid dominoe play:
        - a dominoe is played that hasn't been played yet and for which one of the values on the dominoe matches the next available value
        - in this case, the reward value is either (1+sum(dominoe_values)) or a flat integer rate (determined by value_method)
    2. Valid null token:
        - a null token is played for the first time and no remaining unplayed dominoes match the available one
        - in this case, the reward value is 1

    there is a negative reward in these conditions:
    1. Repeat play
        - a dominoe is played that has already been played
        - reward value is negative but magnitude same as in a valid dominoe play
    2. Non-match play
        - a dominoe is played that hasn't been played yet but the values on it don't match next available
        - reward value is negative but magnitude same as in a valid dominoe play
    3. Invalid null token:
        - a null token is played for the first time but there are still dominoes that match the available one
        - in this case, the reward value is -1
    
    after any negative reward, any remaining plays have a value of 0
    - examples:
        - after first null token, all plays have 0 reward
        - after first repeat play or non-match play, all plays have 0 reward
    - note:
        - I'm considering allowing the agent to keep playing after a repeat or non-match play (and return that failed play back to the hand...)
        - If so, this will get an extra keyword argument so it can be turned on or off    
    """

    assert choices.ndim==2, f"choices should be a (batch_size, max_output) tensor of indices, it is: {choices.shape}"
    batch_size, max_output = choices.shape
    num_in_hand = hands.shape[1]
    null_index = copy(num_in_hand)
    device = transformers.get_device(choices)

    # check verbose
    if verbose is not None:
        debug = True
        assert 0 <= verbose < batch_size, "verbose should be an index corresponding to one of the batch elements"
    else:
        debug = False

    # check value method
    if value_method=='dominoe':
        check_normalize = True
    elif isinstance(value_method, str) and value_method.isdigit() and int(value_method)>0:
        valid_play_value = float(int(value_method))
        check_normalize = False
    else:
        raise ValueError("did not recognize value_method, it has to be either 'dominoe' or a string representation of a positive digit")
    
    # initialize these tracker variables
    next_available = torch.tensor(available, dtype=torch.float).to(device)
    already_played = torch.zeros((batch_size, num_in_hand+1), dtype=torch.bool).to(device) # False until dominoe has been played (including null, even though those are handled differently)
    made_mistake = torch.zeros((batch_size,), dtype=torch.bool).to(device) # False until a mistake is made
    played_null = torch.zeros((batch_size,), dtype=torch.bool).to(device) # False until the null dominoe has been played
    hands = torch.tensor(hands, dtype=torch.float).to(device)

    # keep track of original values -- append the null token as [-1, -1]
    handsOriginal = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1)

    # keep track of remaining playable values -- with null appended -- and will update values of played dominoes
    handsUpdates = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1) 
    
    rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
    if return_direction: 
        direction = -torch.ones((batch_size, max_output), dtype=torch.float).to(device)

    if debug:
        print("Original hand:\n", hands[verbose])
        
    # then for each output:
    for idx in range(max_output):
        if idx==1:
            pass
        # First order checks
        idx_chose_null = choices[:, idx] == null_index # True if chosen dominoe is null token
        idx_repeat = torch.gather(already_played, 1, choices[:, idx].view(-1, 1)).squeeze(1) # True if chosen dominoe has already been played
        chosen_play = torch.gather(handsOriginal, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1) # (batch, 2) size tensor of choice of next play
        idx_match = torch.any(chosen_play.T == next_available, 0) # True if chosen play has a value that matches the next available dominoe
        idx_possible_match = torch.any((handsUpdates == next_available.view(-1, 1, 1)).view(handsUpdates.size(0), -1), dim=1) # True if >0 remaining dominoes matche next available 

        # Valid dominoe play if didn't choose null, didn't repeat a dominoe, matched the available value, hasn't chosen null yet, and hasn't made mistakes
        valid_dominoe_play = ~idx_chose_null & ~idx_repeat & idx_match & ~played_null & ~made_mistake

        # Valid null play if chose null, there aren't possible matches remaining, hasn't chosen null yet, and hasn't made mistakes
        valid_null_play = idx_chose_null & ~idx_possible_match & ~played_null & ~made_mistake
        
        # First invalid dominoe play if didn't choose null, repeated a choice or didn't match available values, and hasn't chosen null yet or hasn't made mistakes
        first_invalid_dominoe_play = ~idx_chose_null & (idx_repeat | ~idx_match) & ~played_null & ~made_mistake

        # First invalid null play if chose null, there are possible matches remaining, and hasn't chosen null yet or hasn't made mistakes
        first_invalid_null_play = idx_chose_null & idx_possible_match & ~played_null & ~made_mistake
        
        # debug block after first order checks
        if debug:
            print('')
            print("\nNew loop in measure reward:\n")
            print("NextAvailable:", next_available[verbose])
            print("PlayAvailable: ", idx_possible_match[verbose])
            print("Choice: ", choices[verbose,idx])
            print("ChosenPlay: ", chosen_play[verbose])
            print("IdxNull: ", idx_chose_null[verbose])
            print("IdxMatch: ", idx_match[verbose])
            print("IdxRepeat: ", idx_repeat[verbose])
            print("ValidDominoePlay: ", valid_dominoe_play[verbose])
            print("ValidNullPlay: ", valid_null_play[verbose])
            print("FirstInvalidDominoePlay: ", first_invalid_dominoe_play[verbose])
            print("FirstInvalidNullPlay: ", first_invalid_null_play[verbose])

        # Figure out what the next available dominoe is for valid plays
        next_value_idx = 1*(chosen_play[:,0]==next_available) # if True, then 1 is index to next value, if False then 0 is index to next value
        new_available = torch.gather(chosen_play, 1, next_value_idx.view(-1,1)).squeeze(1) # next available value (as of now, this includes invalid plays!!!)

        # If valid dominoe play, update next_available
        next_available[valid_dominoe_play] = new_available[valid_dominoe_play]

        # Output direction of play if requested for reconstructing the line
        if return_direction:
            play_direction = 1.0*(next_value_idx==0) # direction is 1 if "forward" and 0 if "backward"
            direction[valid_dominoe_play, idx] = play_direction[valid_dominoe_play].float()

        # Set rewards for dominoe plays
        if value_method == 'dominoe':
            rewards[valid_dominoe_play, idx] += (torch.sum(chosen_play[valid_dominoe_play], dim=1) + 1) # offset by 1 so (0|0) has value
            rewards[first_invalid_dominoe_play, idx] -= (torch.sum(chosen_play[first_invalid_dominoe_play], dim=1) + 1)
        else:
            rewards[valid_dominoe_play, idx] += valid_play_value
            rewards[first_invalid_dominoe_play, idx] -= valid_play_value
        
        # Set rewards for null plays
        rewards[valid_null_play, idx] += 1.0
        rewards[first_invalid_null_play, idx] -= 1.0

        # Now prepare tracking variables for next round
        already_played.scatter_(1, choices[:, idx].view(-1, 1), torch.ones((batch_size, 1), dtype=bool).to(device))
        played_null[idx_chose_null] = True # Played null becomes True if chose null on this round
        made_mistake[idx_repeat | ~idx_match] = True # Made mistake becomes True if chose null on this round

        # Clone chosen play and set it to -1 for any valid dominoe play 
        insert_values = chosen_play.clone()
        insert_values[valid_dominoe_play] = -1

        # Then set the hands updates tensor to the "insert values", will change it to -1's if it's a valid_dominoe_play
        handsUpdates.scatter_(1, choices[:, idx].view(-1,1,1).expand(-1,-1,2), insert_values.unsqueeze(1)) 

        if debug:
            if return_direction:
                print("play_direction:", play_direction[verbose])
            print("NextAvailable: ", next_available[verbose])
            print("HandsUpdated:\n", handsUpdates[verbose])
            print("Rewards[verbose,idx]:", rewards[verbose, idx])
    
    if check_normalize and normalize:
        rewards /= 1
        
    if return_direction:
        return rewards, direction
    else:        
        return rewards


@torch.no_grad()
def measurePossibleRewards_sequencer(available, hands, choices, value_method='dominoe', normalize=True, return_direction=False):
    """
    possible reward function for sequencer
    
    there is a positive reward in two conditions:
    1. Valid dominoe play:
        - a dominoe is played that hasn't been played yet and for which one of the values on the dominoe matches the next available value
        - in this case, the reward value is either (1+sum(dominoe_values)) or a flat integer rate (determined by value_method)
    2. Valid null token:
        - a null token is played for the first time and no remaining unplayed dominoes match the available one
        - in this case, the reward value is 1

    there is a negative reward in these conditions:
    1. Repeat play
        - a dominoe is played that has already been played
        - reward value is negative but magnitude same as in a valid dominoe play
    2. Non-match play
        - a dominoe is played that hasn't been played yet but the values on it don't match next available
        - reward value is negative but magnitude same as in a valid dominoe play
    3. Invalid null token:
        - a null token is played for the first time but there are still dominoes that match the available one
        - in this case, the reward value is -1
    
    after any negative reward, any remaining plays have a value of 0
    - examples:
        - after first null token, all plays have 0 reward
        - after first repeat play or non-match play, all plays have 0 reward
    - note:
        - I'm considering allowing the agent to keep playing after a repeat or non-match play (and return that failed play back to the hand...)
        - If so, this will get an extra keyword argument so it can be turned on or off    
    """

    assert choices.ndim==2, f"choices should be a (batch_size, max_output) tensor of indices, it is: {choices.shape}"
    batch_size, max_output = choices.shape
    num_in_hand = hands.shape[1]
    num_choices = num_in_hand + 1 # there's also a null token
    null_index = copy(num_in_hand)
    device = transformers.get_device(choices)

    # check value method
    if value_method=='dominoe':
        check_normalize = True
    elif isinstance(value_method, str) and value_method.isdigit() and int(value_method)>0:
        valid_play_value = float(int(value_method))
        check_normalize = False
    else:
        raise ValueError("did not recognize value_method, it has to be either 'dominoe' or a string representation of a positive digit")
    
    # initialize these tracker variables
    next_available = torch.tensor(available, dtype=torch.float).to(device)
    already_played = torch.zeros((batch_size, num_in_hand+1), dtype=torch.bool).to(device) # False until dominoe has been played (including null, even though those are handled differently)
    made_mistake = torch.zeros((batch_size,), dtype=torch.bool).to(device) # False until a mistake is made
    played_null = torch.zeros((batch_size,), dtype=torch.bool).to(device) # False until the null dominoe has been played
    hands = torch.tensor(hands, dtype=torch.float).to(device)

    # keep track of original values -- append the null token as [-1, -1]
    handsOriginal = torch.cat((hands, -torch.ones((batch_size,1,2)).to(device)), dim=1)
    if value_method=='dominoe':
        handsValue = torch.cat((torch.sum(hands, dim=2)+1, torch.ones((batch_size,1)).to(device)), dim=1)
    else:
        handsValue = torch.cat((valid_play_value*torch.ones((batch_size, num_choices)).to(device), torch.ones((batch_size,1)).to(device)), dim=1)
        
    # keep track of remaining playable values -- with null appended -- and will update values of played dominoes
    handsUpdates = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1) 
    idx_null = torch.zeros((num_choices,),dtype=torch.bool).scatter(0,torch.tensor(num_choices-1),torch.tensor(True)).to(device)
    
    rewards = torch.zeros((max_output, batch_size, num_choices), dtype=torch.float).to(device)
    if return_direction: 
        direction = -torch.ones((batch_size, max_output), dtype=torch.float).to(device) # this is just of the actual choices
    
    # then for each output:
    for idx in range(max_output):
        # First, measure first order checks for all possible actions    
        idx_match = torch.any(handsOriginal==next_available.view(-1,1,1), dim=2)
        idx_possible_match = torch.any(torch.any(handsUpdates == next_available.view(-1, 1, 1), dim=2), dim=1)

        # Vectorized valid / invalid plays for each possible dominoe
        idx_mistake = already_played & ~idx_match
        valid_dominoe_play = ~idx_null.view(1, -1) & ~idx_mistake & ~played_null.view(-1, 1) & ~made_mistake.view(-1, 1)
        first_invalid_dominoe_play = ~idx_null.view(1, -1) & idx_mistake & ~played_null.view(-1, 1) & ~made_mistake.view(-1, 1)

        # Vectorized valid / invalid null plays
        valid_null_play = idx_null.view(1, -1) & ~idx_possible_match.view(-1, 1) & ~played_null.view(-1, 1) & ~made_mistake.view(-1, 1)
        first_invalid_null_play = idx_null.view(1, -1) & idx_possible_match.view(-1, 1) & ~played_null.view(-1, 1) & ~made_mistake.view(-1, 1)
        
        # Set rewards for dominoe plays
        rewards[idx] += (handsValue * valid_dominoe_play)
        rewards[idx] -= (handsValue * first_invalid_dominoe_play)
    
        # Set rewards for null plays
        # killing reward for valid nulls --- rewards[idx] += (handsValue * valid_null_play)
        rewards[idx] -= (handsValue * first_invalid_null_play)

        # -------- then do accounting for actual choice (which controls game state) -----------
        idx_chose_null = choices[:, idx] == null_index # True if chosen dominoe is null token
        idx_repeat = torch.gather(already_played, 1, choices[:, idx].view(-1, 1)).squeeze(1) # True if chosen dominoe has already been played
        chosen_play = torch.gather(handsOriginal, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1) # (batch, 2) size tensor of choice of next play
        idx_match = torch.any(chosen_play.T == next_available, 0) # True if chosen play has a value that matches the next available dominoe
        idx_possible_match = torch.any((handsUpdates == next_available.view(-1, 1, 1)).view(handsUpdates.size(0), -1), dim=1) # True if >0 remaining dominoes matche next available 
        
        # -- this is the only one we need for choice accounting --
        valid_dominoe_play = ~idx_chose_null & ~idx_repeat & idx_match & ~played_null & ~made_mistake

        # Figure out what the next available dominoe is for valid plays
        next_value_idx = 1*(chosen_play[:,0]==next_available) # if True, then 1 is index to next value, if False then 0 is index to next value
        new_available = torch.gather(chosen_play, 1, next_value_idx.view(-1,1)).squeeze(1) # next available value (as of now, this includes invalid plays!!!)

        # If valid dominoe play, update next_available
        next_available[valid_dominoe_play] = new_available[valid_dominoe_play]

        # Output direction of play if requested for reconstructing the line
        if return_direction:
            play_direction = 1.0*(next_value_idx==0) # direction is 1 if "forward" and 0 if "backward"
            direction[valid_dominoe_play, idx] = play_direction[valid_dominoe_play].float()

        # Now prepare tracking variables for next round
        already_played.scatter_(1, choices[:, idx].view(-1, 1), torch.ones((batch_size, 1), dtype=bool).to(device))
        played_null[idx_chose_null] = True # Played null becomes True if chose null on this round
        made_mistake[idx_repeat | ~idx_match] = True # Made mistake becomes True if chose null on this round

        # Clone chosen play and set it to -1 for any valid dominoe play 
        insert_values = chosen_play.clone()
        insert_values[valid_dominoe_play] = -1

        # Then set the hands updates tensor to the "insert values", will change it to -1's if it's a valid_dominoe_play
        handsUpdates.scatter_(1, choices[:, idx].view(-1,1,1).expand(-1,-1,2), insert_values.unsqueeze(1)) 
    
    if check_normalize and normalize:
        rewards /= normalize
        
    if return_direction:
        return rewards, direction
    else:        
        return rewards



@torch.no_grad()
def measureReward_tsp(dists, choices):
    """reward function for measuring tsp performance"""
    assert choices.ndim==2, "choices should be a 2-d tensor of the sequence of choices for each batch element"
    assert dists.ndim==3, "dists should be a 3-d tensor of the distance matrix across cities for each batch element"
    numCities = dists.size(1)
    batchSize, numChoices = choices.shape
    assert 1 < numChoices <= (numCities+1), "numChoices per batch element should be more than 1 and no more than twice the number of cities"  
    device = transformers.get_device(choices)
    distance = torch.zeros((batchSize, numChoices)).to(device)
    new_city = torch.ones((batchSize, numChoices)).to(device)

    last_location = copy(choices[:, 0]) # last (i.e. initial position) is final step of permutation of cities
    src = torch.ones((batchSize,1), dtype=torch.bool).to(device)
    visited = torch.zeros((batchSize, numChoices), dtype=torch.bool).to(device)
    visited.scatter_(1, last_location.view(batchSize, 1), src) # put first city in to the "visited" tensor
    for nc in range(1, numChoices):
        next_location = choices[:, nc]
        c_dist_possible = torch.gather(dists, 1, last_location.view(batchSize, 1, 1).expand(-1, -1, numCities)).squeeze(1)
        distance[:, nc] = torch.gather(c_dist_possible, 1, next_location.view(batchSize, 1)).squeeze(1)
        c_visited = torch.gather(visited, 1, next_location.view(batchSize, 1)).squeeze(1)
        visited.scatter_(1, next_location.view(batchSize, 1), src)
        new_city[c_visited, nc] = -1.0
        new_city[~c_visited, nc] = 1.0
        last_location = copy(next_location) # update last location

    # add return step (to initial city) to the final choice
    c_dist_possible = torch.gather(dists, 1, choices[:, 0].view(batchSize, 1, 1).expand(-1, -1, numCities)).squeeze(1)
    distance[:, -1] += torch.gather(c_dist_possible, 1, choices[:, -1].view(batchSize, 1)).squeeze(1)

    return distance, new_city



