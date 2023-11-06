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
def measureReward_sequencer(available, hands, choices, normalize=True, return_direction=False, verbose=None):
    assert choices.ndim==2, f"choices should be a (batch_size, max_output) tensor of indices, it is: {choices.shape}"
    batch_size, max_output = choices.shape
    num_in_hand = hands.shape[1]
    device = transformers.get_device(choices)

    # check verbose
    if verbose is not None:
        debug = True
        assert 0 <= verbose < batch_size, "verbose should be an index corresponding to one of the batch elements"
    else:
        debug = False
    
    # initialize these tracker variables
    next_available = torch.tensor(available, dtype=torch.float).to(device)
    havent_played = torch.ones((batch_size, num_in_hand+1), dtype=torch.bool).to(device) # True until dominoe has been played (include null for easier coding b/c out_choices includes idx to null
    hands = torch.tensor(hands, dtype=torch.float).to(device)
    handsOriginal = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1) # keep track of original values
    handsUpdates = torch.cat((hands, -torch.ones((hands.size(0),1,2)).to(device)), dim=1) # Add null choice as [-1,-1]
    
    rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
    if return_direction: 
        direction = -torch.ones((batch_size, max_output), dtype=torch.float).to(device)

    if debug:
        print("Original hand:\n", hands[verbose])
        
    # then for each output:
    for idx in range(max_output):
        # idx of outputs that chose the null token
        idx_null = choices[:,idx] == num_in_hand

        if debug:
            print('')
            print("\nNew loop in measure reward:\n")
            print("next_available:", next_available[verbose])
            print("Choice: ", choices[verbose,idx])
            print("IdxNull: ", idx_null[verbose])
        
        # idx of outputs that have not already been chosen
        idx_not_played = torch.gather(havent_played, 1, choices[:, idx].view(-1,1)).squeeze(1)
        havent_played.scatter_(1, choices[:,idx].view(-1,1), torch.zeros((batch_size,1), dtype=torch.bool).to(device))

        # idx of play (not the null token and a dominoe that hasn't been played yet)
        idx_play = ~idx_null & idx_not_played
        
        # the dominoe that is chosen (including the null token)
        next_play = torch.gather(handsOriginal, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

        # figure out whether play was valid, whcih direction, and next available
        valid_play = torch.any(next_play.T == next_available, 0) # figure out if available matches a value in each dominoe
        next_value_idx = 1*(next_play[:,0]==next_available) # if true, then 1 is next value, if false then 0 is next value (for valid plays)
        new_available = torch.gather(next_play, 1, next_value_idx.view(-1,1)).squeeze(1) # get next available value
        next_available[valid_play] = new_available[valid_play] # update available for all valid plays

        # after getting next play, set any dominoe that has been played to [-1, -1]
        insert_values = next_play.clone()
        insert_values[valid_play] = -1
        handsUpdates.scatter_(1, choices[:, idx].view(-1,1,1).expand(-1,-1,2), insert_values.unsqueeze(1))

        if return_direction:
            play_direction = 1.0*(next_value_idx==0)
            direction[idx_play & valid_play, idx] = play_direction[idx_play & valid_play].float()
            
        # if dominoe that is a valid_play and hasn't been played yet is selected, add reward
        idx_good_play = ~idx_null & idx_not_played & valid_play
        rewards[idx_good_play, idx] += (torch.sum(next_play[idx_good_play], dim=1) + 1)
    
        # if a dominoe is chosen but is invalid, subtract points
        idx_bad_play = ~idx_null & (~idx_not_played | ~valid_play)
        rewards[idx_bad_play, idx] -= (torch.sum(next_play[idx_bad_play], dim=1) + 1)

        # determine which hands still have playable dominoes
        idx_still_playable = torch.any((handsUpdates == next_available.view(-1, 1, 1)).view(handsUpdates.size(0), -1), dim=1)

        # if the null is chosen and no other dominoes are possible, give some reward, otherwise negative reward
        rewards[idx_null & ~idx_still_playable, idx] += 1.0
        rewards[idx_null & idx_still_playable, idx] -= 1.0

        if debug:
            print("Next play: ", next_play[verbose])
            if return_direction:
                print("play_direction:", play_direction[verbose])
            print("next available: ", next_available[verbose])
            print("valid_play:", valid_play[verbose])
            print("idx_still_playable:", idx_still_playable[verbose])
            print("idx_play: ", idx_play[verbose])
            print("Hands updated:\n", handsUpdates[verbose])
            print("Rewards[verbose,idx]:", rewards[verbose, idx])
    
    if normalize:
        rewards /= (highestDominoe+1) 
        
    if return_direction:
        return rewards, direction
    else:        
        return rewards