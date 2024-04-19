from copy import copy
from itertools import repeat
from multiprocessing import Pool, cpu_count
import torch


from .support import get_dominoes, get_best_line, pad_best_lines
from ..utils import named_transpose
from .base import DatasetSL, DatasetRL


class DominoeDataset(DatasetRL, DatasetSL):
    """A dataset for generating dominoe sequences for training and evaluation"""

    def __init__(
        self, task, highest_dominoe, device="cpu", train_fraction=None, loss_function=torch.nn.functional.nll_loss, loss_kwargs={}, **parameters
    ):
        """constructor method"""
        # first add loss function setup to the supervised loss component of this class
        DatasetSL.__init__(self, loss_function=loss_function, loss_kwargs=loss_kwargs)

        self.set_device(device)

        self._check_task(task)
        self.task = task
        self.highest_dominoe = highest_dominoe

        # create base dominoe set
        self.dominoe_set = get_dominoes(self.highest_dominoe, as_torch=True)

        # set the training set if train_fraction is provided
        self.set_train_fraction(train_fraction)

        # check parameters
        self._check_parameters(init=True, **parameters)

        # set parameters to required defaults first, then update
        self.prms = self._required_parameters()
        self.prms = self.parameters(**parameters)

    def _check_task(self, task):
        """
        check if the task is valid and set default parameters for the task
        """
        if task == "sequencer":
            self.null_token = True
            self.available_token = True

        elif task == "sorting":
            self.null_token = False
            self.available_token = False

        elif task is None:
            self.null_token = False
            self.available_token = False

        else:
            raise ValueError("task should be either 'sequencer', 'sorting', or None")

    def _required_parameters(self):
        """
        return the required parameters for the task. This is hard-coded here and only here,
        so if the parameters change, this method should be updated.

        None means the parameter is required and doesn't have a default value. Otherwise,
        the value is the default value for the parameter.

        args:
            task: str, the task for which to get the required parameters

        returns:
            list of str, the required parameters for the task
        """
        # base parameters for all tasks
        params = dict(
            hand_size=None,  # this parameter is required to be set at initialization
            batch_size=1,
            return_target=False,
            ignore_index=-1,
        )
        if self.task == "sequencer":
            params["value_method"] = "length"
            params["value_multiplier"] = 1.0
            return params
        elif self.task == "sorting":
            params["allow_mistakes"] = False
            return params
        else:
            return {}

    @torch.no_grad()
    def set_train_fraction(self, train_fraction):
        """
        Pick a random subset of dominoes to use as the training set.

        args:
            train_fraction: float, the fraction of the dominoes to use for training (0 < train_fraction < 1)

        Will register the training set as self.train_set and the index to them as self.train_index.
        """
        self.train_fraction = train_fraction
        if train_fraction is None:
            self.train_index = None
            self.train_set = None
        else:
            assert train_fraction > 0 and train_fraction < 1, "train_fraction should be a float between 0 and 1"
            self.train_index = torch.randperm(len(self.dominoe_set))[: int(train_fraction * len(self.dominoe_set))]
            self.train_set = self.dominoe_set[self.train_index]

    @torch.no_grad()
    def get_dominoe_set(self, train):
        """ """
        if train and self.train_set is None:
            raise ValueError("Requested training set but it hasn't been made yet, use `set_train_fraction` to make one")
        return self.train_set if train else self.dominoe_set

    @torch.no_grad()
    def generate_batch(self, train=True, device=None, **kwargs):
        """
        generates a batch of dominoes with the required additional outputs

        batch keys:
            input: torch.Tensor, the input to the network, as a binary dominoe representation (and null token)
            train: bool, whether the batch is for training or evaluation
            selection: torch.Tensor, the selection of dominoes in the hand
            target: torch.Tensor, the target for the network (only if requested)

        additional keys for the sequencer task:
            available: torch.Tensor, the available value to play on at the start of the hand
                       only included for the sequencer task
            best_seq: torch.Tensor, the best sequence of dominoes to play (only if requested)
            best_dir: torch.Tensor, the direction of play for the best sequence (only if requested)

        additional keys for the sorting task:
            value: torch.Tensor, the value of each dominoe in the batch (sum of the dominoe)
        """
        # get device
        device = self.get_device(device)

        # choose which set of dominoes to use
        dominoes = self.get_dominoe_set(train)

        # get parameters with potential updates
        prms = self.parameters(**kwargs)

        # get a random dominoe hand
        # will encode the hand as binary representations including null and available tokens if requested
        # will also include the index of the selection in each hand a list of available values for each batch element
        # note that dominoes direction is randomized for the input, but not for the target
        input, selection, available = self._random_dominoe_hand(
            prms["hand_size"],
            self._randomize_direction(dominoes),
            batch_size=prms["batch_size"],
            null_token=self.null_token,
            available_token=self.available_token,
        )

        # construct batch dictionary
        batch = dict(input=input.to(device), train=train, selection=selection)

        # add task specific parameters to the batch dictionary
        batch = self._add_task_parameters(batch, available, **prms)
        batch.update(prms)

        # if target is requested (e.g. for SL tasks) then get target based on registered task
        if prms["return_target"]:
            # get target dictionary
            target_dict = self.set_target(dominoes, selection, available, device=device, **prms)
            # update batch dictionary with target dictionary
            batch.update(target_dict)

        return batch

    def _add_task_parameters(self, batch, available, **prms):
        """Add task specific parameters to the batch dictionary"""
        if self.task == "sequencer":
            batch["available"] = available
        if self.task == "sorting":
            pass
        return batch

    def set_target(self, dominoes, selection, available, device=None, **prms):
        """
        set the target output for the batch based on the registered task
        """
        if self.task == "sequencer":
            return self._gettarget_sequencer(dominoes, selection, available, device=device, **prms)
        elif self.task == "sorting":
            return self._gettarget_sorting(dominoes, selection, device=device, **prms)
        else:
            raise ValueError(f"task {self.task} not recognized")

    @torch.no_grad()
    def reward_function(self, choices, batch, **kwargs):
        """
        measure the reward acquired by the choices made by a set of networks for the current batch
        based on the registered task

        args:
            choice: torch.Tensor, index to the choices made by the network
            batch: tuple, the batch of data generated for this training step
            kwargs: optional kwargs for each task-specific reward function

        returns:
            torch.Tensor, the rewards for the network
        """
        if self.task == "sequencer":
            return self._measurereward_sequencer(choices, batch, **kwargs)
        elif self.task == "sorting":
            return self._measurereward_sorter(choices, batch, **kwargs)
        else:
            raise ValueError(f"task {self.task} not recognized!")

    @torch.no_grad()
    def _gettarget_sequencer(self, dominoes, selection, available, device=None, **prms):
        """
        get the target for the sequencer task

        chooses target based on the best line for each batch element based on either:
        1. the value of the dominoes in the line
        2. the number of dominoes in the line (e.g. the length of the sequence)

        args:
            dominoes: torch.Tensor, the dominoes in the dataset (num_dominoes, 2)
            selection: torch.Tensor, the selection of dominoes in the hand (batch_size, hand_size)
            available: torch.Tensor, the available value to play on (batch_size,)
            device: torch.device, the device to put the target on (optional, default is None)
            prms: dict, the parameters for the batch generation
                  see self.parameters() for more information and look in this method for the specific parameters used

        """
        # get device
        device = self.get_device(device)

        # Depending on the batch size and hand size, doing this with parallel pool is sometimes faster
        if prms.get("parallel", False):
            # Allow user to set the number of processes or fallback onto an agressive default
            num_processes = prms.get("num_processes", cpu_count() - 2)
            with Pool(num_processes) as pool:
                # arguments to get_best_line are (dominoes, available, value_method)
                pool_args = [(dominoes[sel], ava, value) for sel, ava, value in zip(selection, available, repeat(prms["value_method"]))]
                results = pool.starmap(get_best_line, pool_args)
            best_seq, best_dir = named_transpose(results)
        else:
            # Unless the batch size is very large, this is usually faster
            best_seq, best_dir = named_transpose(
                [get_best_line(dominoes[sel], ava, value_method=prms["value_method"]) for sel, ava in zip(selection, available)]
            )

        # hand_size is the index corresponding to the null_token if it exists
        null_index = prms["hand_size"] if self.null_token else prms["ignore_index"]

        # create target and append null_index once, then ignore_index afterwards
        # the idea is that the agent should play the best line, then indicate that the line is over, then anything else doesn't matter
        target = torch.stack(pad_best_lines(best_seq, prms["hand_size"] + 1, null_index, ignore_index=prms["ignore_index"])).long()

        # construct target dictionary
        target_dict = dict(target=target.to(device))

        # add the best sequence and direction if requested
        target_dict["best_seq"] = best_seq
        target_dict["best_dir"] = best_dir

        return target_dict

    @torch.no_grad()
    def _gettarget_sorting(self, dominoes, selection, device=None, **prms):
        """
        target method for the "sorting" task in which dominoes are sorted by value

        args:
            dominoes: torch.Tensor, the dominoes in the dataset (num_dominoes, 2)
            selection: torch.Tensor, the selection of dominoes in the hand (batch_size, hand_size)
            device: torch.device, the device to put the target on (optional, default is None)
            prms: dict, the parameters for the batch generation
                  see self.parameters() for more information and look in this method for the specific parameters used

        returns:
            dict, the target dictionary for the batch
                  containing the target for the batch and optionally the value of each dominoe in the batch
        """
        # get device
        device = self.get_device(device)

        # measure the value of each dominoe in the batch
        value = torch.sum(dominoes[selection], dim=2)

        # set target as the index of descending sorted values
        target = torch.argsort(value, dim=1, descending=True, stable=True).long()

        # return dictionary with target and value
        return dict(target=target.to(device), value=value)

    @torch.no_grad()
    def _measurereward_sequencer(self, choices, batch, return_direction=False, verbose=None):
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
        assert choices.ndim == 2, f"choices should be a (batch_size, max_output) tensor of indices, it is: {choices.shape}"
        batch_size, max_output = choices.shape
        device = choices.device

        # make hands and put on right device
        hands = self.get_dominoe_set(batch["train"])[batch["selection"]].float().to(device)
        num_in_hand = hands.size(1)
        null_index = copy(num_in_hand)

        # check verbose
        if verbose is not None:
            debug = True
            assert 0 <= verbose < batch_size, "verbose should be an index corresponding to one of the batch elements"
        else:
            debug = False

        # check value method
        if not (batch["value_method"] == "dominoe" or batch["value_method"] == "length"):
            raise ValueError("did not recognize value_method, it has to be either 'dominoe' or 'length'")

        # initialize these tracker variables
        next_available = batch["available"].clone().float().to(device)  # next value available to play on
        already_played = torch.zeros((batch_size, num_in_hand + 1), dtype=torch.bool).to(device)  # False until dominoe has been played
        made_mistake = torch.zeros((batch_size,), dtype=torch.bool).to(device)  # False until a mistake is made
        played_null = torch.zeros((batch_size,), dtype=torch.bool).to(device)  # False until the null dominoe has been played

        # keep track of original values -- append the null token as [-1, -1]
        hands_original = torch.cat((hands, -torch.ones((batch_size, 1, 2)).to(device)), dim=1)

        # keep track of remaining playable values -- with null appended -- and will update values of played dominoes
        hands_updates = torch.cat((hands, -torch.ones((batch_size, 1, 2)).to(device)), dim=1)

        rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
        if return_direction:
            direction = -torch.ones((batch_size, max_output), dtype=torch.float).to(device)

        if debug:
            print("Original hand:\n", hands[verbose])

        # then for each output:
        for idx in range(max_output):
            # First order checks
            idx_chose_null = choices[:, idx] == null_index  # True if chosen dominoe is null token
            idx_repeat = torch.gather(already_played, 1, choices[:, idx].view(-1, 1)).squeeze(1)  # True if chosen dominoe has already been played
            # (batch, 2) size tensor of choice of next play
            chosen_play = torch.gather(hands_original, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            idx_match = torch.any(chosen_play.T == next_available, 0)  # True if chosen play has a value that matches the next available dominoe
            # True if >0 remaining dominoes match next available
            idx_possible_match = torch.any((hands_updates == next_available.view(-1, 1, 1)).view(hands_updates.size(0), -1), dim=1)

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
                print("")
                print("\nNew loop in measure reward:\n")
                print("NextAvailable:", next_available[verbose])
                print("PlayAvailable: ", idx_possible_match[verbose])
                print("Choice: ", choices[verbose, idx])
                print("ChosenPlay: ", chosen_play[verbose])
                print("IdxNull: ", idx_chose_null[verbose])
                print("IdxMatch: ", idx_match[verbose])
                print("IdxRepeat: ", idx_repeat[verbose])
                print("ValidDominoePlay: ", valid_dominoe_play[verbose])
                print("ValidNullPlay: ", valid_null_play[verbose])
                print("FirstInvalidDominoePlay: ", first_invalid_dominoe_play[verbose])
                print("FirstInvalidNullPlay: ", first_invalid_null_play[verbose])

            # Figure out what the next available dominoe is for valid plays
            # if True, then 1 is index to next value, if False then 0 is index to next value
            next_value_idx = 1 * (chosen_play[:, 0] == next_available)
            # next available value (as of now, this includes invalid plays!!!)
            new_available = torch.gather(chosen_play, 1, next_value_idx.view(-1, 1)).squeeze(1)

            # If valid dominoe play, update next_available
            next_available[valid_dominoe_play] = new_available[valid_dominoe_play]

            # Output direction of play if requested for reconstructing the line
            if return_direction:
                play_direction = 1.0 * (next_value_idx == 0)  # direction is 1 if "forward" and 0 if "backward"
                direction[valid_dominoe_play, idx] = play_direction[valid_dominoe_play].float()

            # Set rewards for dominoe plays
            if batch["value_method"] == "dominoe":
                valid_play_rewards = torch.sum(chosen_play[valid_dominoe_play], dim=1) + 1.0  # offset by 1 so (0|0) has value
                invalid_play_rewards = torch.sum(chosen_play[first_invalid_dominoe_play], dim=1) + 1.0
                rewards[valid_dominoe_play, idx] += valid_play_rewards * batch["value_multiplier"]
                rewards[first_invalid_dominoe_play, idx] -= invalid_play_rewards * batch["value_multiplier"]
            else:
                rewards[valid_dominoe_play, idx] += 1.0 * batch["value_multiplier"]
                rewards[first_invalid_dominoe_play, idx] -= 1.0 * batch["value_multiplier"]

            # Set rewards for null plays (don't use value multiplier for the null tokens)
            rewards[valid_null_play, idx] += 1.0
            rewards[first_invalid_null_play, idx] -= 1.0

            # Now prepare tracking variables for next round
            already_played.scatter_(1, choices[:, idx].view(-1, 1), torch.ones((batch_size, 1), dtype=bool).to(device))
            played_null[idx_chose_null] = True  # Played null becomes True if chose null on this round
            made_mistake[idx_repeat | ~idx_match] = True  # Made mistake becomes True if chose null on this round

            # Clone chosen play and set it to -1 for any valid dominoe play
            insert_values = chosen_play.clone()
            insert_values[valid_dominoe_play] = -1

            # Then set the hands updates tensor to the "insert values", will change it to -1's if it's a valid_dominoe_play
            hands_updates.scatter_(1, choices[:, idx].view(-1, 1, 1).expand(-1, -1, 2), insert_values.unsqueeze(1))

            if debug:
                if return_direction:
                    print("play_direction:", play_direction[verbose])
                print("NextAvailable: ", next_available[verbose])
                print("HandsUpdated:\n", hands_updates[verbose])
                print("Rewards[verbose,idx]:", rewards[verbose, idx])

        if return_direction:
            return rewards, direction
        else:
            return rewards

    @torch.no_grad()
    def _measurereward_sorter(self, choices, batch, **kwargs):
        """
        measure the reward for the sorting task

        rewards are 1 when a dominoe is chosen that:
        - hasn't been played yet
        - has less than or equal value to the last dominoe played (first dominoe always valid)

        rewards are -1 when a dominoe is chosen that:
        - has already been played
        - has greater value than the last dominoe played

        note: rewards are set to 0 after a mistake is made

        args:
            choices: torch.Tensor, the choices made by the network
            batch: tuple, the batch of data generated for this training step

        returns:
            torch.Tensor, the rewards for the network
        """
        assert choices.ndim == 2, "choices should be a (batch_size, max_output) tensor of indices"
        batch_size, max_output = choices.shape
        device = choices.device

        # make hands and put on right device
        hands = self.get_dominoe_set(batch["train"])[batch["selection"]].float().to(device)
        num_in_hand = hands.size(1)

        # True until dominoe has been played
        havent_played = torch.ones((batch_size, num_in_hand), dtype=torch.bool).to(device)

        # False until the agent makes a mistake (within each batch)
        made_mistake = torch.zeros((batch_size,), dtype=torch.bool).to(device)

        rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
        last_value = torch.inf * torch.ones((batch_size,), dtype=torch.float).to(device)  # initialize last value high

        # then for each output:
        for idx in range(max_output):
            # for next choice, get bool of whether choice has already been played
            idx_not_played = torch.gather(havent_played, 1, choices[:, idx].view(-1, 1)).squeeze(1)

            # update which dominoes have been played
            havent_played.scatter_(1, choices[:, idx].view(-1, 1), torch.zeros((batch_size, 1), dtype=torch.bool).to(device))

            # for dominoes that haven't been played, add their value to rewards
            next_play = torch.gather(hands, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            value_play = torch.sum(next_play, dim=1)

            # check if it's lower in value
            idx_smaller = (value_play <= last_value) & idx_not_played & ~made_mistake
            last_value[idx_smaller] = value_play[idx_smaller]

            # add reward for valid plays, subtract for invalid
            rewards[idx_smaller, idx] += 1.0
            rewards[~idx_smaller & ~made_mistake, idx] -= 1.0

            # update mistake counter (after setting -1 reward for first mistake)
            if not batch["allow_mistakes"]:
                made_mistake |= ~idx_smaller

        return rewards

    @torch.no_grad()
    def _binary_dominoe_representation(self, dominoes, available=None, available_token=False, null_token=False):
        """
        converts a set of dominoes to a stacked two-hot representation (with optional null and available tokens)

        dominoes are paired values (combinations with replacement) of integers
        from 0 to self.highest_dominoe.

        This simple representation is a two-hot vector where the first
        self.highest_dominoe+1 elements represent the first value of the dominoe, and
        the second self.highest_dominoe+1 elements represent the second value of the
        dominoe. Here are some examples for self.highest_dominoe = 3:

        (0 | 0): [1, 0, 0, 0, 1, 0, 0, 0]
        (0 | 1): [1, 0, 0, 0, 0, 1, 0, 0]
        (0 | 2): [1, 0, 0, 0, 0, 0, 1, 0]
        (0 | 3): [1, 0, 0, 0, 0, 0, 0, 1]
        (1 | 0): [0, 1, 0, 0, 1, 0, 0, 0]
        (2 | 1): [0, 0, 1, 0, 0, 1, 0, 0]

        If provided, can also add a null token and an available token to the end of
        the two-hot vector. The null token is a single-hot vector that represents a
        null dominoe. The available token is a single-hot vector that represents the
        available value to play on. If the null token is included, the dimensionality
        of the input is increased by 1. If the available token is included, the
        dimensionality of the input is increased by <highest_dominoe>+1 and the
        available value is represented in the third section of the two-hot vector.

        args:
            dominoes: torch.Tensor, the dominoes to convert to a binary representation
            highest_dominoe: int, the highest value of a dominoe, if None, will use self.highest_dominoe
            available: torch.Tensor, the available value to play on
            available_token: bool, whether to include an available token in the representation
            null_token: bool, whether to include a null token in the representation
        """
        if available_token and (available is None):
            raise ValueError("if with_available=True, then available needs to be provided")

        # create a fake batch dimension if it doesn't exist for consistent code
        with_batch = dominoes.dim() == 3
        if not with_batch:
            dominoes = dominoes.unsqueeze(0)

        # get dataset size
        batch_size = dominoes.size(0)
        num_dominoes = dominoes.size(1)

        # input dimension determined by highest dominoe (twice the number of possible values on a dominoe)
        input_dim = (2 if not available_token else 3) * (self.highest_dominoe + 1) + (1 if null_token else 0)

        # first & second value are index and index shifted by highest_dominoe + 1
        first_value = dominoes[..., 0].unsqueeze(2)
        second_value = dominoes[..., 1].unsqueeze(2) + self.highest_dominoe + 1

        # scatter dominoe data into two-hot vectors
        src = torch.ones((batch_size, num_dominoes, 1), dtype=torch.float)
        binary = torch.zeros((batch_size, num_dominoes, input_dim), dtype=torch.float)
        binary.scatter_(2, first_value, src)
        binary.scatter_(2, second_value, src)

        # add null token to the hand if requested
        if null_token:
            # create a representation of the null token
            rep_null = torch.zeros((batch_size, 1, input_dim), dtype=torch.float)
            rep_null.scatter_(2, torch.tensor(input_dim - 1).view(1, 1, 1).expand(batch_size, -1, -1), torch.ones(batch_size, 1, 1))
            # stack it to the end of each hand
            binary = torch.cat((binary, rep_null), dim=1)

        # add available token to the hand if requested
        if available_token:
            # create a representation of the available token
            available_index = available + 2 * (self.highest_dominoe + 1)
            rep_available = torch.zeros((batch_size, 1, input_dim), dtype=torch.float)
            rep_available.scatter_(2, available_index.view(batch_size, 1, 1), torch.ones(batch_size, 1, 1))
            # stack it to the end of each hand
            binary = torch.cat((binary, rep_available), dim=1)

        # remove batch dimension if it didn't exist
        if not with_batch:
            binary = binary.squeeze(0)

        return binary

    @torch.no_grad()
    def _random_dominoe_hand(self, hand_size, dominoes, batch_size=1, null_token=True, available_token=True):
        """
        general method for creating a random hand of dominoes and encoding it in a two-hot representation

        args:
            hand_size: number of dominoes in each hand
            dominoes: list of dominoes to choose from
            batch_size: number of hands to create
            null_token: whether to include a null token in the input
            available_token: whether to include an available token in the input
        """
        num_dominoes = len(dominoes)

        # choose a hand of hand_size dominoes from the full set for each batch element
        selection = torch.stack([torch.randperm(num_dominoes)[:hand_size] for _ in range(batch_size)])
        hands = dominoes[selection]

        # set available token to a random value from the dataset or None
        if available_token:
            available = torch.randint(0, self.highest_dominoe + 1, (batch_size,))
        else:
            available = None

        # create a binary representation of the hands
        input = self._binary_dominoe_representation(hands, available=available, available_token=available_token, null_token=null_token)

        # return binary representation, the selection indices and the available values
        return input, selection, available

    @torch.no_grad()
    def _randomize_direction(self, dominoes):
        """
        randomize the direction of a dominoes representation in a batch

        Note: doubles don't need to be flipped because they are symmetric, but this method does it anyway
        because it doesn't make a difference and it's easier and just as fast to implement with torch.gather()

        args:
            dominoes: torch.Tensor, the dominoes to randomize with shape (batch_size, num_dominoes, 2) or (num_dominoes, 2)

        returns:
            torch.Tensor, the dominoes with the direction of the dominoes randomized
        """
        # check shape of dominoes dataset
        assert dominoes.size(-1) == 2, "dominoes should have shape (batch_size, num_dominoes, 2) or (num_dominoes, 2)"

        # create a fake batch dimension if it doesn't exist for consistent code
        with_batch = dominoes.dim() == 3
        if not with_batch:
            dominoes = dominoes.unsqueeze(0)

        # get the batch size and number of dominoes
        batch_size = dominoes.size(0)
        num_dominoes = dominoes.size(1)

        # pick a random order for the dominoes (0 means forward order, 1 means reverse)
        order = torch.randint(2, (batch_size, num_dominoes), device=dominoes.device)
        gather_idx = torch.stack([order, 1 - order], dim=2)

        # get randomized dataset
        randomized = torch.gather(dominoes, 2, gather_idx)

        # remove the batch dimension if it wasn't there before
        if not with_batch:
            randomized = randomized.squeeze(0)

        return randomized
