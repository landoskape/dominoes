from abc import ABC, abstractmethod
from copy import copy
import torch

from .support import get_dominoe_set, get_best_line, pad_best_lines
from ..utils import named_transpose


class Dataset(ABC):
    """
    the dataset class is a general purpose dataset for loading and evaluating performance on sequencing problems
    """

    # @abstractmethod
    # def initialize(self, *args, **kwargs):
    #     """required method for initializing the dataset"""

    @abstractmethod
    def parameters(self, **kwargs):
        """
        method for defining parameters for batch generation
        if none provided as kwargs, will use the defaults registered upon dataset creation

        required in all children of this dataset type
        """

    @abstractmethod
    def generate_batch(self, *args, **kwargs):
        """required method for generating a batch"""


class DatasetRL(Dataset):
    """
    A child of the general dataset class that is designed for reinforcement learning tasks
    """

    def create_gamma_transform(self, max_output, gamma, device=None):
        """
        create a gamma transform matrix for the dataset

        args:
            max_output: int, the maximum number of outputs in a sequence
            gamma: float, the gamma value for the transform
            device: torch.device, the device to use for the transform
                    if device is not provided, will use the device registered upon dataset creation

        returns:
            torch.Tensor, the gamma transform matrix
            a toeplitz matrix that can be used to apply exponential discounting to a reward matrix
        """
        # set device to the registered value if not provided
        device = device or self.device

        # create exponent toeplitz matrix for exponential discounting
        exponent = torch.arange(max_output).view(-1, 1) - torch.arange(max_output).view(1, -1)
        exponent = exponent * (exponent >= 0)

        # return the gamma transform matrix
        return (gamma**exponent).to(device)

    def measure_reward(self, scores, choices, batch, gamma_transform):
        """
        measure the reward for a batch of choices

        args:
            scores: list of torch.Tensor, the log scores for the choices for each network
            choices: list of torch.Tensor, index to the choices made by each network
            batch: tuple, the batch of data generated for this training step
            gamma_transform: torch.Tensor, the gamma transform matrix for the reward

        returns:
            list of torch.Tensor, the rewards for each network
        """
        # measure reward for each network
        rewards = [self.reward_function(choice, batch) for choice in choices]

        # measure cumulative discounted rewards for each network
        G = [torch.matmul(reward, gamma_transform) for reward in rewards]

        # measure J for each network
        J = [-torch.sum(score * g) for score, g in zip(scores, G)]

        return rewards, G, J

    @abstractmethod
    def reward_function(self, choices, batch):
        """
        measure the reward acquired by the choices made by a set of networks for the current batch

        args:
            choice: torch.Tensor, index to the choices made by the network
            batch: tuple, the batch of data generated for this training step

        returns:
            torch.Tensor, the rewards for the network
        """


class DominoeDataset(DatasetRL):
    """A dataset for generating dominoe sequences for training and evaluation"""

    def __init__(
        self,
        task,
        highest_dominoe,
        train_fraction=None,
        **parameters,
    ):
        self._check_task(task)
        self.task = task
        self.highest_dominoe = highest_dominoe
        self.train_fraction = train_fraction

        # create base dominoe set
        self.dominoe_set = get_dominoe_set(self.highest_dominoe, as_torch=True)

        # set the training set if train_fraction is provided
        if train_fraction is not None:
            self.set_train_fraction(train_fraction)

        # check parameters
        self._check_parameters(**parameters)

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

    def _check_parameters(self, **task_parameters):
        """
        check if the parameters provided at initialization are valid and complete

        checks two things:
        1. If any parameters are provided that are not recognized for the task, an error will be generated
        2. If any parameters are required for the task but not provided, an error will be generated

        args:
            task_parameters: dict, the parameters provided at initialization
        """
        required_params = self._required_parameters()
        for param in task_parameters:
            if param not in required_params:
                raise ValueError(f"parameter {param} not recognized for task {self.task}")
        for param in required_params:
            if param not in task_parameters and required_params[param] is None:
                raise ValueError(f"parameter {param} not provided for task {self.task}")

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
            return_full=False,
        )
        if self.task == "sequencer":
            params["value_method"] = "length"
            return params
        elif self.task == "sorting":
            return params
        else:
            return {}

    def parameters(self, **prms):
        """
        Helper method for setting parameters for batch generation or just using the defaults
        set at initialization
        """
        prms_to_use = copy(self.prms)
        # check if any parameters are provided that are not recognized
        for param in prms:
            if param not in prms_to_use:
                raise ValueError(f"parameter {param} not recognized for task {self.task}")
        prms_to_use.update(prms)
        return prms_to_use

    def set_train_fraction(self, train_fraction):
        """
        Pick a random subset of dominoes to use as the training set.

        args:
            train_fraction: float, the fraction of the dominoes to use for training (0 < train_fraction < 1)

        Will register the training set as self.train_set and the index to them as self.train_index.
        """
        assert train_fraction > 0 and train_fraction < 1, "train_fraction should be a float between 0 and 1"
        self.train_index = torch.randperm(len(self.dominoe_set))[: int(train_fraction * len(self.dominoe_set))]
        self.train_set = self.dominoe_set[self.train_index]

    def binary_dominoe_representation(self, dominoes, highest_dominoe=None, available=None, available_token=False, null_token=False):
        """
        converts a set of dominoes to a stacked two-hot representation (with optional null and available tokens)

        dominoes are paired values (combinations with replacement) of integers
        from 0 to <highest_dominoe>.

        This simple representation is a two-hot vector where the first
        <highest_dominoe>+1 elements represent the first value of the dominoe, and
        the second <highest_dominoe>+1 elements represent the second value of the
        dominoe. Here are some examples for <highest_dominoe> = 3:

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
        highest_dominoe = highest_dominoe or self.highest_dominoe
        input_dim = (2 if not available_token else 3) * (highest_dominoe + 1) + (1 if null_token else 0)

        # first & second value are index and index shifted by highest_dominoe + 1
        first_value = dominoes[..., 0].unsqueeze(2)
        second_value = dominoes[..., 1].unsqueeze(2) + highest_dominoe + 1

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
            available_index = available + 2 * (highest_dominoe + 1)
            rep_available = torch.zeros((batch_size, 1, input_dim), dtype=torch.float)
            rep_available.scatter_(2, available_index.view(batch_size, 1, 1), torch.ones(batch_size, 1, 1))
            # stack it to the end of each hand
            binary = torch.cat((binary, rep_available), dim=1)

        # remove batch dimension if it didn't exist
        if not with_batch:
            binary = binary.squeeze(0)

        return binary

    def generate_batch(self, train=True, **kwargs):
        """
        generates a batch of dominoes with the required additional outputs
        """
        # choose which set of dominoes to use
        dominoes = self.train_dominoe_set if train else self.dominoe_set

        # randomize direction of the dominoes
        dominoes = self._randomize_direction(dominoes)

        # get parameters with potential updates
        prms = self.parameters(**kwargs)

        # get a random dominoe hand
        # will encode the hand as binary representations including null and available tokens if requested
        # will also include the index of the selection in each hand a list of available values for each batch element
        input, selection, available = self._random_dominoe_hand(
            prms["hand_size"], dominoes, prms["batch_size"], self.null_token, self.available_token
        )

        # make a mask for the input
        mask_tokens = prms["hand_size"] + (1 if self.null_token else 0) + (1 if self.available_token else 0)
        mask = torch.ones((prms["batch_size"], mask_tokens), dtype=torch.float)

        # construct batch dictionary
        batch = dict(input=input, mask=mask)

        # augment batch with more details if requested
        if prms["return_full"]:
            batch["selection"] = selection
            batch["available"] = available

        # if target is requested (e.g. for SL tasks) then get target based on registered task
        if prms["return_target"]:
            # get target dictionary
            target_dict = self.set_target(dominoes, selection, available, **prms)
            # update batch dictionary with target dictionary
            batch.update(target_dict)

        # return batch
        return batch

    def set_target(self, dominoes, selection, available, **prms):
        """
        set the target output for the batch based on the registered task
        """
        if self.task == "sequencer":
            return self._gettarget_sequencer(dominoes, selection, available, **prms)
        elif self.task == "sorting":
            return self._gettarget_sorting(dominoes, selection, available, **prms)
        else:
            raise ValueError(f"task {self.task} not recognized")

    def _gettarget_sequencer(self, dominoes, selection, available, **prms):
        """
        get the target for the sequencer task

        chooses target based on the best line for each batch element based on either:
        1. the value of the dominoes in the line
        2. the number of dominoes in the line (e.g. the length of the sequence)

        args:
            dominoes: torch.Tensor, the dominoes in the dataset (num_dominoes, 2)
            selection: torch.Tensor, the selection of dominoes in the hand (batch_size, hand_size)
            available: torch.Tensor, the available value to play on (batch_size,)
            prms: dict, the parameters for the batch generation
                  see self.parameters() for more information and look in this method for the specific parameters used

        """
        # get best sequence (and direction) for each batch element
        best_seq, best_dir = named_transpose(
            [get_best_line(dominoes[sel], ava, value_method=prms["value_method"]) for sel, ava in zip(selection, available)]
        )

        # hand_size is the index corresponding to the null_token if it exists
        null_index = prms["hand_size"] if self.null_token else prms["ignore_index"]

        # create target and append null_index once, then ignore_index afterwards
        # the idea is that the agent should play the best line, then indicate that the line is over, then anything else doesn't matter
        target = torch.stack(pad_best_lines(best_seq, prms["hand_size"] + 1, null_index, ignore_index=prms["ignore_index"]), dtype=torch.long)

        # construct target dictionary
        target_dict = dict(target=target)

        # add the best sequence and direction if requested
        if prms["return_full"]:
            target_dict["best_seq"] = best_seq
            target_dict["best_dir"] = best_dir

        return target_dict

    # def _gettarget_sorting(self, dominoes, selection, available, **prms):
    #     # sort and pad each list of dominoes by value
    #     def sortPad(val, padTo, ignoreIndex=-1):
    #         s = sorted(range(len(val)), key=lambda i: -val[i])
    #         p = [ignoreIndex] * (padTo - len(val))
    #         return s + p

    #     # create a padded sort index, then turn into a torch tensor as the target vector
    #     sortIdx = [sortPad(val, maxSeqLength, ignoreIndex) for val in value]  # pad with ignore index so nll_loss ignores them
    #     target = torch.stack([torch.LongTensor(idx) for idx in sortIdx])

    def _random_dominoe_hand(self, hand_size, dominoes, highest_dominoe=None, batch_size=1, null_token=True, available_token=True):
        """
        general method for creating a random hand of dominoes and encoding it in a two-hot representation

        args:
            hand_size: number of dominoes in each hand
            dominoes: list of dominoes to choose from
            highest_dominoe: highest value of a dominoe (if not provided, will use self.highest_dominoe)
            batch_size: number of hands to create
            null_token: whether to include a null token in the input
            available_token: whether to include an available token in the input
        """
        num_dominoes = len(dominoes)
        highest_dominoe = highest_dominoe or self.highest_dominoe

        # choose a hand of hand_size dominoes from the full set for each batch element
        selection = torch.stack([torch.randperm(num_dominoes)[:hand_size] for _ in range(batch_size)])
        hands = dominoes[selection]

        # set available token to a random value from the dataset or None
        if available_token:
            available = torch.randint(0, highest_dominoe + 1, (batch_size,))
        else:
            available = None

        # create a binary representation of the hands
        input = self.binary_dominoe_representation(
            hands, highest_dominoe=highest_dominoe, available=available, available_token=available_token, null_token=null_token
        )

        # return binary representation, the selection indices and the available values
        return input, selection, available

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

    def reward_function(self, choices, batch):
        """
        measure the reward acquired by the choices made by a set of networks for the current batch

        args:
            choice: torch.Tensor, index to the choices made by the network
            batch: tuple, the batch of data generated for this training step

        returns:
            torch.Tensor, the rewards for the network
        """
        pass


# @torch.no_grad()
# def measureReward_sortDescend(hands, choices):
#     assert choices.ndim == 2, "choices should be a (batch_size, max_output) tensor of indices"
#     batch_size, max_output = choices.shape
#     num_in_hand = hands.shape[1]
#     device = transformers.get_device(choices)

#     # initialize these tracker variables
#     havent_played = torch.ones((batch_size, num_in_hand), dtype=torch.bool).to(
#         device
#     )  # True until dominoe has been played (include null for easier coding b/c out_choices includes idx to null
#     hands = torch.tensor(hands, dtype=torch.float).to(device)

#     rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
#     last_value = 10000 * torch.ones((batch_size,), dtype=torch.float).to(device)  # initialize last value high

#     # then for each output:
#     for idx in range(max_output):
#         # for next choice, get bool of whether choice has already been played
#         idx_not_played = torch.gather(havent_played, 1, choices[:, idx].view(-1, 1)).squeeze(1)

#         # update which dominoes have been played
#         havent_played.scatter_(1, choices[:, idx].view(-1, 1), torch.zeros((batch_size, 1), dtype=torch.bool).to(device))

#         # for dominoes that haven't been played, add their value to rewards
#         next_play = torch.gather(hands, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
#         value_play = torch.sum(next_play, dim=1)

#         # check if it's lower in value
#         idx_smaller = (value_play <= last_value) & idx_not_played
#         last_value[idx_smaller] = value_play[idx_smaller]

#         # add reward for valid plays, subtract for invalid
#         rewards[idx_smaller, idx] += 1
#         rewards[~idx_smaller, idx] -= 1

#     return rewards


# @torch.no_grad()
# def measureReward_sequencer(available, hands, choices, value_method="dominoe", normalize=True, return_direction=False, verbose=None):
#     """
#     reward function for sequencer

#     there is a positive reward in two conditions:
#     1. Valid dominoe play:
#         - a dominoe is played that hasn't been played yet and for which one of the values on the dominoe matches the next available value
#         - in this case, the reward value is either (1+sum(dominoe_values)) or a flat integer rate (determined by value_method)
#     2. Valid null token:
#         - a null token is played for the first time and no remaining unplayed dominoes match the available one
#         - in this case, the reward value is 1

#     there is a negative reward in these conditions:
#     1. Repeat play
#         - a dominoe is played that has already been played
#         - reward value is negative but magnitude same as in a valid dominoe play
#     2. Non-match play
#         - a dominoe is played that hasn't been played yet but the values on it don't match next available
#         - reward value is negative but magnitude same as in a valid dominoe play
#     3. Invalid null token:
#         - a null token is played for the first time but there are still dominoes that match the available one
#         - in this case, the reward value is -1

#     after any negative reward, any remaining plays have a value of 0
#     - examples:
#         - after first null token, all plays have 0 reward
#         - after first repeat play or non-match play, all plays have 0 reward
#     - note:
#         - I'm considering allowing the agent to keep playing after a repeat or non-match play (and return that failed play back to the hand...)
#         - If so, this will get an extra keyword argument so it can be turned on or off
#     """

#     assert choices.ndim == 2, f"choices should be a (batch_size, max_output) tensor of indices, it is: {choices.shape}"
#     batch_size, max_output = choices.shape
#     num_in_hand = hands.shape[1]
#     null_index = copy(num_in_hand)
#     device = transformers.get_device(choices)

#     # check verbose
#     if verbose is not None:
#         debug = True
#         assert 0 <= verbose < batch_size, "verbose should be an index corresponding to one of the batch elements"
#     else:
#         debug = False

#     # check value method
#     if value_method == "dominoe":
#         check_normalize = True
#     elif isinstance(value_method, str) and value_method.isdigit() and int(value_method) > 0:
#         valid_play_value = float(int(value_method))
#         check_normalize = False
#     else:
#         raise ValueError("did not recognize value_method, it has to be either 'dominoe' or a string representation of a positive digit")

#     # initialize these tracker variables
#     next_available = torch.tensor(available, dtype=torch.float).to(device)
#     already_played = torch.zeros((batch_size, num_in_hand + 1), dtype=torch.bool).to(
#         device
#     )  # False until dominoe has been played (including null, even though those are handled differently)
#     made_mistake = torch.zeros((batch_size,), dtype=torch.bool).to(device)  # False until a mistake is made
#     played_null = torch.zeros((batch_size,), dtype=torch.bool).to(device)  # False until the null dominoe has been played
#     hands = torch.tensor(hands, dtype=torch.float).to(device)

#     # keep track of original values -- append the null token as [-1, -1]
#     handsOriginal = torch.cat((hands, -torch.ones((hands.size(0), 1, 2)).to(device)), dim=1)

#     # keep track of remaining playable values -- with null appended -- and will update values of played dominoes
#     handsUpdates = torch.cat((hands, -torch.ones((hands.size(0), 1, 2)).to(device)), dim=1)

#     rewards = torch.zeros((batch_size, max_output), dtype=torch.float).to(device)
#     if return_direction:
#         direction = -torch.ones((batch_size, max_output), dtype=torch.float).to(device)

#     if debug:
#         print("Original hand:\n", hands[verbose])

#     # then for each output:
#     for idx in range(max_output):
#         if idx == 1:
#             pass
#         # First order checks
#         idx_chose_null = choices[:, idx] == null_index  # True if chosen dominoe is null token
#         idx_repeat = torch.gather(already_played, 1, choices[:, idx].view(-1, 1)).squeeze(1)  # True if chosen dominoe has already been played
#         chosen_play = torch.gather(handsOriginal, 1, choices[:, idx].view(-1, 1, 1).expand(-1, 1, 2)).squeeze(
#             1
#         )  # (batch, 2) size tensor of choice of next play
#         idx_match = torch.any(chosen_play.T == next_available, 0)  # True if chosen play has a value that matches the next available dominoe
#         idx_possible_match = torch.any(
#             (handsUpdates == next_available.view(-1, 1, 1)).view(handsUpdates.size(0), -1), dim=1
#         )  # True if >0 remaining dominoes matche next available

#         # Valid dominoe play if didn't choose null, didn't repeat a dominoe, matched the available value, hasn't chosen null yet, and hasn't made mistakes
#         valid_dominoe_play = ~idx_chose_null & ~idx_repeat & idx_match & ~played_null & ~made_mistake

#         # Valid null play if chose null, there aren't possible matches remaining, hasn't chosen null yet, and hasn't made mistakes
#         valid_null_play = idx_chose_null & ~idx_possible_match & ~played_null & ~made_mistake

#         # First invalid dominoe play if didn't choose null, repeated a choice or didn't match available values, and hasn't chosen null yet or hasn't made mistakes
#         first_invalid_dominoe_play = ~idx_chose_null & (idx_repeat | ~idx_match) & ~played_null & ~made_mistake

#         # First invalid null play if chose null, there are possible matches remaining, and hasn't chosen null yet or hasn't made mistakes
#         first_invalid_null_play = idx_chose_null & idx_possible_match & ~played_null & ~made_mistake

#         # debug block after first order checks
#         if debug:
#             print("")
#             print("\nNew loop in measure reward:\n")
#             print("NextAvailable:", next_available[verbose])
#             print("PlayAvailable: ", idx_possible_match[verbose])
#             print("Choice: ", choices[verbose, idx])
#             print("ChosenPlay: ", chosen_play[verbose])
#             print("IdxNull: ", idx_chose_null[verbose])
#             print("IdxMatch: ", idx_match[verbose])
#             print("IdxRepeat: ", idx_repeat[verbose])
#             print("ValidDominoePlay: ", valid_dominoe_play[verbose])
#             print("ValidNullPlay: ", valid_null_play[verbose])
#             print("FirstInvalidDominoePlay: ", first_invalid_dominoe_play[verbose])
#             print("FirstInvalidNullPlay: ", first_invalid_null_play[verbose])

#         # Figure out what the next available dominoe is for valid plays
#         next_value_idx = 1 * (chosen_play[:, 0] == next_available)  # if True, then 1 is index to next value, if False then 0 is index to next value
#         new_available = torch.gather(chosen_play, 1, next_value_idx.view(-1, 1)).squeeze(
#             1
#         )  # next available value (as of now, this includes invalid plays!!!)

#         # If valid dominoe play, update next_available
#         next_available[valid_dominoe_play] = new_available[valid_dominoe_play]

#         # Output direction of play if requested for reconstructing the line
#         if return_direction:
#             play_direction = 1.0 * (next_value_idx == 0)  # direction is 1 if "forward" and 0 if "backward"
#             direction[valid_dominoe_play, idx] = play_direction[valid_dominoe_play].float()

#         # Set rewards for dominoe plays
#         if value_method == "dominoe":
#             rewards[valid_dominoe_play, idx] += torch.sum(chosen_play[valid_dominoe_play], dim=1) + 1  # offset by 1 so (0|0) has value
#             rewards[first_invalid_dominoe_play, idx] -= torch.sum(chosen_play[first_invalid_dominoe_play], dim=1) + 1
#         else:
#             rewards[valid_dominoe_play, idx] += valid_play_value
#             rewards[first_invalid_dominoe_play, idx] -= valid_play_value

#         # Set rewards for null plays
#         rewards[valid_null_play, idx] += 1.0
#         rewards[first_invalid_null_play, idx] -= 1.0

#         # Now prepare tracking variables for next round
#         already_played.scatter_(1, choices[:, idx].view(-1, 1), torch.ones((batch_size, 1), dtype=bool).to(device))
#         played_null[idx_chose_null] = True  # Played null becomes True if chose null on this round
#         made_mistake[idx_repeat | ~idx_match] = True  # Made mistake becomes True if chose null on this round

#         # Clone chosen play and set it to -1 for any valid dominoe play
#         insert_values = chosen_play.clone()
#         insert_values[valid_dominoe_play] = -1

#         # Then set the hands updates tensor to the "insert values", will change it to -1's if it's a valid_dominoe_play
#         handsUpdates.scatter_(1, choices[:, idx].view(-1, 1, 1).expand(-1, -1, 2), insert_values.unsqueeze(1))

#         if debug:
#             if return_direction:
#                 print("play_direction:", play_direction[verbose])
#             print("NextAvailable: ", next_available[verbose])
#             print("HandsUpdated:\n", handsUpdates[verbose])
#             print("Rewards[verbose,idx]:", rewards[verbose, idx])

#     if check_normalize and normalize:
#         rewards /= 1

#     if return_direction:
#         return rewards, direction
#     else:
#         return rewards


# @torch.no_grad()
# def measureReward_tsp(dists, choices):
#     """reward function for measuring tsp performance"""
#     assert choices.ndim == 2, "choices should be a 2-d tensor of the sequence of choices for each batch element"
#     assert dists.ndim == 3, "dists should be a 3-d tensor of the distance matrix across cities for each batch element"
#     numCities = dists.size(1)
#     batchSize, numChoices = choices.shape
#     assert 1 < numChoices <= (numCities + 1), "numChoices per batch element should be more than 1 and no more than twice the number of cities"
#     device = transformers.get_device(choices)
#     distance = torch.zeros((batchSize, numChoices)).to(device)
#     new_city = torch.ones((batchSize, numChoices)).to(device)

#     last_location = copy(choices[:, 0])  # last (i.e. initial position) is final step of permutation of cities
#     src = torch.ones((batchSize, 1), dtype=torch.bool).to(device)
#     visited = torch.zeros((batchSize, numChoices), dtype=torch.bool).to(device)
#     visited.scatter_(1, last_location.view(batchSize, 1), src)  # put first city in to the "visited" tensor
#     for nc in range(1, numChoices):
#         next_location = choices[:, nc]
#         c_dist_possible = torch.gather(dists, 1, last_location.view(batchSize, 1, 1).expand(-1, -1, numCities)).squeeze(1)
#         distance[:, nc] = torch.gather(c_dist_possible, 1, next_location.view(batchSize, 1)).squeeze(1)
#         c_visited = torch.gather(visited, 1, next_location.view(batchSize, 1)).squeeze(1)
#         visited.scatter_(1, next_location.view(batchSize, 1), src)
#         new_city[c_visited, nc] = -1.0
#         new_city[~c_visited, nc] = 1.0
#         last_location = copy(next_location)  # update last location

#     # add return step (to initial city) to the final choice
#     c_dist_possible = torch.gather(dists, 1, choices[:, 0].view(batchSize, 1, 1).expand(-1, -1, numCities)).squeeze(1)
#     distance[:, -1] += torch.gather(c_dist_possible, 1, choices[:, -1].view(batchSize, 1)).squeeze(1)

#     return distance, new_city
