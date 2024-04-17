from abc import ABC, abstractmethod
from copy import copy
from itertools import repeat
import torch

from multiprocessing import Pool, cpu_count

from .support import get_dominoes, get_best_line, pad_best_lines
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

    def process_reward(self, rewards, scores, choices, gamma_transform):
        """
        process the reward for performing policy gradient

        args:
            rewards: list of torch.Tensor, the rewards for each network (precomputed using `reward_function`)
            scores: list of torch.Tensor, the log scores for the choices for each network
            choices: list of torch.Tensor, index to the choices made by each network
            gamma_transform: torch.Tensor, the gamma transform matrix for the reward

        returns:
            list of torch.Tensor, the rewards for each network
        """
        # measure cumulative discounted rewards for each network
        G = [torch.matmul(reward, gamma_transform) for reward in rewards]

        # measure choice score for each network (the log-probability for each choice)
        choice_scores = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(scores, choices)]

        # measure J for each network
        J = [-torch.sum(cs * g) for cs, g in zip(choice_scores, G)]

        return G, J

    @abstractmethod
    def reward_function(self, choices, batch, **kwargs):
        """
        measure the reward acquired by the choices made by a set of networks for the current batch

        args:
            choice: torch.Tensor, index to the choices made by the network
            batch: tuple, the batch of data generated for this training step
            kwargs: optional kwargs for any additional reward arguments required by a specific task

        returns:
            torch.Tensor, the rewards for the network
            (additional outputs are task dependent)
        """
        pass


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
