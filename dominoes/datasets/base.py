from abc import ABC, abstractmethod
import torch


class Dataset(ABC):
    """
    the dataset class is a general purpose dataset for loading and evaluating performance on sequencing problems
    since it is the master class for RL and SL datasets, the only required method is `generate_batch`
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

    def get_pretemp_score(self, scores, choices, temperature):
        pass

    def _get_choice_score(self, choices, scores):
        """
        get the score for the choices made by the networks

        args:
            choices: torch.Tensor, the choices made by the networks
                     should be 2-d Long tensor of indices
            scores: torch.Tensor, the log scores for the choices
                    should be a 3-d float tensor of scores for each possible choice

        returns:
            torch.Tensor: the score for the choices made by the networks
                          2-d float tensor, same shape as choices
        """
        return torch.gather(scores, 2, choices.unsqueeze(2)).squeeze(2)

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
        choice_scores = [self._get_choice_score(choice, score) for choice, score in zip(choices, scores)]

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
