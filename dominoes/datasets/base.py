from abc import ABC, abstractmethod
from copy import copy
import torch


class Dataset(ABC):
    """
    the dataset class is a general purpose dataset for loading and evaluating performance on sequencing problems
    since it is the master class for RL and SL datasets, the only required method is `generate_batch`
    """

    @abstractmethod
    def _required_parameters(self):
        """
        return the required parameters for the task. This is hard-coded here and only here,
        so if the parameters change, this method should be updated.

        None means the parameter is required and doesn't have a default value. Otherwise,
        the value is the default value for the parameter.

        returns:
            dict, the required parameters for the task
        """
        pass

    def _check_parameters(self, reference=None, init=False, **task_parameters):
        """
        check if parameters provided in the task_parameters are valid (and complete)

        checks two things:
        1. If any parameters are provided that are not recognized for the task, an error will be generated

        ... if init=True, will additionally check:
        2. If any parameters are required for the task but not provided, an error will be generated

        args:
            reference: dict, the reference parameters to check against (if not provided, uses self._required_parameters())
            init: bool, whether this is being called by the constructor's __init__ method
                  in practive, this determines whether any required parameters without defaults are set properly
            task_parameters: dict, the parameters provided at initialization

        raise ValueError if any parameters are not recognized or required parameters are not provided
        """
        if reference is None:
            reference = self._required_parameters()
        for param in task_parameters:
            if param not in reference:
                raise ValueError(f"parameter {param} not recognized for task {self.task}")
        # if init==True, then this is being called by the constructor's __init__ method and
        # we need to check if any required parameters without defaults are set properly
        if init:
            for param in reference:
                if param not in task_parameters and reference[param] is None:
                    raise ValueError(f"parameter {param} not provided for task {self.task}")

    def parameters(self, **prms):
        """
        Helper method for handling default parameters for each task

        The way this is designed is for there to be default parameters set at initialization,
        which never change (unless you edit them directly), and then batch-specific parameters
        that you can update for each batch. Here, the default parameters are copied then updated
        by the optional kwargs for this function, then the updated parameters are returned
        and used by whatever method called ``parameters``.

        For more details on possible inputs, look at "_required_parameters"
        """
        # get registered parameters
        prms_to_use = copy(self.prms)
        # check if updates are valid
        self._check_parameters(reference=prms_to_use, init=False, **prms)
        # update parameters
        prms_to_use.update(prms)
        # return to caller function
        return prms_to_use

    @abstractmethod
    def generate_batch(self, *args, **kwargs):
        """required method for generating a batch"""

    def set_device(self, device):
        """
        set the device for the dataset

        args:
            device: torch.device, the device to use for the dataset
        """
        self.device = torch.device(device)

    def get_device(self, device=None):
        """
        get the device for the dataset (if not provided, will use the device registered upon dataset creation)

        returns:
            torch.device, the device for the dataset
        """
        return torch.device(device) or self.device


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
