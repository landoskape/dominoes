from abc import ABC, abstractmethod
from copy import copy
import torch


class RequiredParameter:
    """
    a class to represent required parameters for a dataset

    each dataset is required to generate a set of parameters during initialization (in get_class_parameters)
    if any of these parameters are an empty initialization of RequiredParameter, then the dataset will raise
    an error in the __post_init__ method if the parameter is not set during initialization.
    """

    def __init__(self):
        pass


class Dataset(ABC):
    """
    the dataset class is a general purpose dataset for loading and evaluating performance on sequencing problems
    since it is the master class for RL and SL datasets, the only required method is `generate_batch`
    """

    def __post_init__(self):
        """a post-init method to ensure that all the required parameters have been set correctly"""
        for key, value in self.prms.items():
            if isinstance(value, RequiredParameter):
                if not hasattr(self, key):
                    raise ValueError(f"required parameter {key} not set for task {self.task}")

    @abstractmethod
    def get_default_parameters(self):
        """
        return the default parameters for the dataset. This is hard-coded here and only here,
        so if the parameters change, this method should be updated.

        None means the parameter is required and doesn't have a default value. Otherwise,
        the value is the default value for the parameter.

        returns:
            dict, the class-specific parameters for this dataset
        """
        pass

    @abstractmethod
    def process_arguments(self, parameters):
        """
        process the arguments for the dataset

        args:
            parameters: dict, the parameters to process for the dataset

        returns:
            dict, the processed parameters for the dataset
        """
        pass

    def parameters(self, **prms):
        """
        Helper method for handling default parameters for each task

        The way this is designed is for there to be default parameters set at initialization,
        which never change (unless you edit them directly), and then batch-specific parameters
        that you can update for each batch. Here, the default parameters are copied then updated
        by the optional kwargs for this function, then the updated parameters are returned
        and used by whatever method called ``parameters``.

        For more details on possible inputs, look at "get_class_parameters"
        """
        # get registered parameters
        prms_to_use = copy(self.prms)
        # update parameters
        prms_to_use.update(prms)
        # return to caller function
        return prms_to_use

    @abstractmethod
    def get_input_dim(self):
        """required method for getting the input dimension of the dataset"""
        pass

    @abstractmethod
    def get_context_parameters(self):
        """required method for getting the context parameters of the dataset for constructing pointer networks"""
        pass

    @abstractmethod
    def get_max_possible_output(self):
        """required method for getting the maximum possible output for the dataset"""
        pass

    @abstractmethod
    def create_training_variables(self, num_nets, **train_parameters):
        """required method for creating training variables for the dataset"""
        pass

    @abstractmethod
    def save_training_variables(self, training_variables, epoch_state, **train_parameters):
        """required method for saving training variables for the dataset"""
        pass

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
        if device is not None:
            return torch.device(device)
        return self.device

    def input_to_device(self, input, device=None):
        """
        move input to the device for the dataset

        args:
            input: torch.Tensor or tuple of torch.Tensor, the input to move to the device
            device: torch.device, the device to use for the input
                    if device is not provided, will use the device registered upon dataset creation

        returns:
            same as input, but moved to requested device
        """
        device = self.get_device(device)
        return input.to(device)


class DatasetSL(Dataset):
    """
    A child of the general dataset class that is designed for supervised learning tasks
    """

    def __init__(self):
        """
        initialize the dataset for supervised learning

        args:
            loss_function: function, the loss function to use for the dataset
        """
        self.loss_function = torch.nn.functional.nll_loss
        self.loss_kwargs = {}

    def set_loss_function(self, loss_function):
        """simple method for setting the loss function"""
        self.loss_function = loss_function

    def set_loss_kwargs(self, **kwargs):
        """simple method for setting the loss function kwargs"""
        self.loss_kwargs = kwargs

    def measure_loss(self, scores, target, check_divergence=True):
        """
        measure the loss between the scores and target for a set of networks

        assumes the scores are log probabilities of each choice with shape (batch, max_output, num_choices)
        assumes the target is the correct choice with shape (batch, max_output)
        unrolls the scores to a 2-d tensors and measures the loss
        """
        # unroll scores and target to fold sequence dimension into batch dimension
        unrolled = [score.view(-1, score.size(2)) for score in scores]
        loss = [self.loss_function(unroll, target.view(-1), **self.loss_kwargs) for unroll in unrolled]
        if check_divergence:
            for i, l in enumerate(loss):
                assert not torch.isnan(l).item(), f"model {i} diverged :("
        return loss


class DatasetRL(Dataset):
    """
    A child of the general dataset class that is designed for reinforcement learning tasks
    """

    def __init__(self):
        pass

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

    def get_pretemp_scores(self, scores, choices, temperature, return_full_score=False):
        """
        get the pre-temperature score for the choices made by the networks

        args:
            scores: list of torch.Tensor, the log scores for the choices for each network
                    should be a 3-d float tensor of scores for each possible choice
            choices: list of torch.Tensor, index to the choices made by each network
                     should be 2-d Long tensor of indices
            temperatures: list of float, the temperature for each network
            return_full_score: bool, whether to return the full score or just the score for the choices

        returns:
            pretemp_policies: list of torch.Tensor, the score for the choices made by the networks
                              2-d float tensor, same shape as choices
            pretemp_scores: list of torch.Tensor, the pre-temperature score for the choices for each network
                            3-d float tensor, same shape as scores, only returned if return_full_score=True
        """
        # Measure the pre-temperature score of each network (this may have an additive offset, but that's okay)
        pretemp_scores = [torch.softmax(score * temperature, dim=2) for score in scores]
        # Get pre-temperature score for the choices made by the networks
        pretemp_policies = [self.get_choice_score(choice, score) for choice, score in zip(choices, pretemp_scores)]
        if return_full_score:
            return pretemp_policies, pretemp_scores
        return pretemp_policies

    def get_choice_score(self, choices, scores):
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

    def process_rewards(self, rewards, scores, choices, gamma_transform, baseline_rewards=None):
        """
        process the rewards for performing policy gradient

        args:
            rewards: list of torch.Tensor, the rewards for each network (precomputed using `reward_function`)
            scores: list of torch.Tensor, the log scores for the choices for each network
            choices: list of torch.Tensor, index to the choices made by each network
            gamma_transform: torch.Tensor, the gamma transform matrix for the reward
            baseline_rewards: list of torch.Tensor, the baseline rewards for each network (if using baselines)
                              -- if provided, will adjust G to be the advantage of the network

        returns:
            list of torch.Tensor, the rewards for each network
        """
        # measure cumulative discounted rewards for each network
        G = [torch.matmul(reward, gamma_transform) for reward in rewards]

        # if using baselines, adjust G to be the advantage of the network
        if baseline_rewards is not None:
            # measure cumulative discounted rewards for each network
            G_baseline = [torch.matmul(reward, gamma_transform) for reward in baseline_rewards]
            # adjust G to be the advantage of the network
            G = [g - gb for g, gb in zip(G, G_baseline)]

        # measure choice score for each network (the log-probability for each choice)
        choice_scores = [self.get_choice_score(choice, score) for choice, score in zip(choices, scores)]

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
