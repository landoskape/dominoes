from copy import copy
from multiprocessing import Pool, cpu_count
import torch


from .base import DatasetRL


class TSPDataset(DatasetRL):
    """A dataset for generating traveling salesman problem environments for training and evaluation"""

    def __init__(self, **parameters):

        # check parameters
        self._check_parameters(init=True, **parameters)

        # set parameters to required defaults first, then update
        self.prms = self._required_parameters()
        self.prms = self.parameters(**parameters)

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
        return params

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

    @torch.no_grad()
    def generate_batch(self, **kwargs):
        """
        ---- fill this in ----
        """
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

        # make a mask for the input
        mask_tokens = prms["hand_size"] + (1 if self.null_token else 0) + (1 if self.available_token else 0)
        mask = torch.ones((prms["batch_size"], mask_tokens), dtype=torch.float)

        # construct batch dictionary
        batch = dict(input=input, mask=mask, train=train, selection=selection)

        # add task specific parameters to the batch dictionary
        batch.update(prms)

        # if target is requested (e.g. for SL tasks) then get target based on registered task
        if prms["return_target"]:
            # get target dictionary
            target_dict = self.set_target(**prms)
            # update batch dictionary with target dictionary
            batch.update(target_dict)

        return batch

    def set_target(self, **prms):
        """
        --- fill this in ---
        """

    @torch.no_grad()
    def reward_function(self, choices, batch, **kwargs):
        """
        --- fill this in ---
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
