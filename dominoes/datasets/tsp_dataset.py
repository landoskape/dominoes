from copy import copy
import torch


from .base import DatasetRL
from .support import get_paths


class TSPDataset(DatasetRL):
    """A dataset for generating traveling salesman problem environments for training and evaluation"""

    def __init__(self, device="cpu", **parameters):
        """constructor method"""
        self.set_device(device)

        # check parameters
        self._check_parameters(init=True, **parameters)

        # set parameters to required defaults first, then update
        self.prms = self._required_parameters()
        self.prms = self.parameters(**parameters)

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
            num_cities=None,  # this parameter is required to be set at initialization
            coord_dims=2,
            batch_size=1,
            return_target=False,
            ignore_index=-1,
            threads=1,
        )
        return params

    @torch.no_grad()
    def generate_batch(self, device=None, **kwargs):
        """
        generates a batch of TSP environments with the specified parameters and additional outputs

        parallelized preparation of batch, better to use 1 thread if num_cities~<10 or batch_size<=256

        batch keys:
            input: torch.Tensor, the input to the network, as a binary dominoe representation (and null token)
            train: bool, whether the batch is for training or evaluation
            selection: torch.Tensor, the selection of dominoes in the hand
            target: torch.Tensor, the target for the network (only if requested)
        """
        # get device
        device = self.get_device(device)

        # get parameters with potential updates
        prms = self.parameters(**kwargs)

        # def tsp_batch(batch_size, num_cities, return_target=True, return_full=False, threads=1):
        input = torch.rand((prms["batch_size"], prms["num_cities"], prms["coord_dims"]), dtype=torch.float)
        dists = torch.cdist(input)

        # define initial position as closest to origin (arbitrary but standard choice)
        init_idx = torch.argmin(torch.sum(input**2, dim=2), dim=1)

        # get representation of initial position (will be fed to decoder)
        init_input = torch.gather(input, 1, init_idx.view(-1, 1, 1).expand(-1, -1, prms["coord_dims"]))

        # construct batch dictionary
        batch = dict(input=input.to(device), dists=dists, init_idx=init_idx, init_input=init_input)

        # add task specific parameters to the batch dictionary
        batch.update(prms)

        if prms["return_target"]:
            batch["target"] = get_paths(input, dists, init_idx, prms["threads"]).to(device)

        return batch

    @torch.no_grad()
    def reward_function(self, choices, batch, **kwargs):
        """
        measure the reward acquired by the choices made by a set of networks for the current batch


        rewards are 1 when a dominoe is chosen that:
        - hasn't been played yet
        - has less than or equal value to the last dominoe played (first dominoe always valid)

        rewards are -1 when a dominoe is chosen that:
        - has already been played
        - has greater value than the last dominoe played

        note: rewards are set to 0 after a mistake is made




        args:
            choice: torch.Tensor, index to the choices made by the network
            batch: tuple, the batch of data generated for this training step
            kwargs: not used, here for consistency with other dataset types

        returns:
            torch.Tensor, the rewards for the network
        """
        assert choices.ndim == 2, "choices should be a 2-d tensor of the sequence of choices for each batch element"
        num_cities = batch["num_cities"]
        batch_size = num_choices = choices.shape
        device = choices.device

        distance = torch.zeros((batch_size, num_choices)).to(device)
        new_city = torch.ones((batch_size, num_choices)).to(device)

        last_location = batch["init_idx"]  # last (i.e. initial position) is final step of permutation of cities

        last_location = copy(choices[:, 0])  # last (i.e. initial position) is final step of permutation of cities
        src = torch.ones((batchSize, 1), dtype=torch.bool).to(device)
        visited = torch.zeros((batchSize, numChoices), dtype=torch.bool).to(device)
        visited.scatter_(1, last_location.view(batchSize, 1), src)  # put first city in to the "visited" tensor
        for nc in range(1, numChoices):
            next_location = choices[:, nc]
            c_dist_possible = torch.gather(dists, 1, last_location.view(batchSize, 1, 1).expand(-1, -1, numCities)).squeeze(1)
            distance[:, nc] = torch.gather(c_dist_possible, 1, next_location.view(batchSize, 1)).squeeze(1)
            c_visited = torch.gather(visited, 1, next_location.view(batchSize, 1)).squeeze(1)
            visited.scatter_(1, next_location.view(batchSize, 1), src)
            new_city[c_visited, nc] = -1.0
            new_city[~c_visited, nc] = 1.0
            last_location = copy(next_location)  # update last location

        # add return step (to initial city) to the final choice
        c_dist_possible = torch.gather(dists, 1, choices[:, 0].view(batchSize, 1, 1).expand(-1, -1, numCities)).squeeze(1)
        distance[:, -1] += torch.gather(c_dist_possible, 1, choices[:, -1].view(batchSize, 1)).squeeze(1)

        return distance, new_city


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
