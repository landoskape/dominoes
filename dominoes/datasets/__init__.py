from .dominoe_dataset import DominoeSequencer, DominoeSorter, DominoeDataset
from .tsp_dataset import TSPDataset


DATASET_REGISTRY = {
    "dominoes": DominoeDataset,  # used for accessing the dominoe dataset class without a task
    "dominoe_sequencer": DominoeSequencer,  # used for sequencing dominoes according to the standard game
    "dominoe_sorter": DominoeSorter,  # used for sorting dominoes according to their values
    "tsp": TSPDataset,  # used for solving the traveling salesman problem
}


def _check_dataset(dataset_name):
    """
    check if a dataset is in the dataset registry
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset ({dataset_name}) is not in DATASET_REGISTRY")


def get_dataset(dataset_name, build=False, **kwargs):
    """
    lookup dataset constructor from dataset registry by name

    if build=True, uses kwargs to build dataset and returns a dataset object
    otherwise just returns the constructor
    """
    _check_dataset(dataset_name)
    dataset = DATASET_REGISTRY[dataset_name]

    # build and return the dataset if requested using the kwargs
    if build:
        return dataset(**kwargs)

    # Otherwise return the constructor
    return dataset
