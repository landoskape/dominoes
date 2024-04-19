from .dominoe_dataset import DominoeDataset
from .tsp_dataset import TSPDataset


DATASET_REGISTRY = {
    "dominoes": DominoeDataset,
    "tsp": TSPDataset,
}


def get_dataset(dataset_name, build=False, **kwargs):
    """
    lookup dataset constructor from dataset registry by name

    if build=True, uses kwargs to build dataset and returns a dataset object
    otherwise just returns the constructor
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset ({dataset_name}) is not in DATASET_REGISTRY")
    dataset = DATASET_REGISTRY[dataset_name]
    if build:
        return dataset(**kwargs)

    # Otherwise return the constructor
    return dataset
