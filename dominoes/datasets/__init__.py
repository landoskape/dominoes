from .dominoe_dataset import DominoeSequencer, DominoeSorting, DominoeDataset
from .tsp_dataset import TSPDataset


DATASET_REGISTRY = {
    "dominoes": DominoeDataset,  # used for accessing the dominoe dataset class without a task
    "dominoe_sequencer": DominoeSequencer,  # used for sequencing dominoes according to the standard game
    "dominoe_sorting": DominoeSorting,  # used for sorting dominoes according to their values
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


def get_dataset_parameters(dataset_name):
    """
    get the parameters for a dataset
    """
    _check_dataset(dataset_name)
    return DATASET_REGISTRY[dataset_name].get_class_parameters()


def get_dataset_kwargs(args):
    """
    return a dictionary of kwargs for a dataset from argparse args
    """
    # first get class parameters for the requested task
    parameters = get_dataset_parameters(args.task)
    # get the required parameters (the parameters that need to be set on initialization)
    required = [key for key, value in parameters.items() if value is None]
    # get the permitted parameters (any other parameters that can be set at initialization)
    permitted = [key for key, value in parameters.items() if value is not None]
    # get required parameters from input args (these have to be present or an error will be raised)
    kwargs = {}
    for key in required:
        if key not in args or args[key] is None:
            raise ValueError(f"Required parameter ({key}) for dataset {args.task} not set in input args!")
        kwargs[key] = args[key]
    # get any other permitted parameters from input args if they are set
    kwargs.update({key: args[key] for key in permitted if key in args})
    # return processed kwargs for the dataset
    return kwargs
