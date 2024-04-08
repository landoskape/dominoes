from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    the dataset class is a general purpose dataset for loading and evaluating performance on sequencing problems
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization function --
        I think I should rewrite this to implement some required functions (e.g. self.initialize()) that is
        a required abstract method in children of the dominoe dataset
        """
        self.initialize(*args, **kwargs)

    # @abstractmethod
    # def check_parameters(self, **prms):
    #     """used to check whether given parameters meet requirements"""

    # @abstractmethod
    # def set_parameters(self, **prms):
    #     """used upon dataset construction to check required parameters for each dataset type"""

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """required method for initializing the dataset"""

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


class DominoeDataset(Dataset):
    def initialize(
        self,
        task,
        highest_dominoe,
        train_fraction,
        batch_size,
        min_seq_length,
        max_seq_length=None,
        null_token=False,
        available_token=False,
        ignore_index=-1,
        return_target=False,
    ):
        self.task = self.check_task(task)
        self.highest_dominoe = highest_dominoe
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.null_token = null_token
        self.available_token = available_token
        self.ignore_index = ignore_index
        self.return_target = return_target

    def generate_batch(self, train=True, **kwargs):
        """
        generates a batch of dominoes with the required additional outputs
        """
        prms = self.parameters(**kwargs)  # get parameters with potential updates
        batch = self.make_batch(**kwargs)  # make batch with requested parameters

        # batch = datasets.generateBatch(highestDominoe, trainDominoes, batchSize, handSize, **batch_inputs)
        # batch_inputs = {
        #     "null_token": False,
        #     "available_token": False,
        #     "ignore_index": ignoreIndex,
        #     "return_full": True,
        #     "return_target": False,
        # }
