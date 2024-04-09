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


def generateBatch(
    highestDominoe,
    dominoes,
    batch_size,
    numInHand,
    return_target=True,
    value_method="dominoe",
    available_token=False,
    null_token=False,
    ignore_index=-1,
    return_full=False,
):

    input, selection, available = randomDominoeHand(
        numInHand, dominoes, highestDominoe, batch_size=batch_size, null_token=null_token, available_token=available_token
    )

    mask_tokens = numInHand + (1 if null_token else 0) + (1 if available_token else 0)
    mask = torch.ones((batch_size, mask_tokens), dtype=torch.float)

    if return_target:
        # then measure best line and convert it to a "target" array
        if available_token:
            bestSequence, bestDirection = getBestLineFromAvailable(dominoes, selection, available, value_method=value_method)
        else:
            bestSequence, bestDirection = getBestLine(dominoes, selection, highestDominoe, value_method=value_method)

        # convert sequence to hand index
        iseq = convertToHandIndex(selection, bestSequence)

        # create target and append null_index once, then ignore_index afterwards
        # the idea is that the agent should play the best line, then indicate that the line is over, then anything else doesn't matter
        null_index = numInHand if null_token else ignore_index
        target = torch.tensor(np.stack(padBestLine(iseq, numInHand + 1, null_index, ignore_index=ignore_index)), dtype=torch.long)
    else:
        # otherwise set these to None so we can use the same return structure
        target, bestSequence, bestDirection = None, None, None

    if return_full:
        return input, target, mask, bestSequence, bestDirection, selection, available
    return input, target, mask
