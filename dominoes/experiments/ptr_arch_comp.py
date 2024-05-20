# standard imports
import torch

# dominoes package
from .. import train
from ..networks import get_pointer_network, get_pointer_methods, get_pointer_arguments
from .base import Experiment
from . import arglib


class PointerArchitectureComparison(Experiment):
    """
    An experiment class for running pointer architecture comparisons

    Can handle a variety of learning algorithms, datasets, and other variations
    (not yet, but it will)
    """

    def get_basename(self):
        """basename for experiment (determines folder, filenames, etc.)"""
        return "ptr_arch_comp"

    def prepare_path(self):
        """prepare the path for the experiment"""
        return [self.args.task, self.args.learning_mode]

    def make_args(self, parser):
        """Method for adding experiment specific arguments to the argument parser"""
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_network_training_metaparameters(parser)
        parser = arglib.add_pointernet_parameters(parser)
        parser = arglib.add_pointernet_encoder_parameters(parser)
        parser = arglib.add_pointernet_decoder_parameters(parser)
        parser = arglib.add_pointernet_pointer_parameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dataset_parameters(parser)
        parser = arglib.add_tsp_parameters(parser)
        parser = arglib.add_dominoe_parameters(parser)
        parser = arglib.add_dominoe_sequencer_parameters(parser)
        parser = arglib.add_dominoe_sorting_parameters(parser)
        return parser

    def pointer_methods(self):
        """get pointer methods for architecture comparisons"""
        # option to use arguments to exclude pointer methods...
        # but for now just use all of them
        return get_pointer_methods()

    def create_networks(self, input_dim, context_parameters):
        """
        method for creating networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """

        # create networks
        embedding_dim, pointer_kwargs = get_pointer_arguments(self.args)

        # Remove pointer method because it's the variable that this experiment is exploring
        _ = pointer_kwargs.pop("pointer_method", None)

        nets = [
            get_pointer_network(
                input_dim,
                embedding_dim,
                pointer_method=pointer_method,
                **context_parameters,
                **pointer_kwargs,
            )
            for pointer_method in self.pointer_methods()
            for _ in range(self.args.replicates)
        ]
        nets = [net.to(self.device) for net in nets]
        optimizers = [torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.wd) for net in nets]

        prms = {}
        return nets, optimizers, prms

    def main(self):
        """
        main experiment loop

        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """
        # load dataset
        dataset = self.prepare_dataset()

        # input dimensionality
        input_dim = dataset.get_input_dim()
        context_parameters = dataset.get_context_parameters()

        # create networks
        nets, optimizers, prms = self.create_networks(input_dim, context_parameters)

        # train networks
        train_parameters = self.make_train_parameters(dataset)

        train_results = train.train(nets, optimizers, dataset, **train_parameters)

        # make full results dictionary
        results = dict(train_results=train_results)

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
        pointer_methods = self.pointer_methods()
