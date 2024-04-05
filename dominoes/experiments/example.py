import torch
from . import arglib
from .experiment import Experiment


class ExampleExperiment(Experiment):
    def get_basename(self):
        return "example_experiment"

    def prepare_path(self):
        return [self.args.network, self.args.dataset, self.args.optimizer]

    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser
        """
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dropout_experiment_details(parser)
        parser = arglib.add_network_metaparameters(parser)
        parser = arglib.add_alignment_analysis_parameters(parser)
        return parser

    def create_networks(self):
        """
        method for creating networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """
        # get optimizer
        if self.args.optimizer == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer}) not recognized")

        nets = [
            get_model(
                self.args.network,
                build=True,
                dataset=self.args.dataset,
                dropout=self.args.default_dropout,
                ignore_flag=self.args.ignore_flag,
            )
            for _ in range(self.args.replicates)
        ]
        nets = [net.to(self.device) for net in nets]

        optimizers = [optim(net.parameters(), lr=self.args.default_lr, weight_decay=self.args.default_wd) for net in nets]

        prms = {
            "vals": [self.args.network],  # require iterable for identifying how many types of networks there are (just one type...)
            "name": "network",
            "dataset": self.args.dataset,
            "dropout": self.args.default_dropout,
            "lr": self.args.default_lr,
            "weight_decay": self.args.default_wd,
        }
        return nets, optimizers, prms

    def main(self):
        """
        main experiment loop

        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """

        # create networks
        nets, optimizers, prms = self.create_networks()

        # load dataset
        dataset = self.prepare_dataset(nets[0])

        # train networks
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

        # do targeted dropout experiment
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do targeted dropout experiment
        evec_dropout_results, evec_dropout_parameters = processing.eigenvector_dropout(self, nets, dataset, eigen_results, train_set=False)

        # make full results dictionary
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            dropout_results=dropout_results,
            dropout_parameters=dropout_parameters,
            eigen_results=eigen_results,
            evec_dropout_results=evec_dropout_results,
            evec_dropout_parameters=evec_dropout_parameters,
        )

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
        plotting.plot_train_results(self, results["train_results"], results["test_results"], results["prms"])
        plotting.plot_dropout_results(
            self,
            results["dropout_results"],
            results["dropout_parameters"],
            results["prms"],
            dropout_type="nodes",
        )
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        plotting.plot_dropout_results(
            self,
            results["evec_dropout_results"],
            results["evec_dropout_parameters"],
            results["prms"],
            dropout_type="eigenvectors",
        )
