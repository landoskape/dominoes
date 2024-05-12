from copy import deepcopy
from scipy.stats import ttest_rel
import torch
from torch import nn

from .net_utils import forward_batch


def make_baseline_nets(nets, dataset, batch_parameters={}, significance=0.05, max_output=None, temperature=1.0, thompson=False):
    """create a copy of each network for baseline calculations"""
    bl_nets = [
        BaselineNetwork(
            net,
            dataset,
            batch_parameters=batch_parameters,
            significance=significance,
            max_output=max_output,
            temperature=temperature,
            thompson=thompson,
        )
        for net in nets
    ]
    return bl_nets


def check_baseline_updates(nets, bl_nets):
    """check if baseline networks should be updated"""
    for net, bl_net in zip(nets, bl_nets):
        bl_net.check_improvement(net)
    return bl_net


class BaselineNetwork(nn.Module):
    def __init__(self, net, dataset, batch_parameters={}, significance=0.05, max_output=None, temperature=1.0, thompson=False):
        """create a baseline network from a pointer network and a reference batch"""
        super().__init__()

        # set network parameters and make a copy of the network
        self.temperature = temperature
        self.thompson = thompson
        self.update_network(net)

        # set dataset and batch parameters
        self.dataset = dataset
        self.batch_parameters = batch_parameters
        self.max_output = max_output
        self.update_reference()

        # set forward kwargs for use in every forward pass
        self.forward_kwargs = dict(
            temperature=self.temperature,
            thompson=self.thompson,
            max_output=self.max_output,
        )

        # set update significance
        self.set_significance(significance)

    def update_reference(self):
        """set the reference batch for the baseline network"""
        self.ref_batch = self.dataset.generate_batch(**self.batch_parameters)
        self.ref_choices = forward_batch([self.net], self.ref_batch, **self.forward_kwargs)[1][0]
        self.ref_rewards = self.dataset.reward_function(self.ref_choices, self.ref_batch)

    @torch.no_grad()
    def update_network(self, net):
        """update the network with a new network"""
        self.net = deepcopy(net)
        self.net.set_temperature(self.temperature)
        self.net.set_thompson(self.thompson)

    def set_significance(self, significance):
        """set the significance level for updating the baseline network"""
        self.significance = significance

    @torch.no_grad()
    def check_improvement(self, net):
        """check if the network should be updated based on the reference batch"""
        choices = forward_batch([net], self.batch, **self.forward_kwargs)[1][0]
        rewards = self.dataset.reward_function(choices, self.batch)
        p = ttest_rel(rewards.view(-1).cpu().numpy(), self.ref_rewards.view(-1).cpu().numpy(), alternative="greater")[1]

        if p < self.significance:
            self.update_network(net)
            self.update_reference()

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """forward pass for baseline network through itself, enforcing temperature and thompson sampling settings"""
        kwargs["temperature"] = self.net.temperature
        kwargs["thompson"] = self.net.thompson
        return self.net.forward(*args, **kwargs)
