import matplotlib.pyplot as plt
import matplotlib as mpl

from .utils import compute_stats_by_type


def plot_train_results(exp, results, labels, name="train"):
    """
    Simplest plot method for plotting loss/reward (or something else eventually) across epochs

    Assumes that the loss/rewards are divided into len(labels) types across the first dimension
    (see below for variable names), will make a plot of the mean across epochs for each type and
    label it accordingly.

    The experiment object passed in as the first argument determines if the plot is saved or shown.
    """
    num_types = len(labels)

    if "loss" in results and results["loss"] is not None:
        loss = compute_stats_by_type(results["loss"], num_types, 1)[0]
        for i in range(num_types):
            plt.plot(loss[:, i], label=labels[i])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{name} loss")
        plt.legend(fontsize=8)

        exp.plot_ready(f"{name}_loss")

    if "reward" in results and results["reward"] is not None:
        reward = compute_stats_by_type(results["reward"], num_types, 1)[0]
        for i in range(num_types):
            plt.plot(reward[:, i], label=labels[i])
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.title(f"{name} reward")
        plt.legend(fontsize=8)

        exp.plot_ready(f"{name}_reward")