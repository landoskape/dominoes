import matplotlib.pyplot as plt
import matplotlib as mpl

from .utils import compute_stats_by_type


def plot_basic_results(results, labels, train=True):
    num_types = len(labels)

    if "loss" in results and results["loss"] is not None:
        loss = compute_stats_by_type(results["loss"], num_types, 1)[0]
        for i in range(num_types):
            plt.plot(loss[:, i], label=labels[i])
        plt.legend(fontsize=8)
        plt.show()

    if "reward" in results and results["reward"] is not None:
        reward = compute_stats_by_type(results["reward"], num_types, 1)[0]
        for i in range(num_types):
            plt.plot(reward[:, i], label=labels[i])
        plt.legend(fontsize=8)
        plt.show()
