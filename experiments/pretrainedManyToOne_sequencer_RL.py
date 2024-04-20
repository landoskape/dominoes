import sys
import os

# add path that contains the dominoes package
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

# standard imports
from copy import copy, deepcopy
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.cuda as torchCuda

# dominoes package
from dominoes import files as fm
from dominoes import utils
from dominoes import datasets
from dominoes import training
from dominoes.networks import transformers as transformers
from dominoes.utils import loadSavedExperiment

device = "cuda" if torchCuda.is_available() else "cpu"

# general variables for experiment
POINTER_METHODS = ["PointerStandard", "PointerDot", "PointerDotLean", "PointerDotNoLN", "PointerAttention", "PointerTransformer"]

# path strings
netPath = fm.netPath()
resPath = fm.resPath()
prmsPath = fm.prmPath()
figsPath = fm.figsPath()

for path in (resPath, prmsPath, figsPath, netPath):
    if not (path.exists()):
        path.mkdir()


# method for returning the name of the saved network parameters (different save for each possible opponent)
def getFileName(extra=None):
    baseName = "pretrainedManyToOne_sequencer_RL"
    if hasattr(args, "nobaseline") and not (args.nobaseline):
        baseName += "_withBaseline"
    baseName += args.pointer_layer
    if args.extraname is not None:
        baseName += f"_{args.extraname}"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName


def handleArguments():
    parser = argparse.ArgumentParser(description="Run pointer dominoe sequencing experiment.")
    parser.add_argument("-hd", "--highest-dominoe", type=int, default=9, help="the highest dominoe in the board")
    parser.add_argument("-hs", "--hand-size", type=int, default=12, help="the maximum tokens per sequence")
    parser.add_argument("-bs", "--batch-size", type=int, default=128, help="number of sequences per batch")
    parser.add_argument("-ne", "--train-epochs", type=int, default=12000, help="the number of training epochs")
    parser.add_argument("-te", "--test-epochs", type=int, default=100, help="the number of testing epochs")
    parser.add_argument("-nr", "--num-runs", type=int, default=8, help="how many runs for each network to train")
    parser.add_argument("--gamma", type=float, default=0.9, help="discounting factor")
    parser.add_argument("--temperature", type=float, default=5.0, help="temperature for training")
    parser.add_argument("--pointer-layer", type=str, default="PointerStandard", help="the pointer layer to replace and specifically train")

    parser.add_argument("--nobaseline", default=False, action="store_true")
    parser.add_argument("--significance", default=0.05, type=float, help="significance of reward improvement for baseline updating")
    parser.add_argument("--baseline-batch-size", default=1024, type=int, help="the size of the baseline batch to use")

    parser.add_argument("--embedding_dim", type=int, default=128, help="the dimensions of the embedding")
    parser.add_argument("--heads", type=int, default=8, help="the number of heads in transformer layers")
    parser.add_argument("--expansion", type=int, default=4, help="the expansion at the MLP part of the transformer")
    parser.add_argument("--encoding-layers", type=int, default=2, help="the number of stacked transformers in the encoder")
    parser.add_argument(
        "--justplot",
        default=False,
        action="store_true",
        help="if used, will only plot the saved results (results have to already have been run and saved)",
    )
    parser.add_argument("--nosave", default=False, action="store_true")
    parser.add_argument("--printargs", default=False, action="store_true")
    parser.add_argument("--extraname", default=None, type=str, help="extra string to be appended to basename for saving")

    args = parser.parse_args()

    assert args.gamma > 0 and args.gamma <= 1, "gamma must be greater than 0 or less than or equal to 1"
    assert args.pointer_layer in POINTER_METHODS, "--pointer-layer must be one of the pointer layer architectures in POINTER_METHODS"

    return args


# method for returning the name of the saved network parameters (different save for each possible opponent)
def pretrainedFileName(extra=None):
    baseName = "ptrArchComp_sequencer_RL"
    if hasattr(args, "nobaseline") and not (args.nobaseline):
        baseName += "_withBaseline"
    if args.extraname is not None:
        baseName += f"_{args.extraname}"
    if extra is not None:
        baseName = baseName + f"_{extra}"
    return baseName


# Parameters of arguments that have to be the same as saved parameters
force_same = ("hand_size", "embedding_dim", "heads", "expansion", "encoding_layers")


def loadNetworks():
    # Load previously stored settings and pretrained networks
    nets = [torch.load(fm.netPath() / pretrainedFileName(extra=f"{method}.pt")) for method in POINTER_METHODS]
    for net in nets:
        net.pointer_method = args.pointer_layer
        net.pointer.pointer_method = args.pointer_layer
        # build pointer (chooses an output)
        if args.pointer_layer == "PointerStandard":
            # output of the network uses a pointer attention layer as described in the original paper
            net.pointer.pointer = transformers.PointerStandard(net.pointer.embedding_dim)

        elif args.pointer_layer == "PointerDot":
            net.pointer.pointer = transformers.PointerDot(net.pointer.embedding_dim)

        elif args.pointer_layer == "PointerDotNoLN":
            net.pointer.pointer = transformers.PointerDotNoLN(net.pointer.embedding_dim)

        elif args.pointer_layer == "PointerDotLean":
            net.pointer.pointer = transformers.PointerDotLean(net.pointer.embedding_dim)

        elif args.pointer_layer == "PointerAttention":
            kwargs = {"heads": net.heads, "kqnorm": net.kqnorm}
            net.pointer.pointer = transformers.PointerAttention(net.pointer.embedding_dim, **kwargs)

        elif args.pointer_layer == "PointerTransformer":
            kwargs = {"heads": net.heads, "expansion": 1, "kqnorm": net.kqnorm, "bias": net.bias}
            net.pointer.pointer = transformers.PointerTransformer(net.pointer.embedding_dim, **kwargs)

        else:
            raise ValueError(f"the pointer_method was not set correctly, {args.pointer_layer} not recognized")

    nets = [net.to(device) for net in nets]

    # Define parameters to be learned and turn off gradients in all other parameters
    learning_parameters = [[] for _ in nets]
    for learn, net in zip(learning_parameters, nets):
        for name, prm in net.named_parameters():
            if "pointer.pointer" in name:
                prm.requires_grad = True
                learn.append(prm)
            else:
                prm.requires_grad = False

    # Prepare optimizer for the specific parameters to be learned
    optimizers = [torch.optim.Adam(learn, lr=1e-3, weight_decay=1e-5) for learn in learning_parameters]

    return nets, optimizers


def resetBaselines(blnets, batchSize, highestDominoe, listDominoes, handSize, num_output, value_method, **kwargs):
    # initialize baseline input (to prevent overfitting with training data)
    batch = datasets.generateBatch(highestDominoe, listDominoes, batchSize, handSize, **kwargs)
    baselineinput, _, _, _, _, baseline_selection, baseline_available = batch
    baselineinput = baselineinput.to(device)  # move to device

    # divide input into main input and context
    baseline_x, baseline_context = baselineinput[:, :-1].contiguous(), baselineinput[:, [-1]]  # input [:, [-1]] is context token
    baseline_input = (baseline_x, baseline_context)

    _, baseline_choices = map(list, zip(*[net(baseline_input, max_output=num_output) for net in blnets]))

    # measure rewards for each sequence
    baseline_rewards = [
        training.measureReward_sequencer(baseline_available, listDominoes[baseline_selection], choice, value_method=value_method, normalize=False)
        for choice in baseline_choices
    ]

    return baseline_input, baseline_selection, baseline_available, baseline_rewards


def get_gamma_transform(gamma, N):
    exponent = torch.arange(N).view(-1, 1) - torch.arange(N).view(1, -1)
    gamma_transform = gamma**exponent * (exponent >= 0)
    return gamma_transform


def trainTestModel():
    # get values from the argument parser
    highestDominoe = args.highest_dominoe
    listDominoes = utils.listDominoes(highestDominoe)

    handSize = args.hand_size
    batchSize = args.batch_size
    null_token = True  # using a null token to indicate end of line
    null_index = copy(handSize)  # index of null token
    available_token = True  # using available token to indicate which value to start on
    ignore_index = -100
    value_method = "1"  # method for generating rewards in reward function

    num_output = copy(handSize)
    gamma = args.gamma
    gamma_transform = get_gamma_transform(gamma, num_output).to(device)

    # other batch parameters
    batchSize = args.batch_size
    baselineBatchSize = args.baseline_batch_size
    significance = args.significance
    do_baseline = not (args.nobaseline)

    # network parameters
    temperature = args.temperature

    # train parameters
    trainEpochs = args.train_epochs
    testEpochs = args.test_epochs
    numRuns = args.num_runs

    numNets = len(POINTER_METHODS)

    print(f"Doing training...")
    trainReward = torch.zeros((trainEpochs, numNets, numRuns))
    trainRewardBaseline = torch.zeros((trainEpochs, numNets, numRuns))
    testReward = torch.zeros((testEpochs, numNets, numRuns))
    testEachReward = torch.zeros((testEpochs, numNets, numRuns, batchSize))
    testMaxReward = torch.zeros((testEpochs, numRuns, batchSize))
    for run in range(numRuns):
        print(f"Training round of networks {run+1}/{numRuns}...")

        # Load pretrained networks with randomly initialized pointer layers
        nets, optimizers = loadNetworks()
        for net in nets:
            net.setTemperature(temperature)
            net.setThompson(True)

        if do_baseline:
            # create baseline nets, initialized as copy of learning nets
            blnets = [deepcopy(net) for net in nets]
            for blnet in blnets:
                blnet.setTemperature(1.0)
                blnet.setThompson(True)

            baseline_kwargs = dict(
                return_target=False, null_token=null_token, available_token=available_token, ignore_index=ignore_index, return_full=True
            )
            baseline_data = resetBaselines(
                blnets, baselineBatchSize, highestDominoe, listDominoes, handSize, num_output, value_method, **baseline_kwargs
            )
            baseline_input, baseline_selection, baseline_available, baseline_rewards = baseline_data

        for epoch in tqdm(range(trainEpochs)):
            # generate input batch
            batch = datasets.generateBatch(
                highestDominoe,
                listDominoes,
                batchSize,
                handSize,
                return_target=False,
                null_token=null_token,
                available_token=available_token,
                ignore_index=ignore_index,
                return_full=True,
            )

            # unpack batch tuple
            input, _, _, _, _, selection, available = batch

            # move to correct device
            input = input.to(device)

            # divide input into main input and context
            x, context = input[:, :-1].contiguous(), input[:, [-1]]  # input [:, [-1]] is context token
            input = (x, context)

            # zero gradients, get output of network
            for opt in optimizers:
                opt.zero_grad()
            log_scores, choices = map(list, zip(*[net(input, max_output=num_output) for net in nets]))

            # measure rewards for each sequence
            rewards = [
                training.measureReward_sequencer(available, listDominoes[selection], choice, value_method=value_method, normalize=False)
                for choice in choices
            ]
            G = [torch.matmul(reward, gamma_transform) for reward in rewards]
            logprob_policy = [
                torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)
            ]  # log-probability for each chosen dominoe

            if do_baseline:
                # - do "baseline rollout" with baseline networks -
                with torch.no_grad():
                    _, bl_choices = map(list, zip(*[net(input, max_output=num_output) for net in blnets]))
                    bl_rewards = [
                        training.measureReward_sequencer(available, listDominoes[selection], choice, value_method=value_method, normalize=False)
                        for choice in bl_choices
                    ]
                    bl_G = [torch.matmul(reward, gamma_transform) for reward in bl_rewards]
                    adjusted_G = [g - blg for g, blg in zip(G, bl_G)]  # baseline corrected G
            else:
                # for a consistent namespace
                adjusted_G = [g for g in G]

            # do backward pass on J and update networks
            for policy, g, opt in zip(logprob_policy, adjusted_G, optimizers):
                J = -torch.sum(policy * g)
                J.backward()
                opt.step()

            # save training data
            for i, reward in enumerate(rewards):
                trainReward[epoch, i, run] = torch.mean(torch.sum(reward, dim=1))

            # check if we should update baseline networks
            if do_baseline:
                with torch.no_grad():
                    # first measure policy on baseline data (in evaluation mode)
                    for net in nets:
                        net.setTemperature(1.0)
                        net.setThompson(False)

                    _, choices = map(list, zip(*[net(baseline_input, max_output=num_output) for net in nets]))
                    rewards = [
                        training.measureReward_sequencer(
                            baseline_available, listDominoes[baseline_selection], choice, value_method=value_method, normalize=False
                        )
                        for choice in choices
                    ]

                    # Store exploitative reward for learning networks throughout training
                    # (I know it's a different dataset, but I have to do this any and want to
                    # avoid too many forward passes...)
                    for i, reward in enumerate(rewards):
                        trainRewardBaseline[epoch, i, run] = torch.mean(torch.sum(reward, dim=1))

                    _, p = map(
                        list,
                        zip(
                            *[
                                ttest_rel(r.view(-1).cpu().numpy(), blr.view(-1).cpu().numpy(), alternative="greater")
                                for r, blr in zip(rewards, baseline_rewards)
                            ]
                        ),
                    )
                    do_update = [pv < significance for pv in p]

                    # for any networks with significantly different values, update them
                    for ii, update in enumerate(do_update):
                        if update:
                            blnets[ii] = deepcopy(nets[ii])
                            blnets[ii].setTemperature(1.0)
                            blnets[ii].setThompson(False)

                    # regenerate baseline data and get baseline network policy
                    if any(do_update):
                        baseline_data = resetBaselines(
                            blnets, baselineBatchSize, highestDominoe, listDominoes, handSize, num_output, value_method, **baseline_kwargs
                        )
                        baseline_input, baseline_selection, baseline_available, baseline_rewards = baseline_data

                    # return nets to training state
                    for net in nets:
                        net.setTemperature(temperature)
                        net.setThompson(True)

        with torch.no_grad():
            print("Testing network...")
            for net in nets:
                net.setTemperature(1.0)
                net.setThompson(False)

            for epoch in tqdm(range(testEpochs)):
                # generate input batch
                batch = datasets.generateBatch(
                    highestDominoe,
                    listDominoes,
                    batchSize,
                    handSize,
                    return_target=True,
                    null_token=null_token,
                    available_token=available_token,
                    ignore_index=ignore_index,
                    return_full=True,
                    value_method="length",
                )

                # unpack batch tuple
                input, target, _, _, _, selection, available = batch
                assert torch.all(torch.sum(target == null_index, dim=1) == 1), "null index is present more or less than once in at least one target"

                # move to correct device
                input, target = input.to(device), target.to(device)
                target = target[:, :num_output].contiguous()
                target[target == ignore_index] = null_index  # need to convert this to a valid index for measuring reward of target

                # divide input into main input and context
                x, context = input[:, :-1].contiguous(), input[:, [-1]]  # input [:, [-1]] is context token
                input = (x, context)

                log_scores, choices = map(list, zip(*[net(input, max_output=num_output) for net in nets]))

                # measure rewards for each sequence
                rewards = [
                    training.measureReward_sequencer(available, listDominoes[selection], choice, value_method=value_method, normalize=False)
                    for choice in choices
                ]

                # save testing data
                for i, reward in enumerate(rewards):
                    testReward[epoch, i, run] = torch.mean(torch.sum(reward, dim=1))
                    testEachReward[epoch, i, run] = torch.sum(reward, dim=1)

                # measure rewards for target (defined as longest possible sequence of the dominoes in the batch
                target_reward = training.measureReward_sequencer(
                    available, listDominoes[selection], target, value_method=value_method, normalize=False
                )
                testMaxReward[epoch, run] = torch.sum(target_reward, dim=1)

    results = {
        "trainReward": trainReward,
        "trainRewardBaseline": trainRewardBaseline,
        "testReward": testReward,
        "testEachReward": testEachReward,
        "testMaxReward": testMaxReward,
    }

    return results, nets


def plotResults(results, args):
    numRuns = args.num_runs
    cmap = mpl.colormaps["tab10"]

    # Process test results in comparison to maximum possible
    minMaxReward = torch.min(results["testMaxReward"])
    maxMaxReward = torch.max(results["testMaxReward"])
    uniqueRewards = torch.arange(minMaxReward, maxMaxReward + 1)
    numUnique = len(uniqueRewards)
    rewPerMax = torch.zeros((len(POINTER_METHODS), numUnique, numRuns))
    for iur, ur in enumerate(uniqueRewards):
        idx_ur = results["testMaxReward"] == ur
        for ii, name in enumerate(POINTER_METHODS):
            for ir in range(numRuns):
                rewPerMax[ii, iur, ir] = torch.mean(results["testEachReward"][:, ii, ir][idx_ur[:, ir, :]])

    # make plot of performance trajectory
    fig, ax = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[2.6, 1], layout="constrained")
    for idx, name in enumerate(POINTER_METHODS):
        adata = median_filter(results["trainReward"][:, idx], size=(10, 1))
        cdata = np.mean(adata, axis=1)
        sdata = np.std(adata, axis=1)
        ax[0].plot(range(args.train_epochs), cdata, color=cmap(idx), lw=1.2, label=name)
        ax[0].fill_between(range(args.train_epochs), cdata + sdata / 2, cdata - sdata / 2, edgecolor="none", facecolor=(cmap(idx), 0.3))
    ax[0].set_ylim(-1, 8)
    ax[0].set_ylabel(f"Total Reward (N={numRuns})")
    ax[0].set_title("Training Performance")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_xticks([0, 2500, 5000, 7500, 10000])
    ylims = ax[0].get_ylim()

    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0] + idx, xOffset[1] + idx]
    for idx, name in enumerate(POINTER_METHODS):
        mnTestReward = torch.mean(results["testReward"][:, idx], dim=0)
        ax[1].plot(get_x(idx), [mnTestReward.mean(), mnTestReward.mean()], color=cmap(idx), lw=4, label=name)
        for mtr in mnTestReward:
            ax[1].plot(get_x(idx), [mtr, mtr], color=cmap(idx), lw=1.5)
        ax[1].plot([idx, idx], [mnTestReward.min(), mnTestReward.max()], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(len(POINTER_METHODS)))
    ax[1].set_xticklabels([pmethod[7:] for pmethod in POINTER_METHODS], rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel(f"Reward (N={numRuns})")
    ax[1].set_title("Testing")
    ax[1].set_xlim(-1, len(POINTER_METHODS))
    ax[1].set_ylim(ylims)

    if not (args.nosave):
        plt.savefig(str(figsPath / getFileName()))

    plt.show()

    # Plot rewards in comparison to maximum possible for each network type
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
    for idx, name in enumerate(POINTER_METHODS):
        adata = rewPerMax[idx]
        cdata = torch.mean(adata, dim=1)
        sdata = torch.std(adata, dim=1)
        ax.plot(uniqueRewards, cdata, color=cmap(idx), lw=1.2, marker="o", markersize=4, label=name)
        ax.fill_between(uniqueRewards, cdata + sdata / 2, cdata - sdata / 2, edgecolor="none", facecolor=(cmap(idx), 0.3))
    ax.plot(uniqueRewards, uniqueRewards, color="k", lw=1.2, linestyle="--", label="max possible")
    ax.set_ylim(0, max(uniqueRewards) + 1)
    ax.set_xticks(uniqueRewards)
    ax.set_xlabel("Maximum Possible Reward")
    ax.set_ylabel("Actual Reward Acquired")
    ax.legend(loc="upper left", fontsize=10)

    if not (args.nosave):
        plt.savefig(str(figsPath / getFileName("maxRewardDifferential")))

    plt.show()

    # Plot baseline rewards to measure how networks are learning without the influence of temperature
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
    for idx, name in enumerate(POINTER_METHODS):
        adata = median_filter(results["trainRewardBaseline"][:, idx], size=(10, 1))
        cdata = np.mean(adata, axis=1)
        sdata = np.std(adata, axis=1)
        ax.plot(range(args.train_epochs), cdata, color=cmap(idx), lw=1.2, label=name)
        ax.fill_between(range(args.train_epochs), cdata + sdata / 2, cdata - sdata / 2, edgecolor="none", facecolor=(cmap(idx), 0.3))
    ax.set_ylim(-1, 8)
    ax.set_ylabel(f"Total Reward (N={numRuns})")
    ax.set_title("Training Performance (Exploit)")
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks([0, 2500, 5000, 7500, 10000])

    if not (args.nosave):
        plt.savefig(str(figsPath / getFileName("trainPerformanceExploit")))

    plt.show()


if __name__ == "__main__":
    args = handleArguments()
    show_results = True

    if args.printargs:
        _, args = loadSavedExperiment(prmsPath, resPath, getFileName(), args=args)
        for key, val in vars(args).items():
            print(f"{key}={val}")
        show_results = False

    elif not (args.justplot):
        # train and test pointerNetwork

        # first load stored arguments and make sure they match currently requested arguments
        _, stored_args = utils.loadSavedExperiment(fm.prmPath(), fm.resPath(), pretrainedFileName())
        for attr in force_same:
            if getattr(args, attr) != getattr(stored_args, attr):
                raise ValueError(f"Requested attribute {attr}={getattr(args, attr)} is different from saved ={getattr(stored_args, attr)}")

        results, nets = trainTestModel()

        # save results if requested
        if not (args.nosave):
            # Save agent parameters
            for net, method in zip(nets, POINTER_METHODS):
                save_name = f"{method}.pt"
                torch.save(net, netPath / getFileName(extra=save_name))
            # Save agent parameters
            np.save(prmsPath / getFileName(), vars(args))
            np.save(resPath / getFileName(), results)

    else:
        results, args = loadSavedExperiment(prmsPath, resPath, getFileName(), args=args)

    if show_results:
        plotResults(results, args)
