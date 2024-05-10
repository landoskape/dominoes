from tqdm import tqdm
import torch
from .utils import named_transpose


def train(nets, optimizers, dataset, **parameters):
    """a generic training function for pointer networks"""
    num_nets = len(nets)
    assert num_nets == len(optimizers), "Number of networks and optimizers must match"

    # get some key training parameters
    epochs = parameters.get("epochs")
    device = parameters.get("device")
    verbose = parameters.get("verbose", True)
    max_possible_output = parameters.get("max_possible_output")  # this is the maximum number of outputs ever
    learning_mode = parameters.get("learning_mode")
    temperature = parameters.get("temperature", 1.0)
    thompson = parameters.get("thompson", True)

    # process the learning_mode and save conditions
    save_loss = parameters.get("save_loss", False)
    save_reward = parameters.get("save_reward", False)
    get_loss = learning_mode == "supervised" or save_loss
    get_reward = learning_mode == "reinforce" or save_reward

    if learning_mode == "reinforce":
        # create gamma transform for processing reward if not provided in parameters
        gamma = parameters.get("gamma")
        gamma_transform = dataset.create_gamma_transform(max_possible_output, gamma, device=device)

    # create some variables for storing data related to supervised loss
    if save_loss:
        train_loss = torch.zeros(epochs, num_nets, device="cpu")

    # create some variables for storing data related to rewards
    if save_reward:
        train_reward = torch.zeros(epochs, num_nets, device="cpu")
        train_reward_by_pos = torch.zeros(epochs, max_possible_output, num_nets, device="cpu")
        confidence = torch.zeros(epochs, max_possible_output, num_nets, device="cpu")

    # create dataset-specified variables for storing data
    dataset_variables = dataset.create_training_variables(num_nets, **parameters)

    # epoch loop
    epoch_loop = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in epoch_loop:
        # generate a batch
        batch = dataset.generate_batch(**parameters)

        # get input for batch
        input = batch["input"]

        # get current max output for batch
        max_output = batch.get("max_output", max_possible_output)

        # get kwargs for forward pass
        net_kwargs = dict(
            mask=batch.get("mask", None),
            init=batch.get("init", None),
            temperature=temperature,
            thompson=thompson,
            max_output=max_output,
        )

        # add context inputs for batch if requested (use *context_inputs for consistent handling)
        context_inputs = []
        if "context" in batch:
            context_inputs.append(batch["context"])
            net_kwargs["context_mask"] = batch.get("context_mask", None)
        if "multimode" in batch:
            context_inputs.append(batch["multimode"])
            net_kwargs["mm_mask"] = batch.get("mm_mask", None)

        # zero gradients
        for opt in optimizers:
            opt.zero_grad()

        # get output of network
        scores, choices = named_transpose([net(input, *context_inputs, **net_kwargs) for net in nets])

        # get loss
        if get_loss:
            loss = dataset.measure_loss(scores, batch["target"], check_divergence=True)

        # get reward
        if get_reward:
            rewards = [dataset.reward_function(choice, batch) for choice in choices]

        # backprop with supervised learning (usually using negative log likelihood loss)
        if learning_mode == "supervised":
            for l in loss:
                l.backward()

        # backprop with reinforcement learning (with the REINFORCE algorithm)
        if learning_mode == "reinforce":
            # get processed rewards, do backprop
            c_gamma_transform = gamma_transform[:max_output][:, :max_output]  # only use the part of gamma_transform that is needed
            J = dataset.process_rewards(rewards, scores, choices, c_gamma_transform)[1]
            for j in J:
                j.backward()

        # update networks
        for opt in optimizers:
            opt.step()

        # save training data
        with torch.no_grad():
            if save_loss:
                for i in range(num_nets):
                    train_loss[epoch, i] = loss[i].detach().cpu()

            if save_reward:
                pretemp_scores = dataset.get_pretemp_scores(scores, choices, temperature)
                for i in range(num_nets):
                    train_reward[epoch, i] = torch.mean(torch.sum(rewards[i], dim=1)).detach().cpu()
                    train_reward_by_pos[epoch, :, i] = torch.mean(rewards[i], dim=0).detach().cpu()
                    confidence[epoch, :, i] = torch.mean(pretemp_scores[i], dim=0).detach().cpu()

            # save dataset-specific variables
            epoch_state = dict(
                epoch=epoch,
                batch=batch,
                scores=scores,
                choices=choices,
                loss=loss if get_loss else None,
                rewards=rewards if get_reward else None,
                gamma_transform=gamma_transform if learning_mode == "reinforce" else None,
                temperature=temperature,
            )
            dataset.save_training_variables(dataset_variables, epoch_state, **parameters)

    # return training data
    results = dict(
        train_loss=train_loss if save_loss else None,
        train_reward=train_reward if save_reward else None,
        train_reward_by_pos=train_reward_by_pos if save_reward else None,
        confidence=confidence if save_reward else None,
        dataset_variables=dataset_variables,
    )

    return results
