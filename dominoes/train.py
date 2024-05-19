from tqdm import tqdm
import torch
from .networks.net_utils import forward_batch
from .networks.baseline import make_baseline_nets, check_baseline_updates
from .utils import train_nets, test_nets


@train_nets
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
    baseline = parameters.get("baseline", True) and learning_mode == "reinforce"

    # process the learning_mode and save conditions
    get_loss = learning_mode == "supervised" or parameters.get("save_loss", False)
    get_reward = learning_mode == "reinforce" or parameters.get("save_reward", False)

    if learning_mode == "reinforce":
        # create gamma transform for processing reward if not provided in parameters
        gamma = parameters.get("gamma")
        gamma_transform = dataset.create_gamma_transform(max_possible_output, gamma, device=device)

    # create some variables for storing data related to supervised loss
    if get_loss:
        train_loss = torch.zeros(epochs, num_nets, device="cpu")

    # create some variables for storing data related to rewards
    if get_reward:
        train_reward = torch.zeros(epochs, num_nets, device="cpu")
        train_reward_by_pos = torch.zeros(epochs, max_possible_output, num_nets, device="cpu")
        confidence = torch.zeros(epochs, max_possible_output, num_nets, device="cpu")

    # prepare baseline networks if required
    if baseline:
        bl_temperature = parameters.get("bl_temperature", 1.0)
        bl_thompson = parameters.get("bl_thompson", False)
        bl_significance = parameters.get("bl_significance", 0.05)
        bl_batch_size = parameters.get("bl_batch_size", 1024)
        bl_duty_cycle = parameters.get("bl_duty_cycle", 1)
        bl_parameters = parameters.copy()
        bl_parameters["batch_size"] = bl_batch_size  # update batch size for baseline reference batch
        bl_nets = make_baseline_nets(
            nets,
            dataset,
            batch_parameters=bl_parameters,
            significance=bl_significance,
            temperature=bl_temperature,
            thompson=bl_thompson,
        )

    # create dataset-specified variables for storing data
    dataset_variables = dataset.create_training_variables(num_nets, **parameters)

    # epoch loop
    epoch_loop = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in epoch_loop:
        # generate a batch
        batch = dataset.generate_batch(**parameters)

        # zero gradients
        for opt in optimizers:
            opt.zero_grad()

        scores, choices = forward_batch(nets, batch, max_possible_output, temperature, thompson)

        # get baseline choices if using them
        if baseline:
            with torch.no_grad():
                bl_choices = forward_batch(bl_nets, batch, max_possible_output, bl_temperature, bl_thompson)[1]

        # get loss
        if get_loss:
            loss = dataset.measure_loss(scores, batch["target"], check_divergence=True)

        # get reward
        if get_reward:
            rewards = [dataset.reward_function(choice, batch) for choice in choices]
            if baseline:
                with torch.no_grad():
                    bl_rewards = [dataset.reward_function(choice, batch) for choice in bl_choices]

        # backprop with supervised learning (usually using negative log likelihood loss)
        if learning_mode == "supervised":
            for l in loss:
                l.backward()

        # backprop with reinforcement learning (with the REINFORCE algorithm)
        if learning_mode == "reinforce":
            # get max output for this batch
            max_output = batch.get("max_output", max_possible_output)
            # get processed rewards, do backprop
            c_gamma_transform = gamma_transform[:max_output][:, :max_output]  # only use the part of gamma_transform that is needed
            _, J = dataset.process_rewards(
                rewards,
                scores,
                choices,
                c_gamma_transform,
                baseline_rewards=bl_rewards if baseline else None,
            )
            for j in J:
                j.backward()

        # update networks
        for opt in optimizers:
            opt.step()

        # update baseline networks if required
        if baseline and epoch % bl_duty_cycle == 0:
            bl_nets = check_baseline_updates(nets, bl_nets)

        # save training data
        with torch.no_grad():
            if get_loss:
                for i in range(num_nets):
                    train_loss[epoch, i] = loss[i].detach().cpu()

            if get_reward:
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
        train_loss=train_loss if get_loss else None,
        train_reward=train_reward if get_reward else None,
        train_reward_by_pos=train_reward_by_pos if get_reward else None,
        confidence=confidence if get_reward else None,
        dataset_variables=dataset_variables,
    )

    return results


# @torch.no_grad()
# @test_nets
# def test(nets, optimizers, dataset, **parameters):
#     """a generic boiler plate function for testing and evaluating pointer networks"""
#     num_nets = len(nets)
#     assert num_nets == len(optimizers), "Number of networks and optimizers must match"

#     # get some key training parameters
#     epochs = parameters.get("epochs")
#     verbose = parameters.get("verbose", False)
#     max_possible_output = parameters.get("max_possible_output")  # this is the maximum number of outputs ever
#     temperature = parameters.get("temperature", 1.0)
#     thompson = parameters.get("thompson", False)

#     # process the learning_mode and save conditions
#     get_loss = parameters.get("save_loss", False)
#     get_reward = parameters.get("save_reward", False)

#     # create some variables for storing data related to supervised loss
#     if get_loss:
#         test_loss = torch.zeros(epochs, num_nets, device="cpu")

#     # create some variables for storing data related to rewards
#     if get_reward:
#         test_reward = torch.zeros(epochs, num_nets, device="cpu")
#         test_reward_by_pos = torch.zeros(epochs, max_possible_output, num_nets, device="cpu")
#         confidence = torch.zeros(epochs, max_possible_output, num_nets, device="cpu")

#     # create dataset-specified variables for storing data
#     dataset_variables = dataset.create_training_variables(num_nets, **parameters)

#     # epoch loop
#     epoch_loop = tqdm(range(epochs)) if verbose else range(epochs)
#     for epoch in epoch_loop:
#         # generate a batch
#         batch = dataset.generate_batch(**parameters)

#         scores, choices = forward_batch(nets, batch, max_possible_output, temperature, thompson)

#         # get loss
#         if get_loss:
#             loss = dataset.measure_loss(scores, batch["target"], check_divergence=True)

#         # get reward
#         if get_reward:
#             rewards = [dataset.reward_function(choice, batch) for choice in choices]

#         # update networks
#         for opt in optimizers:
#             opt.step()

#         # save training data
#         if get_loss:
#             for i in range(num_nets):
#                 test_loss[epoch, i] = loss[i].detach().cpu()

#         if get_reward:
#             pretemp_scores = dataset.get_pretemp_scores(scores, choices, temperature)
#             for i in range(num_nets):
#                 test_reward[epoch, i] = torch.mean(torch.sum(rewards[i], dim=1)).detach().cpu()
#                 test_reward_by_pos[epoch, :, i] = torch.mean(rewards[i], dim=0).detach().cpu()
#                 confidence[epoch, :, i] = torch.mean(pretemp_scores[i], dim=0).detach().cpu()

#         # save dataset-specific variables
#         epoch_state = dict(
#             epoch=epoch,
#             batch=batch,
#             scores=scores,
#             choices=choices,
#             loss=loss if get_loss else None,
#             rewards=rewards if get_reward else None,
#             gamma_transform=None,
#             temperature=temperature,
#         )
#         dataset.save_training_variables(dataset_variables, epoch_state, **parameters)

#     # return training data
#     results = dict(
#         test_loss=test_loss if get_loss else None,
#         test_reward=test_reward if get_reward else None,
#         test_reward_by_pos=test_reward_by_pos if get_reward else None,
#         confidence=confidence if get_reward else None,
#         dataset_variables=dataset_variables,
#     )

#     return results
