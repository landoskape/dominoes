import tqdm
import torch


def train(nets, optimizers, dataset, **parameters):
    """a generic training function for pointer networks"""
    epochs = parameters.get("epochs")
    device = parameters.get("device")
    max_output = parameters.get("max_output")
    gamma = parameters.get("gamma")
    verbose = parameters.get("verbose", True)

    # create gamma transform if not provided in parameters
    if "gamma_transform" in parameters:
        gamma_transform = parameters["gamma_transform"]
    else:
        gamma_transform = dataset.create_gamma_transform(max_output, gamma, device=device)

    epoch_loop = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in epoch_loop:
        batch = dataset.generate_batch(**parameters)

        # unpack batch tuple
        input, _, _, _, _, selection, _ = batch
        input = input.to(device)

        # zero gradients, get output of network
        for opt in optimizers:
            opt.zero_grad()
        log_scores, choices = map(list, zip(*[net(input, max_output=max_output) for net in nets]))

        # log-probability for each chosen dominoe
        logprob_policy = [torch.gather(score, 2, choice.unsqueeze(2)).squeeze(2) for score, choice in zip(log_scores, choices)]

        # measure reward
        rewards = [training.measureReward_sortDescend(trainDominoes[selection], choice) for choice in choices]
        G = [torch.matmul(reward, gamma_transform) for reward in rewards]

        # measure J
        J = [-torch.sum(logpol * g) for logpol, g in zip(logprob_policy, G)]
        for j in J:
            j.backward()

        # update networks
        for opt in optimizers:
            opt.step()

        # save training data
        with torch.no_grad():
            for i in range(numNets):
                trainReward[epoch, i, run] = torch.mean(torch.sum(rewards[i], dim=1)).detach()
                trainRewardByPos[epoch, :, i, run] = torch.mean(rewards[i], dim=0).detach()

                # Measure the models confidence -- ignoring the effect of temperature
                pretemp_score = torch.softmax(log_scores[i] * nets[i].temperature, dim=2)
                pretemp_policy = torch.gather(pretemp_score, 2, choices[i].unsqueeze(2)).squeeze(2)
                trainScoreByPos[epoch, :, i, run] = torch.mean(pretemp_policy, dim=0).detach()
