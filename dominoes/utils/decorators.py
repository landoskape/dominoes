from functools import wraps


def test_nets(func):
    """
    decorator for setting networks to eval mode during testing

    requires that the first argument of the decorated function is a list of networks
    """

    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to eval
        in_training_mode = [set_net_mode(net, training=False) for net in nets]

        # do decorated function
        func_outputs = func(nets, *args, **kwargs)

        # return networks to whatever mode they used to be in
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)

        # return decorated function outputs
        return func_outputs

    # return decorated function
    return wrapper


def train_nets(func):
    """
    decorator for setting networks to training mode during training

    requires that the first argument of the decorated function is a list of networks
    """

    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to train
        in_training_mode = [set_net_mode(net, training=True) for net in nets]

        # do decorated function
        func_outputs = func(nets, *args, **kwargs)

        # return networks to whatever mode they used to be in
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)

        # return decorated function outputs
        return func_outputs

    # return decorated function
    return wrapper


def set_net_mode(net, training=True):
    """helper for setting mode of network and returning current mode"""
    # get current mode of network
    in_training_mode = net.training
    # set to training mode or evaluation mode
    if training:
        net.train()
    else:
        net.eval()
    # return original mode of network
    return in_training_mode
