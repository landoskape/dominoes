import torch


def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))


def named_transpose(list_of_lists):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(list, zip(*list_of_lists))


def compute_stats_by_type(tensor, num_types, dim, method="var"):
    """
    helper method for returning the mean and variance across a certain dimension
    where multiple types are concatenated on that dimension

    for example, suppose we trained 2 networks each with 3 sets of parameters
    and concatenated the loss in a tensor like [set1-loss-net1, set1-loss-net2, set2-loss-net1, ...]
    then this would contract across the nets from each set and return the mean and variance
    """
    num_on_dim = tensor.size(dim)
    num_per_type = int(num_on_dim / num_types)
    tensor_by_type = tensor.unsqueeze(dim)
    expand_shape = list(tensor_by_type.shape)
    expand_shape[dim + 1] = num_per_type
    expand_shape[dim] = num_types
    tensor_by_type = tensor_by_type.view(expand_shape)
    type_means = torch.mean(tensor_by_type, dim=dim + 1)
    if method == "var":
        type_dev = torch.var(tensor_by_type, dim=dim + 1)
    elif method == "std":
        type_dev = torch.std(tensor_by_type, dim=dim + 1)
    elif method == "se":
        type_dev = torch.std(tensor_by_type, dim=dim + 1) / np.sqrt(num_per_type)
    elif method == "range":
        type_dev = ptp(tensor_by_type, dim=dim + 1)
    else:
        raise ValueError(f"Method ({method}) not recognized.")

    return type_means, type_dev
