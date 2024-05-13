import torch
from ..utils import named_transpose


def process_input(input, mask, expected_dim, name="input"):
    """check sizes and create mask if not provided"""
    assert input.ndim == 3, f"{name} should have size: (batch_size, num_tokens, input_dimensionality)"
    batch_size, num_tokens, input_dim = input.size()
    assert input_dim == expected_dim, f"dimensionality of {name} ({input_dim}) doesn't match network ({expected_dim})"

    if mask is not None:
        assert mask.ndim == 2, f"{name} mask must have shape (batch_size, num_tokens)"
        assert mask.size(0) == batch_size and mask.size(1) == num_tokens, f"{name} mask must have same batch size and max tokens as x"
        assert not torch.any(torch.all(mask == 0, dim=1)), f"{name} mask includes rows where all elements are masked, this is not permitted"
    else:
        mask = torch.ones((batch_size, num_tokens), dtype=input.dtype).to(input.device)

    return batch_size, mask


def process_multimodal_input(multimode, mm_mask, num_multimodal, mm_dim):
    """check sizes and create mask for all multimodal inputs if not provided"""
    # first check if multimodal context is a sequence (tuple or list)
    assert type(multimode) == tuple or type(multimode) == list, "context should be a tuple or a list"
    if len(multimode) != num_multimodal:
        raise ValueError(f"this network requires {num_multimodal} context tensors but {len(multimode)} were provided")

    # handle mm_mask
    if mm_mask is None:
        # make a None list for the mask if not provided
        mm_mask = [None for _ in range(num_multimodal)]
    else:
        assert len(mm_mask) == num_multimodal, f"if mm_mask provided, must have {num_multimodal} elements"

    # handle mm_dim
    if type(mm_dim) == int:
        mm_dim = [mm_dim] * num_multimodal
    assert len(mm_dim) == num_multimodal, f"mm_dim must be an integer or a list of integers of length {num_multimodal}"

    # get the batch and mask for each multimode input
    mm_batch_size, mm_mask = named_transpose(
        [process_input(mmc, mmm, mmd, name=f"multimodal input #{imm}") for imm, (mmc, mmm, mmd) in enumerate(zip(multimode, mm_mask, mm_dim))]
    )

    # make sure batch_size is consistent
    assert all([mmb == mm_batch_size[0] for mmb in mm_batch_size]), "batch size of each multimodal input should be the same"

    return mm_batch_size[0], mm_mask


def forward_batch(nets, batch, max_output=None, temperature=None, thompson=None):
    """
    forward pass for a batch of data on a list of pointer networks

    batch is a dictionary with variable inputs and kwargs depending on the dataset.
    This is a one-size fits all method for processing a batch through a list of networks.
    """
    # get input for batch
    input = batch["input"]

    # get current max output for batch
    max_output = batch.get("max_output", max_output)

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

    # get output of network
    scores, choices = named_transpose([net(input, *context_inputs, **net_kwargs) for net in nets])

    # return outputs
    return scores, choices
