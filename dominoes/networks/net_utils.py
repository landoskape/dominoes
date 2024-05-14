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


def get_device(tensor):
    """simple method to get device of input tensor"""
    return "cuda" if tensor.is_cuda else "cpu"


# the following masked softmax methods are from allennlp
# https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#L243
def masked_softmax(
    vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32
) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If the input is completely masked anywere (across the requested dimension), then this will make it
    uniform instead of keeping it masked, which would lead to nans.
    """
    if mask is None:
        return torch.nn.functional.log_softmax(vector, dim=dim)
    mask = mask.float()
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
    with torch.no_grad():
        min_value = vector.min() - 50.0  # make sure it's lower than the lowest value
    vector = vector.masked_fill(mask == 0, min_value)
    # vector = vector + (mask + 1e-45).log()
    # vector = vector.masked_fill(mask==0, float('-inf'))
    # vector[torch.all(mask==0, dim=dim)]=1.0 # if the whole thing is masked, this is needed to prevent nans
    return torch.nn.functional.log_softmax(vector, dim=dim)
