from abc import ABC, abstractmethod
import torch
from torch import nn

from ..utils import get_device, masked_softmax, named_transpose


"""
Almost everything I've learned about machine learning and pytorch has been due
to reading blogs, papers, and people kindly posting good code on github. This
script is no exception, and has drawn heavily from two code sources. 

pointer network code: 
- from the repository: https://github.com/ast0414/pointer-networks-pytorch
- from the original paper: https://papers.nips.cc/paper/5866-pointer-networks

transformer code: 
- from the repository: https://github.com/pbloem/former
  - and the associated (very well-written!) blog: http://peterbloem.nl/blog/transformers
- and of course the paper: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
"""


def _attention_type(contextual, multimodal):
    """
    get the attention type based on the arguments

    (uses the same naming convention as the registry keys)
    """
    attention_type = "A"
    if contextual:
        attention_type = "C" + attention_type
    if multimodal:
        attention_type = "M" + attention_type
    return attention_type


def _get_attention_constructor(contextual, multimodal):
    """get the attention constructor based on the attention type"""
    attention_type = _attention_type(contextual, multimodal)
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(f"Unrecognized attention type: {attention_type}")
    return ATTENTION_REGISTRY[attention_type]


def get_attention_layer(embedding_dim, heads, kqnorm, contextual, multimodal, num_context, residual=False):
    """
    create attention layer with requested arguments

    residual is defaulted to False because transformer layers handle residual connections on their own
    """
    attention_kwargs = dict(heads=heads, kqnorm=kqnorm, residual=residual)
    attention_constructor = _get_attention_constructor(contextual, multimodal)
    if multimodal:
        attention_kwargs["num_context"] = num_context
    return attention_constructor(embedding_dim, **attention_kwargs)


# ---------------------------------
# ----------- attention -----------
# ---------------------------------
class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    Adopted from pbloem/former
    """

    def __init__(self, emb, heads=8, kqnorm=True, residual=False):
        super().__init__()

        # This is a requirement
        assert emb % heads == 0, f"Embedding dimension ({emb}) should be divisible by the # of heads ({heads})"

        # Store parameters
        self.emb = emb
        self.heads = heads
        self.residual = residual

        # Dimensionality of each head
        self.headsize = emb // heads

        # Linear transformations to keys, queries, and values
        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        # Final linear layer after attention
        self.unifyheads = nn.Linear(emb, emb)

        # If requested, apply layer norm to the output of each head
        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([self.headsize])
            self.qln = nn.LayerNorm([self.headsize])

    def _process_primary_input(self, x, mask=None):
        # attention layer forward pass
        assert x.ndim == 3, "x should have size: (batch_size, num_tokens, embedding_dimensionality)"
        batch, tokens, emb = x.size()  # get size of input

        mask = mask if mask is not None else torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))
        assert x.size(0) == mask.size(0) and x.size(1) == mask.size(1), "mask must have same batch_size and num_tokens as x"

        # this is the only requirement on the input (other than the number of dimensions)
        assert emb == self.emb, f"Input embedding dim ({emb}) should match layer embedding dim ({self.emb})"

        return batch, tokens, mask

    def _send_to_kqv(self, x, batch, tokens):
        # convert input tokens to their keys, queries, and values
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        # separate heads
        keys = keys.view(batch, tokens, self.heads, self.headsize)
        queries = queries.view(batch, tokens, self.heads, self.headsize)
        values = values.view(batch, tokens, self.heads, self.headsize)

        # perform layer norm on each heads representation if requested
        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # put each head into batch dimension for straightforward batch dot products
        keys = keys.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)
        queries = queries.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)
        values = values.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)

        return keys, queries, values

    def _measure_attention(self, queries, keys, mask, batch, tokens):
        # scale queries and keys by the fourth root of the embedding size
        # same as dividing the dot product by square root of embedding size (but more memory efficient?)
        queries = queries / (self.emb ** (1 / 4))
        keys = keys / (self.emb ** (1 / 4))

        # dot product between scaled queries and keys is attention
        attention = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert attention.size() == (batch * self.heads, tokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf that are not used
        attention_mask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
        attention_mask = attention_mask.unsqueeze(1).expand(batch, self.heads, tokens, tokens).reshape(batch * self.heads, tokens, tokens)

        # and take softmax to get self-attention probabilities
        attention = masked_softmax(attention, attention_mask, dim=2)

        return attention

    def _measure_head_output(self, attention, values, batch, tokens):
        # return values according how much they are attented
        out = torch.bmm(attention, values).view(batch, self.heads, tokens, self.headsize)

        # unify heads, change view to original input size
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.headsize * self.heads)

        # unify heads with linear layer
        return self.unifyheads(out)

    def forward(self, x, mask=None):
        """core forward method with residual connection for attention mechanism"""
        # create mask if not provided, check input sizes
        batch, tokens, mask = self._process_primary_input(x, mask)
        # convert input tokens to their keys, queries, and values
        keys, queries, values = self._send_to_kqv(x, batch, tokens)
        # measure attention from keys and queries
        attention = self._measure_attention(queries, keys, mask, batch, tokens)
        out = self._measure_head_output(attention, values, batch, tokens)

        # mix output of attention heads with residual channel (when requested)
        return out + x * self.residual


class ContextualAttention(SelfAttention):
    """
    Implementation of attention with contextual inputs not to be transformed

    Treats some inputs as context only and uses them to generate keys and
    values but doesn't generate queries or transformed representations.
    """

    def _send_to_kqv(self, x, context, batch, itokens, tokens):
        """overwrite to include context in key and value calculations"""
        # create full input to keys/values by concatenating input with context
        x_plus_context = torch.cat((x, context), dim=1)

        # convert input tokens to their keys, queries, and values
        queries = self.toqueries(x)
        keys = self.tokeys(x_plus_context)
        values = self.tovalues(x_plus_context)

        # separate heads
        queries = queries.view(batch, itokens, self.heads, self.headsize)
        keys = keys.view(batch, tokens, self.heads, self.headsize)
        values = values.view(batch, tokens, self.heads, self.headsize)

        # perform layer norm on each heads representation if requested
        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # put each head into batch dimension for straightforward batch dot products
        queries = queries.transpose(1, 2).contiguous().view(batch * self.heads, itokens, self.headsize)
        keys = keys.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)
        values = values.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)

        return keys, queries, values

    def _measure_attention(self, queries, keys, mask, context_mask, batch, itokens, tokens):
        """overwrite attention measurement to include contextual information"""

        # scale queries and keys by the fourth root of the embedding size
        # same as dividing the dot product by square root of embedding size (but more memory efficient?)
        queries = queries / (self.emb ** (1 / 4))
        keys = keys / (self.emb ** (1 / 4))

        # dot product between scaled queries and keys is attention
        attention = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert attention.size() == (batch * self.heads, itokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf that are not used
        mask_plus_context_mask = torch.cat((mask, context_mask), dim=1)

        attention_mask = torch.bmm(mask.unsqueeze(2), mask_plus_context_mask.unsqueeze(1))
        attention_mask = attention_mask.unsqueeze(1).expand(batch, self.heads, itokens, tokens).reshape(batch * self.heads, itokens, tokens)

        # and take softmax to get self-attention probabilities
        attention = masked_softmax(attention, attention_mask, dim=2)

        return attention

    def forward(self, x, context, mask=None, context_mask=None):
        """core forward method with residual connection for contextual attention mechanism"""
        batch, itokens, mask = self._process_primary_input(x, mask)
        cbatch, ctokens, context_mask = self._process_primary_input(context, context_mask)
        assert batch == cbatch, "batch size of x and context should match"

        tokens = itokens + ctokens
        keys, queries, values = self._send_to_kqv(x, context, batch, itokens, tokens)
        attention = self._measure_attention(queries, keys, mask, batch, itokens, tokens)
        out = self._measure_head_output(attention, values, batch, itokens)

        # mix output of attention heads with residual channel (when requested)
        return out + x * self.residual


class MultimodalBaseClass(ABC):
    def _init_multimodal(self, emb, num_mm_context=1, kqnorm=True):
        """initialize multimodal attention mechanisms"""

        # Then add requirements for the multimodal attention mechanisms
        self.num_mm_context = num_mm_context
        self.to_mm_keys = nn.ModuleList([nn.Linear(emb, emb, bias=False) for _ in range(num_mm_context)])
        self.to_mm_values = nn.ModuleList([nn.Linear(emb, emb, bias=False) for _ in range(num_mm_context)])

        # If requested, apply layer norm to the output of each head
        if kqnorm:
            self.mm_kln = nn.ModuleList([nn.LayerNorm([self.headsize]) for _ in range(num_mm_context)])

    def _process_multimodal_inputs(self, mm_context, mm_mask=None):
        # first check if multimodal context is a sequence (tuple or list)
        assert type(mm_context) == tuple or type(mm_context) == list, "context should be a tuple or a list"
        # check if the right number of contexts are provided
        msg = f"this network requires {self.num_mm_context} context tensors but only {len(mm_context)} were provided"
        assert len(mm_context) == self.num_mm_context, msg
        # check if the mask is provided and has the right number of elements or is NOne
        assert mm_mask is None or len(mm_mask) == self.num_mm_context, f"if mm_mask provided, must have {self.num_mm_context} elements"
        # make a none list for the mask if not provided
        mm_mask = [None for _ in range(self.num_context)] if mm_mask is None else mm_mask
        # get the batch / tokens / mask for each context input
        mm_batch, mm_tokens, mm_mask = named_transpose([self._process_primary_input(mmc, mmm) for mmc, mmm in zip(mm_context, mm_mask)])
        assert all([mmb == mm_batch[0] for mmb in mm_batch]), "batch size of each mm context tensor should be the same"

        return mm_batch[0], mm_tokens, mm_mask

    @abstractmethod
    def _send_to_kqv(self, x, mm_context, batch, itokens, mmtokens, tokens):
        pass

    @abstractmethod
    def forward(self, x, mm_context, mask=None, mm_mask=None):
        pass


class MultimodalAttention(ContextualAttention, MultimodalBaseClass):
    """
    need docstring
    """

    def __init__(self, emb, num_mm_context=1, heads=8, kqnorm=True, residual=False):
        # implement SelfAttention initialization
        super().__init__(emb, heads=heads, kqnorm=kqnorm, residual=residual)
        self._init_multimodal(emb, num_mm_context=num_mm_context, kqnorm=kqnorm)

    def _send_to_kqv(self, x, mm_context, batch, itokens, mm_tokens, tokens):
        # convert input tokens to their keys, queries, and values
        queries = self.toqueries(x)  # context inputs don't need to query
        keys = self.tokeys(x)
        values = self.tovalues(x)

        # convert context tokens to keys and values
        mm_keys = [to_mm_keys(c) for to_mm_keys, c in zip(self.to_mm_keys, mm_context)]
        mm_values = [to_mm_values(c) for to_mm_values, c in zip(self.to_mm_values, mm_context)]

        # separate heads
        queries = queries.view(batch, itokens, self.heads, self.headsize)
        keys = keys.view(batch, itokens, self.heads, self.headsize)
        values = values.view(batch, itokens, self.heads, self.headsize)

        # separate context heads
        mm_keys = [k.view(batch, mmt, self.heads, self.headsize) for k, mmt in zip(mm_keys, mm_tokens)]
        mm_values = [v.view(batch, mmt, self.heads, self.headsize) for v, mmt in zip(mm_values, mm_tokens)]

        # perform layer norm on each heads representation if requested
        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)
            mm_keys = [mm_kln(mk) for mm_kln, mk in zip(self.mm_kln, mm_keys)]

        # combine representations of main inputs and context inputs
        keys = torch.cat((keys, torch.cat(mm_keys, dim=1)), dim=1)
        values = torch.cat((values, torch.cat(mm_values, dim=1)), dim=1)

        # put each head into batch dimension for straightforward batch dot products
        queries = queries.transpose(1, 2).contiguous().view(batch * self.heads, itokens, self.headsize)
        keys = keys.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)
        values = values.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)

        return keys, queries, values

    def forward(self, x, mm_context, mask=None, mm_mask=None):
        """core forward method with residual connection for multimodal attention mechanism"""
        batch, itokens, mask = self._process_primary_input(x, mask)
        mm_batch, mm_tokens, mm_mask = self._process_multimodal_inputs(mm_context, mm_mask)

        assert mm_batch == batch, "batch size of x and mm_context tensors should be the same"
        tokens = itokens + sum(mm_tokens)

        full_mask = torch.cat((mask, torch.cat(mm_mask, dim=1)), dim=1)
        keys, queries, values = self._send_to_kqv(x, mm_context, batch, itokens, mm_tokens, tokens)
        attention = self._measure_attention(queries, keys, mask, full_mask, batch, itokens, tokens)
        out = self._measure_head_output(attention, values, batch, itokens)

        # mix output of attention heads with residual channel (when requested)
        return out + x * self.residual


class MultimodalContextualAttention(MultimodalAttention):
    """
    need docstring
    """

    def _send_to_kqv(self, x, context, mm_context, batch, itokens, ctokens, mm_tokens, tokens):
        # process input and context together
        x_plus_context = torch.cat((x, context), dim=1)

        # convert input tokens to their keys, queries, and values
        queries = self.toqueries(x)
        keys = self.tokeys(x_plus_context)
        values = self.tovalues(x_plus_context)

        # separate heads
        queries = queries.view(batch, itokens, self.heads, self.headsize)
        keys = keys.view(batch, itokens + ctokens, self.heads, self.headsize)
        values = values.view(batch, itokens + ctokens, self.heads, self.headsize)

        # generate keys and values for multimodal context inputs
        mm_keys = [to_mmkeys(c) for to_mmkeys, c in zip(self.to_mm_keys, mm_context)]
        mm_values = [to_mmvalues(c) for to_mmvalues, c in zip(self.to_mm_values, mm_context)]

        # separate context heads
        mm_keys = [k.view(batch, mmt, self.heads, self.headsize) for k, mmt in zip(mm_keys, mm_tokens)]
        mm_values = [v.view(batch, mmt, self.heads, self.headsize) for v, mmt in zip(mm_values, mm_tokens)]

        # perform layer norm on each heads representation if requested
        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)
            mm_keys = [mmkln(mk) for mmkln, mk in zip(self.mm_kln, mm_keys)]

        # combine representations of main inputs and context inputs
        keys = torch.cat((keys, torch.cat(mm_keys, dim=1)), dim=1)
        values = torch.cat((values, torch.cat(mm_values, dim=1)), dim=1)

        # put each head into batch dimension for straightforward batch dot products
        queries = queries.transpose(1, 2).contiguous().view(batch * self.heads, itokens, self.headsize)
        keys = keys.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)
        values = values.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)

        return keys, queries, values

    def forward(self, x, context, mm_context, mask=None, context_mask=None, mm_mask=None):
        batch, itokens, mask = self._process_primary_input(x, mask)
        cbatch, ctokens, context_mask = self._process_primary_input(context, context_mask)
        mm_batch, mm_tokens, mm_mask = self._process_multimodal_inputs(mm_context, mm_mask)

        assert mm_batch == batch, "batch size of x and mm_context tensors should be the same"
        assert batch == cbatch, "batch size of x and context should match"
        tokens = itokens + ctokens + sum(mm_tokens)

        full_mask = torch.cat((mask, context_mask, torch.cat(mm_mask, dim=1)), dim=1)
        keys, queries, values = self._send_to_kqv(x, context, mm_context, batch, itokens, ctokens, mm_tokens, tokens)
        attention = self._measure_attention(queries, keys, mask, full_mask, batch, itokens, tokens)
        out = self._measure_head_output(attention, values, batch, itokens)

        # mix output of attention heads with residual channel (when requested)
        return out + x * self.residual


ATTENTION_REGISTRY = {
    "A": SelfAttention,
    "CA": ContextualAttention,
    "MA": MultimodalAttention,
    "MCA": MultimodalContextualAttention,
}
