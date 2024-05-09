from abc import ABC, abstractmethod
import torch
from torch import nn

from ..utils import masked_softmax

from . import _process_input, _process_multimodal_input

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


def get_attention_layer(
    embedding_dim,
    num_heads,
    kqnorm=True,
    contextual=False,
    multimodal=False,
    num_multimodal=0,
    bias=False,
    residual=False,
):
    """
    create attention layer with requested arguments

    residual is defaulted to False because transformer layers handle residual connections on their own
    """
    attention_kwargs = dict(
        kqnorm=kqnorm,
        bias=bias,
        residual=residual,
        contextual=contextual,
        multimodal=multimodal and num_multimodal > 0,
    )
    if multimodal and num_multimodal > 0:
        attention_kwargs["num_multimodal"] = num_multimodal
    attention_constructor = _get_attention_constructor(contextual, multimodal)
    return attention_constructor(embedding_dim, num_heads, **attention_kwargs)


# ---------------------------------
# ----------- attention -----------
# ---------------------------------
class AttentionBaseClass(nn.Module, ABC):
    """
    Canonical implementation of multi-head self attention.
    Adopted from pbloem/former
    """

    def __init__(self, embedding_dim, num_heads, kqnorm=True, bias=False, residual=False, num_multimodal=0):
        super().__init__()

        # This is a requirement
        assert embedding_dim % num_heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the # of num_heads ({num_heads})"

        # Store parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.bias = bias
        self.kqnorm = kqnorm
        self.residual = residual
        self.num_multimodal = num_multimodal

        # Dimensionality of each head
        self.num_headsize = embedding_dim // num_heads

        # Build attention matrices for sending input to queries, keys, and values
        self._build_attention_matrices()

        # Build multimodal matrices for sending additional inputs to keys and values
        self._build_multimodal_attention_matrices(num_multimodal)

        # Build supporting matrices
        self._build_layer_norms()

        # Build mixing matrix for unifying num_heads
        self._build_mixing_matrix()

    def _build_attention_matrices(self):
        """
        method for building attention matrices for sending input to queries, keys, and values

        optionally ignore queries for contextual or multimodal inputs
        name will be inserted into attribute in the format "to_{name}_keys/queries/value" if provided
        """
        self.to_queries = nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias)
        self.to_keys = nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias)
        self.to_values = nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias)

    def _build_multimodal_attention_matrices(self, num_multimodal):
        """
        method for building attention matrices for sending input to queries, keys, and values for multimodal inputs
        """
        if num_multimodal > 0:
            self.to_mm_keys = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias) for _ in range(num_multimodal)])
            self.to_mm_values = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias) for _ in range(num_multimodal)])

    def _build_layer_norms(self):
        """method for building layer norm matrices for each head"""
        if self.kqnorm:
            self.kln = nn.LayerNorm([self.num_headsize])
            self.qln = nn.LayerNorm([self.num_headsize])
            if self.num_multimodal > 0:
                self.mm_kln = nn.ModuleList([nn.LayerNorm([self.num_headsize]) for _ in range(self.num_multimodal)])

    def _build_mixing_matrix(self):
        """method for building mixing matrix for unifying num_heads"""
        self.unifynum_heads = nn.Linear(self.embedding_dim, self.embedding_dim)

    def _send_to_kqv(self, x, context=None, multimode=None):
        """
        centralized method for sending input to queries, keys, and values

        works for contextual attention and multimodal attention, if not using context or multimodal
        inputs, just don't provide them to this method.
        """
        batch_size, qtokens = x.size(0), x.size(1)

        if context is not None:
            assert context.size(0) == batch_size, "batch size of x and context should match"
            ctokens = context.size(1)
        else:
            ctokens = 0

        if multimode is not None:
            assert len(multimode) == self.num_multimodal, "number of multimodal contexts should match num_multimodal"
            assert all([mm.size(0) == batch_size for mm in multimode]), "batch size of x and multimode tensors should match"
            mm_tokens = [mm.size(1) for mm in multimode]
            using_multimode = True
        else:
            mm_tokens = [0]
            using_multimode = False

        # process input and context together
        xkv = torch.cat((x, context), dim=1) if context is not None else x

        # convert input tokens to their keys, queries, and values
        queries = self.to_queries(x)
        keys = self.to_keys(xkv)
        values = self.to_values(xkv)

        # separate num_heads
        queries = queries.view(batch_size, qtokens, self.num_heads, self.num_headsize)
        keys = keys.view(batch_size, qtokens + ctokens, self.num_heads, self.num_headsize)
        values = values.view(batch_size, qtokens + ctokens, self.num_heads, self.num_headsize)

        if using_multimode:
            # generate keys and values for multimodal inputs
            mm_keys = [to_mmkeys(mm) for to_mmkeys, mm in zip(self.to_mm_keys, multimode)]
            mm_values = [to_mmvalues(mm) for to_mmvalues, mm in zip(self.to_mm_values, multimode)]

            # separate context num_heads
            mm_keys = [k.view(batch_size, mmt, self.num_heads, self.num_headsize) for k, mmt in zip(mm_keys, mm_tokens)]
            mm_values = [v.view(batch_size, mmt, self.num_heads, self.num_headsize) for v, mmt in zip(mm_values, mm_tokens)]

        if self.kqnorm:
            # perform layer norm on each num_heads representation if requested
            keys = self.kln(keys)
            queries = self.qln(queries)
            if using_multimode:
                mm_keys = [mmkln(mk) for mmkln, mk in zip(self.mm_kln, mm_keys)]

        if using_multimode:
            # combine keys & values with multimodal keys & values
            keys = torch.cat((keys, torch.cat(mm_keys, dim=1)), dim=1)
            values = torch.cat((values, torch.cat(mm_values, dim=1)), dim=1)

        # measure totak number of key/value tokens
        kvtokens = qtokens + ctokens + sum(mm_tokens)

        # put each head into batch dimension for straightforward batch dot products
        queries = queries.transpose(1, 2).contiguous().view(batch_size * self.num_heads, qtokens, self.num_headsize)
        keys = keys.transpose(1, 2).contiguous().view(batch_size * self.num_heads, kvtokens, self.num_headsize)
        values = values.transpose(1, 2).contiguous().view(batch_size * self.num_heads, kvtokens, self.num_headsize)

        return keys, queries, values

    def _make_key_mask(self, mask, context_mask=None, mm_mask=None):
        """
        create a mask for keys, which will be used to mask out context and multimodal tokens

        doesn't do any checks, because it is assumed that the mask, context_mask, and mm_mask
        are all already checked by the _process_input method, and then the respective forward
        pass method checks the outputs of each _process_input call.
        """
        key_mask = torch.cat((mask, context_mask), dim=1) if context_mask is not None else mask
        key_mask = torch.cat((key_mask, torch.cat(mm_mask, dim=1)), dim=1) if mm_mask is not None else key_mask
        return key_mask

    def _measure_attention(self, queries, keys, mask, key_mask=None):
        """
        measure attention between queries and keys, where keys can have more tokens than queries
        if this is a contextual or a multimodal attention layer.

        if keys have context tokens, key_mask must be provided.
        """
        assert queries.size(0) == keys.size(0), "batch size * head_size of queries and keys should match"
        assert queries.size(0) // self.num_heads == mask.size(0), "batch size of queries and mask should match"

        # handle tokens dimension
        qtokens = queries.size(1)
        ktokens = keys.size(1)

        # check if keys have context tokens and if key_mask is provided
        if ktokens != qtokens:
            assert ktokens > qtokens, "keys should have greater than or equal tokens than queries"
            assert key_mask is not None, "key_mask must be provided if keys have more tokens than queries"

        if key_mask is not None:
            assert mask.size(0) == key_mask.size(0), "batch size of mask and key_mask should match"
        else:
            key_mask = mask

        # check for token size consistency
        assert qtokens == mask.size(1), "num_tokens of queries and mask should match"
        assert ktokens == key_mask.size(1), "num_tokens of keys and key_mask should match"

        # scale queries and keys by the fourth root of the embedding size
        # same as dividing the dot product by square root of embedding size (but more memory efficient?)
        queries = queries / (self.embedding_dim ** (1 / 4))
        keys = keys / (self.embedding_dim ** (1 / 4))

        # dot product between scaled queries and keys is attention
        attention = torch.bmm(queries, keys.transpose(1, 2))

        # create mask for attention matrix, expand and reshape appropriately
        attention_mask = torch.bmm(mask.unsqueeze(2), key_mask.unsqueeze(1))
        attention_mask = attention_mask.unsqueeze(1).expand(mask.size(0), self.num_heads, qtokens, ktokens)
        attention_mask = attention_mask.reshape(mask.size(0) * self.num_heads, qtokens, ktokens)

        # and take softmax to get self-attention probabilities
        attention = masked_softmax(attention, attention_mask, dim=2)

        return attention

    def _measure_head_output(self, attention, values, batch_size):
        """
        combine attention with values to get the output of each head, and then unify num_heads
        """
        assert attention.size(0) == values.size(0), "batch size * head_size of attention and values should match"

        # return values according how much they are attented
        out = torch.bmm(attention, values).view(batch_size, self.num_heads, attention.size(1), self.num_headsize)

        # unify num_heads, change view to original input size
        out = out.transpose(1, 2).contiguous().view(batch_size, attention.size(1), self.num_headsize * self.num_heads)

        # unify num_heads with linear layer
        return self.unifynum_heads(out)

    @abstractmethod
    def forward(self, x, mask=None):
        """core forward method with residual connection for attention mechanism"""
        raise NotImplementedError


class SelfAttention(AttentionBaseClass):
    """
    Implementation of multi-head self attention mechanism

    Simplest attention model where attention is measured between all input tokens
    """

    def __init__(self, embedding_dim, num_heads=8, kqnorm=True, bias=False, residual=False):
        """overwriting to prevent user from providing multimodal inputs for this attention layer"""
        super().__init__(embedding_dim, num_heads=num_heads, kqnorm=kqnorm, bias=bias, residual=residual, num_multimodal=0)

    def forward(self, x, mask=None):
        """core forward method with residual connection for attention mechanism"""
        # create mask if not provided, check input sizes
        batch_size, mask = _process_input(x, mask, self.embedding_dim)

        # convert input tokens to their keys, queries, and values
        keys, queries, values = self._send_to_kqv(x)

        # measure attention from keys and queries
        attention = self._measure_attention(queries, keys, mask)
        out = self._measure_head_output(attention, values, batch_size)

        # mix output of attention num_heads with residual channel (when requested)
        return out + x * self.residual


class ContextualAttention(AttentionBaseClass):
    """
    Implementation of multi-headed attention with contextual inputs

    Main inputs (x) are used to generate queries, keys, and values, while context
    inputs are used to generate keys and values that modulate the main inputs.
    """

    def __init__(self, embedding_dim, num_heads=8, kqnorm=True, bias=False, residual=False):
        """overwriting to prevent user from providing multimodal inputs for this attention layer"""
        super().__init__(embedding_dim, num_heads=num_heads, kqnorm=kqnorm, bias=bias, residual=residual, num_multimodal=0)

    def forward(self, x, context, mask=None, context_mask=None):
        """core forward method with residual connection for attention mechanism"""
        # create mask if not provided, check input sizes
        batch_size, mask = _process_input(x, mask, self.embedding_dim)
        context_batch_size, context_mask = _process_input(context, context_mask, self.embedding_dim, name="context")
        assert batch_size == context_batch_size, "batch size of x and context should match"

        # convert input tokens to their keys, queries, and values
        keys, queries, values = self._send_to_kqv(x, context=context)
        key_mask = self._make_key_mask(mask, context_mask=context_mask)

        # measure attention from keys and queries
        attention = self._measure_attention(queries, keys, mask, key_mask=key_mask)
        out = self._measure_head_output(attention, values, batch_size)

        # mix output of attention num_heads with residual channel (when requested)
        return out + x * self.residual


class MultimodalAttention(AttentionBaseClass):
    """
    Implementation of multi-headed attention with multimodal inputs

    Main inputs (x) are used to generate queries, keys, and values, while multimodal
    inputs are used to generate keys and values that modulate the main inputs using
    a different set of key and value matrices (and layer norms if requested).
    """

    def forward(self, x, multimode, mask=None, mm_mask=None):
        """core forward method with residual connection for attention mechanism"""
        # create mask if not provided, check input sizes
        batch_size, mask = _process_input(x, mask, self.embedding_dim)
        mm_batch_size, mm_mask = _process_multimodal_input(multimode, mm_mask, self.num_multimodal, self.embedding_dim)

        assert batch_size == mm_batch_size, "batch size of x and multimode inputs should match"

        # convert input tokens to their keys, queries, and values
        keys, queries, values = self._send_to_kqv(x, multimode=multimode)
        key_mask = self._make_key_mask(mask, mm_mask=mm_mask)

        # measure attention from keys and queries
        attention = self._measure_attention(queries, keys, mask, key_mask=key_mask)
        out = self._measure_head_output(attention, values, batch_size)

        # mix output of attention num_heads with residual channel (when requested)
        return out + x * self.residual


class MultimodalContextualAttention(AttentionBaseClass):
    """
    Implementation of multi-headed attention with multimodal inputs and contextual inputs

    Main inputs (x) are used to generate queries, keys, and values, contextual inputs are
    used to generate keys and values with the same matrices as main inputs, while multimodal
    inputs are used to generate keys and values that modulate the main inputs using
    a different set of key and value matrices (and layer norms if requested).
    """

    def forward(self, x, context, multimode, mask=None, context_mask=None, mm_mask=None):
        """core forward method with residual connection for attention mechanism"""
        # create mask if not provided, check input sizes
        batch_size, mask = _process_input(x, mask, self.embedding_dim)
        context_batch_size, context_mask = _process_input(context, context_mask, self.embedding_dim, name="context")
        mm_batch_size, mm_mask = _process_multimodal_input(multimode, mm_mask, self.num_multimodal, self.embedding_dim)

        assert batch_size == context_batch_size, "batch size of x and context should match"
        assert batch_size == mm_batch_size, "batch size of x and multimode inputs should match"

        # convert input tokens to their keys, queries, and values
        keys, queries, values = self._send_to_kqv(x, context=context, multimode=multimode)
        key_mask = self._make_key_mask(mask, context_mask=context_mask, mm_mask=mm_mask)

        # measure attention from keys and queries
        attention = self._measure_attention(queries, keys, mask, key_mask=key_mask)
        out = self._measure_head_output(attention, values, batch_size)

        # mix output of attention num_heads with residual channel (when requested)
        return out + x * self.residual


ATTENTION_REGISTRY = {
    "A": SelfAttention,
    "CA": ContextualAttention,
    "MA": MultimodalAttention,
    "MCA": MultimodalContextualAttention,
}
