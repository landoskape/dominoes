import torch
from torch import nn

from ..utils import get_device, masked_log_softmax, masked_softmax

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


# ---------------------------------
# ----------- attention -----------
# ---------------------------------
class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    Adopted from pbloem/former
    """

    def __init__(self, emb, heads=8, kqnorm=True):
        super().__init__()

        # This is a requirement
        assert emb % heads == 0, f"Embedding dimension ({emb}) should be divisible by the # of heads ({heads})"

        # Store parameters
        self.emb = emb
        self.heads = heads

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

    def forward(self, x, mask=None):
        # attention layer forward pass
        assert x.ndim == 3, "x should have size: (batch_size, num_tokens, embedding_dimensionality)"
        batch, tokens, emb = x.size()  # get size of input

        mask = mask if mask is not None else torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))
        assert x.size(0) == mask.size(0) and x.size(1) == mask.size(1), "mask must have same batch_size and num_tokens as x"

        # this is the only requirement on the input (other than the number of dimensions)
        assert emb == self.emb, f"Input embedding dim ({emb}) should match layer embedding dim ({self.emb})"

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

        # scale queries and keys by the fourth root of the embedding size
        # same as dividing the dot product by square root of embedding size (but more memory efficient?)
        queries = queries / (emb ** (1 / 4))
        keys = keys / (emb ** (1 / 4))

        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch * self.heads, tokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf that are not used
        dotMask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
        dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, tokens, tokens).reshape(batch * self.heads, tokens, tokens)

        # and take softmax to get self-attention probabilities
        dot = masked_softmax(dot, dotMask, dim=2)

        # return values according how much they are attented
        out = torch.bmm(dot, values).view(batch, self.heads, tokens, self.headsize)

        # unify heads, change view to original input size
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.headsize * self.heads)

        # forward pass ends with a linear layer
        return self.unifyheads(out)


class ContextualAttention(nn.Module):
    """
    Implementation of attention with contextual inputs not to be transformed

    Treats some inputs as context only and uses them to generate keys and
    values but doesn't generate queries or transformed representations.

    I don't know if this kind of attention layer exists. If you read this and
    have seen this kind of attention layer before, please let me know!
    """

    def __init__(self, emb, heads=8, kqnorm=True):
        super().__init__()

        # This is a requirement
        assert emb % heads == 0, f"Embedding dimension ({emb}) should be divisible by the # of heads ({heads})"

        # Store parameters
        self.emb = emb
        self.heads = heads

        # Dimensionality of each head
        self.headsize = emb // heads

        # Linear transformations to keys, queries, and values
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        # Final linear layer after attention
        self.unifyheads = nn.Linear(emb, emb)

        # If requested, apply layer norm to the output of each head
        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([self.headsize])
            self.qln = nn.LayerNorm([self.headsize])

    def forward(self, x, context, mask=None, contextMask=None):
        # attention layer forward pass
        assert x.ndim == 3, "x should have size: (batch_size, num_input_tokens, embedding_dimensionality)"
        assert context.ndim == 3, "context should have size: (batch_size, num_context_tokens, embedding_dimensionality)"
        batch, itokens, emb = x.size()  # get size of input
        cbatch, ctokens, cemb = context.size()  # get size of context
        tokens = itokens + ctokens  # total number of tokens to process

        # concatenate input and context
        x_plus_context = torch.cat((x, context), dim=1)

        # Handle masks
        mask = mask if mask is not None else torch.ones((batch, itokens), dtype=x.dtype).to(get_device(x))
        contextMask = contextMask if contextMask is not None else torch.ones((batch, ctokens), dtype=x.dtype).to(get_device(x))

        assert cbatch == batch, "batch size of x and context should be the same"
        assert (
            emb == cemb == self.emb
        ), f"Input embedding dim ({emb}) and context embedding dim ({cemb}) should both match layer embedding dim ({self.emb})"

        # convert input tokens to their keys, queries, and values
        queries = self.toqueries(x)  # context inputs don't need to query
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

        # scale queries and keys by the fourth root of the embedding size
        # same as dividing the dot product by square root of embedding size (but more memory efficient?)
        queries = queries / (emb ** (1 / 4))
        keys = keys / (emb ** (1 / 4))

        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch * self.heads, itokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf that are not used
        mask_plus_contextMask = torch.cat((mask, contextMask), dim=1)
        dotMask = torch.bmm(mask.unsqueeze(2), mask_plus_contextMask.unsqueeze(1))
        dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, itokens, tokens).reshape(batch * self.heads, itokens, tokens)

        # and take softmax to get self-attention probabilities
        dot = masked_softmax(dot, dotMask, dim=2)

        # return values according how much they are attented
        out = torch.bmm(dot, values).view(batch, self.heads, itokens, self.headsize)

        # unify heads, change view to original input size
        out = out.transpose(1, 2).contiguous().view(batch, itokens, self.headsize * self.heads)

        # forward pass ends with a linear layer
        return self.unifyheads(out)


class MultiContextAttention(nn.Module):
    """
    Implementation of attention with contextual inputs not to be transformed
    Each type of context inputs gets their own tokey and tovalue transforms

    Treats some inputs as context only and uses them to generate keys and
    values but doesn't generate queries or transformed representations.

    I don't know if this kind of attention layer exists. If you read this and
    have seen this kind of attention layer before, please let me know!
    """

    def __init__(self, emb, num_context=1, heads=8, kqnorm=True):
        super().__init__()

        # This is a requirement
        assert emb % heads == 0, f"Embedding dimension ({emb}) should be divisible by the # of heads ({heads})"

        # Store parameters
        self.emb = emb
        self.heads = heads
        self.num_context = num_context

        # Dimensionality of each head
        self.headsize = emb // heads

        # Linear transformations to keys, queries, and values
        self.to_queries = nn.Linear(emb, emb, bias=False)
        self.to_keys = nn.Linear(emb, emb, bias=False)
        self.to_values = nn.Linear(emb, emb, bias=False)

        # Linear transformation for each context element
        self.to_context_keys = nn.ModuleList([nn.Linear(emb, emb, bias=False) for _ in range(num_context)])
        self.to_context_values = nn.ModuleList([nn.Linear(emb, emb, bias=False) for _ in range(num_context)])

        # Final linear layer after attention
        self.unifyheads = nn.Linear(emb, emb)

        # If requested, apply layer norm to the output of each head
        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([self.headsize])
            self.qln = nn.LayerNorm([self.headsize])
            self.ckln = nn.ModuleList([nn.LayerNorm([self.headsize]) for _ in range(num_context)])

    def forward(self, x, context, mask=None, contextMask=None):
        # attention layer forward pass
        assert x.ndim == 3, "x should have size: (batch_size, num_input_tokens, embedding_dimensionality)"
        assert type(context) == tuple or type(context) == list, "context should be a tuple or a list"
        assert len(context) == self.num_context, f"this network requires {self.num_context} context tensors but only {len(context)} were provided"
        assert all([c.ndim == 3 for c in context]), "context should have size: (batch_size, num_context_tokens, embedding_dimensionality)"
        assert contextMask is None or len(contextMask) == self.num_context, f"if contextMask provided, must have {self.num_context} elements"
        batch, itokens, emb = x.size()  # get size of input
        cbatch, ctokens, cemb = map(list, zip(*[c.size() for c in context]))  # get size of context
        tokens = itokens + sum(ctokens)  # total number of tokens to process

        assert all([batch == cb for cb in cbatch]), "batch size of x and context tensors should be the same"
        assert all([emb == ce for ce in cemb]), "Input embedding dim and context tensor embedding dim should be the same"

        # Handle masks
        mask = mask if mask is not None else torch.ones((batch, itokens), dtype=x.dtype).to(get_device(x))
        contextMask = [None for _ in range(self.num_context)] if contextMask is None else contextMask
        contextMask = [
            cm if cm is not None else torch.ones((cb, ct), dtype=x.dtype).to(get_device(x)) for cm, cb, ct in zip(contextMask, cbatch, ctokens)
        ]

        # convert input tokens to their keys, queries, and values
        queries = self.to_queries(x)  # context inputs don't need to query
        keys = self.to_keys(x)
        values = self.to_values(x)

        # convert context tokens to keys and values
        ckeys = [to_context_keys(c) for to_context_keys, c in zip(self.to_context_keys, context)]
        cvalues = [to_context_values(c) for to_context_values, c in zip(self.to_context_values, context)]

        # separate heads
        queries = queries.view(batch, itokens, self.heads, self.headsize)
        keys = keys.view(batch, itokens, self.heads, self.headsize)
        values = values.view(batch, itokens, self.heads, self.headsize)

        # separate context heads
        ckeys = [k.view(batch, ct, self.heads, self.headsize) for k, ct in zip(ckeys, ctokens)]
        cvalues = [v.view(batch, ct, self.heads, self.headsize) for v, ct in zip(cvalues, ctokens)]

        # perform layer norm on each heads representation if requested
        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)
            ckeys = [ckln(ck) for ckln, ck in zip(self.ckln, ckeys)]

        # combine representations of main inputs and context inputs
        keys = torch.cat((keys, torch.cat(ckeys, dim=1)), dim=1)
        values = torch.cat((values, torch.cat(cvalues, dim=1)), dim=1)
        mask_plus_contextMask = torch.cat((mask, torch.cat(contextMask, dim=1)), dim=1)

        # put each head into batch dimension for straightforward batch dot products
        queries = queries.transpose(1, 2).contiguous().view(batch * self.heads, itokens, self.headsize)
        keys = keys.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)
        values = values.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.headsize)

        # scale queries and keys by the fourth root of the embedding size
        # same as dividing the dot product by square root of embedding size (but more memory efficient?)
        queries = queries / (emb ** (1 / 4))
        keys = keys / (emb ** (1 / 4))

        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch * self.heads, itokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf (very small number) that are not used
        dotMask = torch.bmm(mask.unsqueeze(2), mask_plus_contextMask.unsqueeze(1))
        dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, itokens, tokens).reshape(batch * self.heads, itokens, tokens)

        # and take softmax to get self-attention probabilities
        dot = masked_softmax(dot, dotMask, dim=2)

        # return values according how much they are attented
        out = torch.bmm(dot, values).view(batch, self.heads, itokens, self.headsize)

        # unify heads, change view to original input size
        out = out.transpose(1, 2).contiguous().view(batch, itokens, self.headsize * self.heads)

        # forward pass ends with a linear layer
        return self.unifyheads(out)


class PointerStandard(nn.Module):
    """
    PointerStandard Module (as specified in the original paper)
    """

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.W1 = nn.Linear(emb, emb, bias=False)
        self.W2 = nn.Linear(emb, emb, bias=False)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forwardEncoded(self, encoded):
        self.transformEncoded = self.W1(encoded)

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # first transform encoded representations and decoder states
        transformDecoded = self.W2(decoder_state)

        # then combine them and project to a new space
        u = self.vt(torch.tanh(self.transformEncoded + transformDecoded.unsqueeze(1))).squeeze(2)
        return masked_log_softmax(u / temperature, mask, dim=1)


class PointerDot(nn.Module):
    """
    PointerDot Module (variant of the paper, using a simple dot product)
    """

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.W1 = nn.Linear(emb, emb, bias=False)
        self.W2 = nn.Linear(emb, emb, bias=False)
        self.eln = nn.LayerNorm(emb, bias=False)
        self.dln = nn.LayerNorm(emb, bias=False)

    def forwardEncoded(self, encoded):
        self.transformEncoded = self.eln(self.W1(encoded))

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # first transform encoded representations and decoder states
        transformDecoded = self.dln(self.W2(decoder_state))

        # instead of add, tanh, and project on learnable weights,
        # just dot product the encoded representations with the decoder "pointer"
        u = torch.bmm(self.transformEncoded, transformDecoded.unsqueeze(2)).squeeze(2)
        return masked_log_softmax(u / temperature, mask, dim=1)


class PointerDotNoLN(nn.Module):
    """
    PointerDotNoLN Module (variant of the paper, using a simple dot product)

    log_softmax is preferable if the probabilities of each token are not
    needed. However, if token embeddings are combined via their probabilities,
    then softmax is required, so log_softmax should be set to `False`.
    """

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.W1 = nn.Linear(emb, emb, bias=False)
        self.W2 = nn.Linear(emb, emb, bias=False)

        # still need to normalize to prevent gradient blowups -- but without additional affine!
        self.eln = nn.LayerNorm(emb, bias=False, elementwise_affine=False)
        self.dln = nn.LayerNorm(emb, bias=False, elementwise_affine=False)

    def forwardEncoded(self, encoded):
        self.transformEncoded = self.eln(self.W1(encoded))

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # first transform encoded representations and decoder states
        transformDecoded = self.dln(self.W2(decoder_state))

        # instead of add, tanh, and project on learnable weights,
        # just dot product the encoded representations with the decoder "pointer"
        u = torch.bmm(self.transformEncoded, transformDecoded.unsqueeze(2)).squeeze(2)
        return masked_log_softmax(u / temperature, mask, dim=1)


class PointerDotLean(nn.Module):
    """
    PointerDotLean Module (variant of the paper, using a simple dot product and even less weights)
    """

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.eln = nn.LayerNorm(emb, bias=False)
        self.dln = nn.LayerNorm(emb, bias=False)

    def forwardEncoded(self, encoded):
        self.transformEncoded = self.eln(encoded)

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # first transform encoded representations and decoder states
        transformDecoded = self.dln(decoder_state)

        # instead of add, tanh, and project on learnable weights,
        # just dot product the encoded representations with the decoder "pointer"
        u = torch.bmm(self.transformEncoded, transformDecoded.unsqueeze(2)).squeeze(2)
        return masked_log_softmax(u / temperature, mask, dim=1)


class PointerAttention(nn.Module):
    """
    PointerAttention Module (variant of paper, using standard attention layer)
    """

    def __init__(self, emb, **kwargs):
        super().__init__()
        self.emb = emb
        self.attention = MultiContextAttention(emb, num_context=1, **kwargs)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forwardEncoded(self, encoded):
        pass

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # attention on encoded representations with decoder_state
        updated = self.attention(encoded, [decoder_state], mask=mask)
        project = self.vt(torch.tanh(updated)).squeeze(2)
        return masked_log_softmax(project / temperature, mask, dim=1)


class PointerTransformer(nn.Module):
    """
    PointerTransformer Module (variant of paper using a transformer)
    """

    def __init__(self, emb, **kwargs):
        super().__init__()
        self.emb = emb

        if "contextual" in kwargs:
            kwargs["contextual"] = True
        self.transform = ContextualTransformerLayer(emb, num_context=1, **kwargs)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forwardEncoded(self, encoded):
        pass

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # transform encoded representations with decoder_state
        updated = self.transform(encoded, [decoder_state], mask=mask)
        project = self.vt(torch.tanh(updated)).squeeze(2)
        return masked_log_softmax(project / temperature, mask, dim=1)


# ---------------------------------
# ------------ networks -----------
# ---------------------------------
class TransformerLayer(nn.Module):
    """
    Standard implementation of a transformer layer, from:
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

    Performs multi-headed attention on input, then layer normalization, then
    two-stage feedforward processing with an optional expansion, then layer
    normalization, with residual connections before each layer normalization.

    This transformer layer has the option of using contextual attention, where
    some inputs are only used to generate keys and values that modulate the
    primary inputs. This form of contextual attention, if used, processes the
    main inputs and the context inputs using the same key & value matrices.
    (See ContextualAttention above).
    """

    def __init__(self, embedding_dim, heads=8, expansion=1, contextual=False, kqnorm=True, bias=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.contextual = contextual
        self.kqnorm = kqnorm
        self.bias = bias
        assert type(expansion) == int and expansion >= 1, f"expansion ({expansion}) must be a positive integer"
        assert embedding_dim % heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({heads})"

        if contextual:
            self.attention = ContextualAttention(embedding_dim, heads=heads, kqnorm=kqnorm)
        else:
            self.attention = SelfAttention(embedding_dim, heads=heads, kqnorm=kqnorm)

        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * expansion, bias=bias), nn.ReLU(), nn.Linear(embedding_dim * expansion, embedding_dim, bias=bias)
        )
        self.layerNorm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None, contextMask=None):
        if self.contextual:
            x, context = x
            withAttention = self.attention(x, context, mask=mask, contextMask=contextMask)
        else:
            withAttention = self.attention(x, mask=mask)

        x = self.layerNorm1(x + withAttention)
        withTransformation = self.ff(x)

        return self.layerNorm2(x + withTransformation)


class ContextualTransformerLayer(nn.Module):
    """
    Variant implementation of a transformer layer

    Performs multi-headed attention on input, then layer normalization, then
    two-stage feedforward processing with an optional expansion, then layer
    normalization, with residual connections before each layer normalization.

    This transformer layer uses contextual attention, where some inputs are
    only used to generate keys and values that modulate the primary inputs.
    This form of contextual attention uses distinct tokey and tovalue matrices
    to transform each kind of context input (see MultiContextAttention above).
    """

    def __init__(self, embedding_dim, heads=8, expansion=1, num_context=1, kqnorm=True, bias=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.num_context = num_context
        self.kqnorm = kqnorm
        self.bias = bias
        assert type(expansion) == int and expansion >= 1, f"expansion ({expansion}) must be a positive integer"
        assert embedding_dim % heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({heads})"

        self.attention = MultiContextAttention(embedding_dim, num_context=num_context, heads=heads, kqnorm=kqnorm)

        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * expansion, bias=bias), nn.ReLU(), nn.Linear(embedding_dim * expansion, embedding_dim, bias=bias)
        )
        self.layerNorm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, context, mask=None, contextMask=None):
        withAttention = self.attention(x, context, mask=mask, contextMask=contextMask)

        x = self.layerNorm1(x + withAttention)
        withTransformation = self.ff(x)

        return self.layerNorm2(x + withTransformation)
