import torch
from torch import nn
import torch.nn.functional as F

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
# ----------- utilities -----------
# ---------------------------------
def get_device(tensor):
    """simple method to get device of input tensor"""
    return 'cuda' if tensor.is_cuda else 'cpu'
    

# the following masked softmax methods are from allennlp
# https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#L243
def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
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


def masked_log_softmax(vector: torch.Tensor, 
                       mask: torch.Tensor, 
                       dim: int = -1) -> torch.Tensor:
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
    with torch.no_grad(): min_value = vector.min() - 50.0 # make sure it's lower than the lowest value
    vector = vector.masked_fill(mask==0, min_value)
    #vector = vector + (mask + 1e-45).log()
    #vector = vector.masked_fill(mask==0, float('-inf'))
    #vector[torch.all(mask==0, dim=dim)]=1.0 # if the whole thing is masked, this is needed to prevent nans
    return torch.nn.functional.log_softmax(vector, dim=dim)


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
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by the # of heads ({heads})'

        # Store parameters 
        self.emb = emb
        self.heads = heads

        # Dimensionality of each head
        self.headsize = emb // heads

        # Linear transformations to keys, queries, and values
        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        # Final linear layer after attention
        self.unifyheads = nn.Linear(emb, emb)

        # If requested, apply layer norm to the output of each head
        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([self.headsize])
            self.qln = nn.LayerNorm([self.headsize])

    def forward(self, x, mask=None):
        # attention layer forward pass
        assert x.ndim==3, "x should have size: (batch_size, num_tokens, embedding_dimensionality)"
        batch, tokens, emb = x.size() # get size of input
        
        mask = mask if mask is not None else torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))
        assert x.size(0)==mask.size(0) and x.size(1)==mask.size(1), "mask must have same batch_size and num_tokens as x"
            
        # this is the only requirement on the input (other than the number of dimensions)
        assert emb == self.emb, f'Input embedding dim ({emb}) should match layer embedding dim ({self.emb})'

        # convert input tokens to their keys, queries, and values
        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        # separate heads
        keys    = keys.view(batch, tokens, self.heads, self.headsize)
        queries = queries.view(batch, tokens, self.heads, self.headsize)
        values  = values.view(batch, tokens, self.heads, self.headsize)

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
        queries = queries / (emb ** (1/4))
        keys    = keys / (emb ** (1/4))
        
        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch*self.heads, tokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf that are not used
        dotMask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
        dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, tokens, tokens).reshape(batch*self.heads, tokens, tokens)
        
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
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by the # of heads ({heads})'

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
        assert x.ndim==3, "x should have size: (batch_size, num_input_tokens, embedding_dimensionality)"
        assert context.ndim==3, "context should have size: (batch_size, num_context_tokens, embedding_dimensionality)"
        batch, itokens, emb = x.size() # get size of input
        cbatch, ctokens, cemb = context.size() # get size of context
        tokens = itokens + ctokens # total number of tokens to process

        # concatenate input and context
        x_plus_context = torch.cat((x, context), dim=1) 

        # Handle masks
        mask = mask if mask is not None else torch.ones((batch, itokens), dtype=x.dtype).to(get_device(x))
        contextMask = contextMask if contextMask is not None else torch.ones((batch, ctokens), dtype=x.dtype).to(get_device(x))
        
        assert cbatch==batch, "batch size of x and context should be the same"
        assert emb == cemb == self.emb, f'Input embedding dim ({emb}) and context embedding dim ({cemb}) should both match layer embedding dim ({self.emb})'
        
        # convert input tokens to their keys, queries, and values
        queries = self.toqueries(x) # context inputs don't need to query
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
        queries = queries / (emb ** (1/4))
        keys = keys / (emb ** (1/4))
        
        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch*self.heads, itokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf that are not used
        mask_plus_contextMask = torch.cat((mask, contextMask), dim=1)
        dotMask = torch.bmm(mask.unsqueeze(2), mask_plus_contextMask.unsqueeze(1))
        dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, itokens, tokens).reshape(batch*self.heads, itokens, tokens)
            
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
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by the # of heads ({heads})'

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
        assert x.ndim==3, "x should have size: (batch_size, num_input_tokens, embedding_dimensionality)"
        assert type(context)==tuple or type(context)==list, "context should be a tuple or a list"
        assert len(context)==self.num_context, f"this network requires {self.num_context} context tensors but only {len(context)} were provided"
        assert all([c.ndim==3 for c in context]), "context should have size: (batch_size, num_context_tokens, embedding_dimensionality)"
        assert contextMask is None or len(contextMask)==self.num_context, f"if contextMask provided, must have {self.num_context} elements"
        batch, itokens, emb = x.size() # get size of input
        cbatch, ctokens, cemb = map(list, zip(*[c.size() for c in context])) # get size of context
        tokens = itokens + sum(ctokens) # total number of tokens to process

        assert all([batch==cb for cb in cbatch]), "batch size of x and context tensors should be the same"
        assert all([emb==ce for ce in cemb]), "Input embedding dim and context tensor embedding dim should be the same"
        
        # Handle masks
        mask = mask if mask is not None else torch.ones((batch, itokens), dtype=x.dtype).to(get_device(x))
        contextMask = [None for _ in range(self.num_context)] if contextMask is None else contextMask
        contextMask = [cm if cm is not None else torch.ones((cb, ct), dtype=x.dtype).to(get_device(x)) 
                       for cm, cb, ct in zip(contextMask, cbatch, ctokens)]
        
        # convert input tokens to their keys, queries, and values
        queries = self.to_queries(x) # context inputs don't need to query
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
        queries = queries / (emb ** (1/4))
        keys = keys / (emb ** (1/4))
        
        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch*self.heads, itokens, tokens), "somehow the query-key dot product is an unexpected size"

        # mask query key products to -inf (very small number) that are not used
        dotMask = torch.bmm(mask.unsqueeze(2), mask_plus_contextMask.unsqueeze(1))
        dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, itokens, tokens).reshape(batch*self.heads, itokens, tokens)
        
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
        return masked_log_softmax(u/temperature, mask, dim=1)


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
        return masked_log_softmax(u/temperature, mask, dim=1)


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

    def forwardEncoded(self, encoded):
        self.transformEncoded = self.W1(encoded)

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # first transform encoded representations and decoder states 
        transformDecoded = self.W2(decoder_state)
            
        # instead of add, tanh, and project on learnable weights, 
        # just dot product the encoded representations with the decoder "pointer"
        u = torch.bmm(self.transformEncoded, transformDecoded.unsqueeze(2)).squeeze(2)
        return masked_log_softmax(u/temperature, mask, dim=1)


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
        return masked_log_softmax(u/temperature, mask, dim=1)
            

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
        return masked_log_softmax(project/temperature, mask, dim=1)
        
class PointerTransformer(nn.Module):
    """
    PointerTransformer Module (variant of paper using a transformer)
    """
    def __init__(self, emb, **kwargs):
        super().__init__()
        self.emb = emb

        if 'contextual' in kwargs: kwargs['contextual']=True
        self.transform = ContextualTransformerLayer(emb, num_context=1, **kwargs)
        self.vt = nn.Linear(emb, 1, bias=False)
    
    def forwardEncoded(self, encoded):
        pass

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        # transform encoded representations with decoder_state
        updated = self.transform(encoded, [decoder_state], mask=mask)
        project = self.vt(torch.tanh(updated)).squeeze(2)
        return masked_log_softmax(project/temperature, mask, dim=1)

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
        assert type(expansion)==int and expansion>=1, f"expansion ({expansion}) must be a positive integer"
        assert embedding_dim % heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({heads})"

        if contextual:
            self.attention = ContextualAttention(embedding_dim, heads=heads, kqnorm=kqnorm)
        else:
            self.attention = SelfAttention(embedding_dim, heads=heads, kqnorm=kqnorm)
            
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*expansion, bias=bias),
            nn.ReLU(),
            nn.Linear(embedding_dim*expansion, embedding_dim, bias=bias)
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
        assert type(expansion)==int and expansion>=1, f"expansion ({expansion}) must be a positive integer"
        assert embedding_dim % heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({heads})"

        self.attention = MultiContextAttention(embedding_dim, num_context=num_context, heads=heads, kqnorm=kqnorm)
            
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*expansion, bias=bias),
            nn.ReLU(),
            nn.Linear(embedding_dim*expansion, embedding_dim, bias=bias)
        )
        self.layerNorm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, context, mask=None, contextMask=None):
        withAttention = self.attention(x, context, mask=mask, contextMask=contextMask)
        
        x = self.layerNorm1(x + withAttention)
        withTransformation = self.ff(x)
        
        return self.layerNorm2(x + withTransformation)


class PointerModule(nn.Module):
    """
    Implementation of the decoder part of the pointer network
    """
    def __init__(self, embedding_dim, heads=8, expansion=1, kqnorm=True, thompson=False,  bias=True, 
                 decoder_method='transformer', pointer_method='PointerStandard', temperature=1.0, permutation=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.expansion = expansion
        self.kqnorm = kqnorm
        self.bias = bias
        self.decoder_method = decoder_method
        self.pointer_method = pointer_method
        self.thompson = thompson
        self.temperature = temperature
        self.permutation = permutation

        # build decoder (updates the context vector)
        if self.decoder_method == 'gru':
            # if using GRU, then make a GRU cell for the decoder
            self.decoder = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim, bias=self.bias)

        elif self.decoder_method == 'transformer':
            # if using contextual transformer, make one
            self.decoder = ContextualTransformerLayer(self.embedding_dim, heads=self.heads, expansion=self.expansion, 
                                                      num_context=2, kqnorm=self.kqnorm)
        else: 
            raise ValueError(f"decoder_method={decoder_method} not recognized!")

        # build pointer (chooses an output)
        if self.pointer_method == 'PointerStandard':
            # output of the network uses a pointer attention layer as described in the original paper
            self.pointer = PointerStandard(self.embedding_dim)

        elif self.pointer_method == 'PointerDot':
            self.pointer = PointerDot(self.embedding_dim)

        elif self.pointer_method == 'PointerDotNoLN':
            self.pointer = PointerDotNoLN(self.embedding_dim)

        elif self.pointer_method == 'PointerDotLean':
            self.pointer = PointerDotLean(self.embedding_dim)
            
        elif self.pointer_method == 'PointerAttention':
            kwargs = {'heads':self.heads, 'kqnorm':self.kqnorm}
            self.pointer = PointerAttention(self.embedding_dim, **kwargs)

        elif self.pointer_method == 'PointerTransformer':
            # output of the network uses a pointer attention layer with a transformer
            kwargs = {'heads':self.heads, 'expansion':1, 'kqnorm':self.kqnorm, 'bias':self.bias}
            self.pointer = PointerTransformer(self.embedding_dim, **kwargs)
            
        else:
            raise ValueError(f"the pointer_method was not set correctly, {self.pointer_method} not recognized")
            

    def decode(self, encoded, decoder_input, decoder_context, mask):
        # update decoder context using RNN or contextual transformer
        if self.decoder_method == 'gru':
            decoder_context = self.decoder(decoder_input, decoder_context)
        elif self.decoder_method == 'transformer':
            contextInputs = (encoded, decoder_input.unsqueeze(1))
            contextMask = (mask, None)
            decoder_context = self.decoder(decoder_context.unsqueeze(1), contextInputs, contextMask=contextMask).squeeze(1)
        else:
            raise ValueError("decoder_method not recognized")
            
        return decoder_context

    def get_decoder_state(self, decoder_input, decoder_context):
        if self.pointer_method == 'PointerStandard':
            decoder_state = decoder_context
        elif self.pointer_method == 'PointerDot':
            decoder_state = decoder_context
        elif self.pointer_method == 'PointerDotNoLN':
            decoder_state = decoder_context
        elif self.pointer_method == 'PointerDotLean':
            decoder_state = decoder_context
        elif self.pointer_method == 'PointerAttention':
            decoder_state = torch.cat((decoder_input.unsqueeze(1), decoder_context.unsqueeze(1)), dim=1)
        elif self.pointer_method == 'PointerTransformer':
            decoder_state = torch.cat((decoder_input.unsqueeze(1), decoder_context.unsqueeze(1)), dim=1)
        else:
            raise ValueError(f"Pointer method not recognized, somehow it has changed to {self.pointer_method}")
        return decoder_state
        
    def forward(self, encoded, decoder_input, decoder_context, mask=None, max_output=None): 
        """
        forward method for pointer module
        
        x should be an input tensor with shape (batchSize, maxTokens, input_dim)
        mask should be a binary input tensor with shape (batchSize, maxTokens) where a 1 indicates a valid token and 0 indicates padded data
        checks on the mask only care about the shape, so make sure your mask is as described!!!

        max_output should be an integer determining when to cut off decoder output
        """
        batch_size, max_num_tokens, embed_dim = encoded.shape
        assert encoded.ndim==3, "encoded should be (batch_size, num_tokens, embed_dim) size tensor"
        assert decoder_input.ndim==2 and decoder_context.ndim==2, "decoder input and context should be (batch_size, embed_dim) tensors"
        assert decoder_input.size(0) == batch_size, "decoder_input has wrong number of batches"
        assert decoder_input.size(1) == embed_dim, "decoder_input has incorrect embedding dim"
        assert decoder_context.size(0) == batch_size, "decoder_context has wrong number of batches"
        assert decoder_context.size(1) == embed_dim, "decoder_context has incorrect embedding dim"

        if mask is not None: 
            assert mask.ndim == 2, "mask must have shape (batch, tokens)"
            assert mask.size(0)==batch_size and mask.size(1)==max_num_tokens, "mask must have same batch size and max tokens as x"
        else:
            mask = torch.ones((batch_size, max_num_tokens), dtype=encoded.dtype).to(get_device(encoded))

        if max_output is None:
            max_output = max_num_tokens
            
        if self.permutation:
            # prepare source for scattering if required
            src = torch.zeros((batch_size, 1), dtype=mask.dtype).to(get_device(mask))

        # For some pointer layers, the encoded transform happens out of the loop (for others this is a pass)
        self.pointer.forwardEncoded(encoded)

        # Decoding stage
        pointer_log_scores = []
        pointer_choices = []
        for i in range(max_output):
            # update context representation
            decoder_context = self.decode(encoded, decoder_input, decoder_context, mask)
            
            # use pointer attention to evaluate scores of each possible input given the context
            decoder_state = self.get_decoder_state(decoder_input, decoder_context)
            log_score = self.pointer(encoded, decoder_state, mask=mask, temperature=self.temperature)
            
            # choose token for this sample
            if self.thompson:
                # choose probabilistically
                choice = torch.multinomial(torch.exp(log_score)*mask, 1)
            else:
                # choose based on maximum score
                choice = torch.argmax(log_score, dim=1, keepdim=True)

            # next decoder_input is whatever token had the highest probability
            index_tensor = choice.unsqueeze(-1).expand(batch_size, 1, self.embedding_dim)
            decoder_input = torch.gather(encoded, dim=1, index=index_tensor).squeeze(1)
                
            if self.permutation:
                # mask previously chosen tokens (don't include this in the computation graph)
                with torch.no_grad():
                    mask = mask.scatter(1, choice, src)

            # Save output of each decoding round
            pointer_log_scores.append(log_score)
            pointer_choices.append(choice)
            
        log_scores = torch.stack(pointer_log_scores, 1)
        choices = torch.stack(pointer_choices, 1).squeeze(2)

        return log_scores, choices
    

class PointerNetwork(nn.Module):
    """
    Implementation of a pointer network, including an encoding stage and a
    decoding stage. Adopted from the original paper: 
    https://papers.nips.cc/paper/5866-pointer-networks

    With support from: https://github.com/ast0414/pointer-networks-pytorch
    
    This implementation deviates from the original in two ways: 
    1. In the original presentation, the encoding layer uses a bidirectional 
    LSTM. Here, I'm using a transformer on the full sequence to produce 
    encoded representations for each token, followed by an average across 
    token encodings to get a contextual representation. 
    2. The paper uses an LSTM at the decoding stage to process the context 
    representation (<last_output_of_model>=input, <context_representation>
    = <hidden_state>). Here you can do that by setting `decode_with_gru=True`,
    or you can use a contextual transfomer that transforms the context 
    representation using the encoded representations and the last output to 
    make a set of keys and values (but not queries, as they are not 
    transformed). 

    The paper suggests feeding the decoder a dot product between the encoded
    representations and the scores. However, in some cases it may be better to
    use the "greedy" choice and only feed the decoder the token that had the 
    highest probability. That's how this implementation is decoded.
    """
    def __init__(self, input_dim, embedding_dim, heads=8, expansion=1, kqnorm=True, encoding_layers=1, 
                 bias=True, pointer_method='PointerStandard', contextual_encoder=False, decoder_method='transformer', 
                 thompson=False, temperature=1.0, permutation=True):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.expansion = expansion
        self.kqnorm = kqnorm
        self.bias = bias
        self.encoding_layers = encoding_layers
        self.decoder_method = decoder_method
        self.thompson = thompson
        self.pointer_method = pointer_method
        self.temperature = temperature
        self.permutation = permutation

        # this can either be True, False, or 'multicontext'
        if contextual_encoder == True:
            self.contextual_encoder = True
            self.multicontext_encoder = False
        elif contextual_encoder == 'multicontext':
            self.contextual_encoder = True
            self.multicontext_encoder = True
            self.num_context = 1
        elif isinstance(contextual_encoder, tuple) and contextual_encoder[0] == 'multicontext':
            assert isinstance(contextual_encoder[1], int) and contextual_encoder[1]>0, "second element of tuple must be int"
            self.contextual_encoder = True
            self.multicontext_encoder = True
            self.num_context = contextual_encoder[1]
        else:
            self.contextual_encoder = False
            self.multicontext_encoder = False

        self.embedding = nn.Linear(in_features=input_dim, out_features=self.embedding_dim, bias=self.bias)

        if self.multicontext_encoder:    
            self.encodingLayers = nn.ModuleList([ContextualTransformerLayer(self.embedding_dim, 
                                                                heads=self.heads, 
                                                                expansion=self.expansion,  
                                                                kqnorm=self.kqnorm,
                                                                num_context=self.num_context) 
                                                for _ in range(self.encoding_layers)])
        else:
            self.encodingLayers = nn.ModuleList([TransformerLayer(self.embedding_dim, 
                                                                heads=self.heads, 
                                                                expansion=self.expansion,  
                                                                kqnorm=self.kqnorm,
                                                                contextual=self.contextual_encoder) 
                                                for _ in range(self.encoding_layers)])
        
        self.encoding = self.forwardEncoder # since there are masks, it's easier to write a custom forward than wrangle the transformer layers to work with nn.Sequential

        self.pointer = PointerModule(self.embedding_dim, heads=self.heads, expansion=self.expansion, kqnorm=self.kqnorm,
                                     bias=self.bias, decoder_method=self.decoder_method,
                                     pointer_method=self.pointer_method, temperature=self.temperature,
                                     thompson=self.thompson, permutation=permutation)

    def setTemperature(self, temperature):
        self.temperature = temperature
        self.pointer.temperature = temperature

    def setThompson(self, thompson):
        self.thompson = thompson
        self.pointer.thompson = thompson
        
    def forwardEncoder(self, x, mask=None):
        """
        instead of using nn.Sequential just call each layer in sequence
        this solves the problem of passing in the optional mask to the transformer layer 
        (it would have to be passed as an output by the transformer layer and I'd rather keep the transformer layer code clean)
        """
        if self.contextual_encoder:
            context = x[1:] # get either single context or all context inputs (if multicontext)
            x = x[0]
            if self.multicontext_encoder:
                assert len(context)==self.num_context, "number of context inputs doesn't match expected"
            else:
                context = context[0] # pull context out of tuple
            
        for elayer in self.encodingLayers:
            if self.multicontext_encoder:
                x = elayer(x, context, mask=mask)
            elif self.contextual_encoder and not self.multicontext_encoder:
                x = elayer((x, context), mask=mask)
            else:
                x = elayer(x, mask=mask)
                
        return x
        
    def forward(self, x, mask=None, decode_mask=None, max_output=None, init=None): 
        """
        forward method for pointer network with possible masked input
        
        x should be an input tensor with shape (batchSize, maxTokens, input_dim)
        mask should be a binary input tensor with shape (batchSize, maxTokens) where a 1 indicates a valid token and 0 indicates padded data
        checks on the mask only care about the shape, so make sure your mask is as described!!!

        max_output should be an integer determining when to cut off decoder output

        init_position is the initial choice (if provided, masks that choice and sets the initial "decoder_input")
        """
        if self.multicontext_encoder:
            assert isinstance(x, tuple), "if using contextual encoding, input 'x' must be tuple of (mainInputs, contextInputs)"
            assert len(x)==self.num_context+1, "number of context inputs doesn't match expected"
            context = x[1:]
            x = x[0]

        if self.contextual_encoder and not self.multicontext_encoder:
            assert isinstance(x, tuple), "if using contextual encoding, input 'x' must be tuple of (mainInputs, contextInputs)"
            x, context = x
            assert context.ndim == 3, "context (x[1]) must have shape (batch, tokens, input_dim)"
            
        assert x.ndim == 3, "x must have shape (batch, tokens, input_dim)"
        batch, tokens, inp_dim = x.size()
        assert inp_dim == self.input_dim, "input dim of x doesn't match network"
        
        if max_output is None: 
            max_output = tokens

        if self.permutation and max_output > tokens:
            raise ValueError(f"if using permutation mode, max output ({max_output}) must be less than or equal to the number of tokens ({tokens})")
        
        if mask is not None: 
            assert mask.ndim == 2, "mask must have shape (batch, tokens)"
            assert mask.size(0)==batch and mask.size(1)==tokens, "mask must have same batch size and max tokens as x"
        else:
            mask = torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))

        if decode_mask is not None:
            assert decode_mask.ndim == 2, "decode_mask must have shape (batch, tokens)"
            assert decode_mask.size(0)==batch and decode_mask.size(1)==tokens, "decode_mask must have same batch size and max tokens as x"
        else:
            decode_mask = torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))
            decode_mask *= mask

        # Encoding stage
        embeddedRepresentation = self.embedding(x) # embed each token to right dimensionality
        if self.multicontext_encoder:
            embeddedContext = [self.embedding(ctx) for ctx in context]
            embeddedRepresentation = (embeddedRepresentation, *embeddedContext)
        
        if self.contextual_encoder and not self.multicontext_encoder:
            embeddedContext = self.embedding(context) # embed the context to the right dimensionality
            embeddedRepresentation = (embeddedRepresentation, embeddedContext) # add context to transformer input
            
        encodedRepresentation = self.encoding(embeddedRepresentation, mask=mask) # perform N-layer self-attention on inputs
                    
        # Get the masked mean of any encoded tokens
        numerator = torch.sum(encodedRepresentation*mask.unsqueeze(2), dim=1)
        denominator = torch.sum(mask, dim=1, keepdim=True)
        decoder_context = numerator / denominator

        if init is not None:
            rinit = init.view(batch, 1, 1).expand(-1, -1, self.embedding_dim)
            decoder_input = torch.gather(encodedRepresentation, 1, rinit).squeeze(1)
            mask = mask.scatter(1, init.view(batch, 1), torch.zeros((batch,1), dtype=mask.dtype).to(get_device(mask)))
        else:
            decoder_input = torch.zeros((batch, self.embedding_dim)).to(get_device(x)) # initialize decoder_input as zeros

        # Then do pointer network (sequential decode-choice for max_output's)
        log_scores, choices = self.pointer(encodedRepresentation, decoder_input, decoder_context, mask=mask, max_output=max_output)

        return log_scores, choices

