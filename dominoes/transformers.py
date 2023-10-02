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
  - and the associated (very well-written!) blog: 
  http://peterbloem.nl/blog/transformers
- and of course the paper: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
"""


# ---------------------------------
# ----------- utilities -----------
# ---------------------------------
def mask_future(data, maskval=0.0, mask_diagonal=True):
    """
    Masks out values in a given batch of data where i <= j
    Use when you don't want future token j to influence past token i
    Adopted from pbloem/former
    """
    h, w = data.size(-2), data.size(-1)
    mask_index = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    data[..., mask_index[0], mask_index[1]] = maskval

def get_device(tensor):
    """simple method to get device of input tensor"""
    return 'cuda' if tensor.is_cuda else 'cpu'


# ---------------------------------
# ----------- attention -----------
# ---------------------------------
class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    Adopted from pbloem/former
    """
    def __init__(self, emb, heads=8, kqnorm=False):
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
        
        if mask is not None:
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

        if mask is not None:
            # mask query key products to -inf that are not used
            dotMask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
            dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, tokens, tokens).reshape(batch*self.heads, tokens, tokens)
            dot += (dotMask + 1e-45).log()
            # this produces nans for any row that is fully masked 
            #dot.masked_fill_(dotMask==0, float('-inf')) # assign -inf for any value that isn't used
        
        # and take softmax to get self-attention probabilities
        dot = F.softmax(dot, dim=2)
        
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
    def __init__(self, emb, heads=8, kqnorm=False):
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
        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

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
        useMask = (mask is not None) or (contextMask is not None)
        if useMask:
            mask = mask if mask is not None else torch.ones((batch, itokens), dtype=x.dtype).to(get_device(x))
            contextMask = contextMask if contextMask is not None else torch.ones((batch, ctokens), dtype=context.dtype).to(get_device(x))
            assert x.size(0)==mask.size(0)==contextMask.size(0), "masks must have same batch_size as x"
            assert x.size(1)==mask.size(1), "mask must have same num_input_tokens as x"
            assert context.size(1)==contextMask.size(1), "contextMask must have same num_context_tokens as context"
            mask_plus_contextMask = torch.cat((mask, contextMask), dim=1)
        
        assert cbatch==batch, "batch size of x and context should be the same"
        assert emb == cemb == self.emb, f'Input embedding dim ({emb}) and context embedding dim ({cemb}) should both match layer embedding dim ({self.emb})'
        
        # convert input tokens to their keys, queries, and values
        queries = self.toqueries(x) # context inputs don't need to query
        keys    = self.tokeys(x_plus_context)
        values  = self.tovalues(x_plus_context)

        # separate heads
        queries = queries.view(batch, itokens, self.heads, self.headsize)
        keys    = keys.view(batch, tokens, self.heads, self.headsize)
        values  = values.view(batch, tokens, self.heads, self.headsize)

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
        keys    = keys / (emb ** (1/4))
        
        # dot product between scaled queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # check to make sure this is correct
        assert dot.size() == (batch*self.heads, itokens, tokens), "somehow the query-key dot product is an unexpected size"

        if useMask:
            # mask query key products to -inf that are not used
            dotMask = torch.bmm(mask.unsqueeze(2), mask_plus_contextMask.unsqueeze(1))
            dotMask = dotMask.unsqueeze(1).expand(batch, self.heads, itokens, tokens).reshape(batch*self.heads, itokens, tokens)
            dot += (dotMask + 1e-45).log()
            # produces nans for any rows that are fully masked
            #dot.masked_fill_(dotMask==0, float('-inf')) # assign -inf for any value that isn't used
        
        # and take softmax to get self-attention probabilities
        dot = F.softmax(dot, dim=2)
        
        # return values according how much they are attented
        out = torch.bmm(dot, values).view(batch, self.heads, itokens, self.headsize)
        
        # unify heads, change view to original input size
        out = out.transpose(1, 2).contiguous().view(batch, itokens, self.headsize * self.heads)

        # forward pass ends with a linear layer
        return self.unifyheads(out)


class PointerAttention(nn.Module):
    """
    PointerAttention Module (as specified in the original paper)

    log_softmax is preferable if the probabilities of each token are not 
    needed. However, if token embeddings are combined via their probabilities,
    then softmax is required, so log_softmax should be set to `False`.
    """
    def __init__(self, emb, log_softmax=False):
        super().__init__()
        self.emb = emb
        self.log_softmax = log_softmax
        self.W1 = nn.Linear(emb, emb, bias=False)
        self.W2 = nn.Linear(emb, emb, bias=False)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forward(self, encoded, decoder_state, mask=None):
        # first transform encoded representations and decoder states 
        transformEncoded = self.W1(encoded)
        transformDecoded = self.W2(decoder_state).unsqueeze(1) # unsqueeze for broadcasting on token dimension

        # then combine them and project to a new space
        u = self.vt(torch.tanh(transformEncoded + transformDecoded)).squeeze(-1)
        if mask is not None:
            u.masked_fill_(mask==0, float('-inf')) # only use valid tokens
            
        if self.log_softmax:
            # convert to log scores
            return torch.nn.functional.log_softmax(u, dim=-1)
        else:
            # convert to probabilities
            return torch.nn.functional.softmax(u, dim=-1)


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
    primary inputs. 
    """
    def __init__(self, embedding_dim, heads=8, expansion=1, contextual=False, kqnorm=False):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.contextual = contextual
        self.kqnorm = kqnorm
        assert type(expansion)==int and expansion>=1, f"expansion ({expansion}) must be a positive integer"
        assert embedding_dim % heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({heads})"

        if contextual:
            self.attention = ContextualAttention(embedding_dim, heads=heads, kqnorm=kqnorm)
        else:
            self.attention = SelfAttention(embedding_dim, heads=heads, kqnorm=kqnorm)
            
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*expansion),
            nn.ReLU(),
            nn.Linear(embedding_dim*expansion, embedding_dim)
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
    highest probability. If you want to do it that way, set `greedy=True`. If
    you want to be conservative, set `greedy=False` and it will combine 
    representations with probabilities.
    """
    def __init__(self, input_dim, embedding_dim, heads=8, expansion=1, kqnorm=True, encoding_layers=1, bias=False, decode_with_gru=True, greedy=False):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.expansion = expansion
        self.kqnorm = kqnorm
        self.bias = bias
        self.encoding_layers = encoding_layers
        self.decoder_method = 'gru' if decode_with_gru else 'attention'
        self.greedy = greedy

        self.embedding = nn.Linear(in_features=input_dim, out_features=self.embedding_dim, bias=self.bias)
        self.encodingLayers = nn.ModuleList([TransformerLayer(self.embedding_dim, heads=self.heads, expansion=self.expansion, contextual=False, kqnorm=self.kqnorm) for _ in range(self.encoding_layers)])
        self.encoding = self.forwardEncoder # since there are masks, it's easier to write a custom forward than wrangle the transformer layers to work with nn.Sequential
        
        if self.decoder_method == 'gru':
            # if using GRU, then make a GRU cell for the decoder
            self.decoder = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim, bias=self.bias)
        
        elif self.decoder_method == 'attention':
            # if using attention for the decoder, make a contextual transformer for the decoder
            self.decoder = TransformerLayer(self.embedding_dim, heads=self.heads, expansion=self.expansion, contextual=True, kqnorm=self.kqnorm)
        
        else: 
            raise ValueError("an unexpected bug occurred and decoder_method was not set correctly"
                             f"it should be either 'gru' or 'attention', but it is {self.decoder_method}")

        # output of the network uses a pointer attention layer as described in the original paper
        self.pointer = PointerAttention(self.embedding_dim, log_softmax=self.greedy)

    def forwardEncoder(self, x, mask=None):
        """
        instead of using nn.Sequential just call each layer in sequence
        this solves the problem of passing in the optional mask to the transformer layer 
        (it would have to be passed as an output by the transformer layer and I'd rather keep the transformer layer code clean)
        """
        for elayer in self.encodingLayers:
            x = elayer(x, mask=mask)
        return x
        
    def forward(self, x, mask=None): 
        """
        forward method for pointer network with possible masked input
        
        x should be an input tensor with shape (batchSize, maxTokens, input_dim)
        mask should be a 1/0 input tensor with shape (batchSize, maxTokens) where a 1 indicates a valid token and 0 indicates padded data
        checks on the mask only care about the shape, so make sure your mask is as described!!!
        """
        assert x.ndim == 3, "x must have shape (batch, tokens, input_dim)"
        batch, tokens, inp_dim = x.size()
        assert inp_dim == self.input_dim, "input dim of x doesn't match network"
        if mask is not None: 
            assert mask.ndim == 2, "mask must have shape (batch, tokens)"
            assert mask.size(0)==batch and mask.size(1)==tokens, "mask must have same batch size and max tokens as x"
        else:
            mask = torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))
        
        # Encoding stage
        embeddedRepresentation = self.embedding(x) # embed each token to right dimensionality
        encodedRepresentation = self.encoding(embeddedRepresentation, mask=mask) # perform N-layer self-attention on inputs
        
        # Get the masked mean of any encoded tokens
        numerator = torch.sum(encodedRepresentation*mask.unsqueeze(2), dim=1)
        denominator = torch.sum(mask, dim=1, keepdim=True)
        decoder_context = numerator / denominator
        #decoder_context = encodedRepresentation.mean(dim=1) # average encoded representations to get the context
        decoder_input = torch.zeros((batch, self.embedding_dim)).to(get_device(x)) # initialize decoder_input as zeros

        # Decoding stage
        pointer_log_scores = []
        pointer_choices = []
        for i in range(tokens):
            # update decoder context using RNN or contextual transformer
            if self.decoder_method == 'gru':
                decoder_context = self.decoder(decoder_input, decoder_context)
            elif self.decoder_method == 'attention':
                contextInputs = torch.cat((decoder_input.unsqueeze(1), encodedRepresentation), dim=1)
                contextMask = torch.cat((torch.ones((batch,1), dtype=mask.dtype).to(get_device(mask)), mask), dim=1)
                decoder_context = self.decoder((decoder_context.unsqueeze(1), contextInputs), contextMask=contextMask).squeeze(1)
            else:
                raise ValueError("decoder_method not recognized")
                
            # use pointer attention to evaluate scores of each possible input given the context
            score = self.pointer(encodedRepresentation, decoder_context, mask=mask)
            
            # standard loss function (nll_loss) requires log-probabilities
            log_score = score if self.greedy else torch.log(score)

            # choose token for this sample
            choice = torch.argmax(log_score, dim=1, keepdim=True)

            if self.greedy:
                # next decoder_input is whatever token had the highest probability
                index_tensor = choice.unsqueeze(-1).expand(batch, 1, self.embedding_dim)
                decoder_input = torch.gather(encodedRepresentation, dim=1, index=index_tensor).squeeze(1)
            else:
                # next decoder_input is the dot product of token encodings and their probabilities
                decoder_input = torch.bmm(score.unsqueeze(1), encodedRepresentation).squeeze(1)
                
            # Save output of each decoding round
            pointer_log_scores.append(log_score)
            pointer_choices.append(choice)
            
        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        pointer_choices = torch.stack(pointer_choices, 1).squeeze()

        return pointer_log_scores, pointer_choices




# # Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
# def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     """
#     ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
#     masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
#     ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
#     ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
#     broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
#     unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
#     do it yourself before passing the mask into this function.
#     In the case that the input vector is completely masked, the return value of this function is
#     arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
#     of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
#     that we deal with this case relies on having single-precision floats; mixing half-precision
#     floats with fully-masked vectors will likely give you ``nans``.
#     If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
#     lower), the way we handle masking here could mess you up.  But if you've got logit values that
#     extreme, you've got bigger problems than this.
#     """
#     if mask is not None:
#         mask = mask.float()
#         while mask.dim() < vector.dim():
#             mask = mask.unsqueeze(1)
#         # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
#         # results in nans when the whole vector is masked.  We need a very small value instead of a
#         # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
#         # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
#         # becomes 0 - this is just the smallest value we can actually use.
#         vector = vector + (mask + 1e-45).log()
#     return torch.nn.functional.log_softmax(vector, dim=dim)


# # Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
# def masked_max(vector: torch.Tensor,
#                mask: torch.Tensor,
#                dim: int,
#                keepdim: bool = False,
#                min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
#     """
#     To calculate max along certain dimensions on masked values
#     Parameters
#     ----------
#     vector : ``torch.Tensor``
#         The vector to calculate max, assume unmasked parts are already zeros
#     mask : ``torch.Tensor``
#         The mask of the vector. It must be broadcastable with vector.
#     dim : ``int``
#         The dimension to calculate max
#     keepdim : ``bool``
#         Whether to keep dimension
#     min_val : ``float``
#         The minimal value for paddings
#     Returns
#     -------
#     A ``torch.Tensor`` of including the maximum values.
#     """
#     one_minus_mask = (1.0 - mask).bool()
#     replaced_vector = vector.masked_fill(one_minus_mask, min_val)
#     max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
#     return max_value, max_index


# class Encoder(nn.Module):
#     def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
#         super(Encoder, self).__init__()

#         self.batch_first = batch_first
#         self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
#                            batch_first=batch_first, bidirectional=bidirectional)

#     def forward(self, embedded_inputs, input_lengths):
#         # Pack padded batch of sequences for RNN module
#         packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths.cpu(), batch_first=self.batch_first)
#         # Forward pass through RNN
#         outputs, hidden = self.rnn(packed)
#         # Unpack padding
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
#         # Return output and final hidden state
#         return outputs, hidden


# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.vt = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, decoder_state, encoder_outputs, mask):        
#         # (batch_size, max_seq_len, hidden_size)
#         encoder_transform = self.W1(encoder_outputs)

#         # (batch_size, 1 (unsqueezed), hidden_size)
#         decoder_transform = self.W2(decoder_state).unsqueeze(1)

#         # 1st line of Eq.(3) in the paper
#         # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
#         u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

#         # softmax with only valid inputs, excluding zero padded parts
#         # log-softmax for a better numerical stability
#         log_score = masked_log_softmax(u_i, mask, dim=-1)

#         return log_score


# class PointerNet(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_size, bidirectional=True, batch_first=True):
#         super(PointerNet, self).__init__()

#         # Embedding dimension
#         self.embedding_dim = embedding_dim
#         # (Decoder) hidden size
#         self.hidden_size = hidden_size
#         # Bidirectional Encoder
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
#         self.num_layers = 1
#         self.batch_first = batch_first

#         # We use an embedding layer for more complicate application usages later, e.g., word sequences.
#         self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, bias=False)
#         self.encoder = Encoder(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=self.num_layers, 
#                                bidirectional=bidirectional, batch_first=batch_first)
#         self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
#         self.attn = Attention(hidden_size=hidden_size)

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias)

#     def forward(self, input_seq, input_lengths):

#         if self.batch_first:
#             batch_size = input_seq.size(0)
#             max_seq_len = input_seq.size(1)
#         else:
#             batch_size = input_seq.size(1)
#             max_seq_len = input_seq.size(0)

#         # Embedding
#         embedded = self.embedding(input_seq)
#         # (batch_size, max_seq_len, embedding_dim)
#         #print(f"Input_seq.shape: {input_seq.shape}")
#         #print(f"Embedded.shape: {embedded.shape}")
        
#         # encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first else (max_seq_len, batch_size, hidden_size)
#         # hidden_size is usually set same as embedding size
#         # encoder_hidden => (num_layers * num_directions, batch_size, hidden_size) for each of h_n and c_n
#         encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)
        
#         #print(f"encoder_outputs.shape: {encoder_outputs.shape}")
#         #print(f"encoder_hidden.shapes: {encoder_hidden[0].shape} -- {encoder_hidden[1].shape}")
        
#         encoder_outputs.shape: torch.Size([256, 10, 16])
#         encoder_hidden.shapes: torch.Size([2, 256, 8]) -- torch.Size([2, 256, 8])
        
#         if self.bidirectional:
#             # Optionally, Sum bidirectional RNN outputs
#             encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        
#         #print(f"post bidi - encoder_outputs.shape: {encoder_outputs.shape}")
        
#         encoder_h_n, encoder_c_n = encoder_hidden
#         encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
#         encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        
#         #print(f"encoder_h_n.shape: {encoder_h_n.shape}")
#         #print(f"encoder_c_n.shape: {encoder_c_n.shape}")
        
#         # Lets use zeros as an intial input for sorting example
#         decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
#         decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())
        
#         #print(f"decoder_input.shape: {decoder_input.shape}")
#         #print(f"decoder_hidden.shape: {decoder_hidden[0].shape} -- {decoder_hidden[1].shape}")
        
#         range_tensor = torch.arange(max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(batch_size, max_seq_len, max_seq_len)
#         each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)
        
#         #print(f"input_lengths.shape: {input_lengths.shape}")
#         #print(f"each_len_tensor.shape: {each_len_tensor.shape}")
        
#         row_mask_tensor = (range_tensor < each_len_tensor)
#         col_mask_tensor = row_mask_tensor.transpose(1, 2)
#         mask_tensor = row_mask_tensor * col_mask_tensor

#         pointer_log_scores = []
#         pointer_argmaxs = []

#         for i in range(max_seq_len):
#             # We will simply mask out when calculating attention or max (and loss later)
#             # not all input and hiddens, just for simplicity
#             sub_mask = mask_tensor[:, i, :].float()

#             # h, c: (batch_size, hidden_size)
#             h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)
            
#             #if i==0: print(f"H_i.shape: {h_i.shape}, c_i.shape: {c_i.shape}")
#             #if i==0: print(f"encoder_outputs.shape: {encoder_outputs.shape}")
#             #if i==0: print(f"sub_mask.shape: {sub_mask.shape}")
            
#             # next hidden
#             decoder_hidden = (h_i, c_i)

#             # Get a pointer distribution over the encoder outputs using attention
#             # (batch_size, max_seq_len)
#             log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
#             pointer_log_scores.append(log_pointer_score)
            
#             #if i==0: print(f"log_pointer_score.shape: {log_pointer_score.shape}")

#             # Get the indices of maximum pointer
#             _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)
            
#             #if i==0: print(f"Masked_argmax.shape: {masked_argmax.shape}")
            
#             pointer_argmaxs.append(masked_argmax)
#             index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)
            
#             #if i==0: print(f"Index_tensor.shape: {index_tensor.shape}")
            
#             # (batch_size, hidden_size)
#             decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)
            
#             #if i==0: 
#                 #checkDecoderInput = torch.stack([eo[am] for eo,am in zip(encoder_outputs, masked_argmax)]).squeeze(1)
#                 #print(f"check: {checkDecoderInput.shape}")
#                 #print(f"Check data comparison: {torch.equal(checkDecoderInput, decoder_input)}")
                    
#             #if i==0: print(f"decoder_input.shape: {decoder_input.shape}")
            
            
#         pointer_log_scores = torch.stack(pointer_log_scores, 1)
#         pointer_argmaxs = torch.cat(pointer_argmaxs, 1)

#         return pointer_log_scores, pointer_argmaxs, mask_tensor
