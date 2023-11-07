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


class PointerStandard(nn.Module):
    """
    PointerStandard Module (as specified in the original paper)

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

    def forward(self, encoded, decoder_state, mask=None, temperature=1):
        # first transform encoded representations and decoder states 
        transformEncoded = self.W1(encoded)
        transformDecoded = self.W2(decoder_state)

        # then combine them and project to a new space
        u = self.vt(torch.tanh(transformEncoded + transformDecoded.unsqueeze(1))).squeeze(2)
        if mask is not None:
            # u += (mask + 1e-45).log()
            # this produces nans for any row that is fully masked 
            u.masked_fill_(mask==0, -200) # only use valid tokens

        if self.log_softmax:
            # convert to log scores
            return torch.nn.functional.log_softmax(u/temperature, dim=-1)
        else:
            # convert to probabilities
            return torch.nn.functional.softmax(u/temperature, dim=-1)


class PointerDot(nn.Module):
    """
    PointerDot Module (variant of the original paper)

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

    def forward(self, encoded, decoder_state, mask=None, temperature=1):
        # first transform encoded representations and decoder states 
        transformEncoded = self.W1(encoded)
        transformDecoded = self.W2(decoder_state)

        # instead of add, tanh, and project on learnable weights, 
        # just dot product the encoded representations with the decoder "pointer"
        u = torch.bmm(transformEncoded, transformDecoded.unsqueeze(2)).squeeze(2)
        
        if mask is not None:
            # u += (mask + 1e-45).log()
            # this produces nans for any row that is fully masked 
            u.masked_fill_(mask==0, -200) # only use valid tokens

        if self.log_softmax:
            # convert to log scores
            return torch.nn.functional.log_softmax(u/temperature, dim=1)
        else:
            # convert to probabilities
            return torch.nn.functional.softmax(u/temperature, dim=1)
            

class PointerAttention(nn.Module):
    """
    PointerAttention Module (variant of paper using standard attention)

    log_softmax is preferable if the probabilities of each token are not 
    needed. However, if token embeddings are combined via their probabilities,
    then softmax is required, so log_softmax should be set to `False`.
    """
    def __init__(self, emb, log_softmax=False, **kwargs):
        super().__init__()
        self.emb = emb
        self.log_softmax = log_softmax

        self.attention = ContextualAttention(emb, **kwargs)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forward(self, encoded, decoder_state, mask=None, temperature=1):
        # attention on encoded representations with decoder_state
        updated = self.attention(encoded, decoder_state, mask=mask)
        project = self.vt(torch.tanh(updated)).squeeze(2)
        if mask is not None:
            project.masked_fill_(mask==0, -200) # pin masked tokens before softmax

        if self.log_softmax:
            # convert to log scores
            scores = torch.nn.functional.log_softmax(project/temperature, dim=1)
        else:
            # convert to probabilities
            scores = torch.nn.functional.softmax(project/temperature, dim=1)

        return scores
        
class PointerTransformer(nn.Module):
    """
    PointerTransformer Module (variant of paper using a transformer)

    log_softmax is preferable if the probabilities of each token are not 
    needed. However, if token embeddings are combined via their probabilities,
    then softmax is required, so log_softmax should be set to `False`.
    """
    def __init__(self, emb, log_softmax=False, **kwargs):
        super().__init__()
        self.emb = emb
        self.log_softmax = log_softmax

        if 'contextual' in kwargs: kwargs['contextual']=True
        self.transform = TransformerLayer(emb, contextual=True, **kwargs)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forward(self, encoded, decoder_state, mask=None, temperature=1):
        # transform encoded representations with decoder_state
        updated = self.transform((encoded, decoder_state), mask=mask)
        project = self.vt(torch.tanh(updated)).squeeze(2)
        if mask is not None:
            project.masked_fill_(mask==0, -200) # pin masked tokens before softmax

        if self.log_softmax:
            # convert to log scores
            scores = torch.nn.functional.log_softmax(project/temperature, dim=1)
        else:
            # convert to probabilities
            scores = torch.nn.functional.softmax(project/temperature, dim=1)

        return scores

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
    def __init__(self, embedding_dim, heads=8, expansion=1, contextual=False, kqnorm=False, bias=False):
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


class PointerModule(nn.Module):
    """
    Implementation of the decoder part of the pointer network
    """
    def __init__(self, embedding_dim, heads=8, expansion=1, kqnorm=True, encoding_layers=1, thompson=False,
                 bias=False, decode_with_gru=True, pointer_method='PointerStandard', greedy=False, temperature=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.expansion = expansion
        self.kqnorm = kqnorm
        self.bias = bias
        self.decoder_method = 'gru' if decode_with_gru else 'attention'
        self.pointer_method = pointer_method
        self.greedy = greedy
        self.thompson = thompson
        self.temperature = temperature

        # build decoder (updates the context vector)
        if self.decoder_method == 'gru':
            # if using GRU, then make a GRU cell for the decoder
            self.decoder = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim, bias=self.bias)
        
        elif self.decoder_method == 'attention':
            # if using attention for the decoder, make a contextual transformer for the decoder
            self.decoder = TransformerLayer(self.embedding_dim, heads=self.heads, expansion=self.expansion, contextual=True, kqnorm=self.kqnorm)
        
        else: 
            raise ValueError("an unexpected bug occurred and decoder_method was not set correctly"
                             f"it should be either 'gru' or 'attention', but it is {self.decoder_method}")

        # build pointer (chooses an output)
        if self.pointer_method == 'PointerStandard':
            # output of the network uses a pointer attention layer as described in the original paper
            self.pointer = PointerStandard(self.embedding_dim, log_softmax=self.greedy)

        elif self.pointer_method == 'PointerDot':
            self.pointer = PointerDot(self.embedding_dim, log_softmax=self.greedy)
            
        elif self.pointer_method == 'PointerAttention':
            kwargs = {'heads':self.heads, 'kqnorm':self.kqnorm}
            self.pointer = PointerAttention(self.embedding_dim, log_softmax=self.greedy, **kwargs)

        elif self.pointer_method == 'PointerTransformer':
            # output of the network uses a pointer attention layer with a transformer
            kwargs = {'heads':self.heads, 'expansion':1, 'kqnorm':self.kqnorm, 'bias':self.bias}
            self.pointer = PointerTransformer(self.embedding_dim, log_softmax=self.greedy, **kwargs)
            
        else:
            raise ValueError("the pointer_method was not set correctly"
                             f"it should either be 'PointerStandard' or 'PointerTransformer' but it is {self.pointer_method}")
        

    def decode(self, encoded, decoder_input, decoder_context, mask):
        # update decoder context using RNN or contextual transformer
        if self.decoder_method == 'gru':
            decoder_context = self.decoder(decoder_input, decoder_context)
        elif self.decoder_method == 'attention':
            contextInputs = torch.cat((decoder_input.unsqueeze(1), encoded), dim=1)
            contextMask = torch.cat((torch.ones((encoded.size(0),1), dtype=mask.dtype).to(get_device(mask)), mask), dim=1)
            decoder_context = self.decoder((decoder_context.unsqueeze(1), contextInputs), contextMask=contextMask).squeeze(1)
        else:
            raise ValueError("decoder_method not recognized")
            
        return decoder_context

    def get_decoder_state(self, decoder_input, decoder_context):
        if self.pointer_method == 'PointerStandard':
            decoder_state = decoder_context
        elif self.pointer_method == 'PointerDot':
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
            
        # Decoding stage
        pointer_log_scores = []
        pointer_choices = []
        for i in range(max_output):
            # update context representation
            decoder_context = self.decode(encoded, decoder_input, decoder_context, mask)
            
            # use pointer attention to evaluate scores of each possible input given the context
            decoder_state = self.get_decoder_state(decoder_input, decoder_context)
            score = self.pointer(encoded, decoder_state, mask=mask, temperature=self.temperature)
            
            # standard loss function (nll_loss) requires log-probabilities
            log_score = score if self.greedy else torch.log(score)

            # choose token for this sample
            if self.thompson:
                # choose probabilistically
                choice = torch.multinomial(torch.exp(log_score), 1)
            else:
                # choose based on maximum score
                choice = torch.argmax(log_score, dim=1, keepdim=True)

            if self.greedy:
                # next decoder_input is whatever token had the highest probability
                index_tensor = choice.unsqueeze(-1).expand(batch_size, 1, self.embedding_dim)
                decoder_input = torch.gather(encoded, dim=1, index=index_tensor).squeeze(1)
            else:
                # next decoder_input is the dot product of token encodings and their probabilities
                decoder_input = torch.bmm(score.unsqueeze(1), encoded).squeeze(1)
                
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
    highest probability. If you want to do it that way, set `greedy=True`. If
    you want to be conservative, set `greedy=False` and it will combine 
    representations with probabilities.
    """
    def __init__(self, input_dim, embedding_dim,
                 heads=8, expansion=1, kqnorm=True, encoding_layers=1, bias=False, pointer_method='PointerStandard',
                 contextual_encoder=False, decode_with_gru=True, greedy=False, thompson=False, temperature=1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.expansion = expansion
        self.kqnorm = kqnorm
        self.bias = bias
        self.encoding_layers = encoding_layers
        self.contextual_encoder = contextual_encoder
        self.decoder_method = 'gru' if decode_with_gru else 'attention'
        self.greedy = greedy
        self.thompson = thompson
        self.decode_with_gru = decode_with_gru
        self.pointer_method = pointer_method
        self.temperature = temperature

        self.embedding = nn.Linear(in_features=input_dim, out_features=self.embedding_dim, bias=self.bias)
        self.encodingLayers = nn.ModuleList([TransformerLayer(self.embedding_dim, 
                                                              heads=self.heads, 
                                                              expansion=self.expansion,  
                                                              kqnorm=self.kqnorm,
                                                              contextual=self.contextual_encoder) 
                                             for _ in range(self.encoding_layers)])
        
        self.encoding = self.forwardEncoder # since there are masks, it's easier to write a custom forward than wrangle the transformer layers to work with nn.Sequential

        self.pointer = PointerModule(self.embedding_dim, heads=self.heads, expansion=self.expansion, kqnorm=self.kqnorm,
                                     encoding_layers=self.encoding_layers, bias=self.bias, decode_with_gru=self.decode_with_gru,
                                     pointer_method=self.pointer_method, greedy=self.greedy, temperature=self.temperature,
                                     thompson=self.thompson)

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
            context = x[1]
            
        for elayer in self.encodingLayers:
            x = elayer(x, mask=mask)
            if self.contextual_encoder:
                x = (x, context) # re-add context for next pass

        if self.contextual_encoder:
            # only output main inputs, not context
            x = x[0]
                
        return x
        
    def forward(self, x, mask=None, decode_mask=None, max_output=None): 
        """
        forward method for pointer network with possible masked input
        
        x should be an input tensor with shape (batchSize, maxTokens, input_dim)
        mask should be a binary input tensor with shape (batchSize, maxTokens) where a 1 indicates a valid token and 0 indicates padded data
        checks on the mask only care about the shape, so make sure your mask is as described!!!

        max_output should be an integer determining when to cut off decoder output
        """
        if self.contextual_encoder:
            x, context = x
            assert context.ndim == 3, "context (x[1]) must have shape (batch, tokens, input_dim)"
            
        assert x.ndim == 3, "x must have shape (batch, tokens, input_dim)"
        batch, tokens, inp_dim = x.size()
        assert inp_dim == self.input_dim, "input dim of x doesn't match network"
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
            
        if max_output is None: 
            max_output = tokens

        # Encoding stage
        embeddedRepresentation = self.embedding(x) # embed each token to right dimensionality
        if self.contextual_encoder:
            embeddedContext = self.embedding(context) # embed the context to the right dimensionality
            embeddedRepresentation = (embeddedRepresentation, embeddedContext) # add context to transformer input
            
        encodedRepresentation = self.encoding(embeddedRepresentation, mask=mask) # perform N-layer self-attention on inputs
                    
        # Get the masked mean of any encoded tokens
        numerator = torch.sum(encodedRepresentation*mask.unsqueeze(2), dim=1)
        denominator = torch.sum(mask, dim=1, keepdim=True)
        decoder_context = numerator / denominator
        decoder_input = torch.zeros((batch, self.embedding_dim)).to(get_device(x)) # initialize decoder_input as zeros

        # Then do pointer network (sequential decode-choice for max_output's)
        log_scores, choices = self.pointer(encodedRepresentation, decoder_input, decoder_context, mask=mask, max_output=max_output)

        return log_scores, choices

