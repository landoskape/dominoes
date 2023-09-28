import torch
from torch import nn
import torch.nn.functional as F

"""
Almost everything I've learned about machine learning and pytorch has been due
to reading blogs, papers, and people kindly posting good code on github. This
script is no exception, and has drawn heavily from two code sources and a 
paper. 

pointer network code: 
- from the repository: https://github.com/ast0414/pointer-networks-pytorch
- from the original paper: https://papers.nips.cc/paper/5866-pointer-networks

transformer code: 
- from the repository: https://github.com/pbloem/former
  - and the associated blog: http://peterbloem.nl/blog/transformers
"""


def mask_future(data, maskval=0.0, mask_diagonal=True):
    """
    Masks out values in a given batch of data where i <= j
    Use when you don't want future token j to influence past token i
    Adopted from pbloem/former
    """
    h, w = data.size(-2), data.size(-1)
    mask_index = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    data[..., mask_index[0], mask_index[1]] = maskval


class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    Adopted from pbloem/former
    """
    def __init__(self, emb, heads=8, mask=False, kqnorm=False):
        super().__init__()

        # This is a requirement
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by the # of heads ({heads})'

        # Store parameters 
        self.emb = emb
        self.heads = heads
        self.mask = mask

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

    def forward(self, x):
        # attention layer forward pass
        assert x.ndim==3, "x should have size: (batch_size, num_tokens, embedding_dimensionality)"
        batch, tokens, emb = x.size() # get size of input
        
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
        
        # if requested, mask out dot products that inform the past from the future    
        if self.mask: 
            mask_future(dot, maskval=float('-inf'), mask_diagonal=False)

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

    def forward(self, x, context):
        # attention layer forward pass
        assert x.ndim==3, "x should have size: (batch_size, num_tokens, embedding_dimensionality)"
        assert context.ndim==3, "context should have size: (batch_size, num_tokens, embedding_dimensionality)"
        batch, itokens, emb = x.size() # get size of input
        cbatch, ctokens, cemb = context.size() # get size of context
        tokens = itokens + ctokens # total number of tokens to process
        
        assert cbatch==batch, "batch size of x and context should be the same"
        assert emb == cemb == self.emb, f'Input embedding dim ({emb}) and context embedding dim ({cemb}) should both match layer embedding dim ({self.emb})'
        
        # concatenate input and context
        x_plus_context = torch.cat((x, context), dim=1) 
        
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
    Can probably replace with "SelfAttention" above, but let's start here
    """
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.W1 = nn.Linear(emb, emb, bias=False)
        self.W2 = nn.Linear(emb, emb, bias=False)
        self.vt = nn.Linear(emb, 1, bias=False)

    def forward(self, encoded, decoder_state):
        # first transform encoded representations and decoder states 
        transformEncoded = self.W1(encoded)
        transformDecoder = self.W2(decoder_state).unsqueeze(1) # unsqueeze for broadcasting on token dimension

        # then combine them and project to a new space
        u = self.vt(torch.tanh(transformEncoded + transformDecoder)).squeeze(-1)

        # convert to probabilities with softmax
        log_score = torch.nn.functional.log_softmax(u, dim=-1)
        return log_score


class PointerNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, heads=8, kqnorm=True, encoding_layers=1, bias=False, decode_with_gru=True):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.kqnorm = kqnorm
        self.bias = bias
        self.encoding_layers = encoding_layers
        self.decoder_method = 'gru' if decode_with_gru else 'attention'

        self.embedding = nn.Linear(in_features=input_dim, out_features=self.embedding_dim, bias=self.bias)
        self.encodingLayers = [SelfAttention(self.embedding_dim, heads=self.heads, kqnorm=self.kqnorm) for _ in range(self.encoding_layers)]
        self.encoder = nn.Sequential()
        for layer in self.encodingLayers: 
            self.encoder.append(layer)

        self.contextEncoder = SelfAttention(self.embedding_dim, heads=self.heads, kqnorm=self.kqnorm)
        if self.decoder_method == 'gru':
            self.decoder = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim, bias=self.bias)
        
        elif self.decoder_method == 'attention':
            self.decoder = ContextualAttention(self.embedding_dim, heads=self.heads, kqnorm=self.kqnorm)
        
        else: 
            raise ValueError("an unexpected bug occurred and self.decoder_method was not set correctly"
                             f"it should be either 'gru' or 'attention', but it is {self.decoder_method}")
        
        self.pointer = PointerAttention(self.embedding_dim)
    
    def get_device(self, tensor):
        return 'cuda' if tensor.is_cuda else 'cpu'
        
    def forward(self, x): 
        assert x.ndim == 3, "x must have shape (batch, tokens, embedding)"
        batch, tokens, inp_dim = x.size()
        assert inp_dim == self.input_dim, "input dim of x doesn't match network"
        
        embeddedRepresentation = self.embedding(x) # embed each token to right dimensionality
        encodedRepresentation = self.encoder(embeddedRepresentation) # perform N-layer self-attention on inputs
        decoder_input = torch.zeros((batch, self.embedding_dim)).to(self.get_device(x))

        decoder_context = self.contextEncoder(encodedRepresentation).mean(dim=1)

        pointer_log_scores = []
        pointer_choices = []
        
        for i in range(tokens):
            # update decoder context
            if self.decoder_method == 'gru':
                decoder_context = self.decoder(decoder_input, decoder_context)
            elif self.decoder_method == 'attention':
                contextInputs = torch.cat((decoder_input.unsqueeze(1), encodedRepresentation), dim=1)
                decoder_context = self.decoder(decoder_context.unsqueeze(1), contextInputs).squeeze(1)
            else:
                raise ValueError("self.decoder_method not recognized")
                
            # use pointer layer to evaluate scores of each possible input
            log_score = self.pointer(encodedRepresentation, decoder_context)

            # choose token for this sample
            choice = torch.argmax(log_score, dim=1, keepdim=True)

            # convert choice vector to gather index and get encoded representations for this sample
            index_tensor = choice.unsqueeze(-1).expand(batch, 1, self.embedding_dim)
            decoder_input = torch.gather(encodedRepresentation, dim=1, index=index_tensor).squeeze(1)
            
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
