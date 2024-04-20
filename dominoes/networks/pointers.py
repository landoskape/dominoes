import torch
from torch import nn

from ..utils import get_device
from .transformer_modules import (
    ContextualTransformerLayer,
    TransformerLayer,
    PointerStandard,
    PointerDot,
    PointerDotNoLN,
    PointerDotLean,
    PointerAttention,
    PointerTransformer,
)

# This is a list of the available pointer methods that can be used in the pointer network
POINTER_METHODS = [
    "PointerStandard",
    "PointerDot",
    "PointerDotLean",
    "PointerDotNoLN",
    "PointerAttention",
    "PointerTransformer",
]


def get_pointer_methods():
    """method for retrieving the list of pointer methods that can be used in the pointer network"""
    return POINTER_METHODS


class PointerModule(nn.Module):
    """
    Implementation of the decoder part of the pointer network
    """

    def __init__(
        self,
        embedding_dim,
        heads=8,
        expansion=1,
        kqnorm=True,
        thompson=False,
        bias=True,
        decoder_method="transformer",
        pointer_method="PointerStandard",
        temperature=1.0,
        permutation=True,
    ):
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
        if self.decoder_method == "gru":
            # if using GRU, then make a GRU cell for the decoder
            self.decoder = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim, bias=self.bias)

        elif self.decoder_method == "transformer":
            # if using contextual transformer, make one
            self.decoder = ContextualTransformerLayer(
                self.embedding_dim, heads=self.heads, expansion=self.expansion, num_context=2, kqnorm=self.kqnorm
            )
        else:
            raise ValueError(f"decoder_method={decoder_method} not recognized!")

        # build pointer (chooses an output)
        if self.pointer_method == "PointerStandard":
            # output of the network uses a pointer attention layer as described in the original paper
            self.pointer = PointerStandard(self.embedding_dim)

        elif self.pointer_method == "PointerDot":
            self.pointer = PointerDot(self.embedding_dim)

        elif self.pointer_method == "PointerDotNoLN":
            self.pointer = PointerDotNoLN(self.embedding_dim)

        elif self.pointer_method == "PointerDotLean":
            self.pointer = PointerDotLean(self.embedding_dim)

        elif self.pointer_method == "PointerAttention":
            kwargs = {"heads": self.heads, "kqnorm": self.kqnorm}
            self.pointer = PointerAttention(self.embedding_dim, **kwargs)

        elif self.pointer_method == "PointerTransformer":
            # output of the network uses a pointer attention layer with a transformer
            kwargs = {"heads": self.heads, "expansion": 1, "kqnorm": self.kqnorm, "bias": self.bias}
            self.pointer = PointerTransformer(self.embedding_dim, **kwargs)

        else:
            raise ValueError(f"the pointer_method was not set correctly, {self.pointer_method} not recognized")

    def decode(self, encoded, decoder_input, decoder_context, mask):
        # update decoder context using RNN or contextual transformer
        if self.decoder_method == "gru":
            decoder_context = self.decoder(decoder_input, decoder_context)
        elif self.decoder_method == "transformer":
            contextInputs = (encoded, decoder_input.unsqueeze(1))
            contextMask = (mask, None)
            decoder_context = self.decoder(decoder_context.unsqueeze(1), contextInputs, contextMask=contextMask).squeeze(1)
        else:
            raise ValueError("decoder_method not recognized")

        return decoder_context

    def get_decoder_state(self, decoder_input, decoder_context):
        if self.pointer_method == "PointerStandard":
            decoder_state = decoder_context
        elif self.pointer_method == "PointerDot":
            decoder_state = decoder_context
        elif self.pointer_method == "PointerDotNoLN":
            decoder_state = decoder_context
        elif self.pointer_method == "PointerDotLean":
            decoder_state = decoder_context
        elif self.pointer_method == "PointerAttention":
            decoder_state = torch.cat((decoder_input.unsqueeze(1), decoder_context.unsqueeze(1)), dim=1)
        elif self.pointer_method == "PointerTransformer":
            decoder_state = torch.cat((decoder_input.unsqueeze(1), decoder_context.unsqueeze(1)), dim=1)
        else:
            raise ValueError(f"Pointer method not recognized, somehow it has changed to {self.pointer_method}")
        return decoder_state

    def forward(self, encoded, decoder_input, decoder_context, mask=None, max_output=None, store_hidden=False):
        """
        forward method for pointer module

        x should be an input tensor with shape (batchSize, maxTokens, input_dim)
        mask should be a binary input tensor with shape (batchSize, maxTokens) where a 1 indicates a valid token and 0 indicates padded data
        checks on the mask only care about the shape, so make sure your mask is as described!!!

        max_output should be an integer determining when to cut off decoder output
        """
        batch_size, max_num_tokens, embed_dim = encoded.shape
        assert encoded.ndim == 3, "encoded should be (batch_size, num_tokens, embed_dim) size tensor"
        assert decoder_input.ndim == 2 and decoder_context.ndim == 2, "decoder input and context should be (batch_size, embed_dim) tensors"
        assert decoder_input.size(0) == batch_size, "decoder_input has wrong number of batches"
        assert decoder_input.size(1) == embed_dim, "decoder_input has incorrect embedding dim"
        assert decoder_context.size(0) == batch_size, "decoder_context has wrong number of batches"
        assert decoder_context.size(1) == embed_dim, "decoder_context has incorrect embedding dim"

        if mask is not None:
            assert mask.ndim == 2, "mask must have shape (batch, tokens)"
            assert mask.size(0) == batch_size and mask.size(1) == max_num_tokens, "mask must have same batch size and max tokens as x"
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
        pointer_context = []
        for i in range(max_output):
            # update context representation
            decoder_context = self.decode(encoded, decoder_input, decoder_context, mask)

            # use pointer attention to evaluate scores of each possible input given the context
            decoder_state = self.get_decoder_state(decoder_input, decoder_context)
            log_score = self.pointer(encoded, decoder_state, mask=mask, temperature=self.temperature)

            # choose token for this sample
            if self.thompson:
                # choose probabilistically
                choice = torch.multinomial(torch.exp(log_score) * mask, 1)
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
            pointer_context.append(decoder_context)

        log_scores = torch.stack(pointer_log_scores, 1)
        choices = torch.stack(pointer_choices, 1).squeeze(2)

        if store_hidden:
            self.decoder_context = torch.stack(pointer_context, 1)

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

    def __init__(
        self,
        input_dim,
        embedding_dim,
        heads=8,
        expansion=1,
        kqnorm=True,
        encoding_layers=1,
        bias=True,
        pointer_method="PointerStandard",
        contextual_encoder=False,
        decoder_method="transformer",
        thompson=False,
        temperature=1.0,
        permutation=True,
    ):
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
        elif contextual_encoder == "multicontext":
            self.contextual_encoder = True
            self.multicontext_encoder = True
            self.num_context = 1
        elif isinstance(contextual_encoder, tuple) and contextual_encoder[0] == "multicontext":
            assert isinstance(contextual_encoder[1], int) and contextual_encoder[1] > 0, "second element of tuple must be int"
            self.contextual_encoder = True
            self.multicontext_encoder = True
            self.num_context = contextual_encoder[1]
        else:
            self.contextual_encoder = False
            self.multicontext_encoder = False

        self.embedding = nn.Linear(in_features=input_dim, out_features=self.embedding_dim, bias=self.bias)

        if self.multicontext_encoder:
            self.encodingLayers = nn.ModuleList(
                [
                    ContextualTransformerLayer(
                        self.embedding_dim, heads=self.heads, expansion=self.expansion, kqnorm=self.kqnorm, num_context=self.num_context
                    )
                    for _ in range(self.encoding_layers)
                ]
            )
        else:
            self.encodingLayers = nn.ModuleList(
                [
                    TransformerLayer(
                        self.embedding_dim, heads=self.heads, expansion=self.expansion, kqnorm=self.kqnorm, contextual=self.contextual_encoder
                    )
                    for _ in range(self.encoding_layers)
                ]
            )

        # since there are masks, it's easier to write a custom forward than wrangle the transformer layers to work with nn.Sequential
        self.encoding = self.forwardEncoder

        self.pointer = PointerModule(
            self.embedding_dim,
            heads=self.heads,
            expansion=self.expansion,
            kqnorm=self.kqnorm,
            bias=self.bias,
            decoder_method=self.decoder_method,
            pointer_method=self.pointer_method,
            temperature=self.temperature,
            thompson=self.thompson,
            permutation=permutation,
        )

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
            context = x[1:]  # get either single context or all context inputs (if multicontext)
            x = x[0]
            if self.multicontext_encoder:
                assert len(context) == self.num_context, "number of context inputs doesn't match expected"
            else:
                context = context[0]  # pull context out of tuple

        for elayer in self.encodingLayers:
            if self.multicontext_encoder:
                x = elayer(x, context, mask=mask)
            elif self.contextual_encoder and not self.multicontext_encoder:
                x = elayer((x, context), mask=mask)
            else:
                x = elayer(x, mask=mask)

        return x

    def forward(self, x, mask=None, decode_mask=None, max_output=None, init=None, store_hidden=False):
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
            assert len(x) == self.num_context + 1, "number of context inputs doesn't match expected"
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

        if self.permutation:
            msg = f"if using permutation mode, max_output ({max_output}) must be less than or equal to the number of tokens ({tokens})"
            if init is not None:
                if max_output > tokens - 1:
                    raise ValueError(msg + " minus 1 (for the initial token)")
            else:
                if max_output > tokens:
                    raise ValueError(msg)

        if mask is not None:
            assert mask.ndim == 2, "mask must have shape (batch, tokens)"
            assert mask.size(0) == batch and mask.size(1) == tokens, "mask must have same batch size and max tokens as x"
            assert not torch.any(torch.all(mask == 0, dim=1)), "mask includes rows where all elements are masked, this is not permitted"
        else:
            mask = torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))

        if decode_mask is not None:
            assert decode_mask.ndim == 2, "decode_mask must have shape (batch, tokens)"
            assert decode_mask.size(0) == batch and decode_mask.size(1) == tokens, "decode_mask must have same batch size and max tokens as x"
        else:
            decode_mask = torch.ones((batch, tokens), dtype=x.dtype).to(get_device(x))
            decode_mask *= mask

        # Encoding stage
        embeddedRepresentation = self.embedding(x)  # embed each token to right dimensionality
        if self.multicontext_encoder:
            embeddedContext = [self.embedding(ctx) for ctx in context]
            embeddedRepresentation = (embeddedRepresentation, *embeddedContext)

        if self.contextual_encoder and not self.multicontext_encoder:
            embeddedContext = self.embedding(context)  # embed the context to the right dimensionality
            embeddedRepresentation = (embeddedRepresentation, embeddedContext)  # add context to transformer input

        encodedRepresentation = self.encoding(embeddedRepresentation, mask=mask)  # perform N-layer self-attention on inputs

        # Get the masked mean of any encoded tokens
        numerator = torch.sum(encodedRepresentation * mask.unsqueeze(2), dim=1)
        denominator = torch.sum(mask, dim=1, keepdim=True)
        decoder_context = numerator / denominator

        if init is not None:
            rinit = init.view(batch, 1, 1).expand(-1, -1, self.embedding_dim)
            decoder_input = torch.gather(encodedRepresentation, 1, rinit).squeeze(1)
            mask = mask.scatter(1, init.view(batch, 1), torch.zeros((batch, 1), dtype=mask.dtype).to(get_device(mask)))

            # note from refactoring effort:
            # I think if init is used and permutation=True, then we should assert that max_output <= len(tokens)-1 !!!
            raise ValueError("See above comment")
        else:
            decoder_input = torch.zeros((batch, self.embedding_dim)).to(get_device(x))  # initialize decoder_input as zeros

        # Then do pointer network (sequential decode-choice for N=max_output rounds)
        log_scores, choices = self.pointer(
            encodedRepresentation, decoder_input, decoder_context, mask=mask, max_output=max_output, store_hidden=store_hidden
        )

        if store_hidden:
            self.embeddedRepresentation = embeddedRepresentation
            self.encodedRepresentation = encodedRepresentation

        return log_scores, choices
