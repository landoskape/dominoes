import torch
from torch import nn

from .attention_modules import get_attention_layer
from .transformer_modules import get_transformer_layer
from .pointer_layers import get_pointer_layer
from .utils import _check_kwargs


class PointerDecoder(nn.Module):
    """
    Implementation of the decoder part of the pointer network.
    """

    def __init__(
        self,
        embedding_dim,
        decoder_method="transformer",
        pointer_method="standard",
        decoder_kwargs={},
        pointer_kwargs={},
        permutation=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_method = decoder_method
        self.pointer_method = pointer_method
        self.permutation = permutation

        self._build_decoder(decoder_method, decoder_kwargs)
        self.pointer = get_pointer_layer(pointer_method, embedding_dim, **pointer_kwargs)

    def _build_decoder(self, decoder_method, decoder_kwargs):
        """method for building decoder with multiple possible methods and specialized kwargs"""
        # build decoder (updates the context vector)
        if decoder_method == "gru":
            required_kwargs = [
                "gru_bias",
            ]
            _check_kwargs("gru", decoder_kwargs, required_kwargs)
            self.decoder = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim, bias=decoder_kwargs["gru_bias"])

        elif decoder_method == "attention":
            required_kwargs = [
                "num_heads",
                "kqnorm",
                "kqv_bias",
            ]
            _check_kwargs("attention", decoder_kwargs, required_kwargs)
            self.decoder = get_attention_layer(
                self.embedding_dim,
                decoder_kwargs["num_heads"],
                kqnorm=decoder_kwargs["kqnorm"],
                contextual=False,
                multimodal=True,
                num_multimodal=2,
                kqv_bias=decoder_kwargs["kqv_bias"],
                residual=decoder_kwargs.get("residual", True),  # default is to update residual stream
            )
        elif decoder_method == "transformer":
            required_kwargs = [
                "num_heads",
                "expansion",
                "kqnorm",
                "kqv_bias",
                "mlp_bias",
            ]
            _check_kwargs("transformer", decoder_kwargs, required_kwargs)
            self.decoder = get_transformer_layer(
                self.embedding_dim,
                decoder_kwargs["num_heads"],
                expansion=decoder_kwargs["expansion"],
                kqnorm=decoder_kwargs["kqnorm"],
                contextual=False,
                multimodal=True,
                num_multimodal=2,
                kqv_bias=decoder_kwargs["kqv_bias"],
                mlp_bias=decoder_kwargs["mlp_bias"],
            )
        else:
            raise ValueError(f"decoder_method={decoder_method} not recognized!")

    def decode(self, encoded, decoder_input, decoder_context, mask):
        """apply the decoder to update the context for the pointer network"""
        # update decoder context using RNN or contextual transformer
        if self.decoder_method == "gru":
            decoder_context = self.decoder(decoder_input, decoder_context)
        elif self.decoder_method in ["attention", "transformer"]:
            mm_inputs = (encoded, decoder_input.unsqueeze(1))
            mm_mask = (mask, None)
            decoder_context = self.decoder(decoder_context.unsqueeze(1), mm_inputs, mm_mask=mm_mask).squeeze(1)
        else:
            raise ValueError("decoder_method not recognized")

        return decoder_context

    def get_decoder_state(self, decoder_input, decoder_context):
        """create a variable representing the decoder state -- pointer layers require different information"""
        if self.pointer_method in ["standard", "dot", "dot_noln", "dot_lean"]:
            decoder_state = decoder_context
        elif self.pointer_method in ["attention", "transformer"]:
            decoder_state = torch.cat((decoder_input.unsqueeze(1), decoder_context.unsqueeze(1)), dim=1)
        else:
            raise ValueError(f"Pointer method not recognized, somehow it has changed to {self.pointer_method}")
        return decoder_state

    def _process_inputs(self, encoded, decoder_input, decoder_context, mask=None, max_output=None):
        """method for processing inputs to the pointer network, checking shape, creating mask if not provided"""
        batch_size, max_num_tokens, embedding_dim = encoded.shape
        assert encoded.ndim == 3, "encoded should be (batch_size, num_tokens, embedding_dim) size tensor"
        assert decoder_input.ndim == 2 and decoder_context.ndim == 2, "decoder input and context should be (batch_size, embedding_dim) tensors"
        assert decoder_input.size(0) == batch_size, "decoder_input has wrong number of batches"
        assert decoder_input.size(1) == embedding_dim, "decoder_input has incorrect embedding dim"
        assert decoder_context.size(0) == batch_size, "decoder_context has wrong number of batches"
        assert decoder_context.size(1) == embedding_dim, "decoder_context has incorrect embedding dim"

        if mask is not None:
            assert mask.ndim == 2, "mask must have shape (batch, tokens)"
            assert mask.size(0) == batch_size and mask.size(1) == max_num_tokens, "mask must have same batch size and max tokens as x"
        else:
            mask = torch.ones((batch_size, max_num_tokens), dtype=encoded.dtype).to(encoded.device)

        if max_output is None:
            max_output = max_num_tokens

        if max_output > max_num_tokens and self.permutation:
            raise ValueError("max_output must be less than or equal to the number of tokens when using permutation mode")

        if self.permutation:
            # prepare source for scattering if required
            src = torch.zeros((batch_size, 1), dtype=mask.dtype).to(mask.device)
        else:
            src = None

        return mask, max_output, src

    def _pointer_loop(
        self,
        processed_encoding,
        decoder_input,
        decoder_context,
        mask,
        src,
        batch_size,
        thompson,
        temperature,
    ):
        """
        method for running the pointer network loop

        - updates the decoder_context with the decoder module
        - gets the decoder state for the pointer layer (different for each pointer architecture)
        - generates a logit for each token in the encoded representation
        - chooses a token (greedy or thompson sampling)
        - updates the new decoder_input (which corresponds to the previously selected token)
        - updates the mask if permutation is enabled
        """
        # update context representation
        decoder_context = self.decode(processed_encoding, decoder_input, decoder_context, mask)

        # use pointer attention to evaluate scores of each possible input given the context
        decoder_state = self.get_decoder_state(decoder_input, decoder_context)
        log_score = self.pointer(processed_encoding, decoder_state, mask=mask, temperature=temperature)

        # choose token for this sample
        if thompson:
            # choose probabilistically
            choice = torch.multinomial(torch.exp(log_score) * mask, 1)
        else:
            # choose based on maximum score
            choice = torch.argmax(log_score, dim=1, keepdim=True)

        # next decoder_input is whatever token had the highest probability
        index_tensor = choice.unsqueeze(-1).expand(batch_size, 1, self.embedding_dim)
        decoder_input = torch.gather(processed_encoding, dim=1, index=index_tensor).squeeze(1)

        if self.permutation:
            # mask previously chosen tokens (don't include this in the computation graph)
            with torch.no_grad():
                mask = mask.scatter(1, choice, src)

        return log_score, choice, decoder_input, decoder_context, mask

    def forward(
        self,
        encoded,
        decoder_input,
        decoder_context,
        mask=None,
        max_output=None,
        thompson=None,
        temperature=None,
    ):
        """
        forward method for pointer module

        x should be an input tensor with shape (batchSize, maxTokens, input_dim)
        mask should be a binary input tensor with shape (batchSize, maxTokens) where a 1 indicates a valid token and 0 indicates padded data
        checks on the mask only care about the shape, so make sure your mask is as described!!!

        max_output should be an integer determining when to cut off decoder output
        """
        batch_size = encoded.size(0)  # get batch size
        mask, max_output, src = self._process_inputs(encoded, decoder_input, decoder_context, mask=mask, max_output=max_output)

        # For some pointer layers, the encoded transform happens out of the loop just once
        # For others, it happens at each step. The output of this is a special object that
        # forces this initial pass to be performed.
        processed_encoding = self.pointer.process_encoded(encoded)

        # Decoding stage
        pointer_log_scores = []
        pointer_choices = []
        pointer_context = []
        for _ in range(max_output):
            log_score, choice, decoder_input, decoder_context, mask = self._pointer_loop(
                processed_encoding,
                decoder_input,
                decoder_context,
                mask,
                src,
                batch_size,
                thompson,
                temperature,
            )

            # Save output of each decoding round
            pointer_log_scores.append(log_score)
            pointer_choices.append(choice)
            pointer_context.append(decoder_context)

        log_scores = torch.stack(pointer_log_scores, 1)
        choices = torch.stack(pointer_choices, 1).squeeze(2)

        return log_scores, choices
