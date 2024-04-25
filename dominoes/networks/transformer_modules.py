from abc import ABC, abstractmethod
import torch
from torch import nn

from .attention_modules import get_attention_layer


# ---------------------------------
# ------------ networks -----------
# ---------------------------------
class TransformerBaseClass(nn.Module, ABC):
    """
    Performs multi-headed attention on input, then layer normalization, then
    two-stage feedforward processing with an optional expansion, then layer
    normalization, with residual connections before each layer normalization.

    This transformer layer has the option of using contextual attention, where
    some inputs are only used to generate keys and values that modulate the
    primary inputs. This form of contextual attention, if used, processes the
    main inputs and the context inputs using the same key & value matrices.
    (See ContextualAttention above).
    """

    def __init__(self, embedding_dim, contextual, multimodal, heads=8, expansion=1, kqnorm=True, bias=True):
        # check if valid arguments
        self._check_args(embedding_dim, heads, expansion)

        # initialize as a nn module
        super().__init__()

        # store the parameters
        self.embedding_dim = embedding_dim
        self.contextual = contextual
        self.multimodal = multimodal
        self.heads = heads
        self.kqnorm = kqnorm
        self.bias = bias

        # make the attention layer
        self.attention = get_attention_layer(embedding_dim, heads, kqnorm, contextual, multimodal, num_context=1)

        # make the mlp layers
        self.mlp_layers = [
            nn.Linear(embedding_dim, embedding_dim * expansion, bias=bias),
            nn.ReLU(),
            nn.Linear(embedding_dim * expansion, embedding_dim, bias=bias),
        ]
        self.mlp = nn.Sequential(*self.mlp_layers)

        # make the layer norm layers
        self.layer_norm_attended = nn.LayerNorm(embedding_dim)
        self.layer_norm_transformed = nn.LayerNorm(embedding_dim)

    def _check_args(self, embedding_dim, heads, expansion):
        if type(expansion) != int or expansion < 1:
            raise ValueError(f"expansion ({expansion}) must be a positive integer")
        if embedding_dim % heads != 0:
            raise ValueError(f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({heads})")

    def _forward_post_attention(self, x, attended):
        """centralized function to process the output of the attention layer"""
        # mix attended with residual stream and do first layer normalization
        x = self.layer_norm_attended(x + attended)
        # process through mlp layer
        transformed = self.mlp(x)
        # mix transformed with residual stream and do second layer normalization
        return self.layer_norm_transformed(x + transformed)

    @abstractmethod
    def forward(self, x, mask=None, context=None, context_mask=None, mm_context=None, mm_mask=None):
        """forward pass of the transformer"""
        raise NotImplementedError


class Transformer(TransformerBaseClass):
    contextual = False
    multimodal = False

    def __init__(self, *args, **kwargs):
        TransformerBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def forward(self, x, mask=None):
        attended = self.attention(x, mask=mask)
        return self._forward_post_attention(x, attended)


class ContextualTransformer(TransformerBaseClass):
    contextual = True
    multimodal = False

    def __init__(self, *args, **kwargs):
        TransformerBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def forward(self, x, context, mask=None, context_mask=None):
        attended = self.attention(x, context, mask=mask, context_mask=context_mask)
        return self._forward_post_attention(x, attended)


class MultimodalTransformer(TransformerBaseClass):
    contextual = False
    multimodal = True

    def __init__(self, *args, **kwargs):
        TransformerBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def forward(self, x, mm_context, mask=None, mm_mask=None):
        attended = self.attention(x, mm_context, mask=mask, mm_mask=mm_mask)
        return self._forward_post_attention(x, attended)


class MultimodalContextualTransformer(TransformerBaseClass):
    contextual = True
    multimodal = True

    def __init__(self, *args, **kwargs):
        TransformerBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def forward(self, x, context, mm_context, mask=None, context_mask=None, mm_mask=None):
        attended = self.attention(x, context, mm_context, mask=mask, context_mask=context_mask, mm_mask=mm_mask)
        return self._forward_post_attention(x, attended)
