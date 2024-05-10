from abc import ABC, abstractmethod
import torch
from torch import nn

from .attention_modules import get_attention_layer, _attention_type


def _get_transformer_constructor(contextual, multimodal):
    """get the attention constructor based on the attention type"""
    attention_type = _attention_type(contextual, multimodal)
    if attention_type not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Unrecognized transformer type: {attention_type}")
    return TRANSFORMER_REGISTRY[attention_type]


def get_transformer_layer(
    embedding_dim,
    num_heads,
    expansion=4,
    kqnorm=True,
    contextual=False,
    multimodal=False,
    num_multimodal=0,
    kqv_bias=False,
    mlp_bias=True,
):
    """
    create transformer layer with requested arguments
    residual is defaulted to False because transformer layers handle residual connections on their own
    """
    transformer_kwargs = dict(
        num_heads=num_heads,
        expansion=expansion,
        kqnorm=kqnorm,
        kqv_bias=kqv_bias,
        mlp_bias=mlp_bias,
    )
    if multimodal:
        transformer_kwargs["num_multimodal"] = num_multimodal
    transformer_constructor = _get_transformer_constructor(contextual, multimodal)
    return transformer_constructor(embedding_dim, **transformer_kwargs)


# ---------------------------------
# ------------ networks -----------
# ---------------------------------
class TransformerBaseClass(nn.Module, ABC):
    """
    Performs multi-headed attention on input, then layer normalization, then
    two-stage feedforward processing with an optional expansion, then layer
    normalization, with residual connections before each layer normalization.
    """

    def __init__(self, embedding_dim, contextual, multimodal, num_heads=8, expansion=1, kqnorm=True, kqv_bias=False, mlp_bias=True, num_multimodal=0):
        # check if valid arguments
        self._check_args(embedding_dim, num_heads, expansion)

        # initialize as a nn module
        super().__init__()

        # store the parameters
        self.embedding_dim = embedding_dim
        self.contextual = contextual
        self.multimodal = multimodal
        self.num_heads = num_heads
        self.kqnorm = kqnorm
        self.kqv_bias = kqv_bias
        self.mlp_bias = mlp_bias
        self.num_multimodal = num_multimodal * multimodal  # if multimodal is False, num_multimodal is 0

        # make the attention layer
        self.attention = get_attention_layer(
            embedding_dim,
            num_heads,
            kqnorm=kqnorm,
            contextual=contextual,
            multimodal=multimodal,
            num_multimodal=num_multimodal,
            kqv_bias=kqv_bias,
            residual=False,  # residual is handled in the transformer layer's forward pass
        )

        # make the mlp layers
        mlp_layers = [
            nn.Linear(embedding_dim, embedding_dim * expansion, bias=mlp_bias),
            nn.ReLU(),
            nn.Linear(embedding_dim * expansion, embedding_dim, bias=mlp_bias),
        ]
        self.mlp = nn.Sequential(*mlp_layers)

        # make the layer norm layers
        self.layer_norm_attended = nn.LayerNorm(embedding_dim)
        self.layer_norm_transformed = nn.LayerNorm(embedding_dim)

    def _check_args(self, embedding_dim, num_heads, expansion):
        if type(expansion) != int or expansion < 1:
            raise ValueError(f"expansion ({expansion}) must be a positive integer")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embedding_dim}) should be divisible by the number of num_heads ({num_heads})")

    def _forward_post_attention(self, x, attended):
        """centralized function to process the output of the attention layer"""
        # mix attended with residual stream and do first layer normalization
        x = self.layer_norm_attended(x + attended)
        # process through mlp layer
        transformed = self.mlp(x)
        # mix transformed with residual stream and do second layer normalization
        return self.layer_norm_transformed(x + transformed)

    @abstractmethod
    def forward(self, x, mask=None, context=None, context_mask=None, multimode=None, mm_mask=None):
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

    def forward(self, x, multimode, mask=None, mm_mask=None):
        attended = self.attention(x, multimode, mask=mask, mm_mask=mm_mask)
        return self._forward_post_attention(x, attended)


class MultimodalContextualTransformer(TransformerBaseClass):
    contextual = True
    multimodal = True

    def __init__(self, *args, **kwargs):
        TransformerBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def forward(self, x, context, multimode, mask=None, context_mask=None, mm_mask=None):
        attended = self.attention(x, context, multimode, mask=mask, context_mask=context_mask, mm_mask=mm_mask)
        return self._forward_post_attention(x, attended)


TRANSFORMER_REGISTRY = {
    "A": Transformer,
    "CA": ContextualTransformer,
    "MA": MultimodalTransformer,
    "MCA": MultimodalContextualTransformer,
}
