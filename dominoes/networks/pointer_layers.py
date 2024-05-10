from abc import ABC, abstractmethod

import torch
from torch import nn

from ..utils import masked_log_softmax
from .attention_modules import get_attention_layer
from .transformer_modules import get_transformer_layer
from .net_utils import _check_kwargs


def get_pointer_methods():
    """return the available pointer methods"""
    return list(POINTER_REGISTRY.keys())


def _get_pointer_constructor(pointer_method):
    """get the pointer layer constructor based on the pointer method"""
    if pointer_method not in POINTER_REGISTRY:
        raise ValueError(f"Unrecognized pointer method: {pointer_method}")
    return POINTER_REGISTRY[pointer_method]


def get_pointer_layer(pointer_method, embedding_dim, **kwargs):
    """create pointer layer with requested arguments"""
    pointer_constructor = _get_pointer_constructor(pointer_method)
    return pointer_constructor(embedding_dim, **kwargs)


class StoredEncoding:
    """
    StoredEncoding class used to handle the encoded representations of the input for different pointer layers

    Some pointer layers require the encoded representations of the input to be stored for later use. These
    representations are stored in this class and passed to the pointer layer when needed.

    For use, run:
    pl = PointerLayer(**prms) # any of the pointer layers in this module
    encoded = pl.process_encoded(input) # process_encoded return an instance of StoredEncoding
    """

    def __init__(self, encoded):
        self.stored_encoding = encoded


class PointerLayer(ABC, nn.Module):
    """PointerLayer Module (abstract class containing the basic structure of a pointer layer)"""

    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, **kwargs):
        """required method for initializing the pointer layer"""
        raise NotImplementedError

    @abstractmethod
    def process_encoded(self, encoded):
        raise NotImplementedError

    @abstractmethod
    def _get_logits(self, encoded, decoder_state, mask):
        """centralized method of computing logits for the pointer layer"""
        raise NotImplementedError

    def _get_probabilities(self, u, mask, temperature):
        """centralized method of converting logits to probabilities with masking and temperature"""
        return masked_log_softmax(u / temperature, mask, dim=1)

    def forward(self, encoded, decoder_state, mask=None, temperature=1.0):
        """centralized forward pass of the pointer layer"""
        logits = self._get_logits(encoded, decoder_state, mask)
        return self._get_probabilities(logits, mask, temperature)


class PointerStandard(PointerLayer):
    """PointerStandard Module (as specified in the original paper)"""

    def initialize(self, **kwargs):
        """prepare modules for standard pointer layer"""
        self.W1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=kwargs.get("bias", False))
        self.W2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=kwargs.get("bias", False))
        self.vt = nn.Linear(self.embedding_dim, 1, bias=False)

    def process_encoded(self, encoded):
        return StoredEncoding(self.W1(encoded))

    def _get_logits(self, encoded, decoder_state, mask):
        transform_decoded = self.W2(decoder_state)
        logits = self.vt(torch.tanh(encoded + transform_decoded.unsqueeze(1))).squeeze(2)
        return logits


class PointerDot(PointerLayer):
    """PointerDot Module (variant of the paper, using a simple dot product)"""

    def initialize(self, **kwargs):
        """prepare modules for dot product pointer layer"""
        self.W1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=kwargs.get("bias", False))
        self.W2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=kwargs.get("bias", False))
        self.eln = nn.LayerNorm(self.embedding_dim, bias=False)
        self.dln = nn.LayerNorm(self.embedding_dim, bias=False)

    def process_encoded(self, encoded):
        return StoredEncoding(self.eln(self.W1(encoded)))

    def _get_logits(self, encoded, decoder_state, mask):
        transform_decoded = self.dln(self.W2(decoder_state))
        logits = torch.bmm(encoded, transform_decoded.unsqueeze(2)).squeeze(2)
        return logits


class PointerDotNoLN(PointerLayer):
    """PointerDotNoLN Module (variant of the paper, using a simple dot product)"""

    def initialize(self, **kwargs):
        """prepare modules for dot product pointer layer"""
        self.W1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=kwargs.get("bias", False))
        self.W2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=kwargs.get("bias", False))

        # still need to normalize to prevent gradient blowups -- but without additional affine!
        self.eln = nn.LayerNorm(self.embedding_dim, bias=False, elementwise_affine=False)
        self.dln = nn.LayerNorm(self.embedding_dim, bias=False, elementwise_affine=False)

    def process_encoded(self, encoded):
        return StoredEncoding(self.eln(self.W1(encoded)))

    def _get_logits(self, encoded, decoder_state, mask):
        transform_decoded = self.dln(self.W2(decoder_state))
        logits = torch.bmm(encoded, transform_decoded.unsqueeze(2)).squeeze(2)
        return logits


class PointerDotLean(PointerLayer):
    """PointerDotLean Module (variant of the paper, using a simple dot product and even less weights)"""

    def initialize(self, **kwargs):
        """prepare modules for dot product pointer layer"""
        self.eln = nn.LayerNorm(self.embedding_dim, bias=False)
        self.dln = nn.LayerNorm(self.embedding_dim, bias=False)

    def process_encoded(self, encoded):
        return StoredEncoding(self.eln(encoded))

    def _get_logits(self, encoded, decoder_state, mask):
        transform_decoded = self.dln(decoder_state)
        logits = torch.bmm(encoded, transform_decoded.unsqueeze(2)).squeeze(2)
        return logits


class PointerAttention(PointerLayer):
    """PointerAttention Module (variant of paper, using standard attention layer)"""

    def initialize(self, **kwargs):
        """prepare modules for attention pointer layer"""
        _check_kwargs("PointerAttention", kwargs, ["num_heads"])
        kwargs["contextual"] = False
        kwargs["multimodal"] = True
        kwargs["num_multimodal"] = 1
        self.attention_kwargs = kwargs
        num_heads = kwargs.pop("num_heads")
        _ = kwargs.pop("bias", None)  # used for other pointer layers, not PointerAttention
        _ = kwargs.pop("expansion", None)  # used for other pointer layers, not PointerAttention
        _ = kwargs.pop("mlp_bias", None)  # used for other pointer layers, not PointerAttention
        self.attention = get_attention_layer(self.embedding_dim, num_heads, **kwargs)
        self.vt = nn.Linear(self.embedding_dim, 1, bias=False)

    def process_encoded(self, encoded):
        return StoredEncoding(encoded)

    def _get_logits(self, encoded, decoder_state, mask):
        # attention on encoded representations with decoder_state
        attended = self.attention(encoded, [decoder_state], mask=mask)
        logits = self.vt(torch.tanh(attended)).squeeze(2)
        return logits


class PointerTransformer(PointerLayer):
    """PointerTransformer Module (variant of paper using a transformer)"""

    def initialize(self, **kwargs):
        """prepare modules for transformer pointer layer"""
        _check_kwargs("PointerTransformer", kwargs, ["num_heads"])
        kwargs["contextual"] = False
        kwargs["multimodal"] = True
        kwargs["num_multimodal"] = 1
        self.transformer_kwargs = kwargs
        num_heads = kwargs.pop("num_heads")
        _ = kwargs.pop("bias", None)  # used for other pointer layers, not PointerTransformer
        _ = kwargs.pop("residual", None)  # used for other pointer layers, not PointerTransformer
        self.transformer = get_transformer_layer(self.embedding_dim, num_heads, **kwargs)
        self.vt = nn.Linear(self.embedding_dim, 1, bias=False)

    def process_encoded(self, encoded):
        return StoredEncoding(encoded)

    def _get_logits(self, encoded, decoder_state, mask):
        # transform encoded representations with decoder_state
        transformed = self.transformer(encoded, [decoder_state], mask=mask)
        logits = self.vt(torch.tanh(transformed)).squeeze(2)
        return logits


POINTER_REGISTRY = {
    "standard": PointerStandard,
    "dot": PointerDot,
    "dot_noln": PointerDotNoLN,
    "dot_lean": PointerDotLean,
    "attention": PointerAttention,
    "transformer": PointerTransformer,
}
