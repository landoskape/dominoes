from abc import ABC, abstractmethod
import torch
from torch import nn

from .attention_modules import get_attention_layer, _attention_type
from .transformer_modules import get_transformer_layer
from .pointer_decoder import PointerDecoder
from ..utils import check_args, process_arguments
from .net_utils import process_input, process_multimodal_input


def _get_pointernet_constructor(contextual, multimodal):
    """get the attention constructor based on the attention type"""
    attention_type = _attention_type(contextual, multimodal)
    if attention_type not in POINTERNET_REGISTRY:
        raise ValueError(f"Unrecognized pointernet type: {attention_type}")
    return POINTERNET_REGISTRY[attention_type]


def get_pointer_network(
    input_dim,
    embedding_dim,
    embedding_bias=True,
    contextual=False,
    multimodal=False,
    num_multimodal=0,
    mm_input_dim=None,
    num_encoding_layers=1,
    require_init=False,
    encoder_method="transformer",
    decoder_method="transformer",
    pointer_method="standard",
    encoder_kwargs={},
    decoder_kwargs={},
    pointer_kwargs={},
    thompson=False,
    temperature=1.0,
    permutation=True,
):
    """create pointernetwork with requested arguments"""
    pointernet_kwargs = dict(
        embedding_bias=embedding_bias,
        num_encoding_layers=num_encoding_layers,
        require_init=require_init,
        encoder_method=encoder_method,
        decoder_method=decoder_method,
        pointer_method=pointer_method,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        pointer_kwargs=pointer_kwargs,
        thompson=thompson,
        temperature=temperature,
        permutation=permutation,
    )
    if multimodal:
        pointernet_kwargs["num_multimodal"] = num_multimodal
        pointernet_kwargs["mm_input_dim"] = mm_input_dim
    pointernet_constructor = _get_pointernet_constructor(contextual, multimodal)
    return pointernet_constructor(input_dim, embedding_dim, **pointernet_kwargs)


def get_pointer_arguments(args):
    """get pointer network arguments from arguments"""
    return PointerArguments(vars(args).copy()).get_args()


class PointerArguments:
    """
    Supporting class for processing arguments from an ArgumentParser to construct a PointerNetwork

    The input to the PointerArguments constructor is a dictionary of arguments from an ArgumentParser,
    and the initialization method processes these arguments to get the required and optional kwargs
    for the PointerNetwork. It knows the relationship between the name of the argument as defined in
    the ArgumentParser and the name of the associated keyword argument for the relevant constructor
    (e.g. "num_heads" is used in several places, so the map is from "{encoder/decoder/pointer}_num_heads"
    to "num_heads"). There are overlapping and nonoverlapping arguments for different possible constructors,
    so the user should think carefully about how to send inputs into the network.

    Usage:
    ```python
    args = ArgumentParser().parse_args() # (this will be defined elsewhere, usually in the experiment class)
    embedding_dim, pointernet_kwargs = PointerArguments(vars(args)).get_args()
    ptrnet = PointerNetwork(*other_args, embedding_dim, **other_kwargs, **pointernet_kwargs)
    """

    def __init__(self, args):
        self.args = args
        self._get_pointernet_kwargs()
        self._get_encoder_kwargs()
        self._get_decoder_kwargs()
        self._get_pointer_kwargs()

    def get_args(self):
        """return stored arguments"""
        pointernet_kwargs = self.pointernet_kwargs
        pointernet_kwargs["encoder_kwargs"] = self.encoder_kwargs
        pointernet_kwargs["decoder_kwargs"] = self.decoder_kwargs
        pointernet_kwargs["pointer_kwargs"] = self.pointer_kwargs
        return self.embedding_dim, pointernet_kwargs

    def _get_pointernet_kwargs(self):
        """method for getting the pointer network kwargs from a dictionary of arguments"""
        required_args = ["embedding_dim"]

        # these are the possible kwargs that can be passed to the pointer network
        # key is the argument name in the ArgParser, value is the pointer network keyword
        required_kwargs = dict(
            embedding_bias="embedding_bias",
            num_encoding_layers="num_encoding_layers",
            encoder_method="encoder_method",
            decoder_method="decoder_method",
            pointer_method="pointer_method",
        )
        possible_kwargs = dict(
            train_temperature="temperature",
            thompson="thompson",
        )
        # get arguments for pointer network
        (embedding_dim,), pointernet_kwargs = process_arguments(
            self.args,
            required_args,
            required_kwargs,
            possible_kwargs,
            name="PointerNetwork-MainNetwork",
        )

        # store arguments in self
        self.embedding_dim = embedding_dim
        self.pointernet_kwargs = pointernet_kwargs

    def _get_encoder_kwargs(self):
        """method for getting the encoder kwargs from a dictionary of arguments"""
        required_args = []

        # these are the possible kwargs that can be passed to the encoder
        # key is the argument name in the ArgParser, value is the encoder keyword
        required_kwargs = dict(
            encoder_num_heads="num_heads",
        )
        possible_kwargs = dict(
            encoder_expansion="expansion",
            encoder_kqnorm="kqnorm",
            encoder_kqv_bias="kqv_bias",
            encoder_mlp_bias="mlp_bias",
            encoder_residual="residual",
        )
        # get arguments for encoder
        _, encoder_kwargs = process_arguments(
            self.args,
            required_args,
            required_kwargs,
            possible_kwargs,
            name="PointerNetwork-Encoder",
        )

        # store arguments in self
        self.encoder_kwargs = encoder_kwargs

    def _get_decoder_kwargs(self):
        """method for getting the decoder kwargs from a dictionary of arguments"""
        required_args = []

        # these are the possible kwargs that can be passed to the decoder
        # key is the argument name in the ArgParser, value is the decoder keyword
        required_kwargs = {}
        possible_kwargs = dict(
            decoder_num_heads="num_heads",
            decoder_expansion="expansion",
            decoder_gru_bias="gru_bias",
            decoder_kqnorm="kqnorm",
            decoder_kqv_bias="kqv_bias",
            decoder_mlp_bias="mlp_bias",
            decoder_residual="residual",
        )
        # get arguments for decoder
        _, decoder_kwargs = process_arguments(
            self.args,
            required_args,
            required_kwargs,
            possible_kwargs,
            name="PointerNetwork-Decoder",
        )

        # store arguments in self
        self.decoder_kwargs = decoder_kwargs

    def _get_pointer_kwargs(self):
        """method for getting the pointer kwargs from a dictionary of arguments"""
        required_args = []

        # these are the possible kwargs that can be passed to the pointer
        # key is the argument name in the ArgParser, value is the pointer keyword
        required_kwargs = {}
        possible_kwargs = dict(
            pointer_bias="bias",
            pointer_num_heads="num_heads",
            pointer_expansion="expansion",
            pointer_kqnorm="kqnorm",
            pointer_kqv_bias="kqv_bias",
            pointer_mlp_bias="mlp_bias",
            pointer_residual="residual",
        )

        # get arguments for pointer
        _, pointer_kwargs = process_arguments(
            self.args,
            required_args,
            required_kwargs,
            possible_kwargs,
            name="PointerNetwork-Pointer",
        )

        # store arguments in self
        self.pointer_kwargs = pointer_kwargs


class PointerNetworkBaseClass(nn.Module, ABC):
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
        embedding_bias=True,
        contextual=False,
        multimodal=False,
        num_multimodal=0,
        mm_input_dim=None,
        num_encoding_layers=1,
        require_init=False,
        encoder_method="transformer",
        decoder_method="transformer",
        pointer_method="standard",
        encoder_kwargs={},
        decoder_kwargs={},
        pointer_kwargs={},
        thompson=False,
        temperature=1.0,
        permutation=True,
    ):
        super().__init__()

        # network metaparameters
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.contextual = contextual
        self.multimodal = multimodal
        self.num_multimodal = num_multimodal * multimodal
        self.mm_input_dim = self._check_multimodal_input_dim(mm_input_dim, self.num_multimodal) if multimodal else None
        self.num_encoding_layers = num_encoding_layers
        self.require_init = require_init
        self.encoder_method = encoder_method
        self.decoder_method = decoder_method
        self.pointer_method = pointer_method
        self.encoder_kwargs = encoder_kwargs
        self.decoder_kwargs = decoder_kwargs
        self.pointer_kwargs = pointer_kwargs
        self.thompson = thompson
        self.temperature = temperature
        self.permutation = permutation

        # create embedding of input to embedding dimension
        self.embedding = nn.Linear(in_features=input_dim, out_features=self.embedding_dim, bias=self.embedding_bias)

        # create embedding for multimodal inputs to embedding dimension
        if self.multimodal:
            self.mm_embedding = nn.ModuleList(
                [nn.Linear(in_features=mid, out_features=self.embedding_dim, bias=self.embedding_bias) for mid in self.mm_input_dim]
            )

        # build encoder layers
        self._build_encoder(num_encoding_layers, encoder_method, encoder_kwargs)

        # build decoder module
        self.decoder = PointerDecoder(
            self.embedding_dim,
            decoder_method=decoder_method,
            pointer_method=pointer_method,
            decoder_kwargs=decoder_kwargs,
            pointer_kwargs=pointer_kwargs,
            permutation=permutation,
        )

        # if require_init is False, then we need to learn the initial input to the decoder
        if not self.require_init:
            self.decoder_input = nn.Parameter(torch.randn(1, self.embedding_dim))  # Learnable tensor

    def set_temperature(self, temperature):
        """method for setting the temperature of the pointer network"""
        self.temperature = temperature

    def set_thompson(self, thompson):
        """method for setting the thompson sampling flag of the pointer network"""
        self.thompson = thompson

    def _check_multimodal_input_dim(self, mm_input_dim, num_multimodal):
        """helper for setting input dim of multimodal inputs"""
        assert type(mm_input_dim) == tuple or type(mm_input_dim) == list, "mm_input_dim must be a tuple or list"
        assert len(mm_input_dim) == num_multimodal, f"mm_input_dim must have {num_multimodal} elements"
        assert all([type(mid) == int for mid in mm_input_dim]), "all elements of mm_input_dim must be integers"
        return mm_input_dim

    def _build_encoder(self, num_encoding_layers, encoder_method, encoder_kwargs):
        """flexible method for creating encoding layers for pointer network"""
        # build encoder
        required_kwargs = [
            "num_heads",
            "kqnorm",
        ]
        if encoder_method == "attention":
            required_kwargs += [
                "kqv_bias",
                "residual",
            ]
            check_args(encoder_method, encoder_kwargs, required_kwargs)
            self.encoding_layers = nn.ModuleList(
                [
                    get_attention_layer(
                        self.embedding_dim,
                        encoder_kwargs["num_heads"],
                        encoder_kwargs["kqnorm"],
                        contextual=self.contextual,
                        multimodal=self.multimodal,
                        num_multimodal=self.multimodal,
                        kqv_bias=encoder_kwargs["kqv_bias"],
                        residual=encoder_kwargs["residual"],
                    )
                    for _ in range(num_encoding_layers)
                ]
            )

        elif encoder_method == "transformer":
            required_kwargs += [
                "expansion",
                "kqv_bias",
                "mlp_bias",
            ]
            check_args(encoder_method, encoder_kwargs, required_kwargs)
            self.encoding_layers = nn.ModuleList(
                [
                    get_transformer_layer(
                        self.embedding_dim,
                        encoder_kwargs["num_heads"],
                        encoder_kwargs["expansion"],
                        encoder_kwargs["kqnorm"],
                        contextual=self.contextual,
                        multimodal=self.multimodal,
                        num_multimodal=self.multimodal,
                        kqv_bias=encoder_kwargs["kqv_bias"],
                        mlp_bias=encoder_kwargs["mlp_bias"],
                    )
                    for _ in range(num_encoding_layers)
                ]
            )

        else:
            raise ValueError(f"encoder_method={encoder_method} not recognized!")

    def _get_decoder_state(self, temperature, thompson):
        """method for setting the temperature and thompson sampling flag of the pointer network"""
        temperature = temperature or self.temperature
        thompson = thompson or self.thompson
        return temperature, thompson

    def _get_max_output(self, x, init, max_output=None):
        """method for getting the maximum number of outputs for the pointer network"""
        if self.require_init and init is None:
            raise ValueError("init must be provided if require_init is True!")

        # get number of tokens in input
        num_tokens = x.size(1)

        # set max_output to num_tokens if not provided (and subtract 1 if using an initial input)
        max_output = max_output or num_tokens - 1 * (init is not None)
        if self.permutation:
            msg = f"if using permutation mode, max_output ({max_output}) must be less than or equal to the number of tokens ({num_tokens})"
            if init is not None:
                if max_output > num_tokens - 1:
                    raise ValueError(msg + " minus 1 (for the initial token)")
            else:
                if max_output > num_tokens:
                    raise ValueError(msg)

        return max_output

    @abstractmethod
    def _forward_encoder(self):
        """
        each PointerNetwork class should implement this method to process the input tensor x and
        any additional inputs required by the network (e.g. context or multimodal)
        """
        raise NotImplementedError

    def _get_decoder_context(self, encoded, mask):
        """decoder context is defined as the masked mean of encoded tokens"""
        # Get the masked mean of any encoded tokens
        numerator = torch.sum(encoded * mask.unsqueeze(2), dim=1)
        denominator = torch.sum(mask, dim=1, keepdim=True)
        decoder_context = numerator / denominator
        return decoder_context

    def _get_decoder_input(self, encoded, batch_size, init, mask):
        """decoder input is either selected by the init argument or learned as a parameter"""
        # Initialize decoder input (either with initial embedding or with learned vector)
        if self.require_init or init is not None:
            # get representation of init token
            rep_init = init.view(batch_size, 1, 1).expand(-1, -1, self.embedding_dim)
            decoder_input = torch.gather(encoded, 1, rep_init).squeeze(1)
            # mask out the initial token if using a permutation
            if self.permutation:
                mask = mask.scatter(1, init.view(batch_size, 1), torch.zeros((batch_size, 1), dtype=mask.dtype).to(mask.device))
        else:
            # use learned tensor as initial input
            decoder_input = self.decoder_input.expand(batch_size, -1).to(encoded.device)

        return decoder_input, mask

    def _forward_decoder(self, encoded, mask, max_output, batch_size, init, temperature, thompson):
        """central method for creating decoder context and input and running the decoder loop"""
        # Establish the decoder context and input(=initial state)
        decoder_context = self._get_decoder_context(encoded, mask)
        decoder_input, mask = self._get_decoder_input(encoded, batch_size, init, mask)

        # Then do pointer network (sequential decode-choice for N=max_output rounds)
        log_scores, choices = self.decoder(
            encoded,
            decoder_input,
            decoder_context,
            max_output,
            mask=mask,
            temperature=temperature,
            thompson=thompson,
        )

        return log_scores, choices

    @abstractmethod
    def forward(self):
        """
        forward pass of the pointer network (required in children)
        this contains a skeleton of the required code in comments
        """
        # get decoder state (use user provided temperature and thompson sampling or use default if not provided)
        # process inputs to make sure they have the right sizes, create masks if not provided (use helpers for this!)
        # embed input tokens
        # run input through the encoder layers
        # run encoded input through the decoder layers
        raise NotImplementedError


class PointerNetwork(PointerNetworkBaseClass):
    contextual = False
    multimodal = False

    def __init__(self, *args, **kwargs):
        PointerNetworkBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def _forward_encoder(self, x, mask=None):
        """go through encoding layers with potentially masked input"""
        for layer in self.encoding_layers:
            x = layer(x, mask=mask)
        return x

    def forward(self, x, mask=None, init=None, temperature=None, thompson=None, max_output=None):
        """
        forward method of PointerNetwork without contextual or multimodal inputs

        x is input, a 3-D tensor with shape (batch_size, max_tokens, input_dim)
        mask is a binary mask with shape (batch_size, max_tokens) where 1 indicates a valid token and 0 indicates padding
        init is the initial token to start the decoding process
        - required if require_init is True, otherwise optional
        - if not provided, the decoder will start with a learned parameter vector

        temperature is the temperature of the softmax function (default is self.temperature)
        thompson is a boolean flag for thompson sampling (default is self.thompson)
        """
        # get decoder state
        temperature, thompson = self._get_decoder_state(temperature, thompson)

        # process main input
        batch_size, mask = process_input(x, mask, self.input_dim)

        # get max output
        max_output = self._get_max_output(x, init, max_output=max_output)

        # embed main input
        x = self.embedding(x)

        # this  will go in each forward pass method of PointerNetwork children
        encoded = self._forward_encoder(x, mask=mask)

        # return the output of the decoder
        log_scores, choices = self._forward_decoder(encoded, mask, max_output, batch_size, init, temperature, thompson)
        return log_scores, choices


class ContextualPointerNetwork(PointerNetworkBaseClass):
    contextual = True
    multimodal = False

    def __init__(self, *args, **kwargs):
        PointerNetworkBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def _forward_encoder(self, x, context, mask=None, context_mask=None):
        """
        go through encoding layers with potentially masked input and context inputs

        NOTE / TODO: context inputs are not updated, which may limit the representational capacity of the
        encoding layers, since the key/value transforms have to continue representing the embedded context inputs
        in the same space across each layer...
        """
        for layer in self.encoding_layers:
            x = layer(x, context, mask=mask, context_mask=context_mask)
        return x

    def forward(self, x, context, mask=None, context_mask=None, init=None, temperature=None, thompson=None, max_output=None):
        """
        forward method of PointerNetwork with contextual inputs

        x is input, a 3-D tensor with shape (batch_size, max_tokens, input_dim)
        mask is a binary mask with shape (batch_size, max_tokens) where 1 indicates a valid token and 0 indicates padding
        context is a 3-D tensor with shape (batch_size, num_context_tokens, input_dim)
        context_mask is a binary mask with shape (batch_size, num_context_tokens) where 1 indicates a valid token and 0 indicates padding
        init is the initial token to start the decoding process
        - required if require_init is True, otherwise optional
        - if not provided, the decoder will start with a learned parameter vector

        temperature is the temperature of the softmax function (default is self.temperature)
        thompson is a boolean flag for thompson sampling (default is self.thompson)
        """
        # get decoder state
        temperature, thompson = self._get_decoder_state(temperature, thompson)

        # process main input
        batch_size, mask = process_input(x, mask, self.input_dim)
        context_batch_size, context_mask = process_input(context, context_mask, self.input_dim, name="context")

        # check for consistency in batch sizes
        assert batch_size == context_batch_size, "batch sizes of x and context must match"

        # get max output
        max_output = self._get_max_output(x, init, max_output=max_output)

        # embed inputs
        x = self.embedding(x)
        context = self.embedding(context)

        # this  will go in each forward pass method of PointerNetwork children
        encoded = self._forward_encoder(x, context, mask=mask, context_mask=context_mask)

        # return the output of the decoder
        log_scores, choices = self._forward_decoder(encoded, mask, max_output, batch_size, init, temperature, thompson)
        return log_scores, choices


class MultimodalPointerNetwork(PointerNetworkBaseClass):
    contextual = False
    multimodal = True

    def __init__(self, *args, **kwargs):
        PointerNetworkBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def _forward_encoder(self, x, multimode, mask=None, mm_mask=None):
        """
        go through encoding layers with potentially masked input and multimodal inputs
        multimode inputs are not transformed, they are just inserted later in the network (for multilayer encoding)
        """
        for layer in self.encoding_layers:
            x = layer(x, multimode, mask=mask, mm_mask=mm_mask)
        return x

    def forward(self, x, multimode, mask=None, mm_mask=None, init=None, temperature=None, thompson=None, max_output=None):
        """
        forward method of PointerNetwork with multimodal inputs

        x is input, a 3-D tensor with shape (batch_size, max_tokens, input_dim)
        mask is a binary mask with shape (batch_size, max_tokens) where 1 indicates a valid token and 0 indicates padding
        multimode is a list/tuple of 3-D tensors each with shape (batch_size, num_multimodal_tokens[i], input_dim)
        mm_mask is a list/tuple of binary masks with shape (batch_size, num_multimodal_tokens[i]) where 1 indicates a valid token and 0 indicates padding
        - mm_mask can just be None if no mask is provided

        init is the initial token to start the decoding process
        - required if require_init is True, otherwise optional
        - if not provided, the decoder will start with a learned parameter vector

        temperature is the temperature of the softmax function (default is self.temperature)
        thompson is a boolean flag for thompson sampling (default is self.thompson)
        """
        # get decoder state
        temperature, thompson = self._get_decoder_state(temperature, thompson)

        # process main input
        batch_size, mask = process_input(x, mask, self.input_dim)
        mm_batch_size, mm_mask = process_multimodal_input(multimode, mm_mask, self.num_multimodal, self.mm_input_dim)

        # check for consistency in batch sizes
        assert batch_size == mm_batch_size, "batch sizes of x multimodal inputs must match"

        # get max output
        max_output = self._get_max_output(x, init, max_output=max_output)

        # embed inputs
        x = self.embedding(x)
        multimode = [mm_embedding(mmx) for mm_embedding, mmx in zip(self.mm_embedding, multimode)]

        # this  will go in each forward pass method of PointerNetwork children
        encoded = self._forward_encoder(x, multimode, mask=mask, mm_mask=mm_mask)

        # return the output of the decoder
        log_scores, choices = self._forward_decoder(encoded, mask, max_output, batch_size, init, temperature, thompson)
        return log_scores, choices


class MultimodalContextualPointerNetwork(PointerNetworkBaseClass):
    contextual = True
    multimodal = True

    def __init__(self, *args, **kwargs):
        PointerNetworkBaseClass.__init__(self, *args, contextual=self.contextual, multimodal=self.multimodal, **kwargs)

    def _forward_encoder(self, x, context, multimode, mask=None, context_mask=None, mm_mask=None):
        """
        go through encoding layers with potentially masked input, context inputs, and multimodal inputs

        NOTE / TODO: context inputs are not updated, which may limit the representational capacity of the
        encoding layers, since the key/value transforms have to continue representing the embedded context inputs
        in the same space across each layer...

        multimode inputs are not transformed, they are just inserted later in the network (for multilayer encoding)
        """
        for layer in self.encoding_layers:
            x = layer(x, context, multimode, mask=mask, context_mask=context_mask, mm_mask=mm_mask)
        return x

    def forward(self, x, context, multimode, mask=None, context_mask=None, mm_mask=None, init=None, temperature=None, thompson=None, max_output=None):
        """
        forward method of PointerNetwork with contextual and multimodal inputs

        x is input, a 3-D tensor with shape (batch_size, max_tokens, input_dim)
        mask is a binary mask with shape (batch_size, max_tokens) where 1 indicates a valid token and 0 indicates padding
        context is a 3-D tensor with shape (batch_size, num_context_tokens, input_dim)
        context_mask is a binary mask with shape (batch_size, num_context_tokens) where 1 indicates a valid token and 0 indicates padding
        multimode is a list/tuple of 3-D tensors each with shape (batch_size, num_multimodal_tokens[i], input_dim)
        mm_mask is a list/tuple of binary masks with shape (batch_size, num_multimodal_tokens[i]) where 1 indicates a valid token and 0 indicates padding
        - mm_mask can just be None if no mask is provided

        init is the initial token to start the decoding process
        - required if require_init is True, otherwise optional
        - if not provided, the decoder will start with a learned parameter vector

        temperature is the temperature of the softmax function (default is self.temperature)
        thompson is a boolean flag for thompson sampling (default is self.thompson)
        """
        # get decoder state
        temperature, thompson = self._get_decoder_state(temperature, thompson)

        # process main input
        batch_size, mask = process_input(x, mask, self.input_dim)
        context_batch_size, context_mask = process_input(context, context_mask, self.input_dim, name="context")
        mm_batch_size, mm_mask = process_multimodal_input(multimode, mm_mask, self.num_multimodal, self.mm_input_dim)

        # check for consistency in batch sizes
        assert batch_size == context_batch_size, "batch sizes of x and context must match"
        assert batch_size == mm_batch_size, "batch sizes of x multimodal inputs must match"

        # get max output
        max_output = self._get_max_output(x, init, max_output=max_output)

        # embed inputs
        x = self.embedding(x)
        context = self.embedding(context)
        multimode = [mm_embedding(mmx) for mm_embedding, mmx in zip(self.mm_embedding, multimode)]

        # this  will go in each forward pass method of PointerNetwork children
        encoded = self._forward_encoder(x, multimode, mask=mask, mm_mask=mm_mask)

        # return the output of the decoder
        log_scores, choices = self._forward_decoder(encoded, mask, max_output, batch_size, init, temperature, thompson)
        return log_scores, choices


POINTERNET_REGISTRY = {
    "A": PointerNetwork,
    "CA": ContextualPointerNetwork,
    "MA": MultimodalPointerNetwork,
    "MCA": MultimodalContextualPointerNetwork,
}
