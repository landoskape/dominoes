from ..utils import argbool


def add_standard_training_parameters(parser):
    """
    arguments for defining the network type, dataset, optimizer, and other metaparameters
    """
    parser.add_argument("--task", type=str, required=True, help="which task to use (the dataset to load), required")
    parser.add_argument("--learning_mode", type=str, default="reinforce", help="which learning mode to use (default=reinforce)")
    parser.add_argument("--optimizer", type=str, default="Adam", help="what optimizer to train with (default=Adam)")
    parser.add_argument("--batch_size", type=int, default=128, help="what batch size to pass to DataLoader")
    parser.add_argument("--train_epochs", type=int, default=2000, help="how many epochs to train the networks on")
    parser.add_argument("--test_epochs", type=int, default=100, help="how many epochs to train the networks on")
    parser.add_argument("--replicates", type=int, default=2, help="how many replicates of each network to train")
    parser.add_argument("--silent", type=argbool, default=False, help="whether or not to print training progress (default=False)")
    parser.add_argument("--save_loss", type=argbool, default=False, help="whether to save loss during training (default=False)")
    parser.add_argument("--save_reward", type=argbool, default=False, help="whether to save reward during training (default=False)")
    return parser


def add_network_training_metaparameters(parser):
    """
    arguments for determining default network & training metaparameters
    """
    parser.add_argument("--lr", type=float, default=1e-3)  # default learning rate
    parser.add_argument("--wd", type=float, default=0)  # default weight decay
    parser.add_argument("--gamma", type=float, default=1.0)  # default gamma for reward processing
    parser.add_argument("--train_temperature", type=float, default=5.0, help="temperature for training")
    parser.add_argument("--thompson", type=argbool, default=True, help="whether to use Thompson sampling during training (default=True)")
    parser.add_argument("--baseline", type=argbool, default=True, help="whether to use a baseline correction during training (default=True)")
    parser.add_argument("--bl_temperature", type=float, default=1.0, help="temperature for baseline networks during training")
    parser.add_argument("--bl_thompson", type=argbool, default=False, help="whether to use Thompson sampling for baseline networks (default=False)")
    parser.add_argument("--bl_significance", type=float, default=0.05, help="significance level for updating baseline networks (default=0.05)")
    parser.add_argument("--bl_batch_size", type=int, default=1024, help="batch size for baseline networks (default=1024)")
    parser.add_argument("--bl_duty_cycle", type=int, default=10, help="how many epochs to wait before checking baseline improvement (default=10)")
    return parser


def add_pointernet_parameters(parser):
    """arguments for the PointerNet"""
    parser.add_argument("--embedding_dim", type=int, default=128, help="the dimensions of the embedding (default=128)")
    parser.add_argument("--embedding_bias", type=argbool, default=True, help="whether to use embedding_bias (default=True)")
    parser.add_argument("--num_encoding_layers", type=int, default=1, help="the number of encoding layers in the PointerNet (default=1)")
    parser.add_argument("--encoder_method", type=str, default="transformer", help="PointerNet encoding layer method (default='transformer')")
    parser.add_argument("--decoder_method", type=str, default="transformer", help="PointerNet decoding layer method (default='transformer')")
    parser.add_argument("--pointer_method", type=str, default="standard", help="PointerNet pointer layer method (default='standard')")
    return parser


def add_pointernet_encoder_parameters(parser):
    """arguments for the encoder layers in a PointerNet"""
    parser.add_argument("--encoder_num_heads", type=int, default=8, help="the number of heads in ptrnet encoding layers (default=8)")
    parser.add_argument("--encoder_kqnorm", type=argbool, default=True, help="whether to use kqnorm in the encoder (default=True)")
    parser.add_argument("--encoder_expansion", type=int, default=1, help="the expansion of the FF layers in the encoder (default=1)")
    parser.add_argument("--encoder_kqv_bias", type=argbool, default=False, help="whether to use bias in the attention kqv layers (default=False)")
    parser.add_argument("--encoder_mlp_bias", type=argbool, default=True, help="use bias in the MLP part of transformer encoders (default=True)")
    parser.add_argument("--encoder_residual", type=argbool, default=True, help="use residual connections in the attentional encoders (default=True)")
    return parser


def add_pointernet_decoder_parameters(parser):
    """arguments for the decoder layers in a PointerNet"""
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="the number of heads in ptrnet decoding layers (default=8)")
    parser.add_argument("--decoder_kqnorm", type=argbool, default=True, help="whether to use kqnorm in the decoder (default=True)")
    parser.add_argument("--decoder_expansion", type=int, default=1, help="the expansion of the FF layers in the decoder (default=1)")
    parser.add_argument("--decoder_gru_bias", type=argbool, default=True, help="whether to use bias in the gru decoder method (default=True)")
    parser.add_argument("--decoder_kqv_bias", type=argbool, default=False, help="whether to use bias in the attention layer (default=True)")
    parser.add_argument("--decoder_mlp_bias", type=argbool, default=True, help="use bias in the MLP part of transformer decoders (default=True)")
    parser.add_argument("--decoder_residual", type=argbool, default=True, help="use residual connections in the attentional decoders (default=True)")
    return parser


def add_pointernet_pointer_parameters(parser):
    """arguments for the pointer layer in a PointerNet"""
    parser.add_argument("--pointer_num_heads", type=int, default=8, help="the number of heads in ptrnet decoding layers (default=8)")
    parser.add_argument("--pointer_kqnorm", type=argbool, default=True, help="whether to use kqnorm in the decoder (default=True)")
    parser.add_argument("--pointer_expansion", type=int, default=1, help="the expansion of the FF layers in the decoder (default=1)")
    parser.add_argument("--pointer_bias", type=argbool, default=False, help="whether to use bias in pointer projection layers (default=False)")
    parser.add_argument("--pointer_kqv_bias", type=argbool, default=False, help="use bias in the attention layer of pointers (default=True)")
    parser.add_argument("--pointer_mlp_bias", type=argbool, default=True, help="use bias in the MLP part of transformer pointers (default=True)")
    parser.add_argument("--pointer_residual", type=argbool, default=True, help="use residual connections in the attentional pointer (default=True)")
    return parser


def add_checkpointing(parser):
    """
    arguments for managing checkpointing when training networks

    TODO: probably add some arguments for controlling the details of the checkpointing
        : e.g. how often to checkpoint, etc.
    """
    parser.add_argument("--use_prev", default=False, action="store_true", help="if used, will pick up training off previous checkpoint")
    parser.add_argument("--save_ckpts", default=False, action="store_true", help="if used, will save checkpoints of models")
    parser.add_argument("--ckpt_frequency", default=1, type=int, help="frequency (by epoch) to save checkpoints of models")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="if used, will log experiment to WandB")

    return parser


def add_dataset_parameters(parser):
    """add generic dataset parameters"""
    parser.add_argument("--threads", type=int, default=1, help="the number of threads to use for generating batches (default=1)")
    parser.add_argument("--ignore_index", type=int, default=-100, help="the index to ignore in the loss function (default=-100)")
    return parser


def add_tsp_parameters(parser):
    """
    arguments for the traveling salesman problem
    """
    parser.add_argument("--num_cities", type=int, default=10, help="the number of cities in the TSP (default=10)")
    parser.add_argument("--coord_dims", type=int, default=2, help="the number of dimensions for the coordinates (default=2)")
    return parser


def add_dominoe_parameters(parser):
    """
    arguments for any dominoe task
    """
    parser.add_argument("--highest_dominoe", type=int, default=9, help="the highest dominoe value (default=9)")
    parser.add_argument("--train_fraction", type=float, default=0.8, help="the fraction of dominoes to train with (default=0.8)")
    parser.add_argument("--hand_size", type=int, default=8, help="the number of dominoes in the hand (default=8)")
    parser.add_argument("--randomize_direction", type=argbool, default=True, help="randomize the direction of the dominoes (default=True)")
    return parser


def add_dominoe_sequencer_parameters(parser):
    """arguments for the dominoe sequencer task"""
    parser.add_argument("--value_method", type=str, default="length", help="how to calculate the value of a sequence (default=length)")
    parser.add_argument("--value_multiplier", type=float, default=1.0, help="how to scale the value of a sequence (default=1.0)")
    return parser


def add_dominoe_sorting_parameters(parser):
    """arguments for the dominoe sorting task"""
    parser.add_argument("--allow_mistakes", type=argbool, default=False, help="allow mistakes in the sorting task (default=False)")
    return parser
