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
    parser.add_argument("--silent", default=False, action="store_true", help="if used, won't print training progress")
    parser.add_argument(
        "--save_loss",
        default=False,
        action="store_true",
        help="if used, will save loss during training (always True for learning_mode=supervised)",
    )
    parser.add_argument(
        "--save_reward",
        default=False,
        action="store_true",
        help="if used, will save reward during training (always True for learning_mode=reinforce)",
    )
    return parser


def add_network_training_metaparameters(parser):
    """
    arguments for determining default network & training metaparameters
    """
    parser.add_argument("--lr", type=float, default=1e-3)  # default learning rate
    parser.add_argument("--wd", type=float, default=0)  # default weight decay
    parser.add_argument("--gamma", type=float, default=1.0)  # default gamma for reward processing
    parser.add_argument("--train_temperature", type=float, default=5.0, help="temperature for training")
    parser.add_argument(
        "--no_thompson",
        default=False,
        action="store_true",
        help="if used, do greedy instead of Thompson sampling during training (default=False)",
    )
    return parser


def add_pointernet_parameters(parser):
    """arguments for the PointerNet"""
    parser.add_argument("--embedding_dim", type=int, default=128, help="the dimensions of the embedding (default=128)")
    parser.add_argument("--embedding_bias", default=False, action="store_true", help="whether to use embedding_bias (default=False)")
    parser.add_argument("--num_encoding_layers", type=int, default=1, help="the number of encoding layers in the PointerNet (default=1)")
    parser.add_argument("--encoder_method", type=str, default="transformer", help="PointerNet encoding layer method (default='transformer')")
    parser.add_argument("--decoder_method", type=str, default="transformer", help="PointerNet decoding layer method (default='transformer')")
    parser.add_argument("--pointer_method", type=str, default="standard", help="PointerNet pointer layer method (default='standard')")
    return parser


def add_pointernet_encoder_parameters(parser):
    """arguments for the encoder layers in a PointerNet"""
    parser.add_argument("--encoder_num_heads", type=int, default=1, help="the number of heads in ptrnet encoding layers (default=1)")
    parser.add_argument("--encoder_no_kqnorm", default=False, action="store_true", help="if used, won't use kqnorm in the encoder (default=False)")
    parser.add_argument("--encoder_expansion", type=int, default=4, help="the expansion of the FF layers in the encoder (default=4)")
    parser.add_argument(
        "--encoder_no_kqv_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the attention layers (default=False)",
    )
    parser.add_argument(
        "--encoder_no_mlp_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the MLP part of transformer encoders (default=False)",
    )
    parser.add_argument(
        "--encoder_no_residual",
        default=False,
        action="store_true",
        help="if used, won't use residual connections in the encoder (default=False)",
    )
    return parser


def add_pointernet_decoder_parameters(parser):
    """arguments for the decoder layers in a PointerNet"""
    parser.add_argument("--decoder_num_heads", type=int, default=1, help="the number of heads in ptrnet decoding layers (default=1)")
    parser.add_argument("--decoder_no_kqnorm", default=False, action="store_true", help="if used, won't use kqnorm in the decoder (default=False)")
    parser.add_argument("--decoder_expansion", type=int, default=4, help="the expansion of the FF layers in the decoder (default=4)")
    parser.add_argument(
        "--decoder_no_gru_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the gru decoder method (default=False)",
    )
    parser.add_argument(
        "--decoder_no_kqv_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the attention layer (default=False)",
    )
    parser.add_argument(
        "--decoder_no_mlp_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the MLP part of transformer decoders (default=False)",
    )
    parser.add_argument(
        "--decoder_no_residual",
        default=False,
        action="store_true",
        help="if used, won't use residual connections in the decoder (default=False)",
    )
    return parser


def add_pointernet_pointer_parameters(parser):
    """arguments for the pointer layer in a PointerNet"""
    parser.add_argument("--pointer_num_heads", type=int, default=1, help="the number of heads in ptrnet decoding layers (default=1)")
    parser.add_argument("--pointer_no_kqnorm", default=False, action="store_true", help="if used, won't use kqnorm in the decoder (default=False)")
    parser.add_argument("--pointer_expansion", type=int, default=4, help="the expansion of the FF layers in the decoder (default=4)")
    parser.add_argument(
        "--pointer_bias",
        default=False,
        action="store_true",
        help="if used, will use bias in pointer projection layers (default=False)",
    )
    parser.add_argument(
        "--pointer_no_kqv_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the attention layer of pointers (default=False)",
    )
    parser.add_argument(
        "--pointer_no_mlp_bias",
        default=False,
        action="store_true",
        help="if used, won't use bias in the MLP part of transformer pointers (default=False)",
    )
    parser.add_argument(
        "--pointer_no_residual",
        default=False,
        action="store_true",
        help="if used, won't use residual connections in the attentional pointer (default=False)",
    )
    return parser


def add_checkpointing(parser):
    """
    arguments for managing checkpointing when training networks

    TODO: probably add some arguments for controlling the details of the checkpointing
        : e.g. how often to checkpoint, etc.
    """
    parser.add_argument(
        "--use_prev",
        default=False,
        action="store_true",
        help="if used, will pick up training off previous checkpoint",
    )
    parser.add_argument(
        "--save_ckpts",
        default=False,
        action="store_true",
        help="if used, will save checkpoints of models",
    )
    parser.add_argument(
        "--ckpt_frequency",
        default=1,
        type=int,
        help="frequency (by epoch) to save checkpoints of models",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="if used, will log experiment to WandB",
    )

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
    return parser


def add_dominoe_sequencer_parameters(parser):
    """arguments for the dominoe sequencer task"""
    parser.add_argument("--value_method", type=str, default="length", help="how to calculate the value of a sequence (default=length)")
    parser.add_argument("--value_multiplier", type=float, default=1.0, help="how to scale the value of a sequence (default=1.0)")
    return parser


def add_dominoe_sorting_parameters(parser):
    """arguments for the dominoe sorting task"""
    parser.add_argument(
        "--allow_mistakes",
        default=False,
        action="store_true",
        help="if used, will allow mistakes in the sorting task (default=False)",
    )
    return parser
