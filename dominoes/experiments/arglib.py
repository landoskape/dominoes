def add_standard_training_parameters(parser):
    """
    arguments for defining the network type, dataset, optimizer, and other metaparameters
    """
    parser.add_argument("--task", type=str, required=True, help="which task to use (the dataset to load), required")
    parser.add_argument("--optimizer", type=str, default="Adam", help="what optimizer to train with (default=Adam)")
    parser.add_argument("--batch_size", type=int, default=128, help="what batch size to pass to DataLoader")
    parser.add_argument("--train_epochs", type=int, default=2000, help="how many epochs to train the networks on")
    parser.add_argument("--test_epochs", type=int, default=100, help="how many epochs to train the networks on")
    parser.add_argument("--replicates", type=int, default=2, help="how many replicates of each network to train")
    return parser


def add_transformer_parameters(parser):
    """arguments for transformer layers"""
    parser.add_argument("--embedding_dim", type=int, default=48, help="the dimensions of the embedding (default=48)")
    parser.add_argument("--heads", type=int, default=1, help="the number of heads in transformer layers (default=1)")
    parser.add_argument("--encoding_layers", type=int, default=1, help="the number of stacked transformers in the encoder (default=1)")
    parser.add_argument("--expansion", type=int, default=4, help="the expansion of the FF layer in the transformer of the encoder (default=4)")
    return parser


def add_network_metaparameters(parser):
    """
    arguments for determining default network & training metaparameters
    """
    parser.add_argument("--lr", type=float, default=1e-3)  # default learning rate
    parser.add_argument("--wd", type=float, default=0)  # default weight decay
    parser.add_argument("--gamma", type=float, default=1.0)  # default gamma for reward processing
    parser.add_argument("--train_temperature", type=float, default=5.0, help="temperature for training")
    parser.add_argument(
        "--no-thompson", default=False, action="store_true", help="if used, will not use Thompson sampling during training (default=False)"
    )
    parser.add_argument("--no_kqnorm", default=False, action="store_true", help="if used, will not use kqnorm during training (default=False)")
    parser.add_argument("--decoder_method", type=str, default="transformer", help="the method to use for decoding (default=transformer)")
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
    parser.add_argument("--threads", type=int, default=1, help="the number of threads to use for generating batches (default=1)")
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
        "--allow_mistakes", default=False, action="store_true", help="if used, will allow mistakes in the sorting task (default=False)"
    )
    return parser
