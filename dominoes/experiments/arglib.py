def add_standard_training_parameters(parser):
    """
    arguments for defining the network type, dataset, optimizer, and other metaparameters
    """
    parser.add_argument("--network", type=str, default="MLP", help="what base network architecture to use")
    parser.add_argument("--dataset", type=str, default="MNIST", help="what dataset to use")
    parser.add_argument("--optimizer", type=str, default="Adam", help="what optimizer to train with")
    parser.add_argument("--batch-size", type=int, default=1024, help="what batch size to pass to DataLoader")
    parser.add_argument("--epochs", type=int, default=100, help="how many epochs to train the networks on")
    parser.add_argument("--replicates", type=int, default=5, help="how many replicates of each network to train")
    return parser


def add_network_metaparameters(parser):
    """
    arguments for determining default network & training metaparameters
    """
    parser.add_argument("--default-lr", type=float, default=1e-3)  # default learning rate
    parser.add_argument("--default-dropout", type=float, default=0)  # default dropout rate
    parser.add_argument("--default-wd", type=float, default=0)  # default weight decay
    return parser


def add_alignment_analysis_parameters(parser):
    parser.add_argument(
        "--ignore-flag",
        default=False,
        action="store_true",
        help="if used, will omit flagged layers in analyses",
    )
    parser.add_argument(
        "--no-alignment",
        default=False,
        action="store_true",
        help="if used, will not measure alignment during training and testing",
    )
    parser.add_argument(
        "--delta-weights",
        default=False,
        action="store_true",
        help="if used, will measure delta-weights during training",
    )
    parser.add_argument(
        "--frequency",
        default=1,
        type=int,
        help="how frequently to measure alignment etc. (how many minibatches)",
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
        "--ckpt-frequency",
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


def add_dropout_experiment_details(parser):
    """
    add arguments for determining how to run progressive dropout experiments
    """
    parser.add_argument(
        "--num-drops",
        type=int,
        default=9,
        help="number of dropout fractions for progressive dropout",
    )
    parser.add_argument(
        "--dropout-by-layer",
        default=False,
        action="store_true",
        help="whether to do progressive dropout by layer or across all layers",
    )
    return parser
