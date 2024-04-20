from argparse import ArgumentParser
from .ptr_arch_comp import PointerArchitectureComparison

EXPERIMENT_REGISTRY = {
    "ptr_arch_comp": PointerArchitectureComparison,
}


def _check_experiment(experiment_name):
    """
    check if experiment is in the registry
    """
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment ({experiment_name}) is not in EXPERIMENT_REGISTRY")


def get_experiment(experiment_name, build=False, **kwargs):
    """
    lookup experiment constructor from experiment registry by name

    if build=True, builds experiment and returns an experiment object using any kwargs
    otherwise just returns the class constructor
    """
    _check_experiment(experiment_name)
    experiment = EXPERIMENT_REGISTRY[experiment_name]
    if build:
        return experiment(**kwargs)
    return experiment


def create_experiment():
    """
    method to create experiment using initial argument parser

    the argument parser looks for a known argument called "--experiment", and the resulting
    string is used to retrieve an experiment constructor from the EXPERIMENT_REGISTRY

    any remaining arguments (args) are passed to the experiment constructor which has it's
    own argument parser in the class definition (but doesn't define the --experiment argument
    which is why the remaining args need to be passed to it directly)

    note:
    add_help=False so adding a --help argument will show a help message for the specific experiment
    that is requested rather than showing a help message for this little parser then blocking the
    rest of the execution. It means using --help requires a valid 'experiment' positional argument.
    """
    parser = ArgumentParser(description=f"ArgumentParser for loading experiment constructor", add_help=False)
    parser.add_argument("experiment", type=str, help="a string that defines which experiment to run")
    exp_args, args = parser.parse_known_args()
    return get_experiment(exp_args.experiment, build=True, args=args)
