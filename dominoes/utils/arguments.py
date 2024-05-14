def check_args(method_name, args, required_args):
    """method for checking args for required arguments and returning useful errors (args is a dictionary)"""
    for key in required_args:
        if key not in args:
            raise ValueError(f"required arg {key} not found in args ({method_name} requires {required_args})")


def process_arguments(args, required_args, required_kwargs, possible_kwargs, name):
    """method for getting the required and optional kwargs from stored argument dictionary"""
    # if any required args are missing, raise an error
    check_args(name, args, required_args)

    # get required args (in order of list!)
    rq_args = [args[arg] for arg in required_args]

    # get kwargs
    kwargs = {}

    # if any required kwargs are missing, raise an error
    check_args(name, args, required_kwargs)
    for key, value in required_kwargs.items():
        kwargs[value] = args[key]

    # if any kwargs are included in args, add them to the dictionary
    for key, value in possible_kwargs.items():
        if key in args:
            kwargs[value] = args[key]

    return rq_args, kwargs
