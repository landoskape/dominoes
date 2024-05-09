from .pointer_networks import get_pointer_network
from .pointer_layers import get_pointer_methods


def _check_kwargs(method_name, kwargs, required_kwargs):
    """method for checking kwargs for required arguments and returning useful errors"""
    for key in required_kwargs:
        if key not in kwargs:
            raise ValueError(f"required kwarg {key} not found in kwargs ({method_name} requires {required_kwargs})")
