from argparse import ArgumentTypeError


class AttributeDict(dict):
    """
    Use to convert a dictionary to an object with standard attribute lookup.

    Useful for creating a dictionary, then turning it into an object for easy attribute lookup
    where (for example) an ArgumentParser object is expected.

    Example:
    ```
    dictionary = dict(key="value") # whatever you want in the dictionary
    attribute_dictionary = AttributeDict(dictionary)
    print(attribute_dictionary.key) # prints "value"
    ```
    """

    def __getattr__(self, attr):
        """Get an attribute from the dictionary"""
        return self[attr]

    def __setattr__(self, attr, value):
        """Set an attribute in the dictionary"""
        self[attr] = value


def argbool(value):
    """Convert a string to a boolean (for use with ArgumentParser)"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
