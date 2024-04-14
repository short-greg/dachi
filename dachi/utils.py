import typing


class F(object):
    """F is a functor that allows the user to set the args and kwargs
    """

    def __init__(self, f: typing.Callable[[], typing.Any], *args, **kwargs):
        """Create a functor

        Args:
            f (typing.Callable[[], typing.Any]): The function called
        """
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    @property
    def value(self) -> typing.Any:
        return self.f(*self.args, **self.kwargs)


def _is_function(f) -> bool:
    """ 
    Args:
        f: The value to check

    Returns:
        bool: whether f is a function
    """
    f_type = type(f)
    return f_type == type(_is_function) or f_type == type(hasattr)
