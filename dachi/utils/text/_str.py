import string
import re
import typing


class _PartialFormatter(string.Formatter):
    """A partial formatter that does not require all keys to be specified"""

    def __init__(self):
        super().__init__()

    def format(self, format_string, *args, required: bool=True, **kwargs):
        """Format the string

        Args:
            format_string : The string to format
            required (bool, optional): Whether the key is required. Defaults to True.

        Returns:
            str: the formatted string
        """
        if args and kwargs:
            raise ValueError("Cannot mix positional and keyword arguments")

        if kwargs and required:
            difference = set(kwargs.keys()).difference(set(get_str_variables(format_string)))
            if difference:
                raise ValueError(f'Variables specified that are not in the string {difference}')
        self.args = args
        self.kwargs = kwargs
        return super().format(format_string)

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return self.kwargs.get(key, '{' + key + '}')
        if isinstance(key, int):
            return self.args[key] if key < len(self.args) else '{' + str(key) + '}'
        return super().get_value(key, args, kwargs)

    def __call__(self, format_string, *args, **kwargs):
        return self.format(format_string, *args, **kwargs)


def get_str_variables(format_string: str) -> typing.List[str]:
    """Get the variables in a string to format

    Args:
        format_string (str): The string to get variables for 

    Raises:
        ValueError: If the string has both positional and named
        variables

    Returns:
        typing.List[str]: The list of variables
    """
    has_positional = re.search(r'\{\d*\}', format_string)
    has_named = re.search(r'\{[a-zA-Z_]\w*\}', format_string)
    
    if has_positional and has_named:
        raise ValueError("Cannot mix positional and named variables")

    # Extract variables
    if has_positional:
        variables = [int(var) if var.isdigit() else None for var in re.findall(r'\{(\d*)\}', format_string)]
        if None in variables:
            variables = list(range(len(variables)))
    else:
        variables = re.findall(r'\{([a-zA-Z_]\w*)\}', format_string)
    
    return variables


str_formatter = _PartialFormatter() # Use to format a string with keys and values, does not require all keys specified


# def escape_curly_braces(value: typing.Any, render: bool=False) -> str:
#     """Escape curly braces for dictionary-like structures."""

#     if isinstance(value, str):
#         result = f'"{value}"'
#         return result
#     if isinstance(value, typing.Dict):
#         items = ', '.join(f'"{k}": {escape_curly_braces(v)}' for k, v in value.items())
#         return f"{{{{{items}}}}}"
#     if isinstance(value, typing.List):
#         return '[{}]'.format(', '.join(escape_curly_braces(v) for v in value))
#     if render:
#         return render(value)
#     return str(value)


def escape_curly_braces(value: str) -> str:
    """Escape curly braces in a string."""
    # replace { with {{ and } with }} 
    # but do not escape ones that are already escaped how do i do this. I cannot use value.replace and value is a string already. 
    # the code below is wrong because it will double escape already escaped braces
    
    return value.replace('{', '{{').replace('}', '}}')

    


def unescape_curly_braces(value: typing.Any) -> str:
    """Invert the escaping of curly braces."""
    if isinstance(value, str):
        return value.replace('{{', '{').replace('}}', '}')
    return value

