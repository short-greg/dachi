"""
Rendering utilities for converting various objects to text.
"""

# 1st party
import typing
import string
import roman
import re

# 3rd party
import pydantic
from dachi.core._render import render
from dachi.utils.text._render import render as render2

# local




S = typing.TypeVar('S', bound=pydantic.BaseModel)
X = typing.Union[str]


def generate_numbered_list(n, numbering_type='arabic') -> typing.List:
    """Generate a numbered list in arabic or roman numerals

    Args:
        n: The number of numbers to output
        numbering_type (str, optional): The type of numberals to use
        "arabic" or "roman" or "alphabet". Defaults to 'arabic'.

    Raises:
        ValueError: If the amount of numbers is not supported
        ValueError: If the numbering system is incorrect

    Returns:
        : 
    """
    if n < -1:
        raise ValueError('The number in list must be greater than or equal to 0.')
    if numbering_type == 'arabic':
        return [str(i) for i in range(1, n + 1)]
    elif numbering_type == 'roman':
        return [roman.toRoman(i).lower() for i in range(1, n + 1)]
    elif numbering_type == 'alphabet':
        if n > 26:
            raise ValueError("Alphabetic numbering can only handle up to 26 items")
        return [string.ascii_uppercase[i] for i in range(n)]
    else:
        raise ValueError("Unsupported numbering type")


def parse_function_spec(spec):
    """
    Parses a function-style format specifier like `bullet(1, "x", 1.)` and extracts 
    the function name and arguments. Converts numerical values appropriately 
    (int or float) and preserves strings. If the spec is not a function call, 
    it is returned as a plain string with no arguments.

    :param spec: The format specifier string (e.g., "bullet(1, 'x', 1.)" or "bold").
    :return: Tuple (function_name, args) where args is a list of parsed arguments.
    """
    match = re.match(r'(?P<func>[a-zA-Z_][a-zA-Z0-9_]*)\((?P<args>.*)\)', spec)
    if not match:
        return spec, None  # If it's not a function call, return spec as function name with empty args

    func_name = match.group("func")
    raw_args = match.group("args").strip()

    # Handle empty function calls (e.g., "bold()")
    if raw_args is None:
        return func_name, []

    # Parse arguments: convert numbers and preserve strings
    parsed_args = []
    for arg in re.findall(r'\'[^\']*\'|"[^"]*"|[\d.]+|\S+', raw_args):
        arg = arg.strip()
        if arg == ',':
            continue
        if arg.startswith(("'", '"')) and arg.endswith(("'", '"')):  # String arguments
            parsed_args.append(arg[1:-1])  # Remove surrounding quotes
        elif '.' in arg and arg.replace('.', '', 1).isdigit():  # Float argument
            parsed_args.append(float(arg))
        elif arg.isdigit():  # Integer argument
            parsed_args.append(int(arg))
        else:  # Fallback for unrecognized values
            parsed_args.append(arg)

    return func_name, parsed_args


class Styling:

    def __init__(self, value, styles):

        self.value = value
        self.styles = styles
        
    def __format__(self, spec):
        """Apply formatting based on specifiers."""
        spec, args = parse_function_spec(spec)
        args = args or []
        if spec in self.styles:
            return self.styles[spec](self.value, *args)
        elif spec:  # If a standard Python format specifier is given, use it.
            return format(self.value, spec)
        return render(self.value, False)  # Default case


def style_formatter(text, *args, _styles: typing.Dict=None, **kwargs) -> str:
    """
    Formats text with styled arguments using a custom StyleFactory.

    :param text: The format string with placeholders.
    :param args: Positional arguments to be styled.
    :param kwargs: Keyword arguments to be styled.
    :param _style_f: Custom Style Factory (defaults to StyleFactory).
    :return: Styled formatted string.
    """
    _styles = _styles or DEFAULT_STYLE
    updated_args = [ Styling(arg, _styles) for arg in args ]
    updated_kwargs = { k: Styling(v, _styles) for k, v in kwargs.items() }
    return text.format(*updated_args, **updated_kwargs)


def numbered(xs: typing.Iterable[X], indent: int=0, numbering: str='arabic') -> str:
    """Create a numbered list

    Args:
        xs (typing.Iterable[Cue]): A list of strings
        indent (int, optional): The number to start with indenting. Defaults to 0.
        numbering (str, optional): The type of numbering system to use. Defaults to 'arabic'.

    Returns:
        Cue: The resulting Cue
    """
    text = ''
    indent = ' ' * indent
    numbers = generate_numbered_list(len(xs), numbering)
    for i, (x_i, number) in enumerate(zip(xs, numbers)):
        text += f'{indent}{number}. {render(x_i)}'
        if i < (len(numbers) - 1):
            text += "\n"

    return text


def bullet(xs: typing.Iterable[X], bullets: str='-', indent: int=0) -> str:
    """Create a bullet list based on the instructions

    Args:
        xs (typing.Iterable[X]): The cues to bullet
        bullets (str, optional): The string to use for the bullet. Defaults to '-'.

    Returns:
        Cue: The resulting cue
    """
    indent = ' ' * indent
    text = ''
    # out = validate_out(xs)
    text = text + f'\n'.join(
        f'{indent}{bullets} {render(x_i)}' for x_i in xs
    )
    return text


def bold(x: X) -> str:
    """Create a bullet list based on the instructions

    Args:
        xs (typing.Iterable[X]): The cues to bullet
        bullets (str, optional): The string to use for the bullet. Defaults to '-'.

    Returns:
        str: The resulting cue
    """
    return f'**{render(x)}**'


def italic(x: X) -> str:
    """Create a bullet list based on the instructions

    Args:
        xs (typing.Iterable[X]): The cues to bullet
        bullets (str, optional): The string to use for the bullet. Defaults to '-'.

    Returns:
        str: The resulting cue
    """
    return f'*{render(x)}*'


DEFAULT_STYLE = {
    "bold": bold,
    "italic": italic,
    "bullet": bullet,
    "numbered": numbered
}
