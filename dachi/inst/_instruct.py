# 1st party
import typing
import string
import pydantic
import roman
import re
from abc import abstractmethod, ABC
from typing import Self

# 3rd party
import pydantic

# local
from ..core._render import render, render_multi
from ..core import Trainable, Renderable
from ..utils import is_primitive, str_formatter

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



# Engine: typing.TypeAlias = Assist | AsyncAssist | StreamAssist | AsyncStreamAssist

S = typing.TypeVar('S')

# TODO: MOVE OUT OF HERE

class Instruct(ABC):
    """
    """
    @abstractmethod
    def i(self) -> 'Cue':
        """Create an Instruct class used for instructions

        Returns:
            Cue: Get the cue
        """
        pass


class Cue(
    Trainable, 
    Instruct, typing.Generic[S], Renderable
):
    """Specific cue for the model to use
    """
    def __init__(
        self, text: str, name: str='', 
        out: typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None
    ):
        """
        Initializes the instance with the provided text, name, and optional output converter.
        Args:
            text (str): The text to be processed.
            name (str, optional): The name associated with the text. Defaults to an empty string.
            out (Optional[OutConv], optional): The converter to use for processing the output. Defaults to None.
        """
        super().__init__()
        self._out = out
        self.text = text
        self.name = name

    def i(self) -> Self:
        return self

    @pydantic.field_validator('text', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Renderable):
            return v.render()
        if is_primitive(v):
            return str(v)
        return v

    def render(self) -> str:
        """Render the cue

        Returns:
            str: The text for the cue 
        """
        return self.text

    def read(self, data: str) -> typing.Any:
        """Read the data

        Args:
            data (str): The data to read

        Raises:
            RuntimeError: If the cue does not have a out

        Returns:
            S: The result of the read process
        """
        if self._out is None:
            return data
        
        return self._out(data)

    @property
    def fixed_data(self):
        return {"out"}

    def update_param_dict(self, data: typing.Dict) -> bool:
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            # excluded = self.data.dict_excluded()
            # data.update(
            #     excluded
            # )

            self.text = data[self.name]
            return True
        return False

    def param_dict(self):
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            return {self._name: self.text}
        return {}

    def data_schema(self) -> typing.Dict:
        """Get the structure of the object

        Returns:
            typing.Dict: The structure of the object
        """
        return {
            "title": self.name,
            "type": "object",
            "properties": {
                "text": {
                    "title": "Text",
                    "type": "string"
                }
            },
            "required": ["text"]
        }


Y = typing.Union[str, Cue]


def validate_out(cues: typing.List[Y]) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
    """Validate an Out based on several instructions

    Args:
        instructions (typing.List[X]): The instructions 

    Returns:
        Out: The resulting "Out" to use from the instructions
    """
    if isinstance(cues, dict):
        cues = cues.values()

    out = None
    for cue in cues:
        if not isinstance(cue, Cue):
            continue
        if out is None and cue._out is not None:
            out = cue._out
        elif cue._out is not None:
            raise RuntimeError(f'Out cannot be duplicated')
    return out


def fill(x_instr: Y, *args: Y, **kwargs: Y) -> 'Cue':
    """Format a string with variables

    Args:
        x_instr (X): The value to format

    Returns:
        Cue: The resulting cue
    """
    out = validate_out([x_instr])

    kwargs = dict(zip(kwargs.keys(), render_multi(kwargs.values())))
    args = render_multi(args)
    return Cue(
        text=str_formatter(render(x_instr), *args, **kwargs), out=out
    )


def cat(xs: typing.List[Cue], sep: str=' ') -> Cue:
    """Concatenate multiple cues together

    Args:
        xs (typing.List[Cue]): THe cues 
        sep (str): The delimiter to use for the sections

    Raises:
        RuntimeError: 

    Returns:
        Cue: The concatenated cues
    """
    out = validate_out(xs)

    return Cue(text=f'{sep}'.join(
        render(x_i) for x_i in xs
    ), out=out)


def join(x1: Y, x2: Y, sep: str=' ') -> Cue:
    """Join two instructions together

    Args:
        x1 : Cue 1
        x2 : Cue 2
        sep (str): The separator between the two instructions

    Returns:
        Cue: The joined instructions
    """
    out = validate_out([x1, x2])
    return Cue(
        text=render(x1) + sep + render(x2),
        out=out
    )


# def bullet(xs: typing.Iterable[X], bullets: str='-', indent: int=0) -> 'Cue':
#     """Create a bullet list based on the instructions

#     Args:
#         xs (typing.Iterable[X]): The cues to bullet
#         bullets (str, optional): The string to use for the bullet. Defaults to '-'.

#     Returns:
#         Cue: The resulting cue
#     """
#     indent = ' ' * indent
#     text = f'\n{indent}{bullets}'
#     out = validate_out(xs)
#     text = text + f'\n{indent}{bullets}'.join(
#         render(x_i) for x_i in xs
#     )
#     return Cue(
#         text=text, out=out
#     )
