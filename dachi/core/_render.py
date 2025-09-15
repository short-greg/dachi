"""
Rendering utilities for converting various objects to text.
"""

# 1st party
import typing
import inspect
from dataclasses import dataclass
import string
import roman
import re
from typing import Self, get_type_hints

# 3rd party
import pydantic
from ._base import Renderable

# local
from dachi.utils import is_primitive, escape_curly_braces
from dachi.utils import escape_curly_braces, is_primitive


@dataclass
class TemplateField(Renderable):
    """Use for rendering a field in a BaseModel
    """
    type_: str
    description: str
    default: typing.Any = None
    is_required: bool = True

    def to_dict(self) -> typing.Dict:
        """Convert the template to a dict

        Returns:
            typing.Dict: the template
        """
        return {
            'type': self.type_,
            'description': self.description,
            'default': self.default,
            'is_required': self.is_required
        }
    
    def render(self) -> str:
        """Convert the template to a string

        Returns:
            str: The string of the template.
        """
        return str(self.to_dict())


def model_to_text(
    model: pydantic.BaseModel, 
    escape: bool=False
) -> str:
    """Dump the struct to a string

    Returns:
        str: The string
    """
    if escape:  
        return escape_curly_braces(model.model_dump())
    return model.model_dump_json()


def render(
    x: typing.Any, escape_braces: bool=True, 
    template_render: typing.Optional[typing.Callable[['TemplateField'], str]]=None
) -> typing.Union[str, typing.List[str]]:
    """Convert an input to text. Will use the text for a cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    if isinstance(x, TemplateField):
        if template_render is not None:
            x = template_render(x)
        else: 
            x = x.render()

    if isinstance(x, Renderable):
        return x.render()

    elif isinstance(x, pydantic.BaseModel):
        return model_to_text(x, escape_braces)
    elif is_primitive(x):
        return str(x)
    elif isinstance(x, typing.Dict):
        items = {}
        for k, v in x.items():
            if isinstance(v, str):
                v = f'"{v}"'
            else:
                v = render(v, escape_braces)
            items[k] = v    
        items = ', '.join(
            f'"{k}": {v}' 
            for k, v in items.items()
        )

        if escape_braces:
            return f"{{{{{items}}}}}"
        else:
            return f'{{{items}}}'
    elif isinstance(x, typing.List):

        items = []
        for v in x:
            if isinstance(v, str):
                v = f'"{v}"'
            else:
                v = render(v, escape_braces)
            items.append(v)

        return '[{}]'.format(', '.join(render(v) for v in items))
    elif isinstance(x, Renderable):
        return x.render()
    return str(x)


def model_template(model_cls: typing.Type[pydantic.BaseModel]) -> str:
    """Get the template for a pydantic.BaseModel

    Args:
        model_cls (typing.Type[pydantic.BaseModel]): The model to retrieve for

    Returns:
        str: The model template string
    """
    template = {}
    for name, field_type in get_type_hints(model_cls).items():
        
        if inspect.isclass(field_type) and issubclass(field_type, pydantic.BaseModel):
            template[name] = model_template(field_type)
        else:
            template[name] = {
                "is_required": model_cls.model_fields[name].is_required(),
                "type": field_type
            }
    return template


def struct_template(model: pydantic.BaseModel) -> typing.Dict:
    """Get the template for the Struct

    Returns:
        typing.Dict: The template 
    """
    template = {}
    
    base_template = model_template(model)
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        if isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel):

            template[field_name] = struct_template(field_type)
        else:

            if 'is_required' in base_template[field_name]:
                is_required = base_template[field_name]['is_required']
            else:
                is_required = True
            template[field_name] = TemplateField(
                type_=field.annotation,
                description=field.description,
                default=field.default if field.default is not None else None,
                is_required=is_required
            )

    return template


def is_renderable(obj: typing.Any) -> bool:
    """Return whether an object is renderable

    Args:
        obj (typing.Any): The object to check

    Returns:
        bool: whether the object is renderable
    """

    return (
        isinstance(obj, Renderable)
        or is_primitive(obj)
        or isinstance(obj, list)
        or isinstance(obj, dict)
        or isinstance(obj, pydantic.BaseModel)
    )


def render_multi(
    xs: typing.Iterable[typing.Any]
) -> typing.List[str]:
    """Convert an input to text. Will use the text for an cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    return [
        render(x) for x in xs
    ]


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


# def generate_numbered_list(n, numbering_type='arabic') -> typing.List:
#     """Generate a numbered list in arabic or roman numerals

#     Args:
#         n: The number of numbers to output
#         numbering_type (str, optional): The type of numberals to use
#         "arabic" or "roman" or "alphabet". Defaults to 'arabic'.

#     Raises:
#         ValueError: If the amount of numbers is not supported
#         ValueError: If the numbering system is incorrect

#     Returns:
#         : 
#     """
#     if n < -1:
#         raise ValueError('The number in list must be greater than or equal to 0.')
#     if numbering_type == 'arabic':
#         return [str(i) for i in range(1, n + 1)]
#     elif numbering_type == 'roman':
#         return [roman.toRoman(i).lower() for i in range(1, n + 1)]
#     elif numbering_type == 'alphabet':
#         if n > 26:
#             raise ValueError("Alphabetic numbering can only handle up to 26 items")
#         return [string.ascii_uppercase[i] for i in range(n)]
#     else:
#         raise ValueError("Unsupported numbering type")


