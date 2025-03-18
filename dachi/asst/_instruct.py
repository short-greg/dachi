# 1st party
import typing
import string

import pydantic
import roman
import re

# local
from ..utils import (
    render, render_multi
)
from ..proc import Param
from ._instruct_core import Cue, validate_out
from ..utils import (
    str_formatter
)
from ..base import UNDEFINED
from ..proc import Module, AsyncModule, AsyncStreamModule, StreamModule
from ..asst import LLM, OutConv, ToMsg
from ._data import Description

from ..utils import render

from ..utils._utils import str_formatter

S = typing.TypeVar('S', bound=pydantic.BaseModel)
X = typing.Union[str, Description, Cue]


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


def numbered(xs: typing.Iterable[X], indent: int=0, numbering: str='arabic') -> 'Cue':
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
    out = validate_out(xs)
    for i, (x_i, number) in enumerate(zip(xs, numbers)):
        text = f'{indent}{number}. {render(x_i)}'
        if i < (len(numbers) - 1):
            text += "\n"

    return Cue(
        text=text, 
        out=out
    )


def fill(x_instr: X, *args: X, **kwargs: X) -> 'Cue':
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


def join(x1: X, x2: X, sep: str=' ') -> Cue:
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
        if arg.startswith(("'", '"')) and arg.endswith(("'", '"')):  # String arguments
            parsed_args.append(arg[1:-1])  # Remove surrounding quotes
        elif '.' in arg and arg.replace('.', '', 1).isdigit():  # Float argument
            parsed_args.append(float(arg))
        elif arg.isdigit():  # Integer argument
            parsed_args.append(int(arg))
        else:  # Fallback for unrecognized values
            parsed_args.append(arg)

    return func_name, parsed_args


class Style:

    def __init__(self, value, styles):

        self.value = value
        self.styles = styles
        
    def __format__(self, spec):
        """Apply formatting based on specifiers."""
        spec, args = parse_function_spec(spec)
        args = args or []
        if spec in self.styles:
            return self.styles[spec](*args)
        elif spec:  # If a standard Python format specifier is given, use it.
            print(spec, self.value)
            return format(self.value, spec)
        return render(self.value, False)  # Default case


def style_formatter(text, *args, _styles: typing.Dict, **kwargs) -> str:
    """
    Formats text with styled arguments using a custom StyleFactory.

    :param text: The format string with placeholders.
    :param args: Positional arguments to be styled.
    :param kwargs: Keyword arguments to be styled.
    :param _style_f: Custom Style Factory (defaults to StyleFactory).
    :return: Styled formatted string.
    """
    updated_args = [ Style(arg, _styles) for arg in args ]
    updated_kwargs = { k: Style(v, _styles) for k, v in kwargs.items() }
    return text.format(*updated_args, **updated_kwargs)


# TODO: need string args for the context
# other args for things that get inserted into
# the message
class Op(Module, AsyncModule, StreamModule, AsyncStreamModule):
    """
    The Op class allows interaction with a Language Learning Model (LLM) by sending instructions to it.
    """
    def __init__(
        self, llm: LLM, to_msg: ToMsg, out: typing.Optional[OutConv]=None, context: typing.Optional[str]=None
    ): 
        """
        Initialize an Op instance.
        This initializer sets up the necessary components to interact with the LLM and send instructions to it.
        Args:
            llm (LLM): The language model to interact with.
            to_msg (ToMsg): The message converter to use.
            out (typing.Optional[OutConv], optional): The output converter. Defaults to None.
            context (typing.Optional[str], optional): The context for the operation. Defaults to None.
        """
        super().__init__()
        self.to_msg = to_msg
        self.out = out
        self.llm = llm
        self.context = context

    def build_context(self, *args, **kwargs) -> str:
        """
        Builds the context to put the input into.
        If the instance's context attribute is None, it joins the provided 
        positional arguments with newline characters and returns the result 
        as a string. If the context attribute is not None, it formats the 
        context using the provided positional and keyword arguments.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            str: The formatted context or the joined positional arguments.
        """
        if self.context is None:
            assert len(kwargs) == 0
            return '\n'.join(args)
        else:
            return str_formatter(
                self.context, *args, **kwargs
            )

    def forward(self, *args, _out: OutConv=None, **kwargs):
        context = self.build_context(*args, **kwargs)
        msg = self.to_msg(context)
        _, resp = self.llm(msg)
        _out = _out if _out is not None else self._out
        return _out(resp)
    
    async def aforward(self, *args, _out: OutConv=UNDEFINED, **kwargs):

        context = self.build_context(*args, **kwargs)
        msg = self.to_msg(context)
        _, resp = await self.llm.aforward(msg)
        _out = _out if _out is not None else self._out
        return _out(resp)
    
    def stream(self, *args, _out: OutConv=UNDEFINED, **kwargs) -> typing.Iterator:

        context = self.build_context(*args, **kwargs)
        msg = self.to_msg(context)
        _out = _out if _out is not None else self._out
        delta = {}
        for _, resp in self.llm.stream(msg):
            yield _out.delta(resp, delta)

    async def astream(self, *args, _out: OutConv=UNDEFINED, **kwargs) -> typing.AsyncIterator:

        context = self.build_context(*args, **kwargs)
        msg = self.to_msg(context)
        _out = _out if _out is not None else self._out
        delta = {}
        async for _, resp in await self.llm.astream(msg):
            yield _out.delta(resp, delta)

    def spawn(
        self, to_msg: ToMsg=UNDEFINED,
        out: OutConv=UNDEFINED,
        context: str=UNDEFINED,
        **kwargs
    ):
        llm = self.llm if len(kwargs) == 0 else self.llm.spawn(**kwargs)
        to_msg = self.to_msg if self.to_msg is UNDEFINED else to_msg
        out = self.out if out is UNDEFINED else out
        context = self.context if context is UNDEFINED else context
        return Inst(
            llm, to_msg, out, context
        )


class Inst(Module):
    """An operation acts on an cue to produce a new cue
    """

    def __init__(
        self, name: str, cue: X, 
        tunable: bool=False):
        """Create an operation specifying the name

        Args:
            name (str): The name of the operation
            cue (X): The cue for the operation
            tunable: whether the cue is tunable
        """
        self.name = name
        if not isinstance(cue, Cue):
            cue = Param(
                name=self.name, training=tunable, 
                data=Cue(
                    text=render(cue)
            ))
        
        self.cue = cue
        
    def forward(
        self, *args: X, **kwargs: X
    ) -> Cue:
        """Fill in the cue with the inputs

        Returns:
            Cue: 
        """
        cue = render(self.cue)
        out = validate_out(
            [*args, *kwargs.values(), self.cue]
        )

        return Cue(
            text=fill(cue, *args, **kwargs), out=out
        )


def inst(x: typing.Union[typing.Iterable[X], X], cue: X) -> Cue:
    """Execute an operation on a cue

    Args:
        x (typing.Union[typing.Iterable[X], X]): The input
        cue (X): The cue for the operation

    Returns:
        Cue: The resulting cue
    """
    if not isinstance(x, typing.Iterable):
        x = [x]

    out = validate_out([*x, cue])
    resources = ', '.join(render_multi(x))
    # resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {render(cue)} --- With Inputs: {resources}'
    return Cue(
        text=text, out=out
    )


def bullet(xs: typing.Iterable[X], bullets: str='-', indent: int=0) -> 'Cue':
    """Create a bullet list based on the instructions

    Args:
        xs (typing.Iterable[X]): The cues to bullet
        bullets (str, optional): The string to use for the bullet. Defaults to '-'.

    Returns:
        Cue: The resulting cue
    """
    indent = ' ' * indent
    text = f'\n{indent}{bullets}'
    out = validate_out(xs)
    text = text + f'\n{indent}{bullets}'.join(
        render(x_i) for x_i in xs
    )
    return Cue(
        text=text, out=out
    )
