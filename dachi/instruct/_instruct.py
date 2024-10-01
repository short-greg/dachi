# 1st party
import typing
import string

import pydantic
import roman

# local
from .._core import (
    render, render_multi,
    Instruction, render, Param, validate_out 
)
from ..utils import (
    str_formatter
)
from .._core import Module
from ._data import Description


S = typing.TypeVar('S', bound=pydantic.BaseModel)
X = typing.Union[str, Description, Instruction]


def bullet(xs: typing.Iterable[X], bullets: str='-', indent: int=0) -> 'Instruction':
    """Create a bullet list based on the instructions

    Args:
        xs (typing.Iterable[X]): The instructions to bullet
        bullets (str, optional): The string to use for the bullet. Defaults to '-'.

    Returns:
        Instruction: The resulting instruction
    """
    indent = ' ' * indent
    text = f'\n{indent}{bullets}'
    out = validate_out(xs)
    text = text + f'\n{indent}{bullets}'.join(
        render(x_i) for x_i in xs
    )
    return Instruction(
        text=text, out=out
    )


def formatted(x: X, format: str) -> 'Instruction':
    """Format the X with a format. The format string will encapsulate it

    Example:

    formatted('Name', '*') => '*Name*'

    Args:
        x (X): The data to format
        format (str): The format to use

    Returns:
        Instruction: The resulting instruction
    """

    text = render(x)
    if text[:len(format)] == format and text[-len(format):] == format:
        return x
    return Instruction(
        f'{format}{text}{format}',
        out=x.out
    )


def bold(x: X) -> 'Instruction':
    """Format the X with a bold format. The format string will encapsulate it

    Example:

    bold('Name') => '**Name**'

    Args:
        x (X): The data to format

    Returns:
        Instruction: The resulting instruction
    """

    return formatted(x, '**')


def italic(x: X) -> 'Instruction':
    """Format the X with a bold format. The format string will encapsulate it

    Example:

    italic('Name') => '*Name*'

    Args:
        x (X): The data to format

    Returns:
        Instruction: The resulting instruction
    """
    return formatted(x, '*')


def strike(x: X) -> 'Instruction':
    """Format the X with a bold format. The format string will encapsulate it

    Example:

    italic('Name') => '*Name*'

    Args:
        x (X): The data to format

    Returns:
        Instruction: The resulting instruction
    """
    return formatted(x, '~~')


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
        _type_: _description_
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


def numbered(xs: typing.Iterable[X], indent: int=0, numbering: str='arabic') -> 'Instruction':
    """Create a numbered list

    Args:
        xs (typing.Iterable[Instruction]): A list of strings
        indent (int, optional): The number to start with indenting. Defaults to 0.
        numbering (str, optional): The type of numbering system to use. Defaults to 'arabic'.

    Returns:
        Instruction: The resulting Instruction
    """
    text = ''
    indent = ' ' * indent
    numbers = generate_numbered_list(len(xs), numbering)
    out = validate_out(xs)
    for i, (x_i, number) in enumerate(zip(xs, numbers)):
        text = f'{indent}{number}. {render(x_i)}'
        if i < (len(numbers) - 1):
            text += "\n"

    return Instruction(
        text=text, 
        out=out
    )


def fill(x_instr: X, *args: X, **kwargs: X) -> 'Instruction':
    """Format a string with variables

    Args:
        x_instr (X): The value to format

    Returns:
        Instruction: The resulting instruction
    """
    out = validate_out([x_instr])

    kwargs = dict(zip(kwargs.keys(), render_multi(kwargs.values())))
    args = render_multi(args)
    return Instruction(
        text=str_formatter(render(x_instr), *args, **kwargs), out=out
    )


def head(x: X, size: int=1) -> 'Instruction':
    """Add a header to the instruction

    Args:
        x (X): The input to add to
        size (int, optional): The size of the heading. Defaults to 1.

    Returns:
        Instruction: The resulting instruction
    """
    out = validate_out([x])
    heading = '#' * size
    return Instruction(
        text=f'{heading} {render(x)}', out=out
    )


def section(name: X, details: X, size: int=1, linebreak: int=1) -> 'Instruction':
    """Add a section to the instruction

    Args:
        name (X): The name of the section
        details (X): The details for the section
        size (int, optional): The size of the heading. Defaults to 1.
        linebreak (int, optional): How many linebreaks to put between the heading and the details. Defaults to 1.

    Returns:
        Instruction: The inst
    """
    heading = '#' * size
    out = validate_out([name, details])
    linebreak = '\n' * linebreak
    text = f'{heading} {render(name)}{linebreak}' + render(details)
    return Instruction(
        text=text, out=out
    )


def cat(xs: typing.List[Instruction], sep: str=' ') -> Instruction:
    """Concatenate multiple instructions together

    Args:
        xs (typing.List[Instruction]): THe instructions 
        sep (str): The delimiter to use for the sections

    Raises:
        RuntimeError: 

    Returns:
        Instruction: The concatenated instruction
    """
    out = validate_out(xs)

    return Instruction(text=f'{sep}'.join(
        render(x_i) for x_i in xs
    ), out=out)


def join(x1: X, x2: X, sep: str=' ') -> Instruction:
    """Join two instructions together

    Args:
        x1 : Instruction 1
        x2 : Instruction 2
        sep (str): The separator between the two instructions

    Returns:
        Instruction: The joined instructions
    """
    out = validate_out([x1, x2])
    return Instruction(
        text=render(x1) + sep + render(x2),
        out=out
    )


class Operation(Module):
    """An operation acts on an instruction to produce a new instruction
    """

    def __init__(
        self, name: str, instruction: X, 
        tunable: bool=False):
        """Create an operation specifying the name

        Args:
            name (str): The name of the operation
            instruction (X): The instruction for the operation
            tunable: whether the instruction is tunable
        """
        self.name = name
        if not isinstance(instruction, Instruction):
            instruction = Param(
                name=self.name, training=tunable, 
                instruction=Instruction(
                    text=render(instruction)
            ))
        
        self.instruction = instruction
        
    def forward(
        self, *args: X, **kwargs: X
    ) -> Instruction:
        """Fill in the instruction with the inputs

        Returns:
            Instruction: 
        """
        instruction = render(self.instruction)
        out = validate_out(
            [*args, *kwargs.values(), self.instruction]
        )

        return Instruction(
            text=fill(instruction, *args, **kwargs), out=out
        )


def op(x: typing.Union[typing.Iterable[X], X], instruction: X) -> Instruction:
    """Execute an operation on an instruction

    Args:
        x (typing.Union[typing.Iterable[X], X]): The input
        instruction (X): The instruction for the operation

    Returns:
        Instruction: The resulting instruction
    """
    if not isinstance(x, typing.Iterable):
        x = [x]

    out = validate_out([*x, instruction])
    resources = ', '.join(render_multi(x))
    # resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {render(instruction)} --- With Inputs: {resources}'
    return Instruction(
        text=text, out=out
    )
