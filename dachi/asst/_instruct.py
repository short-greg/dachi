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



DEFAULT_STYLES = {
    'bullet': bullet,
    'numbered': numbered
}


def extract_styles(text: str) -> typing.List[typing.Tuple[typing.Optional[str], str]]:

    patterns = [
        r'\{\[\s*(?P<var1>[a-zA-Z_][a-zA-Z0-9_-]*)\s*::?\s*\]\}',
        r'\{\[\s*(?P<pos2>\d*)(?P<style2>[a-zA-Z_][a-zA-Z0-9_-]*)\s*(?:\(\s*(?P<args2>[^)]*)\s*\))?\s*?\s*\]\}',
        r'\{\[\s*(?P<var3>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<style3>[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(\s*(?P<args3>[^)]*)?\s*\))?\s*\]\}',

        r'(?P<pos4>\{\s*\})',
        r'\{(?P<pos5>\d+)(?::(?P<style5>[^}]*))?\}',
        # with style and named argument
        # r'\{(?P<var4>[a-zA-Z_][a-zA-Z0-9_]*)(?::(?P<style>[^}]*))?\}'
        # only with style
        # r'\{¥s*(?P<var5>)(?:(?P<style>[^}]*))?¥s*\}'



        # r'\{(?P<style4>[a-zA-Z_][a-zA-Z0-9_]*(?:::[^}]*)?)\}',
        # r'\{\[\s*(?:(?P<var1>[^:\]]+)\s*:\s*)?(?P<style1>[^\]\(\)]+)\s*(?:\(\s*(?P<args1>[^)]*)\s*\))?\s*\]\}',
        # r'\{\[\s*(?P<var2>[^:\]]+)\s*::\s*\]\}',
        # # r'\{\s*(?P<var3>[^:\}]+)\s*:\s*(?P<style3>[^:\}]+)\s*\}',
        # r'\{\[\s*(?P<var4>[^\]])\s*\]\}',
        # r'\{\s*\}'
    ]

    combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
    matches = re.finditer(combined_pattern, text)
    results = []
    cur_pos = 0

    for match in matches:

        # if match.group('style4'):
        #     var = pos
        #     style = "DEFAULT"
        #     args = None
        #     pos += 1
        if match.group('var1'):
            var = match.group('var1')
            style = "DEFAULT"
            specified = True
            args = None
        elif match.group('style2'):
            print('Style 2')
            pos = match.group('pos2')
            if pos == '':
                specified = False
                var = cur_pos
                cur_pos += 1
            else:
                specified = True
                var = pos
            style = match.group('style2')
            args = match.group('args2')
            if args is not None:
                args = [x.strip() for x in args.split(',')]
        elif match.group('style3'):
            print('style 1')
            var = match.group('var3')
            if var is None:
                var = cur_pos
                cur_pos += 1
            specified = True
            style = match.group('style3')
            args = match.group('args3')
            if args is not None:
                args = [x.strip() for x in args.split(',')]
        elif match.group('pos4'):
            cur_pos += 1
            continue
        else:
            print('Here')
            continue
        
        # elif match.group('var2'):
        #     var = match.group('var2')
        #     style = "DEFAULT"
        #     args = None
        # # elif match.group('style3'):
        # #     var = match.group('var3')
        # #     style = None
        # #     args = None
        # elif match.group('var4'):
        #     var = match.group('var4')
        #     style = None
        #     args = None
        # elif match.group('var5'):
        #     var = match.group('var5')
        #     style = None
        #     args = None
        # else:
        #     var = pos
        #     style = None
        #     args = None
        #     pos += 1
        results.append((var, style, args, specified))
        print(results[-1])
    return results
    # combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
    # matches = re.finditer(combined_pattern, text)
    # results = []
    # pos = 0
    # for match in matches:
    #     if match.group('var1') or match.group('style1'):
    #         var = match.group('var1')
    #         style = match.group('style1')
    #     elif match.group('style2'):
    #         var = pos
    #         style = match.group('style2')
    #         pos += 1
    #     elif match.group('var2'):
    #         var = match.group('var2')
    #         style = "DEFAULT"
    #     elif match.group('var3') or match.group('style3'):
    #         var = match.group('var3')
    #         style = None
    #     elif match.group('var4'):
    #         var = match.group('var4')
    #         style = None
    #     elif match.group('var5'):
    #         var = pos
    #         style = "DEFAULT"
    #         pos += 1
    #     else:
    #         var = pos
    #         style = None
    #         pos += 1
    #     results.append((var, style))
    # return results


def replace_style_formatting(text: str) -> str:
    """
    Convert special formatting patterns in the text to normal Python formatting.

    Args:
        text (str): The text to be converted.

    Returns:
        str: The converted text.
    """
    # Replace {[<var str>:<style str>()]} with {<var str>}
    text = re.sub(r'\{\[\s*([^:\]]+)\s*:\s*([^\]]+)(?:\(\s*(?P<args2>[^\)]*)\s*\))?\s*\]\}', r'{\1}', text)
    
    # Replace {[<var str>::]} with {<var str>}
    text = re.sub(r'\{\[\s*([^:\]]+)\s*::\s*\]\}', r'{\1}', text)
    
    # Replace {[<style str>]} with {}
    text = re.sub(
        r'\{\[\s*([^\]:]+)(?:\(\s*(?P<args2>[^\)]*)\s*\))?\s*\]\}', 
        r'{}', text
    )
    # Replace {[]} with {}
    text = re.sub(r'\{\[\s*]\}', r'{}', text)
    
    return text


def style_format(text: str, *args, styles: typing.Dict, **kwargs) -> str:
    """
    Formats the given text with advanced stylings based on the provided styles dictionary.
    Args:
        text (str): The text to be formatted.
        *args: Positional arguments to be formatted within the text.
        styles (typing.Dict): A dictionary where keys are style names and values are functions that apply the style.
        **kwargs: Keyword arguments to be formatted within the text.
    Returns:
        str: The formatted text with applied styles.
    Example:
        styles = {
            "bold": lambda x: f"**{x}**",
            "italic": lambda x: f"*{x}*"
        }
        formatted_text = style_format("Hello {0} and {name}", "World", name="Alice", styles=styles)
        # formatted_text will be "Hello **World** and *Alice*"
    """

    patterns = extract_styles(text)
    for var, style in patterns:
        if isinstance(var, int):
            if len(args) >= var:
                continue
            if style == "DEFAULT":
                args[var] = render(args[var])
            elif style is None:
                args[var] = args[var]
            else:
                args[var] = styles[style](args[var])
        elif isinstance(var, str):
            if var not in kwargs:
                continue
            if style == "DEFAULT":
                kwargs[var] = render(kwargs[var])
            elif style is None:
                kwargs[var] = kwargs[var]
            else:
                kwargs[var] = styles[style](kwargs[var])
    text = replace_style_formatting(text)
    return str_formatter(text, *args, **kwargs)


# def formatted(x: X, format: str) -> 'Cue':
#     """Format the X with a format. The format string will encapsulate it

#     Example:

#     formatted('Name', '*') => '*Name*'

#     Args:
#         x (X): The data to format
#         format (str): The format to use

#     Returns:
#         Cue: The resulting cue
#     """

#     text = render(x)
#     if text[:len(format)] == format and text[-len(format):] == format:
#         return x
#     return Cue(
#         f'{format}{text}{format}',
#         out=x.out
#     )


# def bold(x: X) -> 'Cue':
#     """Format the X with a bold format. The format string will encapsulate it

#     Example:

#     bold('Name') => '**Name**'

#     Args:
#         x (X): The data to format

#     Returns:
#         Cue: The resulting cue
#     """

#     return formatted(x, '**')


# def italic(x: X) -> 'Cue':
#     """Format the X with a bold format. The format string will encapsulate it

#     Example:

#     italic('Name') => '*Name*'

#     Args:
#         x (X): The data to format

#     Returns:
#         Cue: The resulting cue
#     """
#     return formatted(x, '*')


# def strike(x: X) -> 'Cue':
#     """Format the X with a bold format. The format string will encapsulate it

#     Example:

#     italic('Name') => '*Name*'

#     Args:
#         x (X): The data to format

#     Returns:
#         Cue: The resulting cue
#     """
#     return formatted(x, '~~')


# def section(name: X, details: X, size: int=1, linebreak: int=1) -> 'Cue':
#     """Add a section to the cue

#     Args:
#         name (X): The name of the section
#         details (X): The details for the section
#         size (int, optional): The size of the heading. Defaults to 1.
#         linebreak (int, optional): How many linebreaks to put between the heading and the details. Defaults to 1.

#     Returns:
#         Cue: The inst
#     """
#     heading = '#' * size
#     out = validate_out([name, details])
#     linebreak = '\n' * linebreak
#     text = f'{heading} {render(name)}{linebreak}' + render(details)
#     return Cue(
#         text=text, out=out
#     )

# def head(x: X, size: int=1) -> 'Cue':
#     """Add a header to the cue

#     Args:
#         x (X): The input to add to
#         size (int, optional): The size of the heading. Defaults to 1.

#     Returns:
#         Cue: The resulting cue
#     """
#     out = validate_out([x])
#     heading = '#' * size
#     return Cue(
#         text=f'{heading} {render(x)}', out=out
#     )
