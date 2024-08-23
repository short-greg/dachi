# 1st party
import typing
from functools import wraps, update_wrapper
import inspect
import string

import pydantic
import roman

# local
from ._core import (
    Struct,
    render, render_multi,
    Instruction, render, Param, 
    Instruct, Reader, NullRead,
    Struct, 
    Instruction
)
from ._ai import (
    Dialog, TextMessage, AIModel, AIResponse
)
from ._read import (
    StructRead, PrimRead
)
from ._utils import (
    str_formatter, primitives
)
from ._process import Module
from ._structs import Description


S = typing.TypeVar('S', bound=Struct)
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


def generate_numbered_list(n, numbering_type='arabic'):
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


def validate_out(instructions: typing.List[X]) -> typing.Optional[Reader]:
    """Validate an Out based on several instructions

    Args:
        instructions (typing.List[X]): The instructions 

    Returns:
        Out: The resulting "Out" to use from the instructions
    """
    if isinstance(instructions, dict):
        instructions = instructions.values()

    out = None
    for instruction in instructions:
        if not isinstance(instruction, Instruction):
            continue
        if out is None and instruction.out is not None:
            out = instruction.out
        elif instruction.out is not None:
            raise RuntimeError(f'Out cannot be duplicated')
    return out


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
        Instruction: 
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

    def __init__(self, name: str, instruction: X):
        """Create an operation specifying the name

        Args:
            name (str): The name of the operation
            instruction (X): The instruction for the operation
        """
        if not isinstance(instruction, Instruction):
            instruction = Instruction(
                text=render(instruction)
            )
        self.name = name
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


class SignatureFunc(Module, Instruct):
    """SignatureMethod is a method where you define the instruction in
    the function signature
    """
    def __init__(
        self, f: typing.Callable, engine: AIModel, 
        dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
        is_method: bool=False,
        reader: typing.Optional[Reader]=None,
        train: bool=False, 
        instance=None,
        return_: typing.List[str]=None,
        ai_kwargs: typing.Dict=None
    ):
        """Wrap the signature method with a particular engine and
        dialog factory

        Args:
            f (typing.Callable): The function to wrap
            engine (AIModel): The engine to use for getting the response
            dialog_factory (typing.Optional[typing.Callable[[], Dialog]], optional): The dialog to use. Defaults to None.
            train (bool, optional): Whether to train the instructions or not. Defaults to False.
            is_method (bool, optional): Whether it is a method or not. Defaults to False.
            instance (_type_, optional): The instance. Defaults to None.
        """
        self.f = f
        self.name = f.__name__
        self._is_method = is_method
        self.engine = engine
        self._train = train

        self._docstring = inspect.getdoc(f)
        self.signature = str(inspect.signature(f))
        self.parameters = inspect.signature(f).parameters
        self.return_annotation = inspect.signature(f).return_annotation

        self.return_ = return_
        self._docstring_p = Param(
            name=self.name,
            instruction=self._docstring,
            training=train
        )
        self.out_cls = (
            self.return_annotation if self.return_annotation is not None 
            else AIResponse 
        )

        if reader is None:
            if self.out_cls in primitives:
                reader = PrimRead(name=self.name, out_cls=self.out_cls)
            elif issubclass(self.out_cls, Struct):
                reader = StructRead(name=self.name, out_cls=self.out_cls)
            else:
                reader = NullRead(name=self.name)
        
        self.reader = reader

        update_wrapper(self, f) 
        self.instance = instance
        self._stored = None
        self.dialog_factory = dialog_factory or Dialog
        self.ai_kwargs = ai_kwargs

    def spawn(
        self, 
        engine: AIModel=None, 
        dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
        train: bool=False
    ) -> 'SignatureFunc':
        """Spawn a new SignatureMethod. Especially use to create a trainable one

        Args:
            engine (AIModel, optional): Spawn a new . Defaults to None.
            dialog_factory (typing.Optional[typing.Callable[[], Dialog]], optional): _description_. Defaults to None.
            train (bool, optional): _description_. Defaults to False.

        Returns:
            SignatureMethod: _description_
        """
        return SignatureFunc(
            f=self.f,
            engine=engine or self.engine,
            is_method=self._is_method,
            dialog_factory=dialog_factory or self.dialog_factory,
            reader=self.resp_p,
            train=train,
            return_=self.return_,
            ai_kwargs=self.ai_kwargs
        )

    def i(self, *args, **kwargs) -> Instruction:
        """Get the instruction

        Returns:
            Instruction: Get the instruction
        """
        filled_docstring = self._docstring_p.render()

        filled = set()

        param_values = list(self.parameters.values())

        if self.return_ is not None:
            result = self.f(*args, **kwargs)
            if len(self.return_) == 1:
                values = {
                    self.return_[0]: result
                }
            else:
                values = dict(
                    zip(self.return_, result)
                )
        else:
            values = {}

        if self._is_method:            
            param_values = param_values[1:]

        # what if one of the parameters is an instruction?
        # values.update(dict(zip(args, [v for v in param_values])))
        
        if '{TEMPLATE}' in filled_docstring:
            values['TEMPLATE'] = self.reader.template()

        for value, param in zip(args, param_values):
            values[param.name] = value
            filled.add(param.name)
        
        for k, value in kwargs.items():
            param = self.parameters[k]
            values[param.name] = value
            filled.add(param.name)

        for param in param_values:
            if param.name in filled:
                continue
            if param.default == inspect.Parameter.empty:
                raise RuntimeError('Param has not been defined and no value')
            
            values[param.name] = param.default

        # TODO: Determine how to handle this
        out = validate_out(values)
        values = {key: render(v) for key, v in values.items()}
            # filled_docstring = filled_docstring.replace(
            #     f'{{{param.name}}}', 
            #     str(value) if not isinstance(value, Instruction) else render(value)
            # )
            # filled.add(param.name)
        # for k, value in kwargs.items():
        #     param = self.parameters[k]
            
        #     filled_docstring = filled_docstring.replace(
        #         f'{{{param.name}}}', # str(param.default)
        #         str(value) if not isinstance(value, Instruction) else render(value)
        #     )
        #     filled.add(param.name)

        # for param in param_values:
        #     if param.name in filled:
        #         continue
        #     if param.default == inspect.Parameter.empty:
        #         raise RuntimeError('Param has not been defined and no value')
        #     filled_docstring = filled_docstring.replace(
        #         f'{{{param.name}}}', str(param.default)
        #     )
        #     filled.add(param.name)

        filled_docstring = str_formatter(
            filled_docstring, required=False, **values
        )

        return Instruction(
            text=filled_docstring,
            out=self.reader, # NullRead(name=self.name)
            # out=StructFormatter(name=self.name, out_cls=self.out_cls)
        )

    def forward(
        self, *args,
        _engine: AIModel=None, **kwargs
    ) -> typing.Any:
        """Execute the instruction and get the output 

        Args:
            _engine (AIModel, optional): The engine to override with. Defaults to None.

        Returns:
            typing.Any: The result of processing the instruction
        """
        engine = _engine or self.engine

        instruction = self.i(*args,  **kwargs)
        result = engine(TextMessage('system', instruction), **self.ai_kwargs)
        if self.out_cls is AIResponse:
            return result
        return result.val
        # return self.reader.read(result.content)

    async def async_forward(
        self, *args, 
        _engine: AIModel=None, 
        **kwargs
    ) -> typing.Any:
        """Execute the instruction and get the output

        Args:
            _engine (AIModel, optional): The engine to override with. Defaults to None.

        Returns:
            typing.Any: The result of processing the instruction
        """
        return self.forward(
            *args, 
            _engine=_engine,
            **kwargs
        )

    def __get__(self, instance, owner):
        """Get the SignatureMethod with the instance set

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = SignatureFunc(
            self.f,
            self.engine,
            self.dialog_factory,
            self.reader,
            self._is_method,
            self._train,
            
            instance,
            self.return_,
            self.ai_kwargs

        )
        return self._stored


class InstructFunc(Module, Instruct):
    """InstructMethod is a method where you define the instruction by
    doing operations on that instructions
    """

    def __init__(
        self, f: typing.Callable, engine: AIModel, 
        dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
        is_method: bool=False,
        reader: Reader=None,
        instance=None,
        ai_kwargs=None
    ):
        """Create an InstructMethod that decorates a function that returns 
        an instruction

        Args:
            f (typing.Callable): The function to decorate
            train (bool, optional): Whether to train . Defaults to True.
            instance (, optional): The instance to use. Defaults to None.

        """
        self.f = f
        self._is_method = is_method

        self.engine = engine
        update_wrapper(self, f) 
        self.instance = instance
        self._stored = None
        self.dialog_factory = dialog_factory or Dialog

        # make it so it can automatically set this up
        # rather than using the "Null version"
        self.reader = reader or NullRead(f.__name__)
        self.ai_kwargs = ai_kwargs or {}

    def i(self, *args, **kwargs) -> Instruction:
        """Get the instruction based on the arguments

        Returns:
            Instruction: Get the instruction
        """
        return self.f(*args, **kwargs)

    def forward(self, *args, _engine: AIModel=None, **kwargs) -> typing.Any:        
        """Execute the instruct method and then process the output

        Args:
            _engine (AIModel, optional): Engine to override with. Defaults to None.

        Returns:
            typing.Any: The resulting
        """
        engine = _engine or self.engine
        instruction = self.i(*args, **kwargs)
        result = engine(TextMessage('system', instruction), **self.ai_kwargs)
        if self.out_cls is AIResponse:
            return result
        return result.val

        # instruction = self.i(*args, **kwargs)
        # # return engine(TextMessage('system', instruction)).val
        # return engine(TextMessage('system', instruction), **self.ai_kwargs).val
        # return result
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """

        Returns:
            typing.Any: 
        """
        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):
        """Get the SignatureMethod with the instance specified

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = InstructFunc(
            self.f, 
            self.engine, 
            self.dialog_factory, 
            self._is_method,
            reader=self.reader,
            instance=instance,
            ai_kwargs=self.ai_kwargs
        )
        return self._stored


def instructf(
    engine: AIModel=None, 
    resp_proc: Reader=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # TODO: Use wrapper
        return InstructFunc(f, engine, None, False, None, reader=resp_proc, ai_kwargs=ai_kwargs)
    return _


def instructmethod(
    engine: AIModel=None, 
    reader: Reader=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # TODO: Use wrapper
        return InstructFunc(f, engine, None, True, None, reader=reader, ai_kwargs=ai_kwargs)
    return _


def signaturemethod(
    engine: AIModel=None, 
    resp_p: Reader=None,
    return_: typing.List[str]=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return SignatureFunc(f, engine, None, True, reader=resp_p, return_=return_, ai_kwargs=ai_kwargs)

    return _


def signaturef(
    engine: AIModel=None, 
    reader: Reader=None,
    return_: typing.List[str]=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return SignatureFunc(f, engine, None, False, reader=reader, return_=return_, ai_kwargs=ai_kwargs)

    return _


# class FunctionDetails(object):
#     """FunctionDetails are used to convert the user input into an instruction 
#     """

#     def __init__(
#         self, func: typing.Callable, 
#         is_method: bool=False, 
#         train: bool=False
#     ):
#         """Create FunctionDetails based on the signature of that method or fucntion

#         Args:
#             func (typing.Callable): The function to get the details for
#             is_method (bool, optional): Whether it is a method or not. Defaults to False.
#             train (bool, optional): Whether to train or not train. Defaults to False.
#         """
#         # TODO: I don't want the return type to Out

#         self.func = func
#         self.name = func.__name__
#         self._docstring = inspect.getdoc(func)
#         self.signature = str(inspect.signature(func))
#         self.parameters = inspect.signature(func).parameters
#         self.return_annotation = inspect.signature(func).return_annotation

#         self._docstring_p = Param(
#             name=self.name,
#             instruction=self._docstring,
#             training=train
#         )

#         # origin, generic_type = self.get_generic_type()
#         self.out_cls = (
#             self.return_annotation if self.return_annotation is not None 
#             else str 
#         )# origin
#         # self.parameters = parameters
#         self._is_method = is_method

#     def forward(self, *args, **kwargs) -> Instruction:
#         """Get the instruction based on the input arguments

#         Raises:
#             RuntimeError: If a parameter is not defined or has no value

#         Returns:
#             Instruction: The resulting instruction
#         """
#         filled_docstring = self._docstring_p.render()

#         filled = set()

#         param_values = list(self.parameters.values())

#         if self._is_method:            
#             param_values = param_values[1:]

#         for value, param in zip(args, param_values):
            
#             filled_docstring = filled_docstring.replace(
#                 f'{{{param.name}}}', 
#                 str(value) if not isinstance(value, Instruction) else render(value)
#             )
#             filled.add(param.name)
#         for k, value in kwargs.items():
#             param = self.parameters[k]
            
#             filled_docstring = filled_docstring.replace(
#                 f'{{{param.name}}}', # str(param.default)
#                 str(value) if not isinstance(value, Instruction) else render(value)
#             )
#             filled.add(param.name)

#         for param in param_values:
#             if param.name in filled:
#                 continue
#             if param.default == inspect.Parameter.empty:
#                 raise RuntimeError('Param has not been defined and no value')
#             filled_docstring = filled_docstring.replace(
#                 f'{{{param.name}}}', str(param.default)
#             )
#             filled.add(param.name)

#         return Instruction(
#             text=filled_docstring,
#             out=NullRead(name=self.name)
#             # out=StructFormatter(name=self.name, out_cls=self.out_cls)
#         )

#     # def get_generic_type(self):
#     #     if self.return_annotation is not inspect.Signature.empty:
#     #         origin = getattr(self.return_annotation, '__origin__', None)
#     #         return origin
#     #         # if origin and issubclass(origin, Out):
#     #         #     args = self.return_annotation.__args__ if hasattr(self.return_annotation, '__args__') else ()
#     #         #     return origin, args[0] if args else None
#     #     return None #, None
