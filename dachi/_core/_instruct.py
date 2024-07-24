# 1st party
import typing
from functools import wraps, update_wrapper
import inspect
import string

# local
from ._core import Struct, Out, str_formatter, render, render_multi

from ._process import Module
from ._process import Param
import roman

from ._core import Instruction, Description, render


S = typing.TypeVar('S', bound=Struct)
X = typing.Union[str, Description, Instruction]


def bullet(xs: typing.Iterable[X], bullets: str='-', indent: int=0) -> 'Instruction':
    """

    Args:
        xs (typing.Iterable[Instruction]): 
        bullets (str, optional): . Defaults to '-'.

    Raises:
        RuntimeError: 

    Returns:
        Instruction: 
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
    """

    Args:
        xs (typing.Iterable[Instruction]): 
        indent (int, optional): . Defaults to 0.
        numbering (str, optional): . Defaults to 'arabic'.

    Returns:
        Instruction: 
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


def validate_out(instructions: typing.List[X]) -> Out:

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

    out = validate_out([x_instr])

    kwargs = dict(zip(kwargs.keys(), render_multi(kwargs.values())))
    args = render_multi(args)
    return Instruction(
        text=str_formatter(render(x_instr), *args, **kwargs), out=out
    )


def head(x: X, size: int=1) -> 'Instruction':

    out = validate_out([x])
    heading = '#' * size
    return Instruction(
        text=f'{heading} {render(x)}', out=out
    )


def section(name: X, details: X, size: int=1, linebreak: int=1) -> 'Instruction':

    heading = '#' * size
    out = validate_out([name, details])
    linebreak = '\n' * linebreak
    text = f'{heading} {render(name)}{linebreak}' + render(details)

    return Instruction(
        text=text, out=out
    )


def cat(xs: typing.List[Instruction], delim: str=' ') -> Instruction:
    """

    Args:
        by (str): 
        xs (typing.List[Instruction]): 

    Raises:
        RuntimeError: 

    Returns:
        Instruction: 
    """
    out = validate_out(xs)

    return Instruction(text=f'{delim}'.join(
        render(x_i) for x_i in xs
    ), out=out)


def join(x1: X, x2: X, delim: str=' ') -> Instruction:
    """

    Args:
        x1 : 
        x2 : 
        delim (str): 

    Returns:
        Instruction: 
    """
    out = validate_out([x1, x2])
    return Instruction(
        text=render(x1) + delim + render(x2),
        out=out
    )


class Operation(Module):

    def __init__(self, name: str, instruction: X):
        """

        Args:
            name (str): 
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

    if not isinstance(x, typing.Iterable):
        x = [x]

    out = validate_out([*x, instruction])
    resources = ', '.join(render_multi(x))
    # resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {render(instruction)} --- With Inputs: {resources}'
    return Instruction(
        text=text, out=out
    )


class OutF(Module, typing.Generic[S]):

    def __init__(
        self,
        name: str, 
        signature: str, 
        docstring: str, 
        parameters: typing.Dict,
        is_method: bool=False,
        out_cls: typing.Optional[typing.Type[S]] = None, 
        train: bool=True
    ):
        self.signature = signature
        self.docstring = docstring
        
        self.docstring = Param(
            train=train, name=name, 
            instruction=docstring
        )
        self.out_cls = out_cls
        self.parameters = parameters
        self._is_method = is_method

    def forward(self, *args, **kwargs) -> Instruction:
        filled_docstring = self.docstring.render()

        filled = set()

        param_values = list(self.parameters.values())
        print(param_values)
        if self._is_method:
            param_values = param_values[1:]

        for value, param in zip(args, param_values):
            
            
            filled_docstring = filled_docstring.replace(
                f'{{{param.name}}}', 
                str(value) if not isinstance(value, Instruction) else render(value)
            )
            filled.add(param.name)
        for k, value in kwargs.items():
            param = self.parameters[k]
            
            filled_docstring = filled_docstring.replace(
                f'{{{param.name}}}', # str(param.default)
                str(value) if not isinstance(value, Instruction) else render(value)
            )
            filled.add(param.name)
        for param in param_values:
            if param.name in filled:
                continue
            if param.default == inspect.Parameter.empty:
                raise RuntimeError('Param has not been defined and no value')
            filled_docstring = filled_docstring.replace(
                f'{{{param.name}}}', str(param.default)
            )
            filled.add(param.name)

        return Instruction(
            text=filled_docstring,
            out=self.out_cls
        )


class FunctionDetails:

    def __init__(self, func: typing.Callable):
        self.func = func
        self.name = func.__name__
        self.docstring = inspect.getdoc(func)
        self.signature = str(inspect.signature(func))
        self.parameters = inspect.signature(func).parameters
        self.return_annotation = inspect.signature(func).return_annotation
        if (
            self.return_annotation is not inspect.Signature.empty 
            and not issubclass(self.return_annotation, Out)
        ):
            raise TypeError(f"Expected return type {Out}, got {type(self.return_annotation)} instead")

    def get_generic_type(self):
        if self.return_annotation is not inspect.Signature.empty:
            origin = getattr(self.return_annotation, '__origin__', None)
            if origin and issubclass(origin, Out):
                args = self.return_annotation.__args__ if hasattr(self.return_annotation, '__args__') else ()
                return origin, args[0] if args else None
        return None, None
    
    def out(self, is_method: bool=False, train: bool=True) -> Out:        
        
        origin, generic_type = self.get_generic_type()
        if origin:
            if generic_type:
                return OutF(
                    name=self.name,
                    signature=self.signature,
                    docstring=self.docstring, 
                    parameters=self.parameters,
                    out_cls=self.return_annotation,
                    train=train,
                    is_method=is_method
                )
            else:
                return OutF(
                    name=self.name,
                    signature=self.signature, 
                    docstring=self.docstring, 
                    out_cls=origin,
                    parameters=self.parameters,
                    train=train,
                    is_method=is_method
                )
        return OutF(
            name=self.name,
            signature=self.signature,
            docstring=self.docstring, 
            parameters=self.parameters,
            out_cls=None, 
            train=train,
            is_method=is_method
        )


class _SignatureMethod(Module):

    def __init__(
        self, f: typing.Callable, details: FunctionDetails, 
        train: bool=True, is_method: bool=False, instance=None
    ):
        """

        Args:
            f (typing.Callable): 
            details (FunctionDetails): 
            train (bool, optional): . Defaults to True.
            instance (_type_, optional): . Defaults to None.

        Raises:
            TypeError: 

        Returns:
            _type_: 
        """
        self.f = f
        self._details = details
        self._train = train
        self._out = details.out(is_method, train)

        update_wrapper(self, f) 
        self.instance = instance
        self._stored = None
        self._is_method = is_method

    def forward(self, *args, **kwargs) -> typing.Any:        

        return self._out(*args, **kwargs)

    async def async_forward(self, *args, **kwargs) -> typing.Any:

        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):

        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = _SignatureMethod(
            self.f, self._details, self._train,
            instance
        )
        return self._stored
    

def instructf(train: bool=True):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):
        details = FunctionDetails(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        if hasattr(f, '__self__') or '__self__' in dir(f):
            return _SignatureMethod(f, details, train)
        else:
            return _SignatureMethod(wrapper, details, train)
    return _
