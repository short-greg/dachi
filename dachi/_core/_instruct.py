# 1st party
import typing
from typing_extensions import Self
from abc import abstractmethod, ABC
from functools import wraps, update_wrapper
import inspect
import string
from io import StringIO
import json
import pandas as pd

# local
from ._core import Struct, StructList, Out, str_formatter, to_text

from ._process import Module
from ._process import Param
import roman

from ._core import Instruction, Description

S = typing.TypeVar('S', bound=Struct)
X = typing.Union[str, Description, Instruction]


# class _PartialFormatter(string.Formatter):
#     def __init__(self):
#         super().__init__()

#     def format(self, format_string, **kwargs):
#         self.kwargs = kwargs
#         return super().format(format_string)

#     def get_value(self, key, args, kwargs):
#         if isinstance(key, str):
#             return self.kwargs.get(key, '{' + key + '}')
#         return super().get_value(key, args, kwargs)

#     def __call__(self, format_string, **kwargs):
#         return self.format(format_string, **kwargs)


# str_formatter = _PartialFormatter()



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
        to_text(x_i) for x_i in xs
    )
    return Instruction(
        text=text, out=out
    )


def formatted(x: X, format: str) -> 'Instruction':

    text = to_text(x)
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
        text = f'{indent}{number}. {to_text(x_i)}'
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


def fill(x_instr: X, **kwargs) -> 'Instruction':

    out = validate_out([x_instr])
    print(to_text(x_instr))
    return Instruction(
        text=str_formatter(to_text(x_instr), **kwargs), out=out
    )


def head(x: X, size: int=1) -> 'Instruction':

    out = validate_out([x])
    heading = '#' * size
    return Instruction(
        f'{heading} {to_text(x)}', out=out
    )


def section(name: X, details: X, size: int=1) -> 'Instruction':

    heading = '#' * size
    out = validate_out([name, details])
    text = f'{heading} {to_text(name)}\n\n' + to_text(details)

    return Instruction(
        text=text, out=out
    )


def cat(by: str, xs: typing.List[Instruction]) -> Instruction:
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

    return Instruction(f'{by}'.format(
        to_text(x_i) for x_i in xs
    ), out=out)


def join(x1: X, x2: X, delim: str='\n') -> Instruction:
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
        to_text(x1) + delim + to_text(x2),
        out=out
    )


class Operation(Module):

    def __init__(self, name: str, instruction: X):
        """

        Args:
            name (str): 
        """
        self.name = name
        self.instruction = instruction
        
    def forward(
        self, **kwargs: X
    ) -> Instruction:
        """Fill in the instruction with the inputs

        Returns:
            Instruction: 
        """
        instruction = to_text(self.instruction)
        out = validate_out(
            [*kwargs.values(), self.instruction]
        )

        kwargs = to_text(kwargs.values(), ref=True)

        return Instruction(
            text=fill(instruction, **kwargs), out=out
        )


def op(x: typing.Union[typing.Iterable[X], X], instruction: X) -> Instruction:

    if not isinstance(x, typing.Iterable):
        x = [x]

    out = validate_out([*x, instruction])
    resources = ', '.join(to_text(x, ref=True))
    # resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {to_text(instruction)} --- With Inputs: {resources}'
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
        out_cls: typing.Optional[typing.Type[S]] = None, 
        train: bool=True
    ):
        self.signature = signature
        self.docstring = docstring
        
        self.docstring = Param(
            train=train, name=name, 
            text=docstring
        )
        self.out_cls = out_cls
        self.parameters = parameters

    def forward(self, *args, **kwargs) -> Instruction:
        filled_docstring = self.docstring

        filled = set()

        for value, param in zip(args, self.parameters.values()):
            
            filled_docstring = filled_docstring.replace(
                f'{{{param.name}}}', 
                str(value) if not isinstance(value, Instruction) else to_text(value)
            )
            filled.add(param.name)
        for k, value in kwargs.items():
            param = self.parameters[k]
            filled_docstring = filled_docstring.replace(
                f'{{{param.name}}}', # str(param.default)
                str(value) if not isinstance(value, Instruction) else to_text(value)
            )
            filled.add(param.name)
        for param in self.parameters.values():
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
    
    def out(self, train: bool=True) -> Out:        
        
        origin, generic_type = self.get_generic_type()
        if origin:
            if generic_type:
                return OutF(signature=self.signature, docstring=self.docstring, out_cls=self.return_annotation)
            else:
                return OutF(signature=self.signature, docstring=self.docstring, out_cls=origin)
        return OutF(signature=self.signature, docstring=self.docstring, out_cls=None, train=train)


class _SignatureMethod(Module):

    def __init__(
        self, f: typing.Callable, details: FunctionDetails, 
        train: bool=True, instance=None
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
        self._out = details.out(train)

        update_wrapper(self, f) 
        self.instance = instance
        self._stored = None

    def forward(self, *args, **kwargs) -> typing.Any:        

        return self._out(*args, **kwargs)

    async def async_forward(self, *args, **kwargs) -> typing.Any:

        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):

        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = _SignatureMethod(self.f, instance)
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


    # # Instruction
    # incoming: typing.List['Description'] = Field(default_factory=list)    
    # def __call__(self, **kwargs) -> str:

    #     return self.text(**kwargs).text

    # def traverse(
    #     self, 
    #     visited: typing.Set=None
    # ) -> typing.Iterator['Description']:
    #     """_summary_

    #     Yields:
    #         Description: 
    #     """

    #     visited = visited or set()

    #     if id(self) in visited:
    #         return

    #     yield self
    #     visited.add(id(self))

    #     if self.incoming is None:
    #         return

    #     for inc in self.incoming:
    #         for inc_i in inc.traverse(visited):
    #             yield inc_i


# def get_function_details(func: typing.Callable):
#     docstring = inspect.getdoc(func)
#     signature = inspect.signature(func)
    
#     parameters = signature.parameters
#     return_annotation = signature.return_annotation
    
#     func_name = func.__name__
    
#     return {
#         "name": func_name,
#         "docstring": docstring,
#         "parameters": parameters,
#         "return_annotation": return_annotation
#     }

# CSV[S]
# class Output(typing.Generic[S], Struct):
#     name: str
#     style: Style = None
#     def read(self, text: str) -> S:

#         if self.style is not None:
#             return self.style.load(text)
        
#         Scls = generic_class(self)
#         return Scls.from_text(text)
    
#     @property
#     def text(self) -> str:

#         Scls = generic_class(self)
#         return f"""
#         {self.instruction.text}

#         ===JSON OUTPUT TEMPLATE BELOW===

#         {Scls.template()}
#         """


# class InstructionSet(Struct):

#     instructions: typing.List[Instruction]

#     # def update(self, **kwargs) -> 'ISet':
        
#     #     return ISet(
#     #         [instruction.update(**kwargs) for instruction in self.instructions]
#     #     )
    
#     # @property
#     # def text(self) -> str:
#     #     return '\n\n'.join(
#     #         i.text for i in self.instructions
#     #     )
#     def __call__(self, **kwargs) -> typing.List['str']:
        
#         return [i(**kwargs) for i in self.instructions]
#         # return self.instr(**kwargs)


# class Composition(Description):
#     """
#     """
#     descriptions: typing.List[Description]

#     @property
#     def text(self) -> str:

#         return "\n".join(
#             description.text for description in self.descriptions
#         )

#     def update(self, **kwargs) -> Self:
        
#         descriptions = []
#         for description in self.descriptions: 
#             descriptions.append(description.update(**kwargs))
#         return Composition(
#             descriptions=descriptions, name=self.name
#         )




# def merge_out()


# def traverse(*instructions: Instruction) -> typing.Iterator[Description]:

#     visited = set()
#     for instruction in instructions:
#         for description in instruction.traverse(visited):
#             yield description


# class OutputList(Struct):
    
#     outputs: typing.List[Output]
#     header: Instruction = None
#     footer: Instruction = None

#     def read(self, text: str) -> typing.List:

#         results = []
#         # 1) split by header
#         func_locs = []

#         for output in self.outputs:
#             cur = f"::OUT::{output.name}::"
#             loc = text.find(cur)
#             func_locs.append(
#                 (loc, loc + len(cur))
#             )
#         func_locs.append((-1, -1))

#         for output, p1, p2 in zip(self.outputs, func_locs[:-1], func_locs[1:]):
            
#             print(p1, p2)
#             _ = text[p1[0]:p1[1]]
#             func_response = text[p1[1]:p2[0]]
#             print(func_response)
#             results.append(
#                 output.read(func_response)
#             )
        
#         return results        

#     @property
#     def text(self) -> str:
        
#         output_templates = []
#         for output in self.outputs:
#             output_templates.append(
#                 f"""
#                 ===OUTPUT TEMPLATE {output.name}===

#                 :::OUT:::{output.name}

#                 {output.text}
#                 """
#             )

#         out_text = ''
#         if self.header is not None:
#             out_text = f"""
#             {self.header.text}

#             """
#         out_text = f"""
#         {'\n\n'.join(output_templates)}
#         """

#         if self.footer is not None:
#             out_text = f"""
#             {out_text}

#             {self.footer.text}
#             """
#         return out_text


# class Instructor(Module):

#     @abstractmethod
#     def forward(self, *args, **kwargs) -> Output:
#         pass


# class Parameter(Description):

#     description: Description
#     value: str = None

#     @property
#     def text(self) -> str:
        
#         if self.value is None:
#             return self.description.text
#         return self.value

#     def update(self, **kwargs) -> Self:
        
#         return Parameter(
#             name=self.name,
#             description=self.description.update(**kwargs),
#             value=None
#         )

#     def to_dict(self) -> typing.Dict:
#         return {
#             self.name: {
#                 'name': self.description.name,
#                 'base': self.description.text,
#                 'value': (
#                     self.value 
#                     if self.value is not None 
#                     else self.description.text
#                 )
#             }
#         }




# class JSON(Style[S]):

#     fields: typing.List[str] = None

#     def dumps(self) -> str:
#         pass

#     @classmethod
#     def loads(cls, data: str) -> Self:
#         return cls(**data)

#     def template(self) -> str:
        
#         if self.fields is not None:
#             pass

