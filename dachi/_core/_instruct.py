# 1st party
import typing
from typing_extensions import Self
from abc import abstractmethod, ABC
from functools import wraps, update_wrapper
import inspect
import string


# 3rd party
import pydantic
from pydantic import Field

# local
from . import Struct, Str, StructList
from ..converse import Module
from ._core2 import Param, processf, _DecMethod


S = typing.TypeVar('S', bound=Struct)

X = typing.Union[Str, str, 'Instruction']


class Description(Struct):
    """Provide context in the prompt template
    """
    name: str

    @property
    @abstractmethod
    def text(self) -> str:
        pass

    @abstractmethod
    def update(self, **kwargs) -> Self:
        pass


# Do i want this to be a Pydantic model

class Ref(Struct):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    reference: Description

    @property
    def name(self) -> str:
        return self.reference.name

    @property
    def text(self) -> str:
        return ""

    def update(self, **kwargs) -> Self:
        # doesn't do anything since
        # it is a reference
        return self


class Style(pydantic.BaseModel, typing.Generic[S], ABC):

    @abstractmethod
    def forward(self, struct: S) -> str:
        pass

    @abstractmethod
    def reverse(self, text: str) -> S:
        pass

    def load(self, text: str) -> S:
        return self.reverse(text)

    def __call__(self, struct: S) -> str:
        return self.forward(struct)

    @pydantic.field_validator('*', mode='before')
    def convert_to_string_template(cls, v, info: pydantic.ValidationInfo):
    
        outer_type = cls.model_fields[info.field_name].annotation
        if (inspect.isclass(outer_type) and issubclass(outer_type, Str)) and not isinstance(v, Str):
            return Str(text=v)
        return v


def generic_class(t: typing.TypeVar, idx: int=0):

    return t.__orig_class__.__args__[idx]


class _PartialFormatter(string.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, format_string, **kwargs):
        self.kwargs = kwargs
        return super().format(format_string)

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return self.kwargs.get(key, '{' + key + '}')
        return super().get_value(key, args, kwargs)

    def __call__(self, format_string, **kwargs):
        return self.format(format_string, **kwargs)

str_formatter = _PartialFormatter()


class Out(Struct, ABC):

    name: str
    data: S

    @abstractmethod
    def read(self, data) -> S:
        pass

    @abstractmethod
    def write(self, data) -> str:
        pass

    @abstractmethod
    def template(self) -> str:
        pass


class JSON(Out[S]):

    fields: typing.List[str] = None

    def read(self, data) -> S:
        pass

    def write(self, data) -> str:
        pass

    def template(self) -> str:
        
        if self.fields is not None:
            pass


class CSV(Out[S]):

    fields: typing.List[str] = None

    def read(self, data) -> S:
        pass

    def write(self, data) -> str:
        pass

    def template(self) -> str:

        if self.fields is not None:
            pass


class Merged(Out['StructList']):

    conn: str = '===={name}===='
    
    def read(self, data) -> 'StructList':
        return StructList.load(data)

    def write(self, data) -> str:
        struct_list = StructList.load(data)
        return struct_list.dump()

    def template(self) -> str:
        pass


class Instruction(Struct):
    """Specific instruction for the model to use
    """
    text: typing.Union[str, Str]
    out: Out = None
    # incoming: typing.List['Description'] = Field(default_factory=list)

    def traverse(self, visited: typing.Set=None) -> typing.Iterator['Description']:

        visited = visited or set()

        if id(self) in visited:
            return

        yield self
        visited.add(id(self))

        if self.incoming is None:
            return

        for inc in self.incoming:
            for inc_i in inc.traverse(visited):
                yield inc_i
    
    def __call__(self, **kwargs) -> str:

        return self.text(**kwargs).text


def bullet(xs: typing.Iterable[Instruction], bullets: str='-') -> 'Instruction':

    out = None
    text = f'\n {bullets}'
    for x_i in xs:
        if x_i.out is not None and out is None:
            out = x_i.out
        elif x_i.out is not None:
            raise RuntimeError('The output has already been defined')
    text = text + f'\n {bullets}'.join(
        x_i.to_text() for x_i in xs
    )
    
    return Instruction(
        text=text, out=out
    )


def formatted(x: Instruction, format: str) -> 'Instruction':

    text = x.text()
    if text[:len(format)] == format and text[-len(format):] == format:
        return x
    return Instruction(
        f'{format}{text}{format}',
        out=x.out
    )


def numbered(xs: typing.Iterable[Instruction], starting: str=1) -> 'Instruction':

    out = None
    text = ''
    cur = starting
    for i, x_i in enumerate(xs):
        if x_i.out is not None and out is None:
            out = x_i.out
        elif x_i.out is not None:
            raise RuntimeError('The output has already been defined')
        text = f' {cur}. {x_i.text()}'
        cur += 1
    
    return Instruction(
        text=text, 
        out=out
    )


def fill(x: Instruction, **kwargs) -> 'Instruction':

    return Instruction(
        str_formatter(x.text, **kwargs), 
    )


def head(x: Instruction, size: int=1) -> 'Instruction':

    heading = '#' * size
    return Instruction(
        f'{heading} {x.text()}', out=x.out
    )


def validate_out(instructions: typing.List[Instruction]) -> Out:

    out = None
    for instruction in instructions:
        if out is None and instruction.out is not None:
            out = instruction.out
        elif instruction.out is not None:
            raise RuntimeError(f'Out cannot be duplicated')
    return out


def section(name: Instruction, details: Instruction, size: int=1) -> 'Instruction':

    heading = '#' * size
    out = validate_out([name, details])
    text = f'{heading} {name.text()}\n\n' + details.text()

    return Instruction(
        text=text, out=out
    )


def join(by: str, xs: typing.List[Instruction]) -> Instruction:
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
        x_i.text() for x_i in xs
    ), out=out)


def join(instruction1: Instruction, instruction2: Instruction, delim: str='\n') -> Instruction:
    """

    Args:
        instruction1 : 
        instruction2 : 
        delim (str): 

    Returns:
        Instruction: 
    """
    out = validate_out([instruction1, instruction2])
    return Instruction(
        instruction1.text() + delim + instruction2.text(),
        out=out
    )


class Operation(Module):

    def __init__(self):
        pass

    def forward(
        self, *x: Description, **kwargs
    ) -> Instruction:
        pass


def op(x: typing.Union[typing.List[Description], Description], intruction: str, name: str) -> Instruction:

    if isinstance(x, Description) or isinstance(x, Ref):
        x = [x]
    
    resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {intruction} --- With Inputs: {resources}'
    return Instruction(
        name=name, text=text, incoming=x
    )


class _SignatureMethod(Module):

    def __init__(self, f: typing.Callable, details: typing.Dict, train: bool=True, instance=None):
        self.f = f
        self.details = get_function_details(f)
        update_wrapper(self, f) 
        self._details = details
        self.instance = instance
        self._train = train
        self._stored = None
        return_annotation = self.details["return_annotation"]
        if (
            return_annotation is not inspect.Signature.empty 
            and not issubclass(return_annotation, Out)
        ):
            raise TypeError(f"Expected return type {Out}, got {type(return_annotation)} instead")

    def forward(self, *args, **kwargs) -> typing.Any:        
        pass

    async def async_forward(self, *args, **kwargs) -> typing.Any:

        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):

        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = _SignatureMethod(self.f, instance)
        return self._stored
    
    @classmethod
    def async_(cls, f):

        cls._async_f = f


def get_function_details(func: typing.Callable):
    docstring = inspect.getdoc(func)
    signature = inspect.signature(func)
    
    parameters = signature.parameters
    return_annotation = signature.return_annotation
    
    func_name = func.__name__
    
    return {
        "name": func_name,
        "docstring": docstring,
        "parameters": parameters,
        "return_annotation": return_annotation
    }


def signaturef(train: bool=True):

    def _(f):

        details = get_function_details(f)
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        if hasattr(f, '__self__') or '__self__' in dir(f):
            return _SignatureMethod(f, details, train)
        else:
            return _SignatureMethod(wrapper, details, train)


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


