import pandas as pd
import typing
from dataclasses import dataclass, asdict, fields, field
import pydantic
from abc import abstractmethod


import csv
import pandas as pd
from io import StringIO


class Struct(pydantic.BaseModel):

    pass

    # def __post_init__(self, style: str=None):

    #     self.style = style

    # @property
    # def field_names(self) -> typing.List[str]:

    #     return [field[0].name for field in fields(self)]

    # def to_series(self):
    #     return pd.Series(asdict(self))

    # def to_dict(self, flat: bool=True):

    #     result = {}
        
    #     for k, v in asdict(self).items():
    #         if isinstance(v, Struct) and not flat:
    #             result[k] = v.to_dict()
    #         if isinstance(v, ListStruct) and not flat:
    #             result[k] = v.to_list()
    #         else:
    #             result[k] = v
        
    #     return result

    # # # Think about how to do this best
    # # # I need the styling here
    # def to_text(self):

    #     result = ''
        
    #     for k, v in asdict(self).items():

    #         if isinstance(v, Struct) or isinstance(v, ListStruct):
    #             result = f'{k}: {v.to_text()}'
    #         else:
    #             result = f'{k}: {v}'
        
    #     return result

    # @classmethod
    # def from_text(self):
    #     pass


@dataclass
class Message(Struct):

    role: str
    text: str

    # {role}: {text}


@dataclass
class Doc(Struct):

    name: str
    text: str


T = typing.TypeVar('T', bound=Struct)


class StructList(Struct[T]):

    structs: typing.List[T]


class Chat(StructList[Message]):

    def filter(self, roles: typing.Iterable[str]) -> 'Chat[Message]':

        roles = set(roles)
        
        return Chat(
            s for s in self._structs if s.role in roles
        )


class InstructChat(Chat):

    def __init__(self, structs: typing.List[Message] = None, instructor: str='system'):
        super().__init__(structs)
        self.instructor = instructor

    def chat(self) -> Chat:

        return Chat(

            s for s in self._structs if s.role != self.instructor
        )


class Role(Struct):

    name: str
    descr: str = field(default_factory=lambda: '')


class Text(Struct):

    text: str = ''
    descr: str = field(default_factory=lambda: '')


class Body(Struct):

    sep_before: str='-------'
    sep_after: str='-------'

    def fill(self, struct: Struct):
        pass


# class Instruct(Struct):

#     name: str
#     inputs: typing.List[Struct]
#     descr: str


# class Op(Instruct):

#     output: Struct = field(default_factory=Text)


# class OpFactory(object):

#     def __init__(self, name: str):

#         self.name = name

#     def __call__(self, inputs, descr, output) -> Op:

#         return Op(self.name, inputs, descr, output)


# class _Op:

#     def __getattr__(self, key):

#         return OpFactory(key)


# op = _Op()



S = typing.TypeVar('S', bound=Struct)


# TODO: FINISH THIS
class Style(typing.Generic[S]):

    @abstractmethod
    def write(self, text: str):
        pass
    
    @abstractmethod
    def read(self, text: str) -> S:
        pass



class CSVStyle(Style[StructList]):

    def __init__(self, delim: str=','):

        self.delim = delim

    def read(self, text: str) -> StructList:
        
        io = StringIO(text)
        df = pd.read_csv(io)
        return StructList(structs=df.to_dict())

    def write(self, struct: StructList) -> str:

        d = struct.dict()
        df = pd.DataFrame(d.structs)
        io = StringIO()
        return df.to_csv(io)


class KVStyle(Style):

    def __init__(self, sep: str='::'):

        self.sep = sep

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass


class ListStyle(Style, typing.Generic[S]):

    def __init__(self, sep: str='::'):

        self.sep = sep

    def read(self, text: str):
        
        lines = text.split('\n')
        for line in lines:
            idx, value = line.split('::')
            idx = int(idx)
            value = value.strip()

    def write(self, struct: Struct):

        pass


class TextTemplateStyle(Style):

    def __init__(self, template: str):

        self.template = template

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass


class Context(object):
    
    def __init__(self, struct: Struct, style: Style=None):

        self.struct = struct
        self.style = style

    def __call__(self, style_override: Style=None):

        if style_override is None:
            return self.style.write(self.struct)
        
        return style_override.write(self.struct)


C = typing.TypeVar('C', bound=Struct)


class ContextF(typing.Generic[C]):
    
    def __init__(self, structf: typing.Callable[[], C], style: Style=None):

        self.structf = structf
        self.style = style

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> Context:
        return Context(
            self.structf(*args, **kwargs), self.style
        )


class Var(object):

    def __init__(
        self, producer, incoming: typing.Union['Var', typing.Tuple['Var']]=None
    ):
        self.producer = producer
        self.incoming = incoming

    def detach(self) -> 'Var':

        return Var(self.producer)


class Output(typing.Generic[S]):

    def __init__(
        self, producer, incoming: typing.Union['Var', typing.Tuple['Var']]=None,
        style: Style=None
    ):
        self.producer = producer
        self.style = style
        self.incoming = incoming

    def detach(self) -> 'Output':

        return Output(self.producer)


class Param(typing.Generic[S]):

    def __init__(
        self, struct: S
    ):
        self.struct = struct

    def detach(self) -> 'Param':

        return Param(self.struct)



# TODO: Implement Input/Output, Var
# # Parameters

# 

class Input(object):
    # 
    # 
    pass

class Output(object):
    pass


class Body(object):
    pass


def instrmethod(f):

    def _(self, *args, **kwargs):

        output = f(self, *args, **kwargs)

    return _


def instr(f):

    def _(*args, **kwargs):

        # 
        output = f(*args, **kwargs)

    return _


# .parameters() => 

# can I inherit from "nn.Module?"
# 
# # Have to inherit from Tensor, text.Tensor (?)
# # no grads
# # Don't need backward()
# # 

# add in "ask", ""
# input -> style what comes in
# output -> 
#   # self.output - use this to embed in the context
#          return self.output(value)
#   # # may have compound "output"
#   # # self.output = 
#   # forward decorator.. 
#   # Could use a class decorator also
#   self.forward = self.context.decorate(self.forward)
# context
#   - 
# parameters -> how to get the parameters
#   - make operations "member" variables
#   - otherwise not treated as context
#   - parameters converted to YAML

# def forward(self, ):
#    
#    
#    
#    return self.context(do)

# 

# return self.context.embed([outputs])

# Context
#  - Instance: Defined on the instance. Set an instance variable
#  - Class: Defined on the class. Does not update
#  - Func: Receive a material in the function that it is set with
#      - I think Func and Instance can be the same
#      - 
#   Context()  # 
#   ContextF(...) # context factory 
#   1. Defined on the instance
#   2. Retrieved in the 
#   # pass in the input
#   x = self.contextf(x) # how about for outputs?
#   # If there is no context it will still be used
#       Example, Template, 
#   Context
#   ContextF() # instance conctext
#   Body()
#   Input() => 
#   Output() => Show the template for the output
#      
#      self.output(...) # 
#   # Ignore if not used
    
#    self.role  = self.role(...)
#    role = self.role(...)
#  -    If "split", the output of one function may also
#       need a func material?


# 
# YAML
# CSV
# List
# Keyvals


# class Assistant(object):


# S = typing.TypeVar('S', bound=Struct)


# class ListStruct(BaseStruct, typing.Generic[S]):

#     def __init__(self, structs: typing.List[S]=None, style: str=None, inline: str=None):

#         self._structs = structs or []
#         self.style = style 

#     def to_list(self, flat: bool=True) -> typing.List:

#         if flat:
#             return self._structs
        
#         result = []
#         for struct in self._structs:
#             result.append(struct.to_dict(False))
#         return result
    
#     def to_df(self) -> pd.DataFrame:

#         return pd.DataFrame(
#             [s.to_dict() for s in self._structs]
#         )

#     # want to add styling to the text
#     def to_text(self) -> str:
#         pass

#     def to_text(self):

#         result = ''
        
#         for k, v in asdict(self).items():

#             if isinstance(v, Struct) or isinstance(v, ListStruct):
#                 result = f'{k}: {v.to_text()}'
#             else:
#                 result = f'{k}: {v}'
        
#         return result
    
#     @classmethod
#     def from_text(self):
#         pass

