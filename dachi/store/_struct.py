import pandas as pd
import typing
from dataclasses import dataclass, asdict, fields, field
import pydantic


# TODO: FINISH THIS
class Style(object):

    def __init__(self, template: str):

        self.template = template

    def __call__(self, **vars):

        return self.template(**vars)

    @classmethod
    def create(self, style):

        if isinstance(style, Style):
            return style
        elif isinstance(style, str):
            return Style(style)

        elif style is None:
            pass


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


class CSVStyle(Style):

    def __init__(self, delim: str=','):

        self.delim = delim

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass


class KVStyle(Style):

    def __init__(self, sep: str='::'):

        self.sep = sep

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass


class ListStyle(Style):

    def __init__(self, sep: str='::'):

        self.sep = sep

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass


class MarkdownStyle(Style):

    def __init__(self, template: str):

        self.template = template

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass



# 
# YAML
# CSV
# List
# Keyvals


def instrmethod(f):

    def _(self, *args, **kwargs):

        output = f(self, *args, **kwargs)

    return _


def instr(f):

    def _(*args, **kwargs):

        # 
        output = f(*args, **kwargs)

    return _


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

