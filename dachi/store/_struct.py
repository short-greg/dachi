import pandas as pd
import typing
from dataclasses import dataclass, asdict, fields, field
import pydantic
from abc import abstractmethod

from pydantic import BaseModel, validator, ValidationError
from typing import Any, Dict


import csv
import pandas as pd
from io import StringIO


class Str(pydantic.BaseModel):

    text: str
    vars: typing.List[str]

    def forward(self, **kwargs):

        format = {}
        remaining_vars = []
        for var in self.vars:
            if var in kwargs:
                format[var] = kwargs[var]
            else:
                format[var] = f'{{var}}'
                remaining_vars.append(var)
        return Str(
            self.text.format(**format),
            remaining_vars
        )


class Struct(pydantic.BaseModel):

    @validator('*', pre=True, allow_reuse=True)
    def convert_to_string_template(cls, v, values, field):
        if field.outer_type_ is Str and not isinstance(v, Str):
            v = Str(v)
        return v

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    # fill in the "variables"
    def forward(self, **kwargs) -> 'Struct':
        pass


T = typing.TypeVar('T', bound=Struct)


class Message(Struct):

    role: Str
    text: Str

    # {role}: {text}


class Doc(Struct):

    name: Str
    text: Str


class StructList(Struct[T]):

    structs: typing.List[T]


class Chat(StructList[Message]):

    def filter(self, roles: typing.Iterable[str]) -> 'Chat[Message]':

        roles = set(roles)
        
        return Chat(
            s for s in self._structs if s.role in roles
        )




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

