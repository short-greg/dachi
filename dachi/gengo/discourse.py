import typing
from dataclasses import dataclass, Field

import typing
from dataclasses import dataclass, Field
from typing import Any, Dict
from ..comm._serving import Component, DataStruct


# class Text(DataStruct):
#     """A simple wrapper to use text as a prompt component
#     """

#     def __init__(self, text: str):
#         """Wrap the string 

#         Args:
#             text (str): The string to wrap
#         """
#         self.text = text

#     def as_dict(self) -> Dict:
#         return {
#             "text": self.text
#         }
    
#     def as_text(self, heading: str=None) -> str:
#         return self.structure(
#             self.text, heading
#         )




# Example
# Plan    
# Completion
# 

# @dataclass
# class Arg:
    
#     name: str
#     description: str = Field("")
    

# class Prompt(object):

#     def __init__(self, args: typing.List[typing.Union[Arg, str]], text: str):
        
#         super().__init__()
#         self._args = {}
#         for arg in args:
#             if isinstance(arg, str):
#                 self._args[arg] = Arg(arg)
#             else:
#                 self._args[arg.name] = arg
#         self._text = text

#     def format(self, **kwargs):

#         input_names = set(kwargs.keys())
#         difference = input_names - set(self._args)
#         if len(difference) != 0:
#             raise ValueError(f'Input has keys that are not arguments to the prompt')
#         inputs = {}
#         for k, v in self._args.items():
#             if k in kwargs:
#                 inputs[k] = v
#             else:
#                 inputs[k] = "{{}}"
#         return Prompt(
#             self._text.format(**inputs)
#         )
    
#     @property
#     def text(self) -> str:

#         return self._text


# class Example
# class Plan
# class Document
# class Map
# ...
