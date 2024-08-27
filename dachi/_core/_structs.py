# 1st party
import typing
from abc import ABC, abstractmethod

from ._core import (
    Struct, Renderable
)
from ._utils import generic_class
from ._ai import Message

# 3rd party
import pydantic

# 1st party
import typing
from abc import ABC, abstractmethod
from typing import get_type_hints
from typing import Self
import typing

from uuid import uuid4
from enum import Enum
import asyncio
from dataclasses import dataclass

import inspect
import json

# 3rd party
import pydantic

# local
from ._utils import (
    generic_class
)


S = typing.TypeVar('S', bound=Struct)


class Media:

    descr: str
    data: str


# TODO: Decide whether to keep this

class StructList(Struct, typing.Generic[S]):
    """
    """

    structs: typing.List[S]

    def __getitem__(self, key) -> typing.Any:
        """

        Args:
            key (_type_): 

        Returns:
            typing.Any: 
        """
        return self.structs[key]
    
    def __setitem__(self, key: typing.Optional[int], value: S) -> typing.Any:
        """Set a value in the 

        Args:
            key (str): The key for the value to set
            value : The value to set

        Returns:
            S: the value that was set
        """
        if key is None:
            self.structs.append(value)
        else:
            self.structs[key] = value
        return value
    
    @classmethod
    def load_records(cls, records: typing.List[typing.Dict]) -> 'StructList[S]':
        """Load the struct list from records

        Args:
            records (typing.List[typing.Dict]): The list of records to load

        Returns:
            StructList[S]: The list of structs
        """
        structs = []
        struct_cls: typing.Type[Struct] = generic_class(S)
        for record in records:
            structs.append(struct_cls.load(record))
        return StructList[S](
            structs=structs
        )


class Description(Struct, Renderable, ABC):
    """Provide context in the prompt template
    """
    name: str = pydantic.Field(description='The name of the description.')

    @abstractmethod
    def render(self) -> str:
        pass


class Ref(Struct):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    desc: Description

    @property
    def name(self) -> str:
        """Get the name of the ref

        Returns:
            str: The name of the ref
        """
        return self.desc.name

    def render(self) -> str:
        """Generate the text rendering of the ref

        Returns:
            str: The name for the ref
        """
        return self.desc.name


class MediaMessage(Message):

    def __init__(self, source: str, media: typing.List[Media]):
        """

        Args:
            source (str): 
            media (typing.List[Media]): The media to use
        """
        super().__init__(
            source=source,
            data={
                'media': media
            }
        )

    def render(self) -> str:
        """Render the media message

        Returns:
            str: The rendered message
        """
        return f'{self.source}: Media [{self.media}]'



# class StructList(Struct, typing.Generic[S]):
#     """
#     """

#     structs: typing.List[S]

#     def __init__(self, structs: typing.Iterable):
#         """

#         Args:
#             structs (typing.Iterable): 
#         """
#         super().__init__(structs=structs)

#     def __getitem__(self, key) -> typing.Any:
#         """

#         Args:
#             key (_type_): 

#         Returns:
#             typing.Any: 
#         """
#         return self.structs[key]
    
#     def __setitem__(self, key: typing.Optional[int], value: S) -> typing.Any:
#         """Set a value in the 

#         Args:
#             key (str): The key for the value to set
#             value : The value to set

#         Returns:
#             S: the value that was set
#         """
#         if key is None:
#             self.structs.append(value)
#         else:
#             self.structs[key] = value
#         return value
    
#     @classmethod
#     def load_records(cls, records: typing.List[typing.Dict]) -> 'StructList[S]':
#         """Load the struct list from records

#         Args:
#             records (typing.List[typing.Dict]): The list of records to load

#         Returns:
#             StructList[S]: The list of structs
#         """
#         structs = []
#         struct_cls: typing.Type[Struct] = generic_class(S)
#         for record in records:
#             structs.append(struct_cls.load(record))
#         return StructList[S](
#             structs=structs
#         )

