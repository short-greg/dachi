# 1st party
from abc import abstractmethod, ABC
import asyncio
import typing
from typing_extensions import Self

# 3rd party
import networkx as nx
import functools

import uuid
from dataclasses import dataclass

# from ._core2 import Module, Args, TBase, Src, T, MultiT, IdxSrc



# loop => probe until finished
# Add in "completed"
# 


# class StreamT(TBase):

#     def __init__(
#         self, src: Streamable, val=None,
#         multi: bool=False, name: str=None, annotation: str=None
#     ):

#         self._src = src
#         self._val = val
#         self._multi = multi
#         self._name = name
#         self._annnotation = annotation

#     def label(self, name: str=None, annotation: str=None) -> Self:

#         if name is not None:
#             self._name = name
#         if annotation is not None:
#             self._annnotation = annotation

#     @property
#     def val(self) -> typing.Any:
#         return self._val

#     @property
#     def undefined(self) -> bool:

#         return self._val is None

#     def __getitem__(self, idx: int) -> 'T':
        
#         if not self._multi:
#             raise RuntimeError(
#                 'Object T does not have multiple objects'
#             )
#         else:
#             val = None
#         return T(
#             # have to specify it is "streaming"
#             val, IdxSrc(self, idx)
#         )
    
#     def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:

#         if self._val is not None:
#             return self._val
        
#         if

#     def detach(self):
#         return T(
#             self._val, None, self._multi
#         )

# if not completed the value in by is "Partial"
# 


# class StreamSrc(object):

#     def __init__(self, module: 'StreamModule', args: typing.List['Args']) -> None:
#         super().__init__()

#         self._module = module
#         self._args = args

#     def incoming(self) -> typing.Iterator['TBase']:
        
#         for incoming in arg.incoming:
#             yield incoming

#     def forward(self, by: typing.Dict['TBase', typing.Any]) -> typing.Any:

#         pass        
#         # args = [arg(by) for arg in self.args]
#         # return self._module(args)
    
#     def __getitem__(self, idx) -> 'Module':

#         return self._module[idx]

#     def __iter__(self) -> typing.Iterator['Module']:

#         for mod_i in self._module:
#             yield mod_i

#     def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
#         return self.forward(by)


# class StreamT(TBase):

#     def __init__(self, vals: typing.Tuple, src: 'StreamSrc'):
        
#         # TODO: I 
#         self._vals = vals
#         self._src = src

#     @property
#     def undefined(self) -> bool:

#         return functools.reduce(lambda x, y: x and y.undefined, self._ts)

#     def t(self) -> 'T':

#         pass        
#         # return T(
#         #     self._vals[idx], MultiIdxSrc(self, idx),
#         #     self._src[idx].multi
#         # )

#     def probe(self, by: typing.Dict['T', typing.Any]) -> typing.Any:

#         pass        
#         # if self._vals is not None:
#         #     return self._vals

#         # if self in by:
#         #     return by[self]
    
#         # if self._src is not None:
#         #     for incoming in self._src.incoming():
#         #         incoming.probe(by)
#         #     by[self] = self.src(by)
#         #     return by[self]
        
#         # raise RuntimeError('Val has not been defined and no source for t')

#     def detach(self):
#         return MultiT(
#             self._vals, None, self._multi
#         )


# class StreamModule(object):

#     def __init__(self, streamable: Streamable, args: Args):

#         self.streamable = streamable
#         self.args = args

#     def forward(self, args: Args) -> typing.Any:
#         pass

#     def __iter__(self) -> typing.Iterator[MultiT]:

#         for module_i in self._modules:
#             yield module_i

#     @abstractmethod
#     def __call__(self, args: typing.List[Args]) -> 'MultiT':
#         pass


# class StreamGroup(object):

#     def __init__(self):
#         pass

#     # What if the user forgets to add a T?
#     # What if the user adds a T from another "stream"
#     def add(self, module: Module, args: Args) -> 'T':
#         pass


# class stream_group(object):
#     # use to nest async operations

#     def __init__(self, streamable: Streamable, args: Args):
        
#         self._group = StreamGroup(streamable, args)

#     def __enter__(self):
        
#         return self._group

#     def __exit__(self, exc_type, exc_val, exc_tb):
        
#         self._group._exited = True
#         self._group()

#         if exc_type is not None:
#             raise exc_val



