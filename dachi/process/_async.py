# 1st party
from abc import abstractmethod, ABC
import asyncio
import typing
from typing_extensions import Self

# 3rd party
import networkx as nx
import functools

# local
from ._core2 import Module, Args, T, Src, ParallelModule


class AsyncModule(ParallelModule):
    """A type of Parallel module that makes use of 
    Python's Async
    """

    async def _forward(self, args: typing.List[Args]) -> typing.Tuple:
        """The asynchronous method to use for inputs

        Args:
            args (typing.List[Args]): The list of inputs to the modules 

        Returns:
            typing.Tuple: The output to the paralell module
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:
            
            for module_i, args_i in zip(self._modules, args):
                
                tasks.append(
                    tg.create_task(module_i.async_forward(args_i))
                )

        return tuple(
            t.result() for t in tasks
        )

    def forward(self, *args: Args) -> typing.Tuple:
        """
        Returns:
            typing.Tuple: The output for the paralell module
        """
        return asyncio.run(self._forward(args))

    def forward(self, *args: Args) -> typing.Tuple:
        """

        Returns:
            typing.Tuple: 
        """
        return tuple(
            module_i.forward(*arg_i.args, **arg_i.kwargs)
            for module_i, arg_i in zip(self._modules, args)
        )


# class AsyncGroup(object):

#     def __init__(self):
        
#         self._modules = []
#         self._args = []
#         self._exited = False
#         self._t = None

#     def add(self, module: Module, *args, **kwargs):
        
#         self._modules.append(module)
#         self._args.append(Args(*args, **kwargs))

#     def __call__(self):
        
#         if not self._exited:
#             raise RuntimeError('Group has not exited the context.')

#         module = AsyncModule(
#             self._modules
#         )
#         self._t = module(self._args)

#     def t(self) -> T:

#         if not self._exited:
#             return None

#         return self._t


# class async_group(object):
#     # use to nest async operations

#     def __init__(self):
        
#         self._group = AsyncGroup()

#     def __enter__(self) -> AsyncGroup:
        
#         return self._group

#     def __exit__(self, exc_type, exc_val, exc_tb):
        
#         self._group._exited = True
#         self._group()

#         if exc_type is not None:
#             raise exc_val
