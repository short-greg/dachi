# 1st party
from abc import abstractmethod, ABC
import asyncio
import typing
from typing_extensions import Self
from ._core2 import process

# 3rd party
import networkx as nx
import functools
import numpy as np

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


class Batch(object):

    def __init__(self, data: typing.Iterable, n_samples: int, shuffle: bool, drop_last: bool=True):

        self.data = data
        self.shuffle = shuffle
        self.n_samples = n_samples
        self.drop_last = drop_last

    def __iter__(self) -> typing.Iterator[typing.Any]:

        if self.shuffle:
            order = np.random.permutation(len(self.data))
        else:
            order = np.linspace(0, len(self.data))
        
        n_iterations = len(self.data) // self.n_samples
        if len(self.data) % self.n_samples != 0 and not self.drop_last:
            n_iterations += 1

        start = 0
        upto = self.n_samples
        for _ in range(n_iterations):
            cur_data = self.data[start:upto]
            yield cur_data
            start = upto
            upto += self.n_samples

@process
def batch(data: typing.Iterable, n_samples: int, shuffle: bool, drop_last: bool=True):
    
    return Batch(
        data, n_samples, shuffle, drop_last
    )


@process
def reduce(
    data: typing.Iterable, 
    module: Module, 
    *args, init=None, **kwargs
) -> typing.Any:
    """

    Args:
        data (typing.Iterable): 
        module (Module): 
        init (_type_, optional): . Defaults to None.

    Returns:
        typing.Any: 
    """
    cur = init
    for data_i in data:
        cur = module(cur, data_i)
    return cur


@process
def map(data: typing.Iterable, module: Module, *args, init=None, **kwargs) -> typing.Any:
    
    results = []
    for data_i in data:
        results.append(module(data_i, *args, **kwargs))
    return results


# # TODO: figure out how to do this
@map.async_
async def async_map(data: typing.Iterable, module: Module, *args, init=None, **kwargs):
    
    tasks: asyncio.Task = []
    async with asyncio.TaskGroup() as tg:
        for data_i in data:
            tasks.append(
                tg.create_task(module, data_i, *args, **kwargs)
            )
    
        return tuple(task.result() for task in tasks)
