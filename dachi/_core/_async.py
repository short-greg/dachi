# 1st party
import asyncio
import typing
from ._process import processf

# 3rd party
# import networkx as nx
# import functools
# import numpy as np
# from ._process import Batch, batchf

# local
from ._core import Args
from ._process import Module, ParallelModule


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
                    tg.create_task(module_i.async_forward(
                        *args_i.args, **args_i.kwargs
                    ))
                )

        return tuple(
            t.result() for t in tasks
        )

    def forward(self, args: Args) -> typing.Tuple:
        """
        Returns:
            typing.Tuple: The output for the paralell module
        """
        return asyncio.run(self._forward(args))


class MultiModule(ParallelModule):
    """A type of Parallel module that simply
    makes multiple calls
    """
    def forward(self, *args: Args) -> typing.Tuple:
        """

        Returns:
            typing.Tuple: 
        """
        return tuple(
            module_i.forward(*arg_i.args, **arg_i.kwargs)
            for module_i, arg_i in zip(self._modules, args)
        )


@processf
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


@processf
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


def parallel(m: Module, *args, **kwargs) -> typing.List:
    pass

