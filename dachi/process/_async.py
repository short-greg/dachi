# 1st party
from abc import abstractmethod, ABC
import asyncio
import typing
from typing_extensions import Self


# 3rd party
import networkx as nx
import functools

import uuid


from ._core2 import Module, Args, T, Src, WrapSrc, WrapIdxSrc, MultiT


class Asyncable(object):

    @abstractmethod
    def __async_call__(self, args: Args) -> 'T':
        pass


class ParallelModule(ABC):

    def __init__(self, modules: typing.List[Module]):

        self._modules = modules

    @abstractmethod
    def forward(self, args: typing.List[Args]) -> typing.Tuple:
        pass

    def __getitem__(self, idx: int) -> 'Module':

        return self._modules[idx]

    def __iter__(self) -> typing.Iterator['Module']:

        for module_i in self._modules:
            yield module_i

    @abstractmethod
    def __call__(self, args: typing.List[Args]) -> 'MultiT':
        pass


class AsyncModule(ParallelModule):

    def forward(self, args: typing.List[Args]) -> typing.Tuple:
        
        tasks = []
        with asyncio.TaskGroup() as tg:
            
            for module_i, args_i in zip(self._modules, args):
                
                tasks.append(
                    tg.create_task(module_i.async_forward(args_i))
                )
        
        return tuple(
            t.result() for t in tasks
        )

    def __call__(self, args: typing.List[Args]) -> typing.Tuple:
        
        tasks = []
        with asyncio.TaskGroup() as tg:
            
            for module_i, args_i in zip(self._modules, args):
                
                tasks.append(
                    tg.create_task(module_i.__async_call__(args_i))
                )
        
        return tuple(
            t.result() for t in tasks
        )


class ParallelSrc(WrapSrc):

    def __init__(self, module: 'ParallelModule', args: typing.List['Args']) -> None:
        super().__init__()
        self._module = module
        self._args = args

    def incoming(self) -> typing.Iterator['T']:
        
        for arg in self._args:
            for incoming in arg.incoming:
                yield incoming

    def forward(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        args = [arg(by) for arg in self.args]
        return self._module(args)
    
    def __getitem__(self, idx) -> 'Module':

        return self._module[idx]

    def __iter__(self) -> typing.Iterator['Module']:

        for mod_i in self._module:
            yield mod_i

    def __call__(self, by: typing.Dict['T', typing.Any]) -> typing.Any:
        
        return self.forward(by)


class AsyncGroup(object):

    def __init__(self):
        
        self._modules = []
        self._args = []
        self._exited = False
        self._t = None

    def add(self, module: Module, *args, **kwargs):
        
        self._modules.append(module)
        self._args.append(Args(*args, **kwargs))

    def __call__(self):
        
        if not self._exited:
            raise RuntimeError('Group has not exited the context.')

        module = AsyncModule(
            self._modules
        )
        self._t = module(self._args)

    def t(self) -> 'MultiT':
        return self._t


class async_group(object):
    # use to nest async operations

    def __init__(self):
        
        self._group = AsyncGroup()

    def __enter__(self):
        
        return self._group

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        self._group._exited = True
        self._group()

        if exc_type is not None:
            raise exc_val

