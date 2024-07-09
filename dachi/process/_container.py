from typing import Any
from ._core2 import Module
import asyncio

import typing


class Multi(Module):
    
    def __init__(self, module: Module, split_args: typing.Set[str]) -> None:
        super().__init__()

        self.module = module
        self.split_args = split_args

    def forward(self, n_splits: int, **kwargs) -> Any:
        
        result = []
        for i in range(n_splits):
            kw = {}
            for k, v in kwargs.items():
                if k in self.split_args:
                    count = (len(v) // n_splits)
                    if i == n_splits - 1:
                        kw[k] = v[i * count:]
                    else:
                        kw[k] = v[i * count:(i + 1) * count]
                else:
                    kw[k] = v
            result.append(self.module(**kw))
        return result
    
    async def async_forward(self, n_splits: int, **kwargs) -> Any:
        
        results = []
        async with asyncio.TaskGroup() as tg:
            tasks = []
            for i in range(n_splits):
                kw = {}
                for k, v in kwargs.items():
                    if k in self.split_args:
                        count = (len(v) // n_splits)
                        if i == n_splits - 1:
                            kw[k] = v[i * count:]
                        else:
                            kw[k] = v[i * count:(i + 1) * count]
                    else:
                        kw[k] = v
                
                tasks.append(tg.create_task(self.module.async_forward(**kw)))
            for task in tasks:
                result = await task
                results.append(result)
        return result


class Sequential(Module):
    
    def __init__(self, *module) -> None:
        super().__init__()

        self.modules = module

    def forward(self, *x) -> Any:
        
        first = False
        for module in self.modules:
            if first:
                x = module(*x)
            else:
                x = module(x)
        return x
