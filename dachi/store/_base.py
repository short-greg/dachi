# 1st party
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields, MISSING, field
import typing
from typing_extensions import Self
from functools import wraps
import inspect
import math
from dataclasses import InitVar

# 3rd party
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from pydantic import Field

T = typing.TypeVar('T')

def conceptmethod(func: typing.Callable[..., T]) -> typing.Callable[..., T]:

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if not isinstance(cls, type):
            raise TypeError("This method can only be called on the class, not on instances.")
        return func(cls, *args, **kwargs)

    return classmethod(wrapper)


def abstractconceptmethod(func: typing.Callable[..., T]) -> typing.Callable[..., T]:
    func = abstractmethod(func)

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if not isinstance(cls, type):
            raise TypeError("This method can only be called on the class, not on instances.")
        return func(cls, *args, **kwargs)

    return classmethod(wrapper)
