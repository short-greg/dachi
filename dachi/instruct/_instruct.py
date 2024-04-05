# 1st party
from dataclasses import dataclass, field
import typing
from abc import abstractmethod
from io import StringIO

# 3rd party
import pandas as pd
import pydantic

# local
from ..store import Struct, Str



class Role(Struct):

    name: str
    descr: str = field(default_factory=lambda: '')


class Text(Struct):

    text: str = ''
    descr: str = field(default_factory=lambda: '')


class Body(Struct):

    sep_before: str='-------'
    sep_after: str='-------'

    def fill(self, struct: Struct):
        pass
