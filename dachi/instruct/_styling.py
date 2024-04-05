from io import StringIO
import typing

import pandas as pd

from ._core import StructList, Style, Struct, S


class CSVStyle(Style[StructList]):

    def __init__(self, delim: str=','):

        self.delim = delim

    def read(self, text: str) -> StructList:
        
        io = StringIO(text)
        df = pd.read_csv(io)
        return StructList(structs=df.to_dict())

    def write(self, struct: StructList) -> str:

        d = struct.dict()
        df = pd.DataFrame(d.structs)
        io = StringIO()
        return df.to_csv(io)


class KVStyle(Style):

    def __init__(self, sep: str='::'):

        self.sep = sep

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass


class ListStyle(Style, typing.Generic[S]):

    def __init__(self, sep: str='::'):

        self.sep = sep

    def read(self, text: str):
        
        lines = text.split('\n')
        for line in lines:
            idx, value = line.split('::')
            idx = int(idx)
            value = value.strip()

    def write(self, struct: Struct):

        pass


class TextTemplateStyle(Style):

    def __init__(self, template: str):

        self.template = template

    def read(self, struct_cls: typing.Type[Struct]):
        pass

    def write(self, struct: Struct):

        pass

