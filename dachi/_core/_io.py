# 1st party
import typing
import csv
import pydantic


# 3rd party
from io import StringIO
import pandas as pd

# local
from ._core import (
    Out, ListOut, StructList, 
    Struct, generic_class
)

S = typing.TypeVar('S', bound=Struct)


class CSV(ListOut):

    indexed: bool = True
    delim: str = ','

    def read(self, data: str) -> 'StructList[S]':
        io = StringIO(data)
        df = pd.read_csv(io, sep=self.delim)
        records = df.to_dict(orient='records', index=False)

        return StructList[S].load_records(records)
    
    def stream_read(self, data: str) -> 'StructList[S]':
        io = StringIO(data)
        df = pd.read_csv(io, sep=self.delim)
        records = df.to_dict(orient='records', index=False)

        return StructList[S].load_records(records)

    def write(self, data: StructList[S]) -> StructList[S]:
        io = StringIO()
        data = [
            d_i.dump()
            for d_i in self.data.structs
        ]
        df = pd.DataFrame(data)
        df.to_csv(io, index=True, sep=self.delim)
        
        # Reset the cursor position to the beginning of the StringIO object
        io.seek(0)
        return io.read()

    def out_template(self) -> str:
        s_cls: typing.Type[Struct] = generic_class(S)
        template = s_cls.template()

        result = []
        header = ['Index']
        first = [1]
        last = ['N']
        mid = '...'
        for k, v in template.items():
            header.append(v['name'])
            first.append(f'{v['description']} {v['type_']}')
            last.append(f'{v['description']} {v['type_']}')
        header = f'{self.delim} '.join(header)
        first = f'{self.delim} '.join(first)
        last = f'{self.delim} '.join(first)

        result = [header, first, mid, last]
        return '\n'.join(result)


class KV(Out):

    delim: str = '::'
    
    def read(self, data: str) -> 'StructList[S]':
        pass
    
    def stream_read(self, data: str) -> 'StructList[S]':
        pass

    def write(self, data: StructList[S]) -> StructList[S]:
        pass

    def out_template(self) -> str:
        pass
