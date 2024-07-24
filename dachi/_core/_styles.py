# 1st party
import typing

# 3rd party
from io import StringIO
import pandas as pd

# local
from ._core import Style, StructList, Struct


S = typing.TypeVar('S', bound=Struct)


class CSV(Style[StructList[S]]):

    delim: str = ','
    fields: typing.List[str]
    
    def dumps(self) -> str:
        io = StringIO()
        data = [
            d_i.dump()
            for d_i in self.data.structs
        ]
        df = pd.DataFrame(data)
        df.to_csv(io, index=False, sep=self.delim)

        # Reset the cursor position to the beginning of the StringIO object

        io.seek(0)  
        return io.read()

    @classmethod
    def loads(cls, data: str, delim: str=',') -> StructList[S]:
        io = StringIO(data)
        data = pd.read_csv(io, sep=delim)

        return StructList(structs=data)

    @classmethod
    def template(cls) -> str:

        pass
        # if self.fields is not None:
        #     pass


class Merged(Style['StructList']):

    conn: str = '===={name}===='
    
    def read(self, data) -> 'StructList':
        return StructList.loads(data)

    def write(self, data) -> str:
        struct_list = StructList.loads(data)
        return struct_list.dumps()

    def template(self) -> str:
        pass
