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


class CSV(ListOut, typing.Generic[S]):

    out_cls: StructList[S]
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


class KV(Out, typing.Generic[S]):

    delim: str = '::'
    
    def read(self, data: str) -> 'StructList[S]':
        pass
    
    def stream_read(self, data: str) -> 'StructList[S]':
        pass

    def write(self, data: StructList[S]) -> StructList[S]:
        pass

    def out_template(self) -> str:
        pass


# class Out(object):

#     def __init__(self, types: typing.List[Struct]):
        
#         pass



# class Merged(Style['StructList']):

#     names: typing.List[str] = None
#     conn: str = 'OUT::{name}::'
    
#     @pydantic.field_validator('names', 'data', 'types', mode='before')
#     def validate_names_types_data(cls, values):
#         names = values.get('names', [])
#         types = values.get('types', [])
#         data = values.get('data', None)
        
#         if len(names) != len(types):
#             raise ValueError("The number of names must match the number of types")

#         if data and len(data.structs) != len(names):
#             raise ValueError("The number of names and types must match the number of structs")
        
#         return values

#     @classmethod
#     def loads(cls, data: str) -> 'Merged':
#         lines = data.strip().split('\n')
#         names = []
#         structs = StructList(structs=[])
        
#         for i in range(0, len(lines), 2):
#             conn_line = lines[i]
#             struct_line = lines[i + 1]
            
#             name = conn_line.split('::')[1]
#             names.append(name)
            
#             struct = Struct.from_text(struct_line)
#             structs.add_struct(struct)
        
#         instance = cls()
#         instance.names = names
#         instance.data = structs
#         return instance

#     @classmethod
#     def stream_loads(cls, stream: typing.Iterable[str], types: typing.List[S]) -> typing.Tuple['Merged', int, bool]:
#         names = []
#         structs = StructList(structs=[])
#         lines = []
#         read_count = 0
#         ended_in_failure = False
        
#         i = 0
#         for line in stream:
#             lines.append(line.strip())
#             if len(lines) == 2:
#                 try:
#                     conn_line, struct_line = lines
#                     try:
#                         name = conn_line.split('::')[1]
#                         names.append(name)
#                     except IndexError:
#                         raise ValueError("Invalid format for connection line")

#                     # Create a Struct object from the struct line
#                     types[i].reads()
#                     struct = Struct.from_text(struct_line)
#                     structs.add_struct(struct)

#                     read_count += 1
#                     lines = []
#                 except Exception as e:
#                     ended_in_failure = True
#                     break

#         instance = cls()
#         instance.names = names
#         instance.data = structs
#         return instance, read_count, ended_in_failure

#     def dumps(self) -> str:
#         result = ''
#         if self.names is None:
#             names = []
#         else:
#             names = self.names
#         if len(names) < self.data.structs:
#             residual = range(len(names), len(self.data.structs))
#             names = [*names, *residual]
#         for struct, name in zip(self.data.structs, names):
#             result = result + self.conn.format(name=name)
#             result = f'{result}\n{struct.render()}'

#         return result

#     @classmethod
#     def template(cls) -> str:
#         pass
