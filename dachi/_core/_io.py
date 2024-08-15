# 1st party
import typing
import pydantic
import json

# 3rd party
from io import StringIO
import pandas as pd

# local
from ._core import (
    Out, StructList, 
    Struct, generic_class
)

from ._core import (
    Struct, 
    StructList, Result, 
    StructLoadException
)

S = typing.TypeVar('S', bound=Struct)


class ListOut(Result, typing.Generic[S]):
    """A list of outputs
    """

    name: str
    _out_cls: typing.Type[S] = pydantic.PrivateAttr()

    def __init__(self, out_cls: S, **data):
        super().__init__(**data)
        self._out_cls = out_cls

    def write(self, data: StructList[S]) -> str:
        """

        Args:
            data (StructList[S]): _description_

        Returns:
            str: the data written
        """
        return json.dumps(data)

    def read(self, data: str) -> StructList[S]:
        d = json.loads(data)
        structs = []
        for cur in d['structs']:
            structs.append(self._out_cls.from_dict(cur))

        return StructList(structs=structs)

    def to_text(self, data: StructList[S]) -> str:
        return data.to_text()

    def stream_read(self, data: str) -> S:

        d = json.loads(data)
        structs = []
        for cur in d['structs']:
            structs.append(self._out_cls.from_dict(cur))

        return StructList(structs=structs)
    
    def out_template(self) -> str:
        return self._out_cls.template()


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


class MultiOut(Result):
    """Concatenates multiple outputs Out into one output
    """
    
    outs: typing.List[Out]
    conn: str = '::OUT::{name}::\n'
    signal: str = '\u241E'

    def write(self, data: typing.List[Struct]) -> str:

        result = ''
        for struct, out in zip(data, self.outs):
            result = result + '\n' + self.signal + self.conn.format(name=out.name)
            result = f'{result}\n{struct.render()}'

        return result

    def read(self, data: str) -> typing.List[Struct]:

        structs = []

        d = data
        for out in self.outs:
            from_loc = d.find('\u241E')
            to_loc = d[from_loc + 1:].find('\u241E')
            cur = self.conn.format(name=out.name)
            data_loc = from_loc + len(cur) + 1
            if to_loc != -1:
                data_end_loc = from_loc + 1 + to_loc
            else:
                data_end_loc = None

            data_str = d[data_loc:data_end_loc]
            structs.append(out.read(data_str))
            d = d[data_end_loc:]

        return structs

    def to_text(self, data: typing.List[S]) -> str:

        text = ""
        for data_i, out in zip(data, self.outs):
            cur = data_i.render()
            cur_conn = self.conn.format(out.name)
            text += f"""
            {self.signal}{cur_conn}
            {cur}
            """
        return text

    def stream_read(self, data: str) -> typing.Tuple[S, bool, str]:
        structs = []

        d = data
        for i, t in enumerate(self.outs):
            from_loc = d.find('\u241E')
            to_loc = d[from_loc + 1:].find('\u241E')
            cur = self.conn.format(name=t.name)
            data_loc = from_loc + len(cur) + 1

            if to_loc != -1:
                data_end_loc = from_loc + 1 + to_loc
            else:
                data_end_loc = None

            data_str = d[data_loc:data_end_loc]
            try: 
                structs.append(t.read(data_str))
            except StructLoadException as e:
                return structs, i
            d = d[data_end_loc:]

        return structs, None
    
    def out_template(self) -> str:

        text = ""
        for out in self.outs:
            cur = out.out_template()
            cur_conn = self.conn.format(out.name)
            text += f"""
            {self.signal}{cur_conn}
            {cur}
            """
        return text


class JSONOut(Out):
    """
    """
    def stream_read(self, text: str) -> typing.Tuple[
        typing.Optional[typing.Dict], bool
    ]:
        """

        Args:
            text (str): 

        Returns:
            typing.Tuple[ typing.Optional[typing.Dict], bool ]: 
        """
        try: 
            result = json.loads(text)
            return result, True
        except json.JSONDecodeError:
            return None, False

    def read(self, text: str) -> typing.Dict:
        result = json.loads(text)
        return result

    def write(self, data: Struct) -> str:
        return data.to_text()
    
    def template(self, out_cls: Struct) -> str:
        return out_cls.template()
