# 1st party
import typing
import pydantic
import json

# 3rd party
from io import StringIO
import pandas as pd

# local
from ._core import (
    Reader, 
    Struct, generic_class,
    unescape_curly_braces
)

from ._core import (
    Struct, 
    StructList, 
    Reader, 
    StructLoadException,
    render
)

S = typing.TypeVar('S', bound=Struct)


class StructRead(Reader, typing.Generic[S]):
    """Use for converting an AI response into a struct
    """
    _out_cls: typing.Type[S] = pydantic.PrivateAttr()

    def __init__(self, out_cls: S, **data):
        super().__init__(**data)
        self._out_cls = out_cls

    def dump_data(self, data: S) -> typing.Any:
        return data.to_dict()

    def write_text(self, data: typing.Any) -> str:
        return str(data)

    def to_text(self, data: S) -> str:
        return data.to_text(True)

    def read_text(self, message: str) -> S:
        message = unescape_curly_braces(message)
        return json.loads(message)
        # return self._out_cls.from_text(message, True)
    
    def load_data(self, data) -> S:
        return self._out_cls.from_dict(data)

    def template(self) -> str:
        return self._out_cls.template()
    
    # def example(self, data: S) -> str:
    #     return data.to_text(True)

    # def stream_read(self, message: str) -> S:
    #     return self._out_cls.from_text(message, True)


class StructListRead(Reader, typing.Generic[S]):
    """Convert the output into a list of structs
    """
    name: str
    _out_cls: typing.Type[S] = pydantic.PrivateAttr()

    def __init__(self, out_cls: S, **data):
        """Create a converter specifying the struct class to convert
        to

        Args:
            out_cls (S): The class to convert to
        """
        super().__init__(**data)
        self._out_cls = out_cls

    def dump_data(self, data: StructList[S]) -> typing.Any:
        structs = []
        for cur in data.structs:
            structs.append(self._out_cls.to_dict(cur))

        return structs

    def write_text(self, data: typing.List) -> str:
        return json.dumps(data)

    def read_text(self, message: str) -> StructList[S]:
        """Convert the message into a list of structs

        Args:
            message (str): The AI message to read

        Returns:
            StructList[S]: the list of structs
        """
        return json.loads(message)
    
    def load_data(self, data) -> StructList[S]:
        structs = []
        for cur in data['structs']:
            structs.append(self._out_cls.from_dict(cur))

        return StructList(structs=structs)

    # TODO: Shouldn't this be "render?"
    def to_text(self, data: StructList[S]) -> str:
        """Convert the data to a string

        Args:
            data (StructList[S]): The data to convert to text

        Returns:
            str: the data converted to a string
        """
        return data.to_text()

    def template(self) -> str:
        """Output a template for the struct list

        Returns:
            str: A template of the struct list
        """
        # TODO: This is currently not correct
        #   Needs to output as a list

        return self._out_cls.template()

    # def example(self, data: StructList[S]) -> str:
    #     """Output an example of the output

    #     Args:
    #         data (StructList[S]): The data to output

    #     Returns:
    #         str: the data written
    #     """
    #     return json.dumps(data)

    # def stream_read(self, message: str) -> StructList[S]:
    #     """

    #     Args:
    #         message (str): _description_

    #     Returns:
    #         S: The resulting struct
    #     """
    #     d = json.loads(message)
    #     structs = []
    #     for cur in d['structs']:
    #         structs.append(self._out_cls.from_dict(cur))

    #     return StructList(structs=structs)
    

class CSVRead(Reader):
    """Convert text to a StructList
    """
    indexed: bool = True
    delim: str = ','

    def read_text(self, message: str):

        io = StringIO(message)
        df = pd.read_csv(io, sep=self.delim)
        return df.to_dict(orient='records', index=False)

    def load_data(self, data) -> typing.Dict:
        """Convert the message to a StructList

        Args:
            message (str): The message to convert

        Returns:
            StructList[S]: The result
        """
        return data # StructList[S].load_records(data)

    def dump_data(self, data: typing.List) -> typing.Any:
        return data

    def write_text(self, data: typing.Any) -> str:
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

    def template(self) -> str:
        """Output a template for the CSV

        Returns:
            str: The template for the CSV
        """
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

    # def stream_read(self, message: str) -> 'StructList[S]':
    #     """Convert a message to a struct list one by one

    #     Args:
    #         message (str): The message to convert

    #     Returns:
    #         StructList[S]: The resulting StructList
    #     """
    #     io = StringIO(message)
    #     df = pd.read_csv(io, sep=self.delim)
    #     records = df.to_dict(orient='records', index=False)

    #     return StructList[S].load_records(records)

    # def example(self, data: StructList[S]) -> str:
    #     """Create an example of the CSV

    #     Args:
    #         data (StructList[S]): The data to create an example for

    #     Returns:
    #         StructList[S]: The resulting StructList
    #     """
    #     io = StringIO()
    #     data = [
    #         d_i.dump()
    #         for d_i in self.data.structs
    #     ]
    #     df = pd.DataFrame(data)
    #     df.to_csv(io, index=True, sep=self.delim)
        
    #     # Reset the cursor position to the beginning of the StringIO object
    #     io.seek(0)
    #     return io.read()


class DualRead(Reader):

    text: Reader
    data: Reader

    def read_text(self, message: str) -> typing.Dict:
        return self.text.read_text(message)
    
    def load_data(self, data) -> typing.Any:
        return self.data.load_data(data)

    def dump_data(self, data: typing.Any) -> typing.Any:
        return self.data.dump_data(data)

    def write_text(self, data: typing.Any) -> str:
        return self.text.write_text(data)

    def template(self) -> str:
        return self.text.template()


class KVRead(StructRead):

    sep: str = '::'
    key_descr: typing.Optional[typing.Dict] = None
    
    def read_text(self, message: str) -> typing.Dict:
        lines = message.splitlines()
        result = {}
        for line in lines:
            try:
                key, value = line.split(self.sep)
                result[key] = value
            except ValueError:
                pass
        return result
    
    def load_data(self, data: typing.Dict) -> typing.Dict:
        """

        Args:
            data (typing.Dict): 

        Returns:
            typing.Dict: 
        """
        return data

    def dump_data(self, data: typing.Dict) -> typing.Dict:
        """

        Args:
            data (typing.Dict): 

        Returns:
            typing.Dict: 
        """
        return data

    def write_text(self, data: typing.Dict) -> str:
        """

        Args:
            data (typing.Dict): 

        Returns:
            str: 
        """
        return '\n'.join(
            f'{k}{self.sep}{render(v)}' for k, v in data.items()
        )

    def template(self) -> str:
        """

        Returns:
            str: 
        """
        if self.key_descr is None:
            key_descr = {
                '<Example>': '<The value for the key.>'
            }

        else:
            key_descr = self.key_descr
        return '\n'.join(
            f'{key}{self.sep}{value}' 
            for key, value in key_descr.items()
        )

    # def stream_read(self, message: str) -> 'StructList[S]':
    #     pass


class MultiRead(Reader):
    """Concatenates multiple outputs Out into one output
    """
    outs: typing.List[Reader]
    conn: str = '::OUT::{name}::\n'
    signal: str = '\u241E'

    def dump_data(self, data: typing.Dict) -> typing.Any:
        
        results = []
        for data_i, out in zip(data['data'], self.outs):
            results.append(out.dump_data(data_i))
        return {'data': results, 'i': data['i']}

    def write_text(self, data: typing.Dict) -> str:
        result = ''
        for d, out in zip(data['data'], self.outs):
            result = result + '\n' + self.signal + self.conn.format(name=out.name)
            result = f'{result}\n{render(d)}'

        return result

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

    def read_text(self, message: str) -> typing.Dict:
        """_summary_

        Args:
            message (str): _description_

        Returns:
            typing.Dict: _description_
        """
        structs = []

        d = message
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
                structs.append(t.read_text(data_str))
            except StructLoadException as e:
                return {'data': structs, 'i': i}
            d = d[data_end_loc:]

        return {'data': structs, 'i': i}
    
    def load_data(self, data) -> typing.List:
        """

        Args:
            data (): 

        Returns:
            typing.List: 
        """
        structs = []

        for o, d_i in zip(self.outs, data['data']):
            structs.append(o.load_data(d_i))

        return {'data': structs, 'i': data['i']}

    def template(self) -> str:

        text = ""
        for out in self.outs:
            cur = out.template()
            cur_conn = self.conn.format(out.name)
            text += f"""
            {self.signal}{cur_conn}
            {cur}
            """
        return text

    # def example(self, data: typing.List[Struct]) -> str:

    #     result = ''
    #     for struct, out in zip(data, self.outs):
    #         result = result + '\n' + self.signal + self.conn.format(name=out.name)
    #         result = f'{result}\n{struct.render()}'

    #     return result

    # def read(self, message: str) -> typing.List[Struct]:

    #     structs = []

    #     d = message
    #     for out in self.outs:
    #         from_loc = d.find('\u241E')
    #         to_loc = d[from_loc + 1:].find('\u241E')
    #         cur = self.conn.format(name=out.name)
    #         data_loc = from_loc + len(cur) + 1
    #         if to_loc != -1:
    #             data_end_loc = from_loc + 1 + to_loc
    #         else:
    #             data_end_loc = None

    #         data_str = d[data_loc:data_end_loc]
    #         structs.append(out.read(data_str))
    #         d = d[data_end_loc:]

    #     return structs


class PrimRead(Reader):
    """Use for converting an AI response into a primitive value
    """

    _out: typing.Type

    def __init__(
        self, out: typing.Type,
        **data
    ):
        """

        Args:
            out (typing.Type): 
        """
        super().__init__(**data)
        self._out = out

    def to_text(self, data: typing.Any) -> str:
        return str(data)

    def dump_data(self, data: typing.Any) -> typing.Any:
        return data

    def write_text(self, data: typing.Any) -> str:
        return str(data)

    def read_text(self, message: str) -> typing.Any:
        return self._out(message)
    
    def load_data(self, data) -> typing.Any:
        return data

    def template(self) -> str:
        return f'<{self._out}>'

    # def example(self, data) -> str:
    #     return str(data)

    # def stream_read(self, message: str) -> typing.Any:
    #     return self._out(message)


class JSONRead(StructRead):
    """

    """
    def stream_read(self, message: str) -> typing.Tuple[
        typing.Optional[typing.Dict], bool
    ]:
        """

        Args:
            text (str): 

        Returns:
            typing.Tuple[ typing.Optional[typing.Dict], bool ]: 
        """
        try: 
            result = json.loads(message)
            return result, True
        except json.JSONDecodeError:
            return None, False

    def read(self, message: str) -> typing.Dict:
        result = json.loads(message)
        return result

    def dump_data(self, data: typing.Any) -> typing.Any:
        return data.to_dict()

    def write_text(self, data: typing.Any) -> str:
        return str(data)

    # def example(self, data: Struct) -> str:
    #     return data.to_text()
    
    def template(self, out_cls: Struct) -> str:
        return out_cls.template()
