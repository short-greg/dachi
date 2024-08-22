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
    render,
    escape_curly_braces
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
        print(message)
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
    cols: typing.Optional[typing.List[typing.Tuple[str, str, str]]] = None

    def read_text(self, message: str):

        io = StringIO(message)
        df = pd.read_csv(io, sep=self.delim)
        return df.to_dict(orient='records', index=True)

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
        # s_cls: typing.Type[Struct] = generic_class(S)
        # template = s_cls.template()
        if self.cols is None:
            cols = [['1', 'First Col', ''], ['2', 'Second Col', ''], ['...', '...', ''], ['N', 'Nth Col', '']]
        else:
            cols = self.cols

        result = []
        header = ['Index']
        first = ['1']
        last = ['N']
        mid = '...'
        for name, descr, type_ in cols:
            header.append(name)
            first.append(f'{descr} <{type_}>')
            last.append(f'{descr} <{type_}>')
        header = f'{self.delim}'.join(header)
        first = f'{self.delim}'.join(first)
        last = f'{self.delim}'.join(last)
        result = [header, first, mid, last]
        result = '\n'.join(result)
        return result

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


class KVRead(Reader):

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

    def dump_data(self, data: typing.Dict) -> typing.Dict:
        """Dump all of the data to a dictionary

        Args:
            data (typing.Dict): A dictionary containing all
            of the data

        Returns:
            typing.Dict: The data all dumped
        """
        results = []
        for data_i, out in zip(data['data'], self.outs):
            results.append(out.dump_data(data_i))
        return {'data': results, 'i': data['i']}

    def write_text(self, data: typing.Dict) -> str:
        """Convert the dictionary data to text

        Args:
            data (typing.Dict): The dictionary data obtained from dump_data

        Returns:
            str: The data written as text
        """
        result = ''
        for d, out in zip(data['data'], self.outs):
            result = result + '\n' + self.signal + self.conn.format(name=out.name)
            result = f'{result}\n{render(d)}'

        return result

    def to_text(self, data: typing.List[S]) -> str:
        """Convert the data to text
        """

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
        """

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
    
    def load_data(self, data: typing.Dict) -> typing.Dict:
        """Load the data for each of the components of the output

        Args:
            data (): The dictionary data to load

        Returns:
            typing.Dict: The data with each element of the output
             loaded
        """
        structs = []

        for o, d_i in zip(self.outs, data['data']):
            structs.append(o.load_data(d_i))

        return {'data': structs, 'i': data['i']}

    def template(self) -> str:
        """Output a template for the output

        Args:
            data (): The dictionary data to load

        Returns:
            typing.Dict: The data with each element of the output
             loaded
        """
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

    _out_cls: typing.Type

    def __init__(
        self, out_cls: typing.Type,
        **data
    ):
        """Create a reader for Primitive values

        Args:
            out (typing.Type): The type of data
        """
        super().__init__(**data)
        self._out_cls = out_cls

    def to_text(self, data: typing.Any) -> str:
        """Convert the data to text

        Args:
            data (typing.Any): 

        Returns:
            str: The 
        """
        return str(data)

    def dump_data(self, data: typing.Any) -> typing.Any:
        """

        Args:
            data (typing.Any): 

        Returns:
            typing.Any: 
        """
        return data

    def write_text(self, data: typing.Any) -> str:
        """Convert the primitive to a string

        Args:
            data (primitive): The primitive

        Returns:
            str: The data converted to a string
        """
        return str(data)

    def read_text(self, message: str) -> typing.Any:
        """Read the message from text

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The string converted to the primitive
        """
        return self._out_cls(message)
    
    def load_data(self, data) -> typing.Any:
        """Doesn't do anything because the data should be in
        the right form

        Args:
            data: The primitive

        Returns:
            typing.Any: The primitive value
        """
        return data

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self._out_cls}>'


class JSONRead(Reader):
    """
    """

    key_descr: typing.Optional[typing.Dict] = None

    def read_text(self, message: str) -> typing.Dict:
        """Read in the JSON

        Args:
            text (str): The JSON to read in

        Returns:
            typing.Dict: The result - if it fails, will return an empty dict
        """
        try: 
            result = json.loads(message)
            return result
        except json.JSONDecodeError:
            return {}
    
    def load_data(self, data: typing.Dict) -> typing.Dict:
        """_summary_

        Args:
            data (typing.Dict): _description_

        Returns:
            typing.Dict: Convert the
        """
        return data

    def dump_data(self, data: typing.Dict) -> typing.Dict:
        """Does not do anything 

        Args:
            data (typing.Any): The data 

        Returns:
            typing.Any: The data
        """
        return data

    def write_text(self, data: typing.Any) -> str:
        """Write the data to a a string

        Args:
            data (typing.Any): The data to write

        Returns:
            str: The string version of the data
        """
        return json.dumps(data)

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return escape_curly_braces(self.key_descr)
