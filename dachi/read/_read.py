# 1st party
import typing
import json
import inspect

# 3rd party
from io import StringIO
import pandas as pd
import pydantic

# local
from .._core._core import (
    Reader, 
    render,
)
from ..utils import (
    struct_template,
    model_to_text,
    escape_curly_braces, 
)
from ..data._structs import DataList


S = typing.TypeVar('S', bound=pydantic.BaseModel)


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

    def dump_data(self, data: DataList[S]) -> typing.Any:
        structs = []
        for cur in data.data:
            # 
            # structs.append(self._out_cls.to_dict(cur))
            structs.append(
                cur.model_dump()
            )

        return structs

    def write_text(self, data: typing.List) -> str:
        return json.dumps(data)

    def read_text(self, message: str) -> DataList[S]:
        """Convert the message into a list of structs

        Args:
            message (str): The AI message to read

        Returns:
            StructList[S]: the list of structs
        """
        return json.loads(message)
    
    def load_data(self, data) -> DataList[S]:
        structs = []
        for cur in data['data']:
            structs.append(self._out_cls(**cur))

        return DataList(data=structs)

    # TODO: Shouldn't this be "render?"
    def to_text(self, data: DataList[S]) -> str:
        """Convert the data to a string

        Args:
            data (StructList[S]): The data to convert to text

        Returns:
            str: the data converted to a string
        """
        return model_to_text(data)

    def template(self) -> str:
        """Output a template for the struct list

        Returns:
            str: A template of the struct list
        """
        # TODO: This is currently not correct
        #   Needs to output as a list

        return struct_template(self._out_cls)


class CSVRead(Reader):
    """Convert text to a StructList
    """
    indexed: bool = True
    delim: str = ','
    cols: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.List[typing.Tuple[str, str, str]]]] = None

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
        if (
            isinstance(self.cols, pydantic.BaseModel) or 
            (inspect.isclass(self.cols) and issubclass(self.cols, pydantic.BaseModel))
        ):
            temp = struct_template(self.cols)
            cols = []
            for k, v in temp.items():                
                # if 'description' not in v:
                #     raise RuntimeError(f'Cannot create CSV template for {self.cols}')
                cols.append((k, v.description, v.type_))
        elif self.cols is None:
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
    """Create a Reader of a list of key values
    """

    sep: str = '::'
    key_descr: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.Dict]] = None
    
    def read_text(self, message: str) -> typing.Dict:
        """Read in the list of key values

        Args:
            message (str): The message to read

        Returns:
            typing.Dict: A dictionary of keys and values
        """
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
        """Load data does not do anything as the result
        is a dictionary.

        Args:
            data (typing.Dict): The data to load

        Returns:
            typing.Dict: The dictionary of data
        """
        return data

    def dump_data(self, data: typing.Dict) -> typing.Dict:
        """Convert the data to a dictionary

        Args:
            data (typing.Dict): The data to load

        Returns:
            typing.Dict: The dumped data
        """
        return data

    def write_text(self, data: typing.Dict) -> str:
        """Write data as text

        Args:
            data (typing.Dict): The data to write

        Returns:
            str: The keys and values as text
        """
        return '\n'.join(
            f'{k}{self.sep}{render(v)}' for k, v in data.items()
        )

    def template(self) -> str:
        """Get the template for the Keys and Values

        Returns:
            str: The template
        """
        if self.key_descr is None:
            key_descr = {
                '<Example>': '<The value for the key.>'
            }
        elif (
            inspect.isclass(self.key_descr) and 
            issubclass(self.key_descr, pydantic.BaseModel)
        ):
            temp = struct_template(self.key_descr)
            key_descr = {}
            for k, v in temp.items():
                description =  v.description or 'value'
                key_descr[k] = f'<{description}>'
        else:
            key_descr = self.key_descr
        return '\n'.join(
            f'{key}{self.sep}{value}' 
            for key, value in key_descr.items()
        )


class JSONRead(Reader):
    """Use to read from a JSON
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
        """Load the data from a dictionary. Since JSONs are just a dict
        this does nothing

        Args:
            data (typing.Dict): The data to load

        Returns:
            typing.Dict: The result
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
