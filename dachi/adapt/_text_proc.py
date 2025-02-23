# 1st party
import typing
import json
import inspect

# 3rd party
from io import StringIO
import pandas as pd
import pydantic
import yaml
import typing
import json

# local
from ._read import TextConv, struct_template
from .._core._core import render, END_TOK
from ..utils import (
    escape_curly_braces
)



class CSVConv(TextConv):
    """Convert text to a StructList
    """
    indexed: bool = True
    delim: str = ','
    cols: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.List[typing.Tuple[str, str, str]]]] = None

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
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

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        io = StringIO(message)
        df = pd.read_csv(io, sep=self.delim)
        return df.to_dict(orient='records', index=True)
    
    def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if message is None:
            return None
        if 'message' not in delta_store:
            delta_store['message'] = ''
            delta_store['row'] = 0
            delta_store['fields'] = None
        if message is not END_TOK:
            delta_store['message'] += message
        io = StringIO(delta_store['message'])
        df = pd.read_csv(io, sep=self.delim)
        # print(df)
        if (
            len(df.index) > 0 
            and delta_store['row'] == 0
        ):
            delta_store['fields'] = list(df.columns.values)
        if len(df.index) <= delta_store['row'] and message is not END_TOK:
            return None
        
        delta_store['row'] = len(df.index)
        if len(df.index) == 1:
            return None
        if message is END_TOK:
            df = df.iloc[delta_store['row'] - 1:]
        else:
            df = df.iloc[delta_store['row'] - 2:delta_store['row'] - 1]
        return df.to_dict(orient='records', index=True)

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

        header = []
        first = []
        last = []
        mid = '...'
        if self.indexed:
            header.append('Index')
            first.append('1')
            last.append('N')
            
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


class KVConv(TextConv):
    """Create a Reader of a list of key values
    """

    sep: str = '::'
    key_descr: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.Dict]] = None

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return '\n'.join(
            f'{k}{self.sep}{render(v)}' for k, v in data.items()
        )

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
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


class IndexConv(TextConv):
    """Create a Reader of a list of key values
    """

    sep: str = '::'
    key_descr: typing.Optional[str] = None
    key_type: typing.Optional[typing.Type] = None
    
    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return '\n'.join(
            f'{k}{self.sep}{render(v)}' for k, v in enumerate(data)
        )

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        lines = message.splitlines()
        result = []
        for line in lines:
            try:
                idx, value = line.split(self.sep)
                result.append(value)
            except ValueError:
                pass
        return result

    def template(self, count: int=None) -> str:
        """Get the template for the Keys and Values

        Returns:
            str: The template
        """
        key_descr = (
            'The value for the key.' if self.key_descr is None else self.key_descr
        )
        
        if self.key_type is not None:
            key_descr = key_descr + f' ({self.key_type})'
        key_descr = f'<{key_descr}>'
        
        if count is None:
            lines = [
                f'1{self.sep}{key_descr}',
                f'...',
                f'N{self.sep}{key_descr}',
            ]
        else:
            lines = [
                f'1{self.sep}{key_descr}',
                f'...',
                f'count{self.sep}{key_descr}',
            ]
        
        return '\n'.join(lines)


class JSONConv(TextConv):
    """Use to read from a JSON
    """

    key_descr: typing.Optional[typing.Dict] = None


    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return json.dumps(data)

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
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

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return escape_curly_braces(self.key_descr)


class YAMLConv(TextConv):
    """Use to read from a Yaml
    """

    key_descr: typing.Optional[typing.Dict] = None

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return json.dumps(data)

    def __call__(self, message: str) -> typing.Any:
        """Read in the JSON

        Args:
            text (str): The JSON to read in

        Returns:
            typing.Dict: The result - if it fails, will return an empty dict
        """
        try: 
            result = yaml.safe_load(message)
            return result
        except yaml.YAMLError:
            return {}

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return yaml.safe_dump(self.key_descr)

