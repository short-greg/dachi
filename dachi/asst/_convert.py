# 1st party
import typing
import json
from abc import ABC, abstractmethod
import yaml
import typing
import json
import inspect
import io
import csv

from collections import deque

# 3rd party
import pydantic
from io import StringIO
import pandas as pd

# local
from ..base import Templatable, render, TemplateField, struct_template
from ..utils import (
    unescape_curly_braces
)
from ..msg._messages import END_TOK, Msg
from pydantic_core import PydanticUndefined
from ..utils import (
    escape_curly_braces
)
from .. import utils


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class RespConv(ABC):
    """Use to process the resoponse from an LLM
    """
    def __init__(self, resp: bool):
        """
        Initialize the instance.
        Args:
            resp (bool): Indicates if the response processor responds with data.
        """
        super().__init__()
        self._resp = resp

    @property
    def resp(self) -> bool:
        """Choose whether to include a response

        Returns:
            bool: Whether to respond with a value
        """
        return self._resp

    @abstractmethod
    def __call__(self, response, msg: Msg) -> typing.Any:
        pass

    @abstractmethod
    def delta(
        self, response, msg: Msg, delta_store: typing.Dict
    ) -> typing.Any: 
        pass

    def prep(self) -> typing.Dict:
        return {}


class Parser(ABC):
    """
    Parser is an abstract base class designed to process and parse the output of a 
    large language model (LLM) into discrete units. These units can then be 
    converted into a structured output that is sent to the calling function.
    The class provides a framework for implementing custom parsing logic through 
    the `delta` and `template` methods, which must be defined in subclasses. It 
    also includes a convenience method `__call__` to handle parsing of the entire 
    response in one step.
    """

    @abstractmethod
    def delta(self, resp, delta_store: typing.Dict, last: bool=False) -> typing.List | None:
        """Parse the response one by one

        Args:
            resp: The response
            delta_store (typing.Dict): Dictionary to accumulate updates
            last (bool, optional): Whether it is the last value or not. Defaults to False.

        Returns:
            typing.List: Will return a list if value defined else UNDEFINED
        """
        pass

    def __call__(self, resp) -> typing.List[typing.Any] | None:
        """Convenience function to parse based on the whole set of data

        Args:
            resp: 

        Returns:
            typing.List[typing.Any]: The parsed values. Will return undefined if
            cannot parse
        """
        return self.delta(
            resp, {}, True
        )

    @abstractmethod
    def render(self, data) -> str:
        pass


class CSVRowParser:
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """
    
    def __init__(self, delimiter: str = ',', use_header: bool = False):
        """
        Initializes the CSV parser with the specified delimiter and header usage.
        This class is designed to dynamically parse CSV data, returning new rows 
        as they are accumulated. It supports customization of the delimiter and 
        whether the CSV file includes a header row.
        Args:
            delimiter (str): The character used to separate values in the CSV. 
                             Defaults to ','.
            use_header (bool): Indicates whether the CSV file includes a header row. 
                               Defaults to False.
        """

        self._delimiter = delimiter
        self._use_header = use_header

    def delta(self, resp, delta_store: typing.Dict, last: bool=False) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        delta_store = delta_store or {
            'val': '',        # Accumulates entire CSV content
            'row': 0,  # Tracks last returned row index
            'header': None     # Stores header if enabled
        }

        # Process accumulated data using csv.reader
        # csv_data = io.StringIO(delta_store['val'])
        rows = list(csv.reader(delta_store['val'], delimiter=self._delimiter))

        new_rows = []
        for i, row in enumerate(rows[delta_store['row']:]):  # Only return new rows
            new_rows.append(row)

        if not last:
            new_rows.pop()

        if self._use_header is True and delta_store['header'] is None:
            delta_store['header'] = new_rows.pop(0)
        
        header = delta_store['header']
        utils.increment(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return None
        
        if self._use_header:
            return [(header, row) for row in new_rows]
        return new_rows

    def template(self, data) -> str:
        pass


class CSVCellParser:

    def __init__(self, delimiter: str = ',', use_header: bool = False):
        """
        Initializes the converter for parsing CSV data.
        Args:
            delimiter (str): The character used to separate values in the CSV. Defaults to ','.
            use_header (bool): Indicates whether the CSV includes a header row. If True, 
                the header row will be used to label the cells. Defaults to False.
        This class parses CSV data and returns all cells. If `use_header` is True, 
        the cells will be labeled using the values from the header row.
        """
        self._delimiter = delimiter
        self._use_header = use_header

    def delta(self, resp, delta_store: typing.Dict, last: bool=False) -> typing.Any:
        """
        Parses a single-row CSV and returns one cell at a time.
        """
        if 'val' not in delta_store:
            delta_store.update({
                'val': '',       # Accumulates current cell value
                'row': 0,       # Stores the single data row
                'header': [],  # Stores header if provided
                'data': [],
                'col': 0     # Tracks current column index
            })
        
        utils.call_or_set(
            delta_store, 'val', resp, lambda x, dx: x + dx
        )

        rows = list(csv.reader(delta_store['val'], delimiter=self._delimiter))
        cells = []

        new_rows = []
        for i, row in enumerate(rows[delta_store['row']:]):  # Only return new rows
            new_rows.append(row)

        if self._use_header and len(rows) > 1:
            delta_store['header'] = new_rows.pop(0)

        if not last and len(new_rows) > 1:
            new_rows[-1].pop(-1)

        cells = []

        i = delta_store['row']
        j = delta_store['col']
        header = delta_store['header']
        for i, row in enumerate(new_rows):
            for j, cell in enumerate(row):
                if (i == 0 and j > delta_store['col']) or i != 0:
                    if self._use_header:
                        cells.append[(header[j], cell)]
        delta_store['row'] = i
        delta_store['col'] = j
        if len(cells) == 0:
            return None
        return cells
    
    def template(self, data) -> str:
        pass


class CharDelimParser(Parser):
    
    def __init__(self, c: str=','):
        super().__init__()

        self._c = c

    def delta(self, resp, delta_store: typing.Dict, last: bool=False) -> typing.List | None:
        
        delta_store = delta_store or {'val': ''}
        if resp is END_TOK:
            res = delta_store['val'].split(self._c)
            delta_store['val'] = ''
            return res
        
        utils.get_or_set(delta_store, 'val', resp)
        val = utils.call_or_set(delta_store, 'val', resp, lambda x, d: x + d)
        res = val.split(self._c)
        
        if len(res) > 1:
            return_val = res[:-1]
            delta_store['val'] = res[-1]

            return [return_val]

        return None


class NullParser(Parser):
    """
    A parser that does not perform any parsing or transformation on the input.
    Instead, it simply returns the input response as-is.
    """

    def delta(self, resp, delta_store: typing.Dict, last: bool=False) -> typing.Any:
        return [resp]
    
    def render(self, data):
        pass


class FullParser(Parser):
    """
    A parser that accumulates input data until the end of a stream is reached.
    """
    
    def delta(self, resp, delta_store: typing.Dict, last: bool=False) -> typing.Any:
        
        utils.get_or_set(delta_store['val'], '')
        delta_store['val'] += resp
        if last:
            val = delta_store['val']
            delta_store.clear()
            return [val]
        return None
        

# use the deque for the buffer

# parse => deque => pop => out conv.. Continue popping after finished
# how to handle non-delta cases... Perhaps I will switch back to having a
# non-delta version.. But I still need parsing even in that case

# out_conv.template() => (outputs the template data) => parser.render(data)
# out_conv.example() => (outputs the example data) => parser.render(data)

class ReadError(Exception):
    """
    Exception used when a text processor (TextProc) fails to read data.
    This class supports rethrowing errors while preserving the original exception details.
    """
    def __init__(self, message="Read operation failed", original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
    
    def __str__(self):
        if self.original_exception:
            return f"{super().__str__()} (Caused by: {repr(self.original_exception)})"
        return super().__str__()
    
    @staticmethod
    def rethrow(original_exception, message="Read operation failed"):
        """Utility method to raise a ReadError while preserving the original exception."""
        raise ReadError(message, original_exception) from original_exception


class OutConv(pydantic.BaseModel, Templatable, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """

    name: str = ''

    @abstractmethod
    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        pass
        # return self.write_text(self.dump_data(data))

    def dump_data(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        pass

    @abstractmethod
    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass
        # return self.load_data(self.read_text(message))

    @abstractmethod
    def template(self) -> str:
        """Get the template for the reader

        Returns:
            str: The template as a string
        """
        pass


class NullOutConv(OutConv):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """
    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)

    def dump_data(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return resp

    def template(self) -> str:
        return None


class PrimConv(OutConv):
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

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(self.dump_data())

    def dump_data(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if message is None:
            message = ''
        
        if self._out_cls is bool:
            return message.lower() in ('true', 'y', 'yes', '1', 't')
        return self._out_cls(message)

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self._out_cls}>'


class PydanticConv(OutConv, typing.Generic[S]):
    """Use for converting an AI response into a Pydantic BaseModel
    """
    _out_cls: typing.Type[S] = pydantic.PrivateAttr()

    def __init__(self, out_cls: S, **data):
        """Read in a 

        Args:
            out_cls (S): The class to read in
        """
        super().__init__(**data)
        self._out_cls = out_cls

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data.model_dump_json()

    def dump_data(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data.model_dump_json()

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
    # def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        message = unescape_curly_braces(resp)
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            raise ReadError(
                f'Could not read in {message} as a JSON', e
            )
        return self._out_cls(**data)

    def template_renderer(self, template: TemplateField) -> str:
        """Render the template for the BaseModel

        Args:
            template (TemplateField): The template to render

        Returns:
            str: The template
        """
        t = f'<{template.description}> - type: {template.type_}'
        if template.default is not None or template.default == PydanticUndefined:
            t = f'{t} {template.default}>'
        else:
            t = f'{t}>'
        return t

    def template(self, escape_braces: bool=False) -> str:
        """Convert the object to a template

        Args:
            escape_braces (bool, optional): Whether to escape curly brances. Defaults to False.

        Returns:
            str: Escape the braces
        """
        return render(
            struct_template(self._out_cls), 
            escape_braces, self.template_renderer
        ) 


class CSVConv(OutConv):
    """Convert text to a StructList
    """
    indexed: bool = True
    delim: str = ','
    user_header: bool = True
    
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

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        columns = delta_store.get('columns') if self.user_header else None
        
        if columns is not None:
            io = StringIO(resp)
            df = pd.read_csv(io, sep=self.delim, header=columns)
            return df.to_dict(orient='records', index=True)
        else:
            io = StringIO(resp)
            df = pd.read_csv(io, sep=self.delim, header=columns)
            if self.user_header:
                delta_store['columns'] = list(df.columns.values)
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


class KVConv(OutConv):
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

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        lines = resp.splitlines()
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


class IndexConv(OutConv):
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

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        lines = resp.splitlines()
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


class JSONConv(OutConv):
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

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
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
            result = json.loads(resp)
            return result
        except json.JSONDecodeError:
            return {}

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return escape_curly_braces(self.key_descr)


class YAMLConv(OutConv):
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

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the JSON

        Args:
            text (str): The JSON to read in

        Returns:
            typing.Dict: The result - if it fails, will return an empty dict
        """
        try: 
            result = yaml.safe_load(resp)
            return result
        except yaml.YAMLError:
            return {}

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return yaml.safe_dump(self.key_descr)
