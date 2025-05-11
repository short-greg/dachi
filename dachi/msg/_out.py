# 1st party
import typing
import json
import inspect
from abc import abstractmethod
import csv
import io
from collections import OrderedDict

# 3rd party
import pydantic

# local
from ..inst import render, struct_template
from ._messages import Msg
from ._msg import MsgProc
from ..base import TemplateField, Templatable, ExampleMixin
from ..utils import unescape_curly_braces
from pydantic_core import PydanticUndefined
from ..utils import (
    escape_curly_braces
)
from ._parse import LineParser, Parser
from .. import store
from .. import utils

S = typing.TypeVar('S', bound=pydantic.BaseModel)


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


class OutConv(MsgProc, Templatable, ExampleMixin):
    """Use for converting an AI response into a primitive value
    """
    
    @abstractmethod
    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        pass

    @abstractmethod
    def template(self) -> str:
        """Get the template for the reader

        Returns:
            str: The template as a string
        """
        pass

    def coalesce_resp(self, resp, default='') -> str:
        """_summary_

        Args:
            resp : 

        Returns:
            str: 
        """
        if resp is utils.UNDEFINED or resp is None:
            return default
        
        return resp


class PrimOut(OutConv):
    """Use for converting an AI response into a primitive value
    """

    def __init__(
        self, out_cls: typing.Type, name: str, from_: str='content'
    ):
        """Create a reader for Primitive values

        Args:
            out (typing.Type): The type of data
        """
        super().__init__(
            name=name, from_=from_
        )
        self._out_cls = out_cls

    def delta(self, resp, delta_store: typing.Dict, is_streamed: bool=False, is_last: bool=True) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if resp is None or resp is utils.UNDEFINED:
            resp = ''

        val = store.acc(delta_store, 'val', resp)

        if not is_last:
            return utils.UNDEFINED
        
        if self._out_cls is bool:
            return (
                val.lower() in ('true', 'y', 'yes', '1', 't')
            )
        return self._out_cls(val)

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self._out_cls.__name__}>'


class PydanticOut(OutConv, typing.Generic[S]):
    """Use for converting an AI response into a Pydantic BaseModel
    """
    def __init__(
        self, out_cls: typing.Type[S], 
        name: str, from_: str='content'
    ):
        """Read in a Pydantic BaseModel

        Args:
            out_cls (S): The class to read in
        """
        super().__init__(name, from_)
        self._out_cls = out_cls

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data.model_dump_json()

    def delta(
        self, resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
    ) -> Msg:
    # def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if resp is None or resp is utils.UNDEFINED:
            resp = ''
        val = store.acc(delta_store, 'val', resp)

        if not is_last:
            return utils.UNDEFINED
        
        message = unescape_curly_braces(val)
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


class KVOut(OutConv):
    """Create a Reader of a list of key values
    """

    def __init__(
        self, name: str, from_: str='content',
        sep: str='::', key_descr: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.Dict]] = None
    ):
        super().__init__(name, from_)
        self.sep = sep
        self.key_descr = key_descr
        self.line_parser = LineParser()

    def delta(
        self, resp, delta_store: typing.Dict, is_streamed: bool=False, is_last: bool=True
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        resp = self.coalesce_resp(resp)
        line_store = store.get_or_set(delta_store, 'lines', {})
        res = self.line_parser(resp, line_store, is_streamed, is_last)

        if res is utils.UNDEFINED or res is None:
            return res
        
        result = {}
        for line in res:
            try:

                key, value = line.split(self.sep)
                result[key] = value
            except ValueError as e:
                raise RuntimeError(
                    f"Could not split the line {line} with separator {self.sep}."
                ) from e
    
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


class IndexOut(OutConv):
    """Create a Reader of a list of key values
    """

    def __init__(
        self, name: str, from_: str='content',
        sep: str='::', key_descr: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.Dict]] = None
    ):
        super().__init__(name, from_)
        self.sep = sep
        self.key_descr = key_descr
        self.sep = sep
        self.key_descr = key_descr
        self.key_type = None
        self.line_parser = LineParser()

    def delta(self, resp, delta_store: typing.Dict, is_streamed: bool=False, is_last: bool=True) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        resp = self.coalesce_resp(resp)
        line_store = store.get_or_set(delta_store, 'lines', {})
        resp = self.line_parser(resp, line_store, is_streamed, is_last)
        if resp is utils.UNDEFINED or resp is None:
            return utils.UNDEFINED
        result = []
        for line in resp:
            try:
                idx, value = line.split(self.sep)
                idx = int(idx) - 1
                if idx > len(result):
                    count = idx - len(result)
                    result = [None] * count
                    result[idx - 1] = value
                elif idx < len(result):
                    result[idx] = value
                else:
                    result.append(value)
            except ValueError as e:
                raise RuntimeError(
                    f"Could not split the line {line} by the separator provided {self.sep}."
                ) from e
        
        return result

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


class JSONOut(OutConv):
    """Use to read from a JSON
    """

    def __init__(
        self, name, from_ = 'content', 
        key_descr: typing.Optional[typing.Dict] = None
    ):
        super().__init__(name, from_)
        self.key_descr = key_descr

    def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=True) -> typing.Any:
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
        
        resp = self.coalesce_resp(resp)
        resp = store.acc(delta_store, 'val', resp)

        if not is_last:
            return utils.UNDEFINED

        try: 
            result = json.loads(resp)
            return result
        except json.JSONDecodeError:
            return {}

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return json.dumps(data)

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return escape_curly_braces(self.key_descr)


class ParsedOut(OutConv):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """
    def __init__(self, parser: Parser, name, from_: str='content'):
        super().__init__(name, from_)
        self._parser = parser

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)

    def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return self._parser.forward(
            resp, delta_store, streamed=streamed,
            is_last=is_last
        )

    def template(self) -> str:

        return ''



class NullOut(OutConv):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """

    def __init__(self, name, from_: str='content', parser: Parser=None):
        super().__init__(name, from_)
        self._parser = parser

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)

    def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if self._parser is not None:
            return self._parser.forward(
                resp, delta_store, streamed=streamed,
                is_last=is_last
            )
        return resp

    def template(self) -> str:
        return ''


class CSVOut(OutConv):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """
    def __init__(self, name: str, from_: str ='content', delimiter: str = ',', use_header: bool = True):
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
        super().__init__(name, from_)
        self._delimiter = delimiter
        self._use_header = use_header

    def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        # resp = self.handle_null(resp, '')

        val = store.acc(delta_store, 'val', resp, '')
        row = store.get_or_set(delta_store, 'row', 0)
        header = store.get_or_set(
            delta_store, 'header', None
        )
        # Process accumulated data using csv.reader
        # csv_data = io.StringIO(delta_store['val'])

        rows = list(
            csv.reader(io.StringIO(val), delimiter=self._delimiter)
        )
        new_rows = []
        for i, row in enumerate(rows[row:]):  
            # Only return new rows
            new_rows.append(row)

        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if not is_last:
            new_rows.pop()

        if len(new_rows) == 0:
            return utils.UNDEFINED

        if (
            self._use_header is True 
            and delta_store['header'] is None
        ):
            delta_store['header'] = new_rows.pop(0)
            store.acc(delta_store, 'row', 1)

        header = delta_store['header']
        store.acc(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if self._use_header:
            return [OrderedDict(zip(header, row)) for row in new_rows]
        return new_rows

    def render(self, data) -> str:
        """Convert the data to a CSV string

        Args:
            data: An iterable of rows. If header is set to true

        Returns:
            str: the rendered CSV
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self._delimiter)
        
        if self._use_header:
            header = [key for key, _ in data[0]]
            writer.writerow(header)
            for row in data:
                writer.writerow([value for _, value in row])
        else:
            for row in data:
                writer.writerow(row)
        
        return output.getvalue()

    def example(self, data):
        return super().example(data)
    
    def template(self):
        return super().template()


def conv_to_out(
    out, 
    name: str='out', 
    from_: str='content'
) -> OutConv:
    """

    Args:
        out: 
        name (str, optional): . Defaults to 'out'.
        from_ (str, optional): . Defaults to 'content'.

    Raises:
        ValueError: 

    Returns:
        OutConv: 
    """
    if isinstance(out, OutConv):
        return out
    
    if out is bool:
        return PrimOut(bool, name, from_)
    if out is str:
        return PrimOut(str, name, from_)
    if out is int:
        return PrimOut(int, name, from_)
    if out is None:
        return None
    if inspect.isclass(out) and issubclass(out, pydantic.BaseModel):
        return PydanticOut(out, name, from_)
    raise ValueError
