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
from pydantic_core import PydanticUndefined

# local
from ._msg import RespProc
from ..core import (
    TemplateField, Templatable, ExampleMixin, ModuleList,
    render, struct_template, Msg,
    Resp
)
from ..utils import unescape_curly_braces, escape_curly_braces
from ._parse import LineParser, Parser
# from .. import store
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
    def rethrow(
        original_exception, 
        message="Read operation failed"
    ):
        """Utility method to raise a ReadError while preserving the original exception."""
        raise ReadError(message, original_exception) from original_exception


class ToOut(
    RespProc, 
    Templatable, 
    ExampleMixin
):
    """Use for converting an AI response into a primitive value
    """    
    name: str
    from_: str = 'content'
    
    @abstractmethod
    def render(self, data: typing.Any) -> str:
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

    def forward(
        self, 
        resp: Resp, 
        delta_store: typing.Dict=None,
        is_streamed: bool=False,
        is_last: bool=True
    ) -> Msg:
        """Processes the message

        Args:
            msg (Msg): The message to process
            delta_store (typing.Dict, optional): The delta store. Defaults to None.

        Returns:
            Msg: The processed message
        """
        delta_store = delta_store if delta_store is not None else {}

        resp = [resp.data[r] for r in self.from_]
        is_undefined = all(
            r is utils.UNDEFINED for r in resp
        )
        
        if self._single:
            resp = resp[0]
        
        if is_undefined:
            resp.out[self.name] = utils.UNDEFINED
            return utils.UNDEFINED
        resp.out[self.name] = res = self.delta(
            resp, delta_store, is_streamed, is_last
        )
        self.post(
            resp, 
            res, 
            is_streamed, 
            is_last
        )
        return resp


class PrimOut(ToOut):
    """Use for converting an AI response into a primitive value
    """

    out_cls: str
    prim_map: typing.ClassVar[typing.Dict[str, typing.Callable]] = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool
    }

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if resp is None or resp is utils.UNDEFINED:
            resp = ''

        val = utils.acc(delta_store, 'val', resp)

        if not is_last:
            return utils.UNDEFINED
        
        if self._out_cls == 'bool':
            return (
                val.lower() in ('true', 'y', 'yes', '1', 't')
            )
        return self.prim_map[self.out_cls](val)

    def render(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)
    
    def example(self) -> str:
        if self._out_cls == 'int':
            return self.render(1)
        elif self._out_cls == 'bool':
            return self.render(True)
        elif self._out_cls == 'str':
            return self.render("data")
        elif self._out_cls == 'float':
            return self.render(3.14)
        raise RuntimeError(
            f"Don't know how render for type {self._out_cls}"
        ) 

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self._out_cls.__name__}>'


class PydanticOut(ToOut):
    """Use for converting an AI response into a Pydantic BaseModel
    """
    out_cls: typing.Type[pydantic.BaseModel]

    def render(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data.model_dump_json()

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any:
    # def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if resp is None or resp is utils.UNDEFINED:
            resp = ''
        val = utils.acc(delta_store, 'val', resp)

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

    def example(self):
        data = {
            'x': 'data 1',
            'y': 'data 2',
            'z': 'data 3'
        }
        return self.render(
            data
        )


class KVOut(ToOut):
    """Create a Reader of a list of key values
    """
    sep: str = '::'
    key_descr: typing.Type[pydantic.BaseModel] | None = None

    def __post_init__(
        self
    ):
        super().__post_init__()
        self.line_parser = LineParser()

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        resp = self.coalesce_resp(resp)
        line_store = utils.get_or_set(delta_store, 'lines', {})
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

    def render(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return '\n'.join(
            f'{k}{self.sep}{render(v)}' for k, v in data.items()
        )

    def example(self):
        data = {
            'x': 'data 1',
            'y': 'data 2',
            'z': 'data 3'
        }
        return self.render(data)


class IndexOut(ToOut):
    """Create a Reader of a list of key values
    """
    sep: str = '::'
    key_descr: typing.Type[pydantic.BaseModel] | typing.Dict | None = None

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        resp = self.coalesce_resp(resp)
        line_store = utils.get_or_set(
            delta_store, 'lines', {}
        )
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

    def render(self, data: typing.Any) -> str:
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
    
    def example(self):
        
        data = ["Data 1", "Data 2", "Data 3"]
        return self.render(data)


class JSONOut(ToOut):
    """Use to read from a JSON
    """
    
    key_descr: typing.Dict[str, str] | None = None

    def delta(
        self,
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any:
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
        resp = utils.acc(delta_store, 'val', resp)

        if not is_last:
            return utils.UNDEFINED

        try: 
            result = json.loads(resp)
            return result
        except json.JSONDecodeError:
            return {}

    def render(self, data: typing.Any) -> str:
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
        return escape_curly_braces(
            self.key_descr
        )

    def example(self) -> str:
        
        return json.dumps({
            "key1": "value1",
            "key2": 123,
            "key3": True,
            "key4": [1, 2, 3],
            "key5": {"nestedKey": "nestedValue"}
        }, indent=4)


class ParsedOut(ToOut):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """

    parser: Parser

    def render(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return self._parser(
            resp, delta_store, streamed=streamed,
            is_last=is_last
        )

    def template(self) -> str:

        return self._parser.render(
            ['<Template 1>', '<Template 2>']
        )

    def example(self):
        
        return self._parser.render(
            ['Example 1', 'Example 2']
        )


class TupleOut(ToOut):

    convs: ModuleList[ToOut]
    parser: Parser

    def delta(
        self, 
        resp, 
        delta_store, 
        is_streamed = False, 
        is_last = True
    ) -> typing.Any:
        parsed = utils.sub_dict(delta_store, 'parsed')
        i = utils.get_or_set(delta_store, 'i', 0)
        res = self.parser.forward(
            resp, 
            parsed, 
            is_streamed,
            is_last
        )
        if res is utils.UNDEFINED:
            return utils.UNDEFINED

        outs = []
        if is_last and len(res) != len(self.convs[i:]):
            raise RuntimeError(
                "There are more out processors to retrieve than items."
            )
        
        if len(res) > len(self.convs[i:]):
            raise RuntimeError(
                "There are more items to retrieve than out processors."
            )
        for res_i, conv in zip(res, self.convs[i:]):

            outs.append(conv.delta(res_i, {}))
            utils.acc(delta_store, 'i', 1)
        return outs
        
    def render(self, data) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        datas = [
            conv.render(data_i) 
            for data_i, conv in 
            zip(data, self.convs)
        ]
        return self.parser.render(datas)

    def template(self):
        
        templates = [conv.template() for conv in self.convs]
        return self.parser.render(templates)

    def example(self):
        datas = [
            conv.example() 
            for conv in self.convs
        ]
        return self.parser.render(datas)


class ListOut(ToOut):
    """
    ListOut is an output converter that transforms the output from a language model (LLM) into a list of objects.
    It utilizes a provided OutConv instance to convert each item in the list and a Parser to parse and render the list structure.
        conv (OutConv): The output converter used to process each item in the list.
        parser (Parser): The parser responsible for handling the list structure and rendering.
        name (str, optional): The name of the output converter. Defaults to 'out'.
        from_ (str, optional): The source field for conversion. Defaults to 'content'.
    """

    conv: ToOut
    parser: Parser

    def delta(
        self, 
        resp, 
        delta_store, 
        is_streamed = False, 
        is_last = True
    ) -> typing.Any:
        res = self.parser.forward(
            resp, delta_store, is_streamed,
            is_last
        )
        if res is utils.UNDEFINED:
            return utils.UNDEFINED

        return [self.conv.delta(
            res_i, {}, False, True
        ) for res_i in res]
        
    def render(self, data) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        datas = [
            conv.render(data_i) 
            for data_i, conv in 
            zip(data, self.convs)
        ]
        return self.parser.render(datas)

    def template(self):
        
        templates = [
            self.conv.template() for _ in range(3)]
        return self.parser.render(templates)

    def example(self):
        datas = [
            self.conv.example() 
            for _ in range(2)
        ]
        return self.parser.render(datas)


class NullOut(ToOut):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """

    parser: Parser | None = None

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ) -> typing.Any:
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

    def render(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return str(data)
    
    def template(self) -> str:
        return ''
    
    def example(self):
        return ''


class CSVOut(ToOut):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """
    delimiter: str = ',', 
    use_header: bool = True

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        # resp = self.handle_null(resp, '')

        val = utils.acc(delta_store, 'val', resp, '')
        row = utils.get_or_set(delta_store, 'row', 0)
        header = utils.get_or_set(
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
            utils.acc(delta_store, 'row', 1)

        header = delta_store['header']
        utils.acc(delta_store, 'row', len(new_rows))
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

    def example(self):
        data = [
            ["1", "2", "3"],
            ["2", "3", "4"]
        ]
        return self.render(data)

    def template(self):
        return super().template()


def conv_to_out(
    out, 
    name: str='out', 
    from_: str='content'
) -> ToOut:
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
    if isinstance(out, ToOut):
        return out
    
    if out is bool:
        return PrimOut('bool', name, from_)
    if out is str:
        return PrimOut('str', name, from_)
    if out is int:
        return PrimOut('int', name, from_)
    if out is float:
        return PrimOut('flaot', name, from_)
    if out is None:
        return None
    if inspect.isclass(out) and issubclass(out, pydantic.BaseModel):
        return PydanticOut(out, name, from_)
    raise ValueError
