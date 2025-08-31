
# 1st party 
from abc import abstractmethod
import typing
import inspect

# 3rd party
import pydantic

# local
from ..core import Resp, render
from ._process import Process
from .. import utils


RESPONSE_FIELD = 'response'

# TODO: After updating this
# the _out.py, _parse.py and _resp.py tests must
# be updated. and consolidated into here
# I also want to remove "def post"
# all of these just output a value (They will be ToOuts)
# and that value will 

# Also, I want to update these to take in a Resp rather
# than a message. Then I want to have a delta method
# and a regular forward method (separate them
# Aim to simplify all of these classes

# 1st party
import inspect
import typing
from abc import abstractmethod
import csv
import io
from collections import OrderedDict

# 3rd party
import pydantic

# local
from ..core import (
    Templatable, 
    ExampleMixin, ModuleList,
    render, 
    Resp, struct_template
)
from ._parse import LineParser, Parser
from .. import utils

S = typing.TypeVar('S', bound=pydantic.BaseModel)


# 1st party
import typing
import csv
import io
from abc import abstractmethod

# 3rd party

# local
from .. import utils
from collections import OrderedDict
from dachi.proc import Process

# RespProc => use to covert to "out"
# out = {
#   'x': bool, # will create a boolean RespProc (convert text to bool)
#   'y': int, # will create an integer (PrimConv) RespProc (convert text to int)
#   'd': pydantic.BaseModel # use the actual Pydantic model type..
#        # will use "StructConv" i.e.. Structured output conv
#   'z': ToOut()
# }
# out = tuple(...) # outputs a tuple
# out = dict() # outputs a dict 
# out = ... # outputs a single value
# out = ToOut()
# ToOut(CSV())
# ToOut(JSON())
# ToOut(JSONList())
# # this will loop key by key
# # returning dictionary by dictionary
# ToOut(JSONKeys())
# ToolUse() 
# <- set the name of the tool
# <- allow to execute the tool and write the result
#
# Don't add out to LLM
# add it to op... op


# Adapter... <- keep this separate
# OP(LLM(), out=...)
# llm( out=...)
# llm.forward(out=...)
# llm()
# # set the default out
# op("")
# tool_user("", out=..)
# <- checks tool use and can continue the conversation




# class RespConv(Process, ABC):
#     """Use to process the resoponse from an LLM
#     """

#     name: str

#     @abstractmethod
#     def delta(
#         self, 
#         resp, 
#         delta_store: typing.Dict, 
#         streamed: bool=False, 
#         is_last: bool=False
#     ) -> Msg: 
#         pass

#     def forward(
#         self, 
#         resp: Resp, 
#         is_streamed: bool=False, 
#         is_last: bool=True
#     ):

#         r = resp.val

#         delta_store = utils.get_or_set(
#             resp.delta, self.name, {}
#         )
        
#         resp.data[self.name] = res = self.delta(
#             r, 
#             delta_store, 
#             is_streamed, 
#             is_last
#         )

#         self.post(
#             resp, 
#             res, 
#             resp.delta[self.name], 
#             is_streamed, 
#             is_last
#         )
#         return resp

#     def prep(self) -> typing.Dict:
#         return {}













class JSONObj(pydantic.BaseModel):
    """Wrapper for JSON output configuration"""
    json_schema: typing.Dict | pydantic.BaseModel | None = None
    mode: typing.Literal["json_object", "json_schema"] | None = None
    
    def get_schema_dict(self) -> typing.Dict | None:
        """Convert Pydantic models to schema dict"""
        if self.json_schema is None:
            return None
        elif isinstance(self.json_schema, dict):
            return self.json_schema
        elif isinstance(self.json_schema, pydantic.BaseModel) or (
            inspect.isclass(self.json_schema) and 
            issubclass(self.json_schema, pydantic.BaseModel)
        ):
            return self.json_schema.model_json_schema()
        return None
    
    def to_response_format(self) -> typing.Dict:
        """Convert to OpenAI response_format parameter"""
        if self.json_schema is None:
            return {"type": "json_object"}
        
        schema_dict = self.get_schema_dict()
        if schema_dict is None:
            return {"type": "json_object"}
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": getattr(self.json_schema, '__name__', 'response_schema'),
                "schema": schema_dict
            }
        }






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
    Process,
    Templatable, 
    ExampleMixin
):
    """Use for converting an AI response into a output
    """
    
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

    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming response
        
        Args:
            resp: Complete response to process
            
        Returns:
            Processed result
        """
        # Simple non-streaming processing - override in subclasses
        return resp.text if hasattr(resp, 'text') else str(resp)
    
    @abstractmethod
    def delta(self, resp, delta_store: typing.Dict, is_last: bool = True) -> typing.Any:
        """Process streaming response chunks
        
        Args:
            resp: Response chunk
            delta_store: State storage for accumulating across chunks
            is_last: Whether this is the final chunk
            
        Returns:
            Processed result or UNDEFINED if not ready
        """
        pass




class ToPrim(ToOut):
    """Use for converting an AI response into a primitive value
    """

    out_cls: str
    prim_map: typing.ClassVar[typing.Dict[str, typing.Callable]] = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool
    }
    
    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming response"""
        val = str(resp.text) if hasattr(resp, 'text') else str(resp)
        
        if isinstance(self.out_cls, typing.Type):
            return self.out_cls(val)

        if self.out_cls == 'bool':
            return val.lower() in ('true', 'y', 'yes', '1', 't')
        return self.prim_map[self.out_cls](val)

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming response chunks"""
        if resp is None or resp is utils.UNDEFINED:
            resp = ''

        val = utils.acc(delta_store, 'val', resp)

        if not is_last:
            return utils.UNDEFINED
        
        if isinstance(self.out_cls, typing.Type):
            return self.out_cls(val)

        if self.out_cls == 'bool':
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
        if self.out_cls == 'int':
            return self.render(1)
        elif self.out_cls == 'bool':
            return self.render(True)
        elif self.out_cls == 'str':
            return self.render("data")
        elif self.out_cls == 'float':
            return self.render(3.14)
        raise RuntimeError(
            f"Don't know how render for type {self.out_cls}"
        ) 

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self.out_cls.__name__}>'




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

    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming response"""
        text = str(resp.text) if hasattr(resp, 'text') else str(resp)
        lines = text.strip().split('\n')
        
        result = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                key, value = line.split(self.sep, 1)
                result[key.strip()] = value.strip()
            except ValueError as e:
                raise RuntimeError(
                    f"Could not split the line {line} with separator {self.sep}."
                ) from e
        return result
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming response chunks"""
        resp = self.coalesce_resp(resp)
        line_store = utils.get_or_set(delta_store, 'lines', {})
        res = self.line_parser.forward(resp, line_store, streamed=True, is_last=is_last)

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
    key_descr: str = ''
    key_type: typing.Type[pydantic.BaseModel] | typing.Dict | None = None

    def __post_init__(self):
        super().__post_init__()
        self.line_parser = LineParser()

    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming response"""
        text = str(resp.text) if hasattr(resp, 'text') else str(resp)
        lines = text.strip().split('\n')
        
        result = []
        for line in lines:
            if not line.strip():
                continue
            try:
                idx_str, value = line.split(self.sep, 1)
                idx = int(idx_str) - 1  # Convert to 0-based index
                
                # Extend list if needed
                while len(result) <= idx:
                    result.append(None)
                
                result[idx] = value.strip()
            except ValueError as e:
                raise RuntimeError(
                    f"Could not split the line {line} by the separator provided {self.sep}."
                ) from e
        return result
        
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming response chunks"""
        resp = self.coalesce_resp(resp)
        line_store = utils.get_or_set(
            delta_store, 'lines', {}
        )
        resp = self.line_parser.forward(resp, line_store, streamed=True, is_last=is_last)
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
            'The value for the key.' if self.key_type is None else self.key_descr
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




# TODO: Replace this with ToJSONList
# JSONList will expect the response to be a 
# list of JSON objects. These can be streamed however
# [
#    {}, # object 1
#    {} # object 2
# ]
# it has to detect when 
# also include a way to create the 
# It can take int a ModuleList of StructConvs
# but still be able to return 
class TupleOut(ToOut):

    convs: ModuleList
    parser: Parser

    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming response"""
        # Simple implementation - parse and process each part with its converter
        text = str(resp.text) if hasattr(resp, 'text') else str(resp)
        
        # Use parser to split the text into parts
        parts = self.parser.render([text])
        
        if len(parts) != len(self.convs):
            raise RuntimeError(f"Expected {len(self.convs)} parts, got {len(parts)}")
        
        results = []
        for part, conv in zip(parts, self.convs):
            # Create a mock response for each converter
            mock_resp = type('MockResp', (), {'text': part})()
            results.append(conv.forward(mock_resp))
        
        return tuple(results)
    
    def delta(
        self, 
        resp, 
        delta_store, 
        is_last = True
    ) -> typing.Any:
        parsed = utils.sub_dict(delta_store, 'parsed')
        i = utils.get_or_set(delta_store, 'i', 0)
        res = self.parser.forward(
            resp, 
            parsed, 
            streamed=True,
            is_last=is_last
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

# TODO: Add a JSONValsOut 
# This will expect a JSON that startswith a dictionary
# and will make it possible to stream 
# one value from the JSON at a time alone with the key
# out = (key, value) <-will set the .out to 




# TODO: Review what this can do
# compared to CSVRowParser
class CSVOut(ToOut):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """
    delimiter: str = ','
    use_header: bool = True

    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming CSV response"""
        import csv
        import io
        
        text = str(resp.text) if hasattr(resp, 'text') else str(resp)
        
        rows = list(csv.reader(io.StringIO(text), delimiter=self.delimiter))
        
        if not rows:
            return []
        
        if self.use_header:
            header = rows[0]
            data_rows = rows[1:]
            return [dict(zip(header, row)) for row in data_rows]
        
        return rows
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
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
            csv.reader(io.StringIO(val), delimiter=self.delimiter)
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
            self.use_header is True 
            and delta_store['header'] is None
        ):
            delta_store['header'] = new_rows.pop(0)
            utils.acc(delta_store, 'row', 1)

        header = delta_store['header']
        utils.acc(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if self.use_header:
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
        writer = csv.writer(output, delimiter=self.delimiter)
        
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


class ToolExecConv(ToOut):
    """Use for converting a tool from a response to execute it.
    
    This converter takes tool call objects and executes them, returning
    the execution results. Works with the unified Resp structure.
    """
    
    def forward(self, resp: Resp) -> typing.Any:
        """Process complete non-streaming tool calls"""
        # Assuming resp has tool calls that can be executed
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return [tool() for tool in resp.tool_calls if hasattr(tool, '__call__')]
        return []
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Execute tool calls and return results.

        Args:
            resp: Tool call data from unified response structure
            delta_store: State storage for streaming
            is_streamed: Whether this is a streaming response
            is_last: Whether this is the final chunk

        Returns:
            Execution results or UNDEFINED if no tools to execute
        """
        if resp is None or resp is utils.UNDEFINED:
            return utils.UNDEFINED
        
        # Handle individual ToolCall objects
        if hasattr(resp, '__call__'):
            return resp()
        
        # Handle list of ToolCall objects
        if isinstance(resp, list):
            return [r() for r in resp if hasattr(r, '__call__')]
        
        return utils.UNDEFINED

    def render(self, data: typing.Any) -> str:
        """Render tool execution result as string"""
        return str(data)

    def template(self) -> str:
        """Template for tool execution output"""
        return "{result}"

    def example(self) -> str:
        """Example tool execution output"""
        return "Tool execution result"





class Parser(Process):
    """Base class for parsers. 
    It converts the input text
    into a list of objects
    """
    def forward(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.List | None:
        pass

    @abstractmethod
    def render(self, data) -> str:
        pass

# TODO: Look into how to get CSVRowParser
# combined with CSVout. Ensure CSVOut
# can process row by row or if it already
# can
class CSVRowParser(Parser):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """

    delimiter: str = ','
    use_header: bool = True

    def forward(
        self, 
        resp, 
        delta_store: typing.Dict=None, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        # resp = self.handle_null(resp, '')
        delta_store = delta_store if delta_store is not None else {}

        val = utils.acc(delta_store, 'val', resp, '')
        row = utils.get_or_set(delta_store, 'row', 0)
        header = utils.get_or_set(
            delta_store, 'header', None
        )
        # Process accumulated data using csv.reader
        # csv_data = io.StringIO(delta_store['val'])

        rows = list(
            csv.reader(io.StringIO(val), delimiter=self.delimiter)
        )
        new_rows = []
        for i, row in enumerate(rows[row:]):  # Only return new rows
            new_rows.append(row)

        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if not is_last:
            new_rows.pop()

        if len(new_rows) == 0:
            return utils.UNDEFINED

        if (
            self.use_header is True 
            and delta_store['header'] is None
        ):
            delta_store['header'] = new_rows.pop(0)
            utils.acc(delta_store, 'row', 1)

        header = delta_store['header']
        utils.acc(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if self.use_header:
            return [OrderedDict(zip(header, row)) for row in new_rows]
        return new_rows

    def render(self, data) -> str:
        """

        Args:
            data: An iterable of rows. If header is set to true

        Returns:
            str: the rendered CSV
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter)
        
        if self.use_header:
            header = [key for key, _ in data[0]]
            writer.writerow(header)
            for row in data:
                writer.writerow([value for _, value in row])
        else:
            for row in data:
                writer.writerow(row)
        
        return output.getvalue()

# Review this and see how it differs
# from KVParser
class CharDelimParser(Parser):
    """Parses based on a defined character
    """
    sep: str = ','

    def forward(
        self, 
        resp, 
        delta_store: typing.Dict=None, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.List | None:
        
        delta_store = delta_store if delta_store is not None else {}
        # resp = self.handle_null(resp, '')
        resp = resp or ''
        val = utils.acc(delta_store, 'val', resp)
        res = val.split(self.sep)
        return_val = utils.UNDEFINED
        
        if len(res) > 0:
            if len(val) == 0:
                return_val = []
            elif val[-1] == self.sep:
                return_val = res[:-1]
                delta_store['val'] = ''
            elif is_last:
                return_val = res
                delta_store['val'] = ''
            else:
                return_val = res[:-1]
                delta_store['val'] = res[-1]
                
        return return_val

    def render(self, data) -> str:
        """Renders the data separated by lines

        Args:
            data: The data to render

        Returns:
            str: The rendered data
        """
        
        return f'{self.sep}'.join(data)


# Make a ToOut based on this
class LineParser(Parser):
    """
    Parses line by line. Can have a line continue by putting a backslash at the end of the line.
    """

    def forward(
        self, 
        resp, 
        delta_store: typing.Dict=None, 
        streamed: bool=False, 
        is_last: bool=True
    ) -> typing.List:
        """

        Args:
            resp : 
            delta_store (typing.Dict): 
            is_last (bool, optional): . Defaults to False.

        Returns:
            typing.Any: 
        """ 
        delta_store = delta_store if delta_store is not None else {}
        resp = resp or ''
        utils.acc(delta_store, 'val', resp)
        lines = delta_store['val'].splitlines()
        buffer = []

        if is_last and len(lines) == 0:
            return []
        final_ch = delta_store.get('val', '')[-1] if delta_store.get('val', '') else ''
        buffered_lines = []
        for i, line in enumerate(lines):

            if not line:
                continue  # skip empty lines if that's desired

            if line.endswith("\\"):
                buffer.append(line[:-1])
            else:

                buffer.append(line)
                buffered_lines.append(buffer)
                # logical_line = "".join(buffer)
                # logical_line = "\n".join(buffer)
                # result.append(logical_line)
                buffer = []

        if not is_last and len(buffered_lines) > 0:
            if final_ch == '\n':
                buffered_lines[-1].append('\n')
            delta_store['val'] = ''.join(buffered_lines[-1])
            buffered_lines.pop(-1)

        if len(buffered_lines) == 0:
            return utils.UNDEFINED

        return [
            ''.join(line)
            for line in buffered_lines
        ]

    def render(self, data) -> str:
        """Render the data

        Args:
            data: The data to render

        Returns:
            str: The data rendered in lines
        """
        res = []
        for d in data:
            res.append(d.replace('\n', '\\n'))
        return f'\n'.join(data)





# class ParsedOut(ToOut):
#     """A Reader that does not change the data. 
#     So in most cases will simply output a string
#     """

#     parser: Parser

#     def render(self, data: typing.Any) -> str:
#         """Output an example of the data

#         Args:
#             data (typing.Any): 

#         Returns:
#             str: 
#         """
#         return str(data)

#     def delta(
#         self, 
#         resp, 
#         delta_store: typing.Dict, 
#         streamed: bool=False, 
#         is_last: bool=False
#     ) -> typing.Any:
#         """Read in the output

#         Args:
#             message (str): The message to read

#         Returns:
#             typing.Any: The output of the reader
#         """
#         return self._parser(
#             resp, delta_store, streamed=streamed,
#             is_last=is_last
#         )

#     def template(self) -> str:

#         return self._parser.render(
#             ['<Template 1>', '<Template 2>']
#         )

#     def example(self):
        
#         return self._parser.render(
#             ['Example 1', 'Example 2']
#         )
