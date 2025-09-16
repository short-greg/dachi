"""
This module provides classes for processing and converting AI responses into various output formats. 

They are used to set the "out" member on the Resp class.

Example:
to_out = TextOut()
resp.out = to_out(resp) # the LLM returns a resp, some can also take in a string value

The interface for a ToOut is

- forward(resp: str | None) -> typing.Any
- delta(resp: str | None, delta_store: typing.Dict, is_last: bool) -> typing.Any
- example() -> str # used for prompting with an example of the object
- template() -> str # used for prompting with the template of the object
- render(data) -> str 
"""


# 1st party 
from abc import abstractmethod
import typing
import inspect
import json
import csv
import io
from collections import OrderedDict

# 3rd party
import pydantic

# local
from ..core import Resp, render
from ._process import Process
from .. import utils
from ._parser import LineParser, Parser

from ..core import (
    Templatable, 
    ExampleMixin, ModuleList,
    render, 
    Resp, struct_template, END_TOK
)



# local
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

# TODO: Decide where to put this
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
    """ToOut is used to set an AI response (Resp)'s out member.
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
        """Use for prompting an LLM with the template

        Returns:
            str: The template as a string
        """
        pass

    @abstractmethod
    def example(self) -> str:
        """Generate example output for the data

        Returns:
            str: Example output as a string
        """
        pass    

    def coalesce_resp(self, resp, default='') -> str:
        """Coalesce the response into a single string

        Args:
            resp : 

        Returns:
            str: 
        """
        if resp is utils.UNDEFINED or resp is None:
            return default
        
        return resp
    
    @abstractmethod
    def forward(
        self, 
        resp: str | None
    ) -> typing.Any:
        """Process streaming response chunks
        
        Args:
            resp: Response chunk containing data in resp.data['response']
            delta_store: State storage for accumulating across chunks
            is_last: Whether this is the final chunk
            
        Returns:
            Processed result or UNDEFINED if not ready
        """
        pass

    @abstractmethod
    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool = True
    ) -> typing.Any:
        """Process streaming response chunks
        
        Args:
            resp: Response chunk containing data in resp.data['response']
            delta_store: State storage for accumulating across chunks
            is_last: Whether this is the final chunk
            
        Returns:
            Processed result or UNDEFINED if not ready
        """
        pass



class PrimOut(ToOut):
    """Use for converting an AI response into a primitive value. This will be a str, int, float or bool. When streaming, It will wait until the entire response is received before processing.

    Args:
        out_cls: The name of the primitive class to convert to. One of 'str', 'int', 'float', 'bool' or the actual type.

    """
    out_cls: str
    prim_map: typing.ClassVar[typing.Dict[str, typing.Callable]] = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool
    }
    
    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming response
        
        resp (Resp): Complete response to process
        """
        if resp is None:
            return utils.UNDEFINED

        if isinstance(self.out_cls, typing.Type):
            return self.out_cls(resp)

        if self.out_cls == 'bool':
            return resp.lower() in ('true', 'y', 'yes', '1', 't')
        return self.prim_map[self.out_cls](resp)

    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming response chunks
        
        Args:
            resp (Resp): The response object to process
            delta_store (typing.Dict): The dictionary to store deltas
            is_last (bool, optional): Whether this is the last chunk. Defaults to True. 
        """
        resp = self.coalesce_resp(resp)

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
            data (typing.Any): The primitive data to render

        Returns:
            str: The rendered data
        """
        return str(data)
    
    def example(self) -> str:
        """Generate example output based on the specified primitive type
        Returns:
            str: Example output as a string
        """
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

    Example:
        kv_out = KVOut(sep='::', key_descr={'name': 'The name of the person', 'age': 'The age of the person'})
        resp = Resp(msg=Msg(role='assistant', text='name::John Doe\nage::30'))
        data = kv_out.forward(resp)
        print(data)
    """
    sep: str = '::'
    key_descr: typing.Type[pydantic.BaseModel] | None = None
    
    def __post_init__(
        self
    ):
        """
            Initializes the KVOut instance.
            Creates a LineProcessor for parsing
            into lines that will be used for
            determining the keys and values.
        """
        super().__post_init__()
        self.line_parser = LineParser()

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming response

        Args:
            resp (Resp): The response object to process
        """
        if resp is None:
            return utils.UNDEFINED
        
        lines = resp.strip().split('\n')

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
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """
        Process streaming response chunks
        Args:
            resp (Resp): The response object to process
            delta_store (typing.Dict): The dictionary to store deltas
            is_last (bool, optional): Whether this is the last chunk. Defaults to True.

        Raises:
            RuntimeError: If processing fails

        Returns:
            typing.Any: The processed result
        """
        
        resp = self.coalesce_resp(resp)
        line_store = utils.get_or_set(delta_store, 'lines', {})
        res = self.line_parser.delta(resp, line_store, streamed=True, is_last=is_last)

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


class StructOut(ToOut):
    """
    Unified structured data converter for JSON/structured outputs.
    Works with the unified Resp structure.
    """
    struct: typing.Union[pydantic.BaseModel, typing.Dict, None] = None

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming structured response"""
        if resp is None:
            return utils.UNDEFINED

        try:
            parsed_json = json.loads(resp)

            # If struct is set, use it to process the parsed JSON
            if self.struct is not None:
                if inspect.isclass(self.struct) and issubclass(self.struct, pydantic.BaseModel):
                    # Use pydantic validation
                    return self.struct.model_validate(parsed_json)
                elif isinstance(self.struct, dict):
                    # Use dict schema (for now just return parsed JSON)
                    return parsed_json
            
            return parsed_json
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse JSON: {e}")

    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool = True
    ) -> typing.Any:
        """Process structured data from unified response."""
        resp = self.coalesce_resp(resp)
        utils.acc(delta_store, 'content', resp)

        if is_last:
            try:
                struct = json.loads(delta_store.get('content', '{}'))
                return struct
            except json.JSONDecodeError:
                return {}

        return utils.UNDEFINED

    def render(self, data: typing.Any) -> str:
        """Render structured data as JSON string"""
        return json.dumps(data, indent=2)

    def template(self) -> str:
        """Template for structured output"""
        return '{"key": "value"}'

    def example(self) -> str:
        """Generate example output for structured data"""
        raise RuntimeError("StructOut.example() not implemented - cannot generate examples for arbitrary pydantic models")


class TextOut(ToOut):
    """Use for converting an AI response text - good for streaming text output
    """

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming text response
        
        resp: Complete response to process
        """
        if resp is None:
            return utils.UNDEFINED
        return resp

    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool = True
    ) -> typing.Any:
        """Process streaming text chunks - returns immediately for good streaming"""
        if resp is None:
            return ''
        return resp

    def render(self, data: typing.Any) -> str:
        """Output an example of the data"""
        return str(data)

    def template(self) -> str:
        """Output the template for the string"""
        return '<text>'

    def example(self) -> str:
        """Generate example text output"""
        return "example text"


class IndexOut(ToOut):
    """Create a Reader of a list of key values
    """
    sep: str = '::'
    key_descr: str = ''
    key_type: typing.Type[pydantic.BaseModel] | typing.Dict | None = None

    def __post_init__(self):
        super().__post_init__()
        self.line_parser = LineParser()

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming response
        
        resp (Resp): The response object to process
        """
        if resp is None:
            return utils.UNDEFINED
        lines = resp.strip().split('\n')

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
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming response chunks
        Args:
            resp (Resp): The response object to process
            delta_store (typing.Dict): The dictionary to store deltas
            is_last (bool, optional): Whether this is the last chunk. Defaults to True.
        Raises:
            RuntimeError: If processing fails
        Returns:
            typing.Any: The processed result
        """
        resp = self.coalesce_resp(resp)
        line_store = utils.get_or_set(
            delta_store, 'lines', {}
        )
        resp = self.line_parser.delta(resp, line_store, streamed=True, is_last=is_last)
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
        """Generate an example output for the tuple.

        Returns:
            str: An example output string.
        """
        data = ["Data 1", "Data 2", "Data 3"]
        return self.render(data)


class JSONListOut(ToOut):
    """Processes JSON arrays, handling streaming of individual JSON objects.
    
    Example:
        json_list_out = JSONListOut(model_cls=MyModel)
        resp = Resp(msg=Msg(role='assistant', text='[{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]'))
        data = json_list_out.forward(resp)
        print(data)  # Output: [MyModel(name='John', age=25), MyModel(name='Jane', age=30)]
    """
    
    model_cls: typing.Type[pydantic.BaseModel] | None = None
    
    def forward(self, resp: str | None) -> typing.Any:
        """Process complete JSON array response"""
        if resp is None:
            return utils.UNDEFINED
        
        try:
            data = json.loads(resp)
            if not isinstance(data, list):
                raise RuntimeError("Expected JSON array, got different type")
            
            if self.model_cls:
                return [self.model_cls(**item) if isinstance(item, dict) else item for item in data]
            return data
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON array: {e}") from e
    
    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming JSON array chunks, returning individual objects as they complete"""
        resp = self.coalesce_resp(resp)
        val = utils.acc(delta_store, 'val', resp)
        
        processed_count = utils.get_or_set(delta_store, 'processed_count', 0)
        
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list) and len(parsed) > processed_count:
                new_items = parsed[processed_count:]
                if self.model_cls:
                    new_items = [
                        self.model_cls(**item) if isinstance(item, dict) else item 
                        for item in new_items
                    ]
                delta_store['processed_count'] = len(parsed)
                return new_items
        except json.JSONDecodeError:
            pass
        
        return utils.UNDEFINED
    
    def render(self, data: typing.Any) -> str:
        """Render data as JSON array"""
        return json.dumps(data, indent=2)
    
    def template(self) -> str:
        """Template for JSON array output"""
        if self.model_cls:
            schema = self.model_cls.model_json_schema()
            return f'[{json.dumps(schema.get("properties", {}))}, ...]'
        return '[{"key": "<value>"}, ...]'
    
    def example(self) -> str:
        """Example JSON array output"""
        if self.model_cls:
            try:
                example_obj = self.model_cls()
                return self.render([example_obj.model_dump(), example_obj.model_dump()])
            except Exception:
                pass
        return self.render([{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]) 


class TupleOut(ToOut):
    """Processes a response into a tuple of values using specified processors and a parser.
    
    Each processor must inherit from Process, take in a string, and output some value.
    
    Example:
        class StrProcessor(Process):
            def forward(self, s: str) -> str:
                return s.strip()
                
        class IntProcessor(Process):
            def forward(self, s: str) -> int:
                return int(s.strip())
        
        tuple_out = TupleOut(
            processors=ModuleList([StrProcessor(), IntProcessor()]),
            parser=LineParser()
        )
        resp = Resp(msg=Msg(role='assistant', text='Hello\n42'))
        data = tuple_out.forward(resp)
        print(data)  # Output: ('Hello', 42)

        resp1 = Resp(msg=Msg(role='assistant', text='Hello\n'))
        resp2 = Resp(msg=Msg(role='assistant', text='42'))
        print("Streaming tuple:")
        # For streaming, use delta method
        tuple_out.delta(resp1, delta_store={})
        # Output: ['Hello']
        tuple_out.delta(resp2, delta_store={}, is_last=True)
        # Output: [42]
    """

    processors: ModuleList # a list of the ToOut processors to use
    parser: Parser

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming response"""
        # Simple implementation - parse and process each part with its processor
        if resp is None:
            return utils.UNDEFINED
        
        # Use parser to split the text into parts
        parts = self.parser.forward(resp)

        if len(parts) != len(self.processors):
            raise RuntimeError(f"Expected {len(self.processors)} parts, got {len(parts)}")
        
        results = []
        for part, processor in zip(parts, self.processors):
            # Note: processors take in a string and output some value
            results.append(processor.forward(str(part)))
        
        return tuple(results)
    
    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming response chunks
        
        Args:
            resp (Resp): The response object to process
            delta_store (typing.Dict): The dictionary to store deltas
            is_last (bool, optional): Whether this is the last chunk. Defaults to True.
        Raises:
            RuntimeError: If processing fails
        """
        resp = self.coalesce_resp(resp)
        parsed = utils.sub_dict(delta_store, 'parsed')
        i = utils.get_or_set(delta_store, 'i', 0)
        res = self.parser.delta(
            resp, 
            parsed, 
            streamed=True,
            is_last=is_last
        )
        if res is utils.UNDEFINED:
            return utils.UNDEFINED

        outs = []
        if is_last and len(res) != len(self.processors[i:]):
            raise RuntimeError(
                "There are more out processors to retrieve than items."
            )
        
        if len(res) > len(self.processors[i:]):
            raise RuntimeError(
                "There are more items to retrieve than out processors."
            )
        for res_i, processor in zip(res, self.processors[i:]):

            # Note: processors take in a string and output some value
            outs.append(processor.forward(str(res_i)))
            utils.acc(delta_store, 'i', 1)
        return outs
        
    def render(self, data) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return self.parser.render(data)

    def template(self):
        """Generate template showing expected format"""
        templates = ["<item1>", "<item2>", "<item3>"]
        return self.parser.render(templates)

    def example(self):
        """Generate simple example of tuple output"""
        data = ["value1", "value2", "value3"]
        return self.parser.render(data)


class JSONValsOut(ToOut):
    """Processes JSON objects, streaming key-value pairs one at a time.
    
    Example:
        json_vals_out = JSONValsOut(processors={'name': PrimOut(out_cls='str'), 'age': PrimOut(out_cls='int')})
        resp = Resp(msg=Msg(role='assistant', text='{"name": "John", "age": 30, "city": "New York"}'))
        data = json_vals_out.forward(resp)
        print(data)
        # Output: [('name', 'John'), ('age', 30), ('city', 'New York')]

        resp1 = Resp(msg=Msg(role='assistant', text='{"name": "John", '))
        resp2 = Resp(msg=Msg(role='assistant', text='"age": 30, '))
        resp3 = Resp(msg=Msg(role='assistant', text='"city": "New York"}'))
        resp4 = Resp(msg=Msg(role='assistant', text='}'))
        print("Streaming JSON values:")
        # For streaming, use delta method
        json_vals_out.delta(resp1, delta_store={})
        # Output: [('name', 'John')]
        json_vals_out.delta(resp2, delta_store={})
        # Output: [('age', 30)]
        json_vals_out.delta(resp3, delta_store={})
        # Output: [('city', 'New York')]
        json_vals_out.delta(resp4, delta_store={}, is_last=True)
        # Output: None (no more new key-value pairs)
    """
    processors: typing.Dict[str, ToOut] | None = None

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete JSON object response"""
        if resp is None:
            return utils.UNDEFINED
        try:
            data = json.loads(resp)
            if isinstance(data, dict):
                result = []
                for k, v in data.items():
                    if self.processors and k in self.processors:
                        # Note: processors assume input is a string
                        processed_v = self.processors[k].forward(str(v))
                        result.append((k, processed_v))
                    else:
                        result.append((k, v))
                return result
            else:
                raise RuntimeError("Expected JSON object, got different type")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON object: {e}") from e
    
    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.Any:
        """Process streaming JSON object chunks, returning new key-value pairs as they complete"""
        resp = self.coalesce_resp(resp)
        val = utils.acc(delta_store, 'val', resp)
        
        processed_keys = utils.get_or_set(delta_store, 'processed_keys', set())
        
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                new_pairs = []
                for k, v in parsed.items():
                    if k not in processed_keys:
                        if self.processors and k in self.processors:
                            # Note: processors assume input is a string
                            processed_v = self.processors[k].forward(str(v))
                            new_pairs.append((k, processed_v))
                        else:
                            new_pairs.append((k, v))
                        processed_keys.add(k)
                
                delta_store['processed_keys'] = processed_keys
                if new_pairs:
                    return new_pairs
        except json.JSONDecodeError:
            pass
        
        return utils.UNDEFINED
    
    def render(self, data: typing.Any) -> str:
        """Render key-value pairs as JSON object"""
        if isinstance(data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in data):
            obj = {k: v for k, v in data}
        else:
            obj = data
        return json.dumps(obj, indent=2)
    
    def template(self) -> str:
        """Template for JSON values output"""
        return '{"key1": "<value1>", "key2": "<value2>"}'
    
    def example(self) -> str:
        """Example JSON values output"""
        return self.render([("name", "John"), ("age", 25), ("city", "Boston")]) 


class CSVOut(ToOut):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.

    Example:
        csv_out = CSVOut(delimiter=',', use_header=True)
        resp = Resp(msg=Msg(role='assistant', text='name,age,city\nJohn,30,Boston\nJane,25,New York'))
        data = csv_out.forward(resp
        print(data)
        # Output: [{'name': 'John', 'age': '30', 'city': 'Boston'}, {'name': 'Jane', 'age': '25', 'city': 'New York'}]
    """
    delimiter: str = ','
    use_header: bool = True

    def forward(self, resp: str | None) -> typing.Any:
        """Process complete non-streaming CSV response"""

        if resp is None:
            return utils.UNDEFINED

        rows = list(csv.reader(io.StringIO(resp), delimiter=self.delimiter))

        if not rows:
            return []
        
        if self.use_header:
            header = rows[0]
            data_rows = rows[1:]
            return [dict(zip(header, row)) for row in data_rows]
        
        return rows
    
    def delta(
        self, 
        resp: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=False
    ) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        # resp = self.handle_null(resp, '')

        # TODO: Response is not currently correct, it is
        # not treating it as a Resp class
        # just get the data from resp.msg.text
        resp = self.coalesce_resp(resp, '')

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
