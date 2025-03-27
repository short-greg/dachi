# 1st party
import typing
import json
from abc import ABC, abstractmethod
import inspect

# 3rd party
import pydantic

# local
from ..msg import render, struct_template, Msg
from ._asst import StreamAssist, AsyncAssist, Assist, AsyncStreamAssist
from ._parse import Parser, NullParser
from ..base import Templatable, TemplateField
from ..utils import unescape_curly_braces
from pydantic_core import PydanticUndefined
from ..utils import (
    escape_curly_braces
)
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


class OutConv(pydantic.BaseModel, Templatable, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """

    name: str = ''

    @abstractmethod
    def delta(self, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

    def stream(
        self, resp_iterator: typing.Iterator, 
        parser: Parser=None,
        delta_store: typing.Dict=None, 
        pdelta_store: typing.Dict=None,
        get_msg: bool=False
    ) -> typing.Iterator:
        
        parser = parser or NullParser()
        for msg, resp in parser.stream(
            resp_iterator, pdelta_store, True
        ):
            resp = self.delta(resp, delta_store)
            if resp is not utils.UNDEFINED:
                if get_msg:
                    yield msg, resp
                else:
                    yield resp

    async def astream(
        self, resp_iterator: typing.Iterator, 
        parser: Parser=None,
        delta_store: typing.Dict=None, 
        pdelta_store: typing.Dict=None,
        get_msg: bool=False
    ) -> typing.AsyncIterator:
        
        parser = parser or NullParser()
        async for msg, resp in await parser.astream(
            resp_iterator, pdelta_store, True
        ):
            resp = self.delta(resp, delta_store)
            if resp is not utils.UNDEFINED:
                if get_msg:
                    yield msg, resp
                else:
                    yield resp


# def stream_out(asst: StreamAssist, *args, parser: Parser=None, **kwargs) -> typing.Tuple[Msg, typing.Any]:


        delta_store = delta_store or {}
        for resp in resp_iterator:
            yield self.delta(resp, delta_store)


    def __call__(self, resp) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return self.delta(
            resp, {}
        )

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

    def delta(self, resp, delta_store: typing.Dict) -> typing.Any:
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

    def delta(self, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """

        if self._out_cls is bool:
            return resp.lower() in ('true', 'y', 'yes', '1', 't')
        return self._out_cls(resp)

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

    def delta(self, resp, delta_store: typing.Dict) -> typing.Any:
    # def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        
        # if resp is not END_TOK:
        #     utils.add(delta_store, 'val', resp)
        #     return utils.UNDEFINED
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


class KVConv(OutConv):
    """Create a Reader of a list of key values
    """
    sep: str = '::'
    key_descr: typing.Optional[typing.Union[typing.Type[pydantic.BaseModel], typing.Dict]] = None

    def delta(
        self, resp, delta_store: typing.Dict
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        result = {}
        for line in resp:
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


class IndexConv(OutConv):
    """Create a Reader of a list of key values
    """
    sep: str = '::'
    key_descr: typing.Optional[str] = None
    key_type: typing.Optional[typing.Type] = None

    def delta(self, resp, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        
        result = []
        for line in resp:
            try:
                idx, value = line.split(self.sep)
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


class JSONConv(OutConv):
    """Use to read from a JSON
    """
    key_descr: typing.Optional[typing.Dict] = None

    def delta(self, resp, delta_store: typing.Dict) -> typing.Any:
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
        val = utils.add(
            delta_store, 'val', resp
        )
        try: 
            result = json.loads(val)
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

