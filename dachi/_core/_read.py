# 1st party
import typing
import json
import inspect
from abc import ABC, abstractmethod

# 3rd party
from io import StringIO
import pandas as pd
import pydantic
import pydantic
import yaml
import typing
import json

# local
from ..utils import (
    struct_template, model_to_text, 
    unescape_curly_braces, TemplateField, 
    StructLoadException
)
from ._core import TextProc, render
from ._core import (
    TextProc, Templatable,
    render,
)
from ..utils import (
    struct_template,
    model_to_text,
    escape_curly_braces
)
from pydantic_core import PydanticUndefined


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class TextProc(pydantic.BaseModel, Templatable, ABC):
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

    @abstractmethod
    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass
        # return self.load_data(self.read_text(message))

    @abstractmethod
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

    @abstractmethod
    def template(self) -> str:
        """Get the template for the reader

        Returns:
            str: The template as a string
        """
        pass


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


class NullTextProc(TextProc):
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
        return data

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return message

    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return message

    def template(self) -> str:
        return None


class MultiTextProc(TextProc):
    """Concatenates multiple outputs Out into one output
    """
    outs: typing.List[TextProc]
    conn: str = '::OUT::{name}::\n'
    signal: str = '\u241E'

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        results = []
        for data_i, out in zip(data['data'], self.outs):
            results.append(out.dump_data(data_i))
        data = {'data': results, 'i': data['i']}
        result = ''
        for i, (d, out) in enumerate(zip(data['data'], self.outs)):
            name = out.name or str(i) 
            result = result + '\n' + self.signal + self.conn.format(name=name)
            result = f'{result}\n{render(d)}'

        return result

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """     
        structs = []

        d = message
        for i, out in enumerate(self.outs):
            name = out.name or str(i)
            from_loc = d.find('\u241E')
            to_loc = d[from_loc + 1:].find('\u241E')
            cur = self.conn.format(name=name)
            data_loc = from_loc + len(cur) + 1

            if to_loc != -1:
                data_end_loc = from_loc + 1 + to_loc
            else:
                data_end_loc = None

            data_str = d[data_loc:data_end_loc]
            try: 
                structs.append(out.read_text(data_str))
            except StructLoadException as e:
                return {'data': structs, 'i': i}
            d = d[data_end_loc:]

        data = {'data': structs, 'i': i, 'n': len(self.outs)}
        structs = []

        for o, d_i in zip(self.outs, data['data']):
            structs.append(o.load_data(d_i))

        return {'data': structs, 'i': data['i'], 'n': data['n']}
    
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return message

    def template(self) -> str:
        """Output a template for the output

        Args:
            data (): The dictionary data to load

        Returns:
            typing.Dict: The data with each element of the output
             loaded
        """
        texts = []
        for i, out in enumerate(self.outs):
            print(out.name)
            name = out.name or str(i)
            if isinstance(out, TextProc):
                cur = out.template()
            else:
                cur = struct_template(out)
            cur_conn = self.conn.format(name=name)
            texts.append(f"{self.signal}{cur_conn}\n{cur}")
        text = '\n'.join(texts)
        return text


class PrimProc(TextProc):
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
        return str(data)

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if message is None:
            return ''
        if self._out_cls is bool:
            return message.lower() in ('true', 'y', 'yes', '1', 't')
        return self._out_cls(message)

    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass
        # return message

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self._out_cls}>'


class PydanticProc(TextProc, typing.Generic[S]):
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
        return str(data.model_dump())

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        message = unescape_curly_braces(message)
        data = json.loads(message)
        return self._out_cls(**data)

    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass
        # return message

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
        # return self._out_cls.template()
        return render(
            struct_template(self._out_cls), 
            escape_braces, self.template_renderer
        ) 
