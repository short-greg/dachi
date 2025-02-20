# 1st party
import typing
import json
from abc import ABC, abstractmethod

# 3rd party
import pydantic

# local
from .._core import (
    struct_template,
)
from ..utils import unescape_curly_braces
from .._core import (
    render, Templatable, END_TOK, TemplateField, 
    struct_template
)
from ..data import Msg
from pydantic_core import PydanticUndefined

S = typing.TypeVar('S', bound=pydantic.BaseModel)



class RespProc(ABC):
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
    def delta(self, response, msg: Msg, delta_store: typing.Dict) -> typing.Any: 
        pass

    def prep(self) -> typing.Dict:
        return {}


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

    def dump_data(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        pass

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

    def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
        """Default processing for delta. Will simply wait until the end to return the result

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if message is None:
            return None
        if 'message' not in delta_store:
            delta_store['message'] = ''
        
        if message is not END_TOK:
            delta_store['message'] += (
                message if message is not END_TOK else ''
            )
            return None

        return self(delta_store['message'])

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
        return str(data)

    def dump_data(self, data: typing.Any) -> str:
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

    def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if message is END_TOK:
            return None

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
            results.append(out.example(data_i))
        data = {'data': results, 'i': data['i']}
        result = ''
        for i, (d, out) in enumerate(zip(data['data'], self.outs)):
            name = out.name or str(i) 
            result = result + '\n' + self.signal + self.conn.format(name=name)
            result = f'{result}\n{render(d)}'

        return result

    # def dump_data(self, data: typing.Any) -> str:
    #     """Output an example of the data

    #     Args:
    #         data (typing.Any): 

    #     Returns:
    #         str: 
    #     """
    #     results = []
    #     for data_i, out in zip(data['data'], self.outs):
    #         results.append(out.dump_data(data_i))
    #     data = {'data': results, 'i': data['i']}
    #     result = ''
    #     for i, (d, out) in enumerate(zip(data['data'], self.outs)):
    #         name = out.name or str(i) 
    #         result = result + '\n' + self.signal + self.conn.format(name=name)
    #         result = f'{result}\n{render(d)}'

    #     return result

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
                structs.append(out(data_str))
            except ReadError as e:
                return {'data': structs, 'i': i}
            d = d[data_end_loc:]

        data = {'data': structs, 'i': i, 'n': len(self.outs)}
        return data
        # structs = []

        # for o, d_i in zip(self.outs, data['data']):
        #     structs.append(d_i)

        # return {'data': structs, 'i': data['i'], 'n': data['n']}
    
    def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if 'message' not in delta_store:
            delta_store['i'] = 0
            delta_store['structs'] = []
            delta_store['message'] = ''

        if message is not None and message is not END_TOK:
            delta_store['message'] += message

        i = delta_store['i']
        if i >= len(self.outs):
            return None
        
        d = delta_store['message']
        out = self.outs[i]
        name = out.name or str(i)
        from_loc = d.find('\u241E')
        to_loc = d[from_loc + 1:].find('\u241E')
        if to_loc is -1 and message is not END_TOK:
            return None

        cur = self.conn.format(name=name)
        data_loc = from_loc + len(cur) + 1

        if to_loc != -1:
            data_end_loc = from_loc + 1 + to_loc
        else:
            data_end_loc = None

        data_str = d[data_loc:data_end_loc]
        try:
            delta_store['structs'].append(out(data_str))
            print('RES ')
            print(d)
            print('After')
            print(d[data_end_loc:])
            delta_store['message'] = d[data_end_loc:]
            delta_store['i'] += 1
        except ReadError as e:
            return None
            # raise ReadError('Reading of MultiTextProc has failed.', e)
        
        return delta_store['structs'][i]

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
        return str(self.dump_data())

    def dump_data(self, data: typing.Any) -> str:
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
        if message is None:
            return None
        
        if self._out_cls is bool:
            return message.lower() in ('true', 'y', 'yes', '1', 't')
        return self._out_cls(message)

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
        return data.model_dump_json()

    def dump_data(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return data.model_dump_json()

    def __call__(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        message = unescape_curly_braces(message)
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            raise ReadError(
                f'Could not read in {message} as a JSON', e
            )
        return self._out_cls(**data)

        # try:
        #     message = unescape_curly_braces(delta['message'])
        #     data = json.loads(message)
        #     return self._out_cls(**data)
        # except json.JSONDecodeError as e:
        #     raise ReadError('Reading of Pydantic Base Model has failed.', e)

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
