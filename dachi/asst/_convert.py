# 1st party
import typing
import json
from abc import ABC, abstractmethod

import typing
import json
import inspect

# 3rd party
import pydantic
from io import StringIO
import pandas as pd
import yaml
import typing
import json

# local
from ..base import Templatable, render, TemplateField, struct_template
from ..utils import (
    unescape_curly_braces
)
from ._messages import END_TOK, Msg
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


class Delim(ABC):

    @abstractmethod
    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        pass


class CharDelim(Delim):
    
    def __init__(self, c: str=','):
        super().__init__()

        self._c = c

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:
        
        delta_store = delta_store or {}
        if resp == self._c:
            return delta_store.get('val', '')
        
        utils.get_or_set(delta_store, 'val', resp)
        utils.call_or_set(delta_store, 'val', resp, lambda x, d: x + d)
        # Decide on what to return here
        return utils.UNDEFINED


class NullDelim():

    def __call__(self, msg, resp, delta_store: typing.Dict) -> typing.Any:

        if resp is END_TOK:
            return utils.UNDEFINED
        
        return resp


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

    # def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
    #     """Default processing for delta. Will simply wait until the end to return the result

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Any: The output of the reader
    #     """
    #     if message is None:
    #         return None
    #     if 'message' not in delta_store:
    #         delta_store['message'] = ''
        
    #     if message is not END_TOK:
    #         delta_store['message'] += (
    #             message if message is not END_TOK else ''
    #         )
    #         return None

    #     return self(delta_store['message'])

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

    # def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
    #     """Read in the output

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Any: The output of the reader
    #     """
    #     if message is END_TOK:
    #         return ''

    #     return message

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
    
    # # TODO: Add
    # def delta(self, message: str, delta_store):
        
    #     if message is None:
    #         message = ''

    #     if self._out_cls is str:
    #         # Handle strings cases
    #         if message is END_TOK:
    #             return ''
    #         return message
    #     else:
    #         # Handle non-string cases
    #         if message is END_TOK:
    #             return self(delta_store['delta'])
    #         else:
    #             delta_store['delta'] += message
    #             return ''

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

        # try:
        #     message = unescape_curly_braces(delta['message'])
        #     data = json.loads(message)
        #     return self._out_cls(**data)
        # except json.JSONDecodeError as e:
        #     raise ReadError('Reading of Pydantic Base Model has failed.', e)

    # def delta(self, message: str, delta_store):
        
    #     if message is None:
    #         message = ''

    #     # Handle non-string cases
    #     if message is END_TOK:
    #         return self(delta_store['delta'])
    #     else:
    #         delta_store['delta'] += message
    #         return ''

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
    
    # def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
    #     """Read in the output

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Any: The output of the reader
    #     """
    #     if message is None:
    #         return None
    #     if 'message' not in delta_store:
    #         delta_store['message'] = ''
    #         delta_store['row'] = 0
    #         delta_store['fields'] = None
    #     if message is not END_TOK:
    #         delta_store['message'] += message
    #     io = StringIO(delta_store['message'])
    #     df = pd.read_csv(io, sep=self.delim)
    #     # print(df)
    #     if (
    #         len(df.index) > 0 
    #         and delta_store['row'] == 0
    #     ):
    #         delta_store['fields'] = list(df.columns.values)
    #     if len(df.index) <= delta_store['row'] and message is not END_TOK:
    #         return None
        
    #     delta_store['row'] = len(df.index)
    #     if len(df.index) == 1:
    #         return None
    #     if message is END_TOK:
    #         df = df.iloc[delta_store['row'] - 1:]
    #     else:
    #         df = df.iloc[delta_store['row'] - 2:delta_store['row'] - 1]
    #     return df.to_dict(orient='records', index=True)

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


# class MultiOutConv(OutConv):
#     """Concatenates multiple outputs Out into one output
#     """
#     outs: typing.List[OutConv]
#     conn: str = '::OUT::{name}::\n'
#     signal: str = '\u241E'

#     def example(self, data: typing.Any) -> str:
#         """Output an example of the data

#         Args:
#             data (typing.Any): 

#         Returns:
#             str: 
#         """
#         results = []
#         for data_i, out in zip(data['data'], self.outs):
#             results.append(out.example(data_i))
#         data = {'data': results, 'i': data['i']}
#         result = ''
#         for i, (d, out) in enumerate(zip(data['data'], self.outs)):
#             name = out.name or str(i) 
#             result = result + '\n' + self.signal + self.conn.format(name=name)
#             result = f'{result}\n{render(d)}'

#         return result

#     def __call__(self, message: str) -> typing.Any:
#         """Read in the output

#         Args:
#             message (str): The message to read

#         Returns:
#             typing.Any: The output of the reader
#         """     
#         structs = []

#         d = message
#         for i, out in enumerate(self.outs):
#             name = out.name or str(i)
#             from_loc = d.find('\u241E')
#             to_loc = d[from_loc + 1:].find('\u241E')
#             cur = self.conn.format(name=name)
#             data_loc = from_loc + len(cur) + 1

#             if to_loc != -1:
#                 data_end_loc = from_loc + 1 + to_loc
#             else:
#                 data_end_loc = None

#             data_str = d[data_loc:data_end_loc]
#             try: 
#                 structs.append(out(data_str))
#             except ReadError as e:
#                 return {'data': structs, 'i': i}
#             d = d[data_end_loc:]

#         data = {'data': structs, 'i': i, 'n': len(self.outs)}
#         return data
#         # structs = []

#         # for o, d_i in zip(self.outs, data['data']):
#         #     structs.append(d_i)

#         # return {'data': structs, 'i': data['i'], 'n': data['n']}
    
#     def delta(self, message: str, delta_store: typing.Dict) -> typing.Any:
#         """Read in the output

#         Args:
#             message (str): The message to read

#         Returns:
#             typing.Any: The output of the reader
#         """
#         if 'message' not in delta_store:
#             delta_store['i'] = 0
#             delta_store['structs'] = []
#             delta_store['message'] = ''

#         if message is not None and message is not END_TOK:
#             delta_store['message'] += message

#         i = delta_store['i']
#         if i >= len(self.outs):
#             return None
        
#         d = delta_store['message']
#         out = self.outs[i]
#         name = out.name or str(i)
#         from_loc = d.find('\u241E')
#         to_loc = d[from_loc + 1:].find('\u241E')
#         if to_loc == -1 and message is not END_TOK:
#             return None

#         cur = self.conn.format(name=name)
#         data_loc = from_loc + len(cur) + 1

#         if to_loc != -1:
#             data_end_loc = from_loc + 1 + to_loc
#         else:
#             data_end_loc = None

#         data_str = d[data_loc:data_end_loc]
#         try:
#             delta_store['structs'].append(out(data_str))
#             # print('RES ')
#             # print(d)
#             # print('After')
#             # print(d[data_end_loc:])
#             delta_store['message'] = d[data_end_loc:]
#             delta_store['i'] += 1
#         except ReadError as e:
#             return None
#             # raise ReadError('Reading of MultiTextProc has failed.', e)
        
#         return delta_store['structs'][i]

#     def template(self) -> str:
#         """Output a template for the output

#         Args:
#             data (): The dictionary data to load

#         Returns:
#             typing.Dict: The data with each element of the output
#              loaded
#         """
#         texts = []
#         for i, out in enumerate(self.outs):
#             print(out.name)
#             name = out.name or str(i)
#             if isinstance(out, OutConv):
#                 cur = out.template()
#             else:
#                 cur = struct_template(out)
#             cur_conn = self.conn.format(name=name)
#             texts.append(f"{self.signal}{cur_conn}\n{cur}")
#         text = '\n'.join(texts)
#         return text
