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

    # @abstractmethod
    # def dump_data(self, data: typing.Any) -> typing.Any:
    #     """Convert the data from the output of write_text
    #     to the original format

    #     Args:
    #         data (typing.Any): The data

    #     Returns:
    #         typing.Any: The data
    #     """
    #     pass

    # @abstractmethod
    # def write_text(self, data: typing.Any) -> str:
    #     """Write out the text for the data

    #     Args:
    #         data (typing.Any): The data to write the text for

    #     Returns:
    #         str: The text
    #     """
    #     pass

    # @abstractmethod
    # def read_text(self, message: str) -> typing.Any:
    #     """Read in the text and output to a "json" compatible format or a primitive

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Any: The result of the reading process
    #     """
    #     pass

    # @abstractmethod
    # def load_data(self, data: typing.Dict) -> typing.Any:
    #     """Load the data output from reading the text

    #     Args:
    #         data (typing.Dict): The data to load (JSON format)

    #     Returns:
    #         typing.Any: The result of the reading
    #     """
    #     pass


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

    # def dump_data(self, data: typing.Any) -> typing.Any:
    #     """Convert the data to JSON compatible format

    #     Args:
    #         data (typing.Any): The data to convert to "JSON" compatible format

    #     Returns:
    #         typing.Any: Returns the data passed in
    #     """
    #     return data

    # def write_text(self, data: typing.Any) -> str:
    #     """Output the data to text

    #     Args:
    #         data (typing.Any): The JSON compatible data

    #     Returns:
    #         str: The data converted to text
    #     """
    #     return data

    # def read_text(self, data: str) -> typing.Dict:
    #     """Read in the text as a JSON compatible structure

    #     Args:
    #         data (str): The data to read in

    #     Returns:
    #         typing.Dict: The JSON compatible object (does nothing because it is null)
    #     """
    #     return data
    
    # def load_data(self, data) -> typing.Any:
    #     """Load the data

    #     Args:
    #         data: The data to load

    #     Returns:
    #         typing.Any: The data passed in (since null)
    #     """
    #     return data


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

    # def dump_data(self, data: typing.Dict) -> typing.Dict:
    #     """Dump all of the data to a dictionary

    #     Args:
    #         data (typing.Dict): A dictionary containing all
    #         of the data

    #     Returns:
    #         typing.Dict: The data all dumped
    #     """
    #     results = []
    #     for data_i, out in zip(data['data'], self.outs):
    #         results.append(out.dump_data(data_i))
    #     return {'data': results, 'i': data['i']}

    # def write_text(self, data: typing.Dict) -> str:
    #     """Convert the dictionary data to text

    #     Args:
    #         data (typing.Dict): The dictionary data obtained from dump_data

    #     Returns:
    #         str: The data written as text
    #     """
    #     result = ''
    #     for i, (d, out) in enumerate(zip(data['data'], self.outs)):
    #         name = out.name or str(i) 
    #         result = result + '\n' + self.signal + self.conn.format(name=name)
    #         result = f'{result}\n{render(d)}'

    #     return result

    # def to_text(self, data: typing.List[S]) -> str:
    #     """Convert the data to text
    #     """

    #     text = ""
    #     for i, (data_i, out) in enumerate(zip(data, self.outs)):
    #         name = out.name or str(i) 
    #         cur = render(data_i)
    #         cur_conn = self.conn.format(name)
    #         text += f"""
    #         {self.signal}{cur_conn}
    #         {cur}
    #         """
    #     return text

    # def read_text(self, message: str) -> typing.Dict:
    #     """

    #     Args:
    #         message (str): The message containing multiple outputs

    #     Returns:
    #         typing.Dict: The outputs contained in a dictionary
    #     """
    #     structs = []

    #     d = message
    #     for i, out in enumerate(self.outs):
    #         name = out.name or str(i)
    #         from_loc = d.find('\u241E')
    #         to_loc = d[from_loc + 1:].find('\u241E')
    #         cur = self.conn.format(name=name)
    #         data_loc = from_loc + len(cur) + 1

    #         if to_loc != -1:
    #             data_end_loc = from_loc + 1 + to_loc
    #         else:
    #             data_end_loc = None

    #         data_str = d[data_loc:data_end_loc]
    #         try: 
    #             structs.append(out.read_text(data_str))
    #         except StructLoadException as e:
    #             return {'data': structs, 'i': i}
    #         d = d[data_end_loc:]

    #     return {'data': structs, 'i': i, 'n': len(self.outs)}
    
    # def load_data(self, data: typing.Dict) -> typing.Dict:
    #     """Load the data for each of the components of the output

    #     Args:
    #         data (): The dictionary data to load

    #     Returns:
    #         typing.Dict: The data with each element of the output
    #          loaded
    #     """
    #     structs = []

    #     for o, d_i in zip(self.outs, data['data']):
    #         structs.append(o.load_data(d_i))

    #     return {'data': structs, 'i': data['i'], 'n': data['n']}

    # def template(self) -> str:
    #     """Output a template for the output

    #     Args:
    #         data (): The dictionary data to load

    #     Returns:
    #         typing.Dict: The data with each element of the output
    #          loaded
    #     """
    #     texts = []
    #     for i, out in enumerate(self.outs):
    #         print(out.name)
    #         name = out.name or str(i)
    #         if isinstance(out, TextProc):
    #             cur = out.template()
    #         else:
    #             cur = struct_template(out)
    #         cur_conn = self.conn.format(name=name)
    #         texts.append(f"{self.signal}{cur_conn}\n{cur}")
    #     text = '\n'.join(texts)
    #     return text


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

    # def to_text(self, data: typing.Any) -> str:
    #     """Convert the data to text

    #     Args:
    #         data (typing.Any): The data to convert

    #     Returns:
    #         str: The text
    #     """
    #     return str(data)

    # def dump_data(self, data: typing.Any) -> typing.Any:
    #     """

    #     Args:
    #         data (typing.Any): The data to dump

    #     Returns:
    #         typing.Any: The data (since it is a primitive, does nothing)
    #     """
    #     return data

    # def write_text(self, data: typing.Any) -> str:
    #     """Convert the primitive to a string

    #     Args:
    #         data (primitive): The primitive

    #     Returns:
    #         str: The data converted to a string
    #     """
    #     return str(data)

    # def read_text(self, message: str) -> typing.Any:
    #     """Read the message from text

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Any: The string converted to the primitive
    #     """
    #     if message is None:
    #         return ''
    #     if self._out_cls is bool:
    #         return message.lower() in ('true', 'y', 'yes', '1', 't')
    #     return self._out_cls(message)
    
    # def load_data(self, data) -> typing.Any:
    #     """Doesn't do anything because the data should be in
    #     the right form

    #     Args:
    #         data: The primitive

    #     Returns:
    #         typing.Any: The primitive value
    #     """
    #     return data

    # def template(self) -> str:
    #     """Output the template for the string

    #     Returns:
    #         str: The template for the data
    #     """
    #     return f'<{self._out_cls}>'


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

    # def dump_data(self, data: S) -> typing.Any:
    #     """Convert the data to a dictionary

    #     Args:
    #         data (S): The data to convert

    #     Returns:
    #         typing.Any: The result
    #     """
    #     return data.model_dump()

    # def write_text(self, data: typing.Any) -> str:
    #     """Convert the dumped data to a string

    #     Args:
    #         data (typing.Any): The data to convert to a string

    #     Returns:
    #         str: The convered data
    #     """
    #     return str(data)

    # def to_text(self, data: S) -> str:
    #     """Convert the data to text

    #     Args:
    #         data (S): The data to convert

    #     Returns:
    #         str: The 
    #     """
    #     return model_to_text(data, True)

    # def read_text(self, message: str) -> S:
    #     """Read in text from the message

    #     Args:
    #         message (str): Read in the text from 

    #     Returns:
    #         S: The BaseModel specified by S
    #     """
    #     message = unescape_curly_braces(message)
    #     return json.loads(message)
    #     # return self._out_cls.from_text(message, True)
    
    # def load_data(self, data) -> S:
    #     """Load data from the 

    #     Args:
    #         data: The data to load

    #     Returns:
    #         S: the loaded data
    #     """
    #     return self._out_cls(**data)
    
    # def template_renderer(self, template: TemplateField) -> str:
    #     """Render the template for the BaseModel

    #     Args:
    #         template (TemplateField): The template to render

    #     Returns:
    #         str: The template
    #     """
    #     t = f'<{template.description}> - type: {template.type_}'
    #     if template.default is not None or template.default == PydanticUndefined:
    #         t = f'{t} {template.default}>'
    #     else:
    #         t = f'{t}>'
    #     return t

    # def template(self, escape_braces: bool=False) -> str:
    #     """Convert the object to a template

    #     Args:
    #         escape_braces (bool, optional): Whether to escape curly brances. Defaults to False.

    #     Returns:
    #         str: Escape the braces
    #     """
    #     # return self._out_cls.template()
    #     return render(struct_template(self._out_cls), escape_braces, self.template_renderer) 


class CSVProc(TextProc):
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
    
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

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

    # def read_text(self, message: str) -> typing.Dict[str, typing.Any]:
    #     """Read in the CSV

    #     Args:
    #         message (str): The message to read
    #     """
    #     io = StringIO(message)
    #     df = pd.read_csv(io, sep=self.delim)
    #     return df.to_dict(orient='records', index=True)

    # def load_data(self, data) -> typing.Dict:
    #     """Convert the message to a StructList

    #     Args:
    #         message (str): The message to convert

    #     Returns:
    #         StructList[S]: The result
    #     """
    #     return data # StructList[S].load_records(data)

    # def dump_data(self, data: typing.List) -> typing.Any:
    #     """Doesn't do anything because write_text expects a list

    #     Args:
    #         data (typing.List): the data to dump

    #     Returns:
    #         typing.Any: the data
    #     """
    #     return data

    # def write_text(self, data: typing.Any) -> str:
    #     """Write the data to a CSV

    #     Args:
    #         data (StructList[S]): The data to write

    #     Returns:
    #         str: The data as a CSV
    #     """
    #     io = StringIO()
    #     data = [
    #         d_i.dump()
    #         for d_i in self.data.structs
    #     ]
    #     df = pd.DataFrame(data)
    #     df.to_csv(io, index=True, sep=self.delim)
        
    #     # Reset the cursor position to the beginning of the StringIO object
    #     io.seek(0)
    #     return io.read()



class KVProc(TextProc):
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
    
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

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

    # def read_text(self, message: str) -> typing.Dict:
    #     """Read in the list of key values

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Dict: A dictionary of keys and values
    #     """
    #     lines = message.splitlines()
    #     result = {}
    #     for line in lines:
    #         try:
    #             key, value = line.split(self.sep)
    #             result[key] = value
    #         except ValueError:
    #             pass
    #     return result
    
    # def load_data(self, data: typing.Dict) -> typing.Dict:
    #     """Load data does not do anything as the result
    #     is a dictionary.

    #     Args:
    #         data (typing.Dict): The data to load

    #     Returns:
    #         typing.Dict: The dictionary of data
    #     """
    #     return data

    # def dump_data(self, data: typing.Dict) -> typing.Dict:
    #     """Convert the data to a dictionary

    #     Args:
    #         data (typing.Dict): The data to load

    #     Returns:
    #         typing.Dict: The dumped data
    #     """
    #     return data

    # def write_text(self, data: typing.Dict) -> str:
    #     """Write data as text

    #     Args:
    #         data (typing.Dict): The data to write

    #     Returns:
    #         str: The keys and values as text
    #     """
    #     return '\n'.join(
    #         f'{k}{self.sep}{render(v)}' for k, v in data.items()
    #     )



class IndexProc(TextProc):
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
    
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

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


    # def read_text(self, message: str) -> typing.Dict:
    #     """Read in the list of key values

    #     Args:
    #         message (str): The message to read

    #     Returns:
    #         typing.Dict: A dictionary of keys and values
    #     """
    #     lines = message.splitlines()
    #     result = []
    #     for line in lines:
    #         try:
    #             idx, value = line.split(self.sep)
    #             result.append(value)
    #         except ValueError:
    #             pass
    #     return result
    
    # def load_data(self, data: typing.Dict) -> typing.Dict:
    #     """Load data does not do anything as the result
    #     is a dictionary.

    #     Args:
    #         data (typing.Dict): The data to load

    #     Returns:
    #         typing.Dict: The dictionary of data
    #     """
    #     return data

    # def dump_data(self, data: typing.Dict) -> typing.Dict:
    #     """Convert the data to a dictionary

    #     Args:
    #         data (typing.Dict): The data to load

    #     Returns:
    #         typing.Dict: The dumped data
    #     """
    #     return data

    # def write_text(self, data: typing.List) -> str:
    #     """Write data as text

    #     Args:
    #         data (typing.Dict): The data to write

    #     Returns:
    #         str: The keys and values as text
    #     """
    #     return '\n'.join(
    #         f'{k}{self.sep}{render(v)}' for k, v in enumerate(data)
    #     )

    # def template(self, count: int=None) -> str:
    #     """Get the template for the Keys and Values

    #     Returns:
    #         str: The template
    #     """
    #     key_descr = (
    #         'The value for the key.' if self.key_descr is None else self.key_descr
    #     )
        
    #     if self.key_type is not None:
    #         key_descr = key_descr + f' ({self.key_type})'
    #     key_descr = f'<{key_descr}>'
        
    #     if count is None:
    #         lines = [
    #             f'1{self.sep}{key_descr}',
    #             f'...',
    #             f'N{self.sep}{key_descr}',
    #         ]
    #     else:
    #         lines = [
    #             f'1{self.sep}{key_descr}',
    #             f'...',
    #             f'count{self.sep}{key_descr}',
    #         ]
        
    #     return '\n'.join(lines)


class JSONProc(TextProc):
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
        
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return escape_curly_braces(self.key_descr)


    # def read_text(self, message: str) -> typing.Dict:
    #     """Read in the JSON

    #     Args:
    #         text (str): The JSON to read in

    #     Returns:
    #         typing.Dict: The result - if it fails, will return an empty dict
    #     """
    #     try: 
    #         result = json.loads(message)
    #         return result
    #     except json.JSONDecodeError:
    #         return {}
    
    # def load_data(self, data: typing.Dict) -> typing.Dict:
    #     """Load the data from a dictionary. Since JSONs are just a dict
    #     this does nothing

    #     Args:
    #         data (typing.Dict): The data to load

    #     Returns:
    #         typing.Dict: The result
    #     """
    #     return data

    # def dump_data(self, data: typing.Dict) -> typing.Dict:
    #     """Does not do anything 

    #     Args:
    #         data (typing.Any): The data 

    #     Returns:
    #         typing.Any: The data
    #     """
    #     return data

    # def write_text(self, data: typing.Any) -> str:
    #     """Write the data to a a string

    #     Args:
    #         data (typing.Any): The data to write

    #     Returns:
    #         str: The string version of the data
    #     """
    #     return json.dumps(data)


class YAMLRead(TextProc):
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
        
    def delta(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        pass

    def template(self) -> str:
        """Output the template for the class

        Returns:
            str: The template for the output
        """
        return yaml.safe_dump(self.key_descr)


    # def read_text(self, message: str) -> typing.Dict:
    #     """Read in the JSON

    #     Args:
    #         text (str): The JSON to read in

    #     Returns:
    #         typing.Dict: The result - if it fails, will return an empty dict
    #     """
    #     try: 
    #         result = yaml.safe_load(message)
    #         return result
    #     except yaml.YAMLError:
    #         return {}
    
    # def load_data(self, data: typing.Dict) -> typing.Dict:
    #     """Load the data from a dictionary. Since JSONs are just a dict
    #     this does nothing

    #     Args:
    #         data (typing.Dict): The data to load

    #     Returns:
    #         typing.Dict: The result
    #     """
    #     return data

    # def dump_data(self, data: typing.Dict) -> typing.Dict:
    #     """Does not do anything 

    #     Args:
    #         data (typing.Any): The data 

    #     Returns:
    #         typing.Any: The data
    #     """
    #     return data

    # def write_text(self, data: typing.Any) -> str:
    #     """Write the data to a a string

    #     Args:
    #         data (typing.Any): The data to write

    #     Returns:
    #         str: The string version of the data
    #     """
    #     return yaml.safe_dump(data)

    # def template(self) -> str:
    #     """Output the template for the class

    #     Returns:
    #         str: The template for the output
    #     """
    #     return yaml.safe_dump(self.key_descr)


# class StructListRead(TextProc, typing.Generic[S]):
#     """Convert the output into a list of structs
#     """
#     name: str
#     _out_cls: typing.Type[S] = pydantic.PrivateAttr()

#     def __init__(self, out_cls: S, **data):
#         """Create a converter specifying the struct class to convert
#         to

#         Args:
#             out_cls (S): The class to convert to
#         """
#         super().__init__(**data)
#         self._out_cls = out_cls

#     def dump_data(self, data: DataList[S]) -> typing.Any:
#         structs = []
#         for cur in data.data:
#             # 
#             # structs.append(self._out_cls.to_dict(cur))
#             structs.append(
#                 cur.model_dump()
#             )

#         return structs

#     def write_text(self, data: typing.List) -> str:
#         return json.dumps(data)

#     def read_text(self, message: str) -> DataList[S]:
#         """Convert the message into a list of structs

#         Args:
#             message (str): The AI message to read

#         Returns:
#             StructList[S]: the list of structs
#         """
#         return json.loads(message)
    
#     def load_data(self, data) -> DataList[S]:
#         structs = []
#         for cur in data['data']:
#             structs.append(self._out_cls(**cur))

#         return DataList(data=structs)

#     # TODO: Shouldn't this be "render?"
#     def to_text(self, data: DataList[S]) -> str:
#         """Convert the data to a string

#         Args:
#             data (StructList[S]): The data to convert to text

#         Returns:
#             str: the data converted to a string
#         """
#         return model_to_text(data)

#     def template(self) -> str:
#         """Output a template for the struct list

#         Returns:
#             str: A template of the struct list
#         """
#         # TODO: This is currently not correct
#         #   Needs to output as a list

#         return struct_template(self._out_cls)


# class DualRead(TextProc):
#     """A reader that convert to an intermediate format
#     """

#     text: TextProc
#     data: TextProc

#     def read_text(self, message: str) -> typing.Dict:
#         """Read in the text

#         Args:
#             message (str): The message to read
#         """
#         return self.text.read_text(message)
    
#     def load_data(self, data) -> typing.Any:
#         """Load" the data

#         Args:
#             data (typing.Any): The data to load
#         """
#         return self.data.load_data(data)

#     def dump_data(self, data: typing.Any) -> typing.Any:
#         """Dump the data

#         Args:
#             data (typing.Any): The data to dump
#         """
#         return self.data.dump_data(data)

#     def write_text(self, data: typing.Any) -> str:
#         """Write text data
#         """
#         return self.text.write_text(data)

#     def template(self) -> str:
#         """Output a template for the text output
#         """
#         return self.text.template()
