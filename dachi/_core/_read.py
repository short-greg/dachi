import typing
import json

import pydantic

from ..utils import (
    struct_template, model_to_text, 
    unescape_curly_braces, TemplateField, 
    StructLoadException
)
from ._core import Reader, render
from pydantic_core import PydanticUndefined


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class MultiRead(Reader):
    """Concatenates multiple outputs Out into one output
    """
    outs: typing.List[Reader]
    conn: str = '::OUT::{name}::\n'
    signal: str = '\u241E'

    def dump_data(self, data: typing.Dict) -> typing.Dict:
        """Dump all of the data to a dictionary

        Args:
            data (typing.Dict): A dictionary containing all
            of the data

        Returns:
            typing.Dict: The data all dumped
        """
        results = []
        for data_i, out in zip(data['data'], self.outs):
            results.append(out.dump_data(data_i))
        return {'data': results, 'i': data['i']}

    def write_text(self, data: typing.Dict) -> str:
        """Convert the dictionary data to text

        Args:
            data (typing.Dict): The dictionary data obtained from dump_data

        Returns:
            str: The data written as text
        """
        result = ''
        for i, (d, out) in enumerate(zip(data['data'], self.outs)):
            name = out.name or str(i) 
            result = result + '\n' + self.signal + self.conn.format(name=name)
            result = f'{result}\n{render(d)}'

        return result

    def to_text(self, data: typing.List[S]) -> str:
        """Convert the data to text
        """

        text = ""
        for i, (data_i, out) in enumerate(zip(data, self.outs)):
            name = out.name or str(i) 
            cur = render(data_i)
            cur_conn = self.conn.format(name)
            text += f"""
            {self.signal}{cur_conn}
            {cur}
            """
        return text

    def read_text(self, message: str) -> typing.Dict:
        """

        Args:
            message (str): _description_

        Returns:
            typing.Dict: _description_
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

        return {'data': structs, 'i': i, 'n': len(self.outs)}
    
    def load_data(self, data: typing.Dict) -> typing.Dict:
        """Load the data for each of the components of the output

        Args:
            data (): The dictionary data to load

        Returns:
            typing.Dict: The data with each element of the output
             loaded
        """
        structs = []

        for o, d_i in zip(self.outs, data['data']):
            structs.append(o.load_data(d_i))

        return {'data': structs, 'i': data['i'], 'n': data['n']}

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
            cur = struct_template(out)
            print(out, type(out))
            cur_conn = self.conn.format(name=name)
            texts.append(f"{self.signal}{cur_conn}\n{cur}")
        text = '\n'.join(texts)
        return text


class PrimRead(Reader):
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

    def to_text(self, data: typing.Any) -> str:
        """Convert the data to text

        Args:
            data (typing.Any): The data to convert

        Returns:
            str: The text
        """
        return str(data)

    def dump_data(self, data: typing.Any) -> typing.Any:
        """

        Args:
            data (typing.Any): The data to dump

        Returns:
            typing.Any: The data (since it is a primitive, does nothing)
        """
        return data

    def write_text(self, data: typing.Any) -> str:
        """Convert the primitive to a string

        Args:
            data (primitive): The primitive

        Returns:
            str: The data converted to a string
        """
        return str(data)

    def read_text(self, message: str) -> typing.Any:
        """Read the message from text

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The string converted to the primitive
        """
        if self._out_cls is bool:
            return message.lower() in ('true', 'y', 'yes', '1', 't')
        return self._out_cls(message)
    
    def load_data(self, data) -> typing.Any:
        """Doesn't do anything because the data should be in
        the right form

        Args:
            data: The primitive

        Returns:
            typing.Any: The primitive value
        """
        return data

    def template(self) -> str:
        """Output the template for the string

        Returns:
            str: The template for the data
        """
        return f'<{self._out_cls}>'


class PydanticRead(Reader, typing.Generic[S]):
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

    def dump_data(self, data: S) -> typing.Any:
        """Convert the data to a dictionary

        Args:
            data (S): The data to convert

        Returns:
            typing.Any: The result
        """
        return data.model_dump()

    def write_text(self, data: typing.Any) -> str:
        """Convert the dumped data to a string

        Args:
            data (typing.Any): The data to convert to a string

        Returns:
            str: The convered data
        """
        return str(data)

    def to_text(self, data: S) -> str:
        """Convert the data to text

        Args:
            data (S): The data to convert

        Returns:
            str: The 
        """
        return model_to_text(data, True)

    def read_text(self, message: str) -> S:
        """Read in text from the message

        Args:
            message (str): Read in the text from 

        Returns:
            S: The BaseModel specified by S
        """
        message = unescape_curly_braces(message)
        return json.loads(message)
        # return self._out_cls.from_text(message, True)
    
    def load_data(self, data) -> S:
        """Load data from the 

        Args:
            data: The data to load

        Returns:
            S: the loaded data
        """
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
        # return self._out_cls.template()
        return render(struct_template(self._out_cls), escape_braces, self.template_renderer) 

