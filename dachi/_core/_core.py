# 1st party
from abc import ABC, abstractmethod
from typing import Self
import typing
from ..utils import (
    Renderable, TemplateField, 
    Templatable, model_to_text
)
from uuid import uuid4

# 3rd party
import pydantic

# local
from ..utils import (
    is_primitive, 
    escape_curly_braces
)

S = typing.TypeVar('S', bound=pydantic.BaseModel)


class Storable(ABC):
    """Object to serialize objects to make them easy to recover
    """
    def __init__(self):
        """Create the storable object
        """
        self._id = str(uuid4())

    @property
    def id(self) -> str:
        """The object id of the storable

        Returns:
            str: The ID
        """
        return self._id

    def load_state_dict(self, state_dict: typing.Dict):
        """Load the state dict for the object

        Args:
            state_dict (typing.Dict): The state dict
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                self.__dict__[k] = v.load_state_dict(state_dict[k])
            else:
                self.__dict__[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """Retrieve the state dict for the object

        Returns:
            typing.Dict: The state dict
        """
        cur = {}

        for k, v in self.__dict__.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur


def render(
    x: typing.Any, escape_braces: bool=True, 
    template_render: typing.Optional[typing.Callable[[TemplateField], str]]=None
) -> typing.Union[str, typing.List[str]]:
    """Convert an input to text. Will use the text for a cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    if isinstance(x, TemplateField):
        if template_render is not None:
            x = template_render(x)
        else: 
            x = x.render()

    if isinstance(x, Renderable):
        return x.render()

    elif isinstance(x, pydantic.BaseModel):
        return model_to_text(x, escape_curly_braces)
    elif is_primitive(x):
        return str(x)
    elif isinstance(x, typing.Dict):
        items = {}
        for k, v in x.items():
            v = render(v, escape_braces)
            if isinstance(v, str):
                v = f'"{v}"'
            items[k] = v    
        items = ', '.join(
            f'"{k}": {v}' 
            for k, v in items.items()
        )

        if escape_braces:
            return f"{{{{{items}}}}}"
        else:
            return f'{{{items}}}'
    elif isinstance(x, typing.List):

        items = []
        for v in x:
            v = render(v, escape_braces)
            if isinstance(v, str):
                v = f'"{v}"'
            items.append(v)

        return '[{}]'.format(', '.join(render(v) for v in items))
    elif isinstance(x, Renderable):
        return x.render()
    return str(x)


def render_multi(xs: typing.Iterable[typing.Any]) -> typing.List[str]:
    """Convert an input to text. Will use the text for an cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    return [
        render(x) for x in xs
    ]


class Reader(pydantic.BaseModel, Templatable, ABC):
    """Use a reader to read in data convert data retrieved from
    an LLM to a better format
    """

    name: str = ''

    def example(self, data: typing.Any) -> str:
        """Output an example of the data

        Args:
            data (typing.Any): 

        Returns:
            str: 
        """
        return self.write_text(self.dump_data(data))

    @abstractmethod
    def dump_data(self, data: typing.Any) -> typing.Any:
        """Convert the data from the output of write_text
        to the original format

        Args:
            data (typing.Any): The data

        Returns:
            typing.Any: The data
        """
        pass

    @abstractmethod
    def write_text(self, data: typing.Any) -> str:
        """Write out the text for the data

        Args:
            data (typing.Any): The data to write the text for

        Returns:
            str: The text
        """
        pass

    def read(self, message: str) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        return self.load_data(self.read_text(message))
    
    @abstractmethod
    def read_text(self, message: str) -> typing.Any:
        """Read in the text and output to a "json" compatible format or a primitive

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The result of the reading process
        """
        pass

    @abstractmethod
    def load_data(self, data: typing.Dict) -> typing.Any:
        """Load the data output from reading the text

        Args:
            data (typing.Dict): The data to load (JSON format)

        Returns:
            typing.Any: The result of the reading
        """
        pass

    @abstractmethod
    def template(self) -> str:
        """Get the template for the reader

        Returns:
            str: The template as a string
        """
        pass


class NullRead(Reader):
    """A Reader that does not change the data. 
    So in most cases will simply output a string
    """

    def dump_data(self, data: typing.Any) -> typing.Any:
        """Convert the data to JSON compatible format

        Args:
            data (typing.Any): The data to convert to "JSON" compatible format

        Returns:
            typing.Any: Returns the data passed in
        """
        return data

    def write_text(self, data: typing.Any) -> str:
        """Output the data to text

        Args:
            data (typing.Any): The JSON compatible data

        Returns:
            str: The data converted to text
        """
        return data

    def read_text(self, data: str) -> typing.Dict:
        """Read in the text as a JSON compatible structure

        Args:
            data (str): The data to read in

        Returns:
            typing.Dict: The JSON compatible object (does nothing because it is null)
        """
        return data
    
    def load_data(self, data) -> typing.Any:
        """Load the data

        Args:
            data: The data to load

        Returns:
            typing.Any: The data passed in (since null)
        """
        return data

    def template(self) -> str:
        return None


class Instruct(ABC):
    """
    """
    @abstractmethod
    def i(self) -> 'Cue':
        """Create an Instruct class used for instructions

        Returns:
            Cue: Get the cue
        """
        pass


class Cue(pydantic.BaseModel, Instruct, typing.Generic[S], Renderable):
    """Specific cue for the model to use
    """
    
    text: str
    out: typing.Optional[Reader] = None

    def __init__(self, text: str, name: str='', out: typing.Optional[Reader] = None):

        super().__init__(text=text, name=name, out=out)

    def i(self) -> Self:
        return self

    @pydantic.field_validator('text', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Renderable):
            return v.render()
        if is_primitive(v):
            return str(v)
        return v

    def render(self) -> str:
        """Render the cue

        Returns:
            str: The text for the cue 
        """
        return self.text

    def read(self, data: str) -> S:
        """Read the data

        Args:
            data (str): The data to read

        Raises:
            RuntimeError: If the cue does not have a reader

        Returns:
            S: The result of the read process
        """
        if self.out is None:
            raise RuntimeError(
                "Out has not been specified so can't read it"
            )
        
        return self.out.read(data)

    def state_dict(self) -> typing.Dict:
        
        return {
            'text': self.text,
        }

    def load_state_dict(self, params: typing.Dict):
        
        self.text = params['text']


class Param(pydantic.BaseModel, Renderable, Storable):
    """Use Param to wrap instructions so the instructions
    can update
    """
    
    name: str
    cue: Cue
    training: bool=False
    text: str = None

    @pydantic.field_validator('cue', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Cue):
            return v
        if isinstance(v, Renderable):
            return Cue(text=v.render())
        if is_primitive(v):
            return Cue(text=render(v))
        return v

    def update(self, text: str) -> bool:
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            self.text = text
            return True
        return False

    def render(self) -> str:
        """Convert the Parameter to a string
        IF the text for the paramter has not been 
        updated 

        Returns:
            str: 
        """
        if self.text is None:
            return self.cue.render()
        return self.text

    def read(self, data: typing.Dict) -> S:
        """Read in the data

        Args:
            data (typing.Dict): The data to read in

        Returns:
            S: The result of the reading
        """
        return self.cue.read(data)

    def reads(self, data: str) -> S:
        return self.cue.read_out(data)
    
    def state_dict(self) -> typing.Dict:
        """Get the state dict for the Param

        Returns:
            typing.Dict: the state dict
        """
        
        return {
            'name': self.name,
            'cue': self.cue.state_dict(),
            'training': self.training,
            'text': self.text
        }

    def load_state_dict(self, params: typing.Dict):
        """Load the state dict for the Param
        
        Args:
            params (typing.Dict): the state dict
        """
        self.name = params['name']
        self.cue = self.cue.load_state_dict(params['cue'])
        self.training = params['training']
        self.text = params['text']


class Module(Storable, ABC):
    """Base class for Modules
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        pass

    def __call__(self, *args, **kwargs) -> typing.Any:
        """Execute the module

        Returns:
            typing.Any: The output of the module
        """
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool=True) -> typing.Iterator['Param']:
        """Loop over the parameters for the module

        Yields:
            Param: The parameters for the module
        """
        yielded = set()
        for k, v in self.__dict__.items():
            if isinstance(v, Param):
                if id(v) in yielded:
                    continue
                yielded.add(id(v))
                
                yield v
            if recurse and isinstance(v, Module):
                for v in v.parameters(True):
                    if id(v) in yielded:
                        continue
                    yielded.add(id(v))
                    yield v

    def children(self, recurse: bool=True) -> typing.Iterator['Module']:
        """Loop over all of the child modules

        Yields:
            Module: The child module
        """
        yielded = set()
        print(self.__dict__)
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                if id(v) in yielded:
                    continue
                yield v
                yielded.add(id(v))
                if recurse:
                    for v in v.children(True):
                        if id(v) in yielded:
                            continue
                        yielded.add(id(v))
                        yield v
    
    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """Execute the forward method asynchronously

        Returns:
            typing.Any: 
        """
        res = self.forward(*args, **kwargs)
        return res

    def stream_forward(self, *args, **kwargs) -> typing.Iterator[
        typing.Tuple[typing.Any, typing.Any]
    ]:
        """Stream the output

        Yields:
            Iterator[typing.Iterator[ typing.Tuple[typing.Any, typing.Any] ]]: The current value and the change in the value
        """
        # default behavior doesn't actually stream
        res = self.forward(*args, **kwargs) 
        yield res, res

    async def async_stream_forward(self, *args, **kwargs) -> typing.AsyncIterator:
        """
        Returns:
            Streamer: The Streamer to loop over
        """

        for d, dx in self.stream_forward(*args, **kwargs):
            yield d, dx

    def state_dict(self):
        
        state_dict = {}
        for i, child in enumerate(self.children(False)):
            state_dict[i] = child.state_dict()
        
        params = {}
        for i, param in enumerate(self.parameters(False)):
            params[i] = param.state_dict()
        state_dict['__params__'] = params

        return state_dict
    
    def load_state_dict(self, state_dict):

        for i, child in enumerate(self.children(False)):
            cur_dict = state_dict[i]
            child.load_state_dict(cur_dict)

        params = state_dict['__params__']
        for i, cur in enumerate(self.parameters(False)):
            cur.load_state_dict(params[i])
