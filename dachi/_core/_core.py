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
    escape_curly_braces,
    is_async_function,
    is_generator_function
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


def is_renderable(obj: typing.Any) -> bool:
    """Return whether an object is renderable

    Args:
        obj (typing.Any): The object to check

    Returns:
        bool: whether the object is renderable
    """

    return (
        isinstance(obj, Renderable)
        or is_primitive(obj)
        or isinstance(obj, list)
        or isinstance(obj, dict)
        or isinstance(obj, pydantic.BaseModel)
    )


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
