from abc import ABC, abstractmethod

import typing
from uuid import uuid4
import pydantic


class Trainable(pydantic.BaseModel):

    @property
    def fixed_keys(self) -> typing.Set:
        return set()
    
    def fixed_data(self) -> typing.Dict:
        fixed = set(self.fixed_keys)
        return {
            key: val
            for key, val in self.model_dump().items()
            if key not in fixed
        }

    def unfixed_data(self) -> typing.Dict:
        fixed = set(self.fixed_keys)
        return {
            key: val
            for key, val in self.model_dump().items()
            if key in fixed
        }


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



class Renderable(ABC):
    """Mixin for classes that implement the render()
    method. Render is used to determine how to represent an
    object as a string to send to thte LLM
    """

    @abstractmethod
    def render(self) -> str:
        """Convert an object to a string representation for 
        an llm

        Returns:
            str: the string representation of the object
        """
        pass


class Templatable(ABC):
    """A mixin to indicate that the class 
    has a template function defined. Templates are
    used by the LLM to determine how to output.
    """

    @abstractmethod
    def template(self) -> str:
        """Get the template 

        Returns:
            str: 
        """
        pass


class ExampleMixin(ABC):
    """A mixin to indicate that the class 
    has an example function
    """
    @abstractmethod
    def example(self) -> str:
        """Get the template 

        Returns:
            str: 
        """
        pass

