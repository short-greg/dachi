from abc import abstractmethod, ABC
import pydantic
from ..utils import Renderable


class Description(pydantic.BaseModel, Renderable, ABC):
    """Provide context in the prompt template
    """
    name: str = pydantic.Field(description='The name of the description.')

    @abstractmethod
    def render(self) -> str:
        pass


class Ref(pydantic.BaseModel, Renderable):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    desc: Description

    @property
    def name(self) -> str:
        """Get the name of the ref

        Returns:
            str: The name of the ref
        """
        return self.desc.name

    def render(self) -> str:
        """Generate the text rendering of the ref

        Returns:
            str: The name for the ref
        """
        return self.desc.name
