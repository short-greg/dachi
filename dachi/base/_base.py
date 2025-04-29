from abc import ABC, abstractmethod
import typing
from uuid import uuid4


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


def dict_state_dict(d) -> typing.Dict:
    """Convert the dictionary into a state dict.
    All "Storable" values will be converted into a 
    state dict

    Args:
        d : The dictionary to convert

    Returns:
        typing.Dict: The dictionary
    """
    
    return {
        k: v.state_dict() if isinstance(v, Storable)
        else v
        for k, v in d.items()
    }


def list_state_dict(d) -> typing.List:
    """Convert the list input into a "state dict"
    The actual output of this will be a list, though.

    Args:
        d : Get the state dict for a list

    Returns:
        typing.List: The state "dict" for the list
    """

    return [
        v.state_dict() if isinstance(v, Storable)
        else v
        for v in d
    ]


def load_dict_state_dict(d, state):
    """

    Args:
        d : _description_
        state : _description_
    """
    for k, v in d.items():
        if k not in state:
            continue
        if isinstance(v, Storable):
            v.load_state_dict(state[k])
        else:
            d[k] = state[k]


def load_list_state_dict(d, state):
        
    for i, a in enumerate(d):
        if isinstance(a, Storable):
            a.load_state_dict()
        else:
            d[i] = state[i]


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


class Trainable(ABC):
    """
    Trainable  mixin for objects that can be trained and defines the interface.
    
    Notes:
        - This class is designed to be used as a mixin and should not be 
          instantiated directly.
        - Subclasses must implement all abstract methods to provide the 
          required functionality.
    """

    @abstractmethod
    def update_param_dict(self, param_dict: typing.Dict):
        """Update the state dict for the object

        Args:
            state_dict (typing.Dict): The state dict
        """
        pass

    @abstractmethod
    def param_dict(self) -> typing.Dict:
        """Update the state dict for the object

        """
        pass


    @abstractmethod
    def data_schema(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        pass

    # @abstractmethod
    # def param_structure(self) -> typing.Dict:
    #     """Get the structure of the object

    #     Returns:
    #         typing.Dict: The structure of the object
    #     """
    #     pass

    # @property
    # def fixed_keys(self) -> typing.Set:
    #     return set()
    
    # def fixed_data(self) -> typing.Dict:
    #     fixed = set(self.fixed_keys)
    #     return {
    #         key: val
    #         for key, val in self.model_dump().items()
    #         if key not in fixed
    #     }

    # def unfixed_data(self) -> typing.Dict:
    #     fixed = set(self.fixed_keys)
    #     return {
    #         key: val
    #         for key, val in self.model_dump().items()
    #         if key in fixed
    #     }

# import pydantic
# from ..utils import is_primitive

# def state_dict(data):
    
#     if isinstance(data, Storable):
#         return data.state_dict()
#     if isinstance(data, typing.Dict):
#         return {state_dict(d) for d in data.items()}
#     if isinstance(data, typing.List):
#         return [state_dict(d) for d in data]
#     if isinstance(data, typing.Tuple):
#         return (state_dict(d) for d in data)
#     if isinstance(data, pydantic.BaseModel):
#         cur = {}

#         all_items = {
#             **data.__dict__, 
#             **data.__pydantic_private__
#         }
#         for k, v in all_items.items():
#             if isinstance(v, Storable):
#                 cur[k] = v.state_dict()
#             else:
#                 cur[k] = v
#         return cur
#     if is_primitive(data):
#         return data

#     if hasattr(data, '__dict__'):
#         cur = {}
#         for k, v in data.__dict__.items():
#             if isinstance(v, Storable):
#                 cur[k] = v.state_dict()
#             else:
#                 cur[k] = v
#         return cur
#     return data


# def load_state_dict(data, state_dict):
#     if isinstance(data, Storable):
#         return data.load_state_dict(state_dict)
#     if isinstance(data, pydantic.BaseModel):
#         for k, v in data.__dict__items():
#             if isinstance(v, Storable):
#                 data.__dict__[k] = v.load_state_dict(state_dict[k])
#             else:
#                 data.__dict__[k] = load_state_dict(state_dict[k])
#         for k, v in data.__pydantic_private__.items():
#             if isinstance(v, Storable):
#                 data.__pydantic_private__[k] = v.load_state_dict(state_dict[k])
#             else:
#                 data.__pydantic_private__[k] = load_state_dict(state_dict[k])
#     if hasattr(data, '__dict__'):
#         for k, v in data.__dict__.items():
#             if isinstance(v, Storable):
#                 data.__dict__[k] = v.load_state_dict(state_dict[k])
#             else:
#                 data.__dict__[k] = load_state_dict(v, state_dict[k])

#     return data
