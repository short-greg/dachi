# 1st party
import typing
from typing import TypeVar

# 3rd party
import numpy as np

# local
from ..core import Renderable, Storable, Trainable
from ..core import render


T = TypeVar("T", bound=Trainable)


# TODO: Make this store just a Pydantic BaseModel
# rather than a cue.
class Param(
    Renderable, 
    Storable
):
    """Use Param to wrap instructions so the instructions
    can update
    """
    __store__ = ["data", "training"]

    def __init__(
        self, name: str, 
        data: Trainable, 
        training: bool=False
    ):
        """

        Args:
            name (str): The param name
            data (Trainable): the data in the param
            training (bool, optional): whether training or not. Defaults to False.
        """
        self.name = name
        self.data = data
        self.training = training

    def data_schema(self) -> typing.Dict:

        sub_schema = self.data.data_schema()
        schema = {
            "title": self.name,
            "type": "object",
            "properties": sub_schema,
            # "required": [self.name]
        }
        return schema

    def update_param_dict(self, data: typing.Dict) -> bool:
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            self.data.load_state_dict(data)
            return True
        return False

    def param_dict(self):
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            return self.data.state_dict()
        return {}
    
    def param_structure(self):
        if self.training:
            return self.data.param_structure()
        return {}

    def render(self) -> str:
        """Convert the Parameter to a string
        IF the text for the paramter has not been 
        updated 

        Returns:
            str: 
        """
        if self.data is not None:
            return render(self.data)
        return self.text
    

class ParamSet(object):
    """A set of parameters
    This is used to define a set of parameters
    and their structure
    """

    def __init__(self, params: typing.List[Param]):
        """Instantiate a set of parameters
        Args:
            params (typing.List[Param]): The parameters to set
        """
        super().__init__()
        self.params = params

    def data_schema(self) -> typing.Dict:
        """
        Generates a JSON schema dictionary for the parameters.
        The schema defines the structure of a JSON object with the title "ParamSet".
        It includes the properties and required fields based on the parameters.
        Returns:
            typing.Dict: A dictionary representing the JSON schema.
        """
        schema = {
            "title": "ParamSet",
            "type": "object",
            "properties": {},
            "required": []
        }
        for param in self.params:
            schema["properties"][param.name] = param.data_schema()
            schema["required"].append(param.name)
        return schema

    def update_param_dict(self, data: typing.Dict) -> bool:
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        for param in self.params:
            if param.name in data:
                param.update_param_dict(data[param.name])
    
    def param_dict(self):
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        data = {}
        for param in self.params:
            if param.training:
                data[param.name] = param.param_dict()
        return data
    
    def param_structure(self):
        """Update the text for the parameter
        If not in "training" mode will not update
        Args:
            text (str): The text to update with
        Returns:
            True if updated and Fals if not (not in training mode)
        """

        data = {}
        for param in self.params:
            if param.training:
                data[param.name] = param.param_structure()
        return data


# class ParamSet(object):

#     def __init__(self, params: typing.Iterable[Param]):
#         """

#         Args:
#             params (typing.Iterable[Param]): 
#         """
#         self._params = list(params)

#     def __iter__(self) -> Param:

#         for param in self._params:
#             yield param


def update_params(param_set: ParamSet, update: typing.List[typing.Dict]):
    """_summary_

    Args:
        param_set (ParamSet): 
        update (typing.List[typing.Dict]): 
    """

    for p, u in zip(param_set, update):
        p.update(u)
