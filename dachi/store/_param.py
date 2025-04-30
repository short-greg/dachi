# 1st party
import typing

# 3rd party
import numpy as np
import pydantic

# local
from ..base import Renderable, Storable, Trainable
from ..msg._render import render

from typing import TypeVar


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
            # excluded = self.data.dict_excluded()
            # data.update(
            #     excluded
            # )

            self.data = self.data.__class__(
                **data
            )
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
            return self.data.param_dict()
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
    
    def state_dict(self) -> typing.Dict:
        """Get the state dict for the Param

        Returns:
            typing.Dict: the state dict
        """
        
        return {
            'name': self.name,
            'data': self.data.param_dict(),
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


class ParamSet(object):

    def __init__(self, params: typing.List[Param]):
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

        data = {}
        for param in self.params:
            if param.training:
                data[param.name] = param.param_structure()
        return data
