# 1st party
import typing

# 3rd party
import numpy as np
import pydantic

# local
from ._core import Renderable
from ._core import Storable, render

from typing import TypeVar


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


T = TypeVar("T", bound=Trainable)



# TODO: Make this store just a Pydantic BaseModel
# rather than a cue.
class Param(
    pydantic.BaseModel, 
    Renderable, 
    Storable
):
    """Use Param to wrap instructions so the instructions
    can update
    """
    name: str
    data: Trainable
    training: bool=False

    def update_param(self, data: typing.Dict) -> bool:
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            excluded = self.data.dict_excluded()
            data.update(
                excluded
            )

            self.data = self.data.__class__(
                **data
            )
            return True
        return False

    def dump_param(self):
        """Update the text for the parameter
        If not in "training" mode will not update

        Args:
            text (str): The text to update with
        
        Returns:
            True if updated and Fals if not (not in training mode)
        """
        if self.training:
            return {
                self.name: self.data.dict_excluded()
            }
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
    
    def data_schema(self, exclude_fixed: bool=True) -> typing.Dict:
        """Get thethe schema for the data used by the model

        Returns:
            typing.Dict: The schema for the data
        """
        schema = self.data.__class__.model_json_schema()

        if exclude_fixed:
            excluded = self.data.fixed_keys()
            properties = {
                k: v for k, v in schema["properties"].items() if k not in excluded
            }
            schema['properties'] = properties
        return schema
    
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

    # def read(self, data: typing.Dict) -> S:
    #     """Read in the data

    #     Args:
    #         data (typing.Dict): The data to read in

    #     Returns:
    #         S: The result of the reading
    #     """
    #     return self.cue.read(data)

    # def reads(self, data: str) -> S:
    #     return self.cue.read_out(data)


class ParamSet(object):

    def __init__(self, params: typing.List[Param]):
        super().__init__()
        self.params = params

    def schema(self) -> typing.Dict:
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
    
    def load(self, data: typing.List[typing.Dict]):

        for datum, param in zip(data, self.params):
            param.update(datum)
