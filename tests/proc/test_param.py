import asyncio
import typing
from typing import Any
from dachi.proc import _process as core
from dachi.proc import Param, Module, ParamSet
from dachi.proc import _process
import numpy as np
from dachi.base import Trainable


class Value(Trainable):

    def __init__(self, val: float):

        self.val = val

    def update_param_dict(self, param_dict: typing.Dict):
        """Update the state dict for the object

        Args:
            state_dict (typing.Dict): The state dict
        """
        self.val = param_dict['val']

    def param_dict(self) -> typing.Dict:
        """Update the state dict for the object

        """
        return {
            'val': self.val
        }

    def data_schema(self) -> typing.Dict:
        """Get the structure of the object

        Returns:
            typing.Dict: The structure of the object
        """
        return {
            'val': {
            'type': 'string',
            'required': True
            }
        }


class TestParam(object):

    def test_get_x_from_param(self):

        param = Param(
            name='X', data=Value(1.0), training=True
        )
        assert param.data.val == 1.0

    def test_get_param_dict(self):

        param = Param(
            name='X', data=Value(1.0), training=True
        )
        assert param.param_dict()['val'] == 1.0

    def test_set_param_dict(self):
        param = Param(
            name='X', data=Value(1.0), training=True
        )
        param.update_param_dict(
            {'val': 2.0}
        )
        assert param.param_dict()['val'] == 2.0


class TestParamSet(object):

    def test_param_set_returns_val(self):

        param_set = ParamSet([Param(
            name='X', data=Value(1.0), training=True
        )])
        param_dict = param_set.param_dict()
        assert param_dict['X']['val'] == 1.0

    def test_update_param_dict_returns_value(self):

        param_set = ParamSet([Param(
            name='X', data=Value(1.0), training=True
        )])
        param_set.update_param_dict(
            {'X': {'val': 3.0}}
        )
        assert param_set.params[0].data.val == 3.0

    def test_data_schema_returns_x(self):

        param_set = ParamSet([Param(
            name='X', data=Value(1.0), training=True
        )])
        schema = param_set.data_schema()
        assert 'X' in schema['properties']

    def test_data_schema_returns_val(self):

        param_set = ParamSet([Param(
            name='X', data=Value(1.0), training=True
        )])
        schema = param_set.data_schema()
        assert 'val' in schema['properties']['X']['properties']
