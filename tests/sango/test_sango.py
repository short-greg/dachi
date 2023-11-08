from kaijin import sango
from kaijin.sango._sango import Data
import pytest


class SampleHook(sango.DataHook):

    def __init__(self):
        super().__init__('sample')
        self.called = False

    def __call__(self, data: Data, prev_value):
        self.called = True


class TestData:

    def test_set_data_value_is_1(self):

        data = sango.Data('x', 1)
        assert data.value == 1

    def test_set_data_name_is_correct(self):

        data = sango.Data('x', 1)
        assert data.name == 'x'

    def test_set_data_value_is_correct_after_updating(self):

        data = sango.Data('x', 1)
        data.update(2)
        assert data.value == 2

    def test_error_raised_if_check_fails(self):
        check = lambda x: (x >= 0, 'Value must be greater than or equal to 0')

        with pytest.raises(ValueError):
            sango.Data('x', -1, check)


class TestDataHook:
    
    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        data = sango.Data('x', 1)
        data.register_hook(hook)
        data.update(2)
        assert hook.called is True
