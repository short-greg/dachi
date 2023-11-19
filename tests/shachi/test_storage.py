from dachi.shachi import _storage as sango
from dachi.shachi._storage import Data
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

    def test_data_hook_is_registered(self):

        data = sango.Data('x', 1)
        hook = SampleHook()
        data.register_hook(hook)
        assert data.has_hook(hook)

    def test_data_hook_is_removed(self):

        data = sango.Data('x', 1)
        hook = SampleHook()
        data.register_hook(hook)
        data.remove_hook(hook)
        assert not data.has_hook(hook)



class TestSynched:

    def test_set_synched_value_is_1(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        assert synched.value == 1

    def test_set_data_name_is_correct(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        assert synched.name == 'x'

    def test_set_data_value_is_correct_after_updating(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        data.update(2)
        assert synched.value == 2

    def test_error_raised_if_check_fails(self):
        check = lambda x: (x >= 0, 'Value must be greater than or equal to 0')

        data = sango.Data('x', 1, check)
        synched = sango.Synched('x', data)
        with pytest.raises(ValueError):
            synched.update(-1) 

    def test_data_hook_is_registered(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        hook = SampleHook()
        synched.register_hook(hook)
        assert synched.has_hook(hook)

    def test_data_hook_is_removed(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        hook = SampleHook()
        synched.register_hook(hook)
        synched.remove_hook(hook)
        assert not synched.has_hook(hook)


class TestDataHook:
    
    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        data = sango.Data('x', 1)
        data.register_hook(hook)
        data.update(2)
        assert hook.called is True


class TestCompositeHook:
    
    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        hook2 = SampleHook()
        composite = sango.CompositeHook('x', [hook, hook2])
        data = sango.Data('x', 1)
        data.register_hook(composite)
        data.update(2)
        assert hook.called is True
        assert hook2.called is True


class TestDataStore:

    def test_add_to_datastore_adds_value(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert data_store['x'] == 1

    def test_get_returns_none_if_not_in_data_store(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        value = data_store.get('y')
        assert value is None

    def test_add_to_datastore_adds_value(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert data_store.get('x') == 1

    def test_in_returns_true_if_x_is_there(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert 'x' in data_store

    def test_in_returns_false_if_x_is_not_there(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert 'y' not in data_store

    def test_update_changes_the_value(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        data_store.update('x', 2)
        assert data_store['x'] == 2

    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        data_store = sango.DataStore()
        data_store.add('x', 1)
        data_store.register_hook('x', hook)
        data_store.update('x', 2)
        assert hook.called is True
