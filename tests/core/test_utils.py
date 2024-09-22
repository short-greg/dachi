import pytest
from .test_core import SimpleStruct
from dachi._core import _utils as utils


class TestStrFormatter(object):

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{} {}', 1, 2
        ) == '1 2'

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{0} {1}', 1, 2
        ) == '1 2'

    def test_formatter_formats_named_variables(self):

        assert utils.str_formatter(
            '{x} {y}', x=1, y=2
        ) == '1 2'

    def test_formatter_raises_error_if_positional_and_named_variables(self):

        with pytest.raises(ValueError):
            utils.str_formatter(
                '{0} {y}', 1, y=2
            )

    def test_get_variables_gets_all_pos_variables(self):

        assert utils.get_str_variables(
            '{0} {1}'
        ) == [0, 1]

    def test_get_variables_gets_all_named_variables(self):

        assert utils.get_str_variables(
            '{x} {y}'
        ) == ['x', 'y']


class TestGetOrSpawn(object):

    def test_get_or_spawn_doesnt_spawn_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = utils.get_or_spawn(state, 'child')
        assert child is target

    def test_get_or_spawn_spawns_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = utils.get_or_spawn(state, 'other')
        assert not child is target


class TestGetOrSet(object):

    def test_get_or_set_doesnt_set_new_value(self):

        state = {'val': 2}
        target = state['val']
        child = utils.get_or_set(state, 'val', 3)
        assert child is target

    def test_get_or_spawn_sets_a_new_value(self):

        state = {}
        child = utils.get_or_set(state, 'val', 3)
        assert child == 3



class TestShared:

    def test_shared_sets_the_data(self):

        val = utils.Shared(1)
        assert val.data == 1

    def test_shared_sets_the_data_to_the_default(self):

        val = utils.Shared(default=1)
        assert val.data == 1
    
    def test_shared_sets_the_data_to_the_default_after_reset(self):

        val = utils.Shared(2, default=1)
        val.reset()
        assert val.data == 1
    
    def test_callback_is_called_on_update(self):

        val = utils.Shared(3, default=1)
        x = 2

        def cb(data):
            nonlocal x
            x = data

        val.register(cb)
        val.data = 4

        assert x == 4


class TestBuffer:

    def test_buffer_len_is_correct(self):
        buffer = utils.Buffer()
        assert len(buffer) == 0

    def test_value_added_to_the_buffer(self):
        buffer = utils.Buffer()
        buffer.add(2)
        assert len(buffer) == 1

    def test_value_not_added_to_the_buffer_after_closing(self):
        buffer = utils.Buffer()
        buffer.close()
        with pytest.raises(RuntimeError):
            buffer.add(2)

    def test_value_added_to_the_buffer_after_opening(self):
        buffer = utils.Buffer()
        buffer.close()
        buffer.open()
        buffer.add(2)
        assert len(buffer) == 1

    # def test_buffer_len_0_after_resetting(self):
    #     buffer = _core.Buffer()
    #     buffer.add(2)
    #     buffer.reset()
    #     assert len(buffer) == 0

    def test_read_from_buffer_reads_all_data(self):
        buffer = utils.Buffer()
        buffer.add(2, 3)
        data = buffer.it().read_all()
        
        assert data == [2, 3]

    def test_read_from_buffer_reads_all_data_after_one_read(self):
        buffer = utils.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        data = it.read_all()
        buffer.add(1, 2)
        data = it.read_all()
        
        assert data == [1, 2]

    def test_read_returns_nothing_if_iterator_finished(self):
        buffer = utils.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        it.read_all()
        data = it.read_all()
        
        assert len(data) == 0

    def test_it_informs_me_if_at_end(self):
        buffer = utils.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        it.read_all()
        assert it.end()

    def test_it_informs_me_if_buffer_closed(self):
        buffer = utils.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        it.read_all()
        buffer.close()

        assert not it.is_open()

    def test_it_reduce_combines(self):
        buffer = utils.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        val = it.read_reduce(lambda cur, x: cur + str(x), '')
        assert val == '23'

