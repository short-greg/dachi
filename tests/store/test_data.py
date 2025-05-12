import pytest
from dachi import store


class TestGetOrSpawn(object):

    def test_get_or_spawn_doesnt_spawn_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = store.get_or_spawn(state, 'child')
        assert child is target

    def test_get_or_spawn_spawns_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = store.get_or_spawn(state, 'other')
        assert not child is target


class TestGetOrSet(object):

    def test_get_or_set_doesnt_set_new_value(self):

        state = {'val': 2}
        target = state['val']
        child = store.get_or_set(state, 'val', 3)
        assert child is target

    def test_get_or_spawn_sets_a_new_value(self):

        state = {}
        child = store.get_or_set(state, 'val', 3)
        assert child == 3


class TestShared:

    def test_shared_sets_the_data(self):

        val = store.Shared(1)
        assert val.data == 1

    def test_shared_sets_the_data_to_the_default(self):

        val = store.Shared(default=1)
        assert val.data == 1
    
    def test_shared_sets_the_data_to_the_default_after_reset(self):

        val = store.Shared(2, default=1)
        val.reset()
        assert val.data == 1
    
    def test_callback_is_called_on_update(self):

        val = store.Shared(3, default=1)
        x = 2

        def cb(data):
            nonlocal x
            x = data

        val.register(cb)
        val.data = 4

        assert x == 4


class TestBuffer:

    def test_buffer_len_is_correct(self):
        buffer = store.Buffer()
        assert len(buffer) == 0

    def test_value_added_to_the_buffer(self):
        buffer = store.Buffer()
        buffer.add(2)
        assert len(buffer) == 1

    def test_value_not_added_to_the_buffer_after_closing(self):
        buffer = store.Buffer()
        buffer.close()
        with pytest.raises(RuntimeError):
            buffer.add(2)

    def test_value_added_to_the_buffer_after_opening(self):
        buffer = store.Buffer()
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
        buffer = store.Buffer()
        buffer.add(2, 3)
        data = buffer.it().read_all()
        
        assert data == [2, 3]

    def test_read_from_buffer_reads_all_data_after_one_read(self):
        buffer = store.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        data = it.read_all()
        buffer.add(1, 2)
        data = it.read_all()
        
        assert data == [1, 2]

    def test_read_returns_nothing_if_iterator_finished(self):
        buffer = store.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        it.read_all()
        data = it.read_all()
        
        assert len(data) == 0

    def test_it_informs_me_if_at_end(self):
        buffer = store.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        it.read_all()
        assert it.end()

    def test_it_informs_me_if_buffer_closed(self):
        buffer = store.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        it.read_all()
        buffer.close()

        assert not it.is_open()

    def test_it_reduce_combines(self):
        buffer = store.Buffer()
        buffer.add(2, 3)
        it = buffer.it()
        
        val = it.read_reduce(lambda cur, x: cur + str(x), '')
        assert val == '23'
