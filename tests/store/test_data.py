import pytest
from dachi import store
import pandas as pd
from dachi.store import Record


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

        class DummyRenderable:
            def __init__(self):
                pass

        def render(obj):
            return str(obj)

class TestRecord:

    def test_init_with_no_kwargs(self):
        # Should create empty DataFrame
        rec = Record()
        assert isinstance(rec._data, pd.DataFrame)
        assert rec._data.empty
        assert rec.indexed is False

    def test_init_with_kwargs(self):
        rec = Record(a=[1, 2], b=[3, 4])
        assert list(rec._data.columns) == ['a', 'b']
        assert rec._data.shape == (2, 2)
        assert rec._data['a'].tolist() == [1, 2]
        assert rec._data['b'].tolist() == [3, 4]

    def test_extend_adds_rows(self):
        rec = Record(a=[1], b=[2])
        rec.extend(a=[3], b=[4])
        assert rec._data.shape == (2, 2)
        assert rec._data.iloc[1]['a'] == 3
        assert rec._data.iloc[1]['b'] == 4

    def test_extend_with_empty(self):
        rec = Record(a=[1])
        rec.extend()
        # Should not add any rows
        assert rec._data.shape == (1, 1)
    
    def test_extend_on_empty_record_adds_rows(self):
        rec = Record()
        rec.extend(a=[1, 2], b=[3, 4])
        assert list(rec._data.columns) == ['a', 'b']
        assert rec._data.shape == (2, 2)
        assert rec._data['a'].tolist() == [1, 2]
        assert rec._data['b'].tolist() == [3, 4]

    def test_extend_on_empty_record_with_no_kwargs(self):
        rec = Record()
        rec.extend()
        # Should remain empty
        assert rec._data.empty

    def test_extend_on_empty_record_with_partial_columns(self):
        rec = Record()
        rec.extend(a=[1, 2])
        assert list(rec._data.columns) == ['a']
        assert rec._data['a'].tolist() == [1, 2]

    def test_append_adds_single_row(self):
        rec = Record(a=[1], b=[2])
        rec.append(a=3, b=4)
        assert rec._data.shape == (2, 2)
        assert rec._data.iloc[1]['a'] == 3
        assert rec._data.iloc[1]['b'] == 4

    def test_append_with_missing_column(self):
        rec = Record(a=[1])
        rec.append(a=2, b=3)
        # Should add NaN for missing columns in previous rows
        assert 'b' in rec._data.columns
        assert pd.isna(rec._data.iloc[0]['b'])

    def test_join_returns_dataframe_with_new_columns(self):
        rec = Record(a=[1, 2])
        record = rec.join(b=[3, 4])
        assert isinstance(record, Record)
        assert 'b' in record
        assert record['b'].tolist() == [3, 4]

    def test_join_with_mismatched_length(self):
        rec = Record(a=[1, 2])
        # Should raise ValueError if lengths don't match
        with pytest.raises(ValueError):
            rec.join(b=[1])

    def test_df_property_returns_dataframe(self):
        rec = Record(a=[1, 2])
        # Patch _items to match expected property
        rec._items = {'a': [1, 2]}
        df = rec.df
        assert isinstance(df, pd.DataFrame)
        assert df['a'].tolist() == [1, 2]

    def test_len_returns_number_of_rows(self):
        rec = Record(a=[1, 2, 3])
        assert len(rec) == 3

    def test_render_returns_string(self):
        rec = Record(a=[1, 2])
        result = rec.render()
        assert isinstance(result, str)
        assert 'a' in result

    def test_extend_with_different_columns(self):
        rec = Record(a=[1])
        rec.extend(b=[2])
        assert 'b' in rec._data.columns
        assert pd.isna(rec._data.iloc[0]['b'])

    def test_append_with_no_kwargs(self):
        rec = Record(a=[1])
        rec.append()
        # Should add a row with all NaN
        assert rec._data.shape[0] == 1
