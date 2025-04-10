import pytest
from ..utils.test_core import SimpleStruct
from dachi.asst import _out as _out
from dachi.msg import model_to_text


from dachi.msg import model_to_text, END_TOK, Msg, StreamMsg
from dachi.asst import _out as text_proc
from .._structs import SimpleStruct2
from dachi import utils


class TestPydanticConv(object):

    def test_pydanticconv_handles_valid_json(self):
        """Test that PydanticConv correctly parses valid JSON into the Pydantic model."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        valid_json = '{"x": "hello"}'
        msg = Msg(
            role='user',
            meta={
                'data': [valid_json]
            }
        )
        result = out(msg)['meta']['F1']
        assert result.x == 'hello'

    def test_pydanticconv_raises_error_on_invalid_json(self):
        """Test that PydanticConv raises an error for invalid JSON."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        invalid_json = '{"x": "hello"'
        msg = Msg(
            role='user',
            meta={
                'data': [invalid_json]
            }
        )
        with pytest.raises(_out.ReadError):
            out(msg)

    def test_pydanticconv_handles_streamed_json(self):
        """Test that PydanticConv correctly handles streamed JSON data."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['{"x": "he']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['llo"}']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store)['meta']['F1']
        assert result.x == 'hello'

    def test_pydanticconv_handles_empty_data(self):
        """Test that PydanticConv handles empty data gracefully."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        msg = Msg(
            role='user',
            meta={
                'data': []
            }
        )
        result = out(msg).m['F1']
        assert result is utils.UNDEFINED

    def test_pydanticconv_template_contains_field(self):
        """Test that the template output contains the expected field."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        template = out.template()
        assert 'x' in template

    def test_pydanticconv_example_output(self):
        """Test that PydanticConv generates the correct example JSON."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        example = out.example(SimpleStruct(x='example'))
        assert example == '{"x":"example"}'

    def test_pydanticconv_handles_partial_streamed_data(self):
        """Test that PydanticConv handles partial streamed data correctly."""
        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['{"x": "exa']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['mple"}']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store)['meta']['F1']
        assert result.x == 'example'


    def test_out_creates_out_class(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = simple.model_dump_json()
        msg = Msg(
            role='user',
            meta={
                'data': [d],  # Wrap the JSON string in a list
            }
        )
        simple2 = out.forward(msg)['meta']['F1']
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': [d[:4]],
            }, is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': [d[4:]],
            }, is_last=True
        )
        delta = {}
        out(msg1, delta_store=delta)
        simple2 = out(msg2, delta_store=delta)['meta']['F1']
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2


class TestPrimRead(object):

    def test_prim_read_handles_empty_data(self):
        """Test that PrimConv handles empty data gracefully."""
        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        msg = Msg(
            role='user',
            meta={
                'data': []
            }
        )
        result = out(msg)['meta']['F1']
        assert result is utils.UNDEFINED

    def test_prim_read_handles_invalid_data(self):
        """Test that PrimConv raises an error for invalid data."""
        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['invalid']
            }
        )
        with pytest.raises(ValueError):
            out(msg)

    def test_prim_read_handles_float_conversion(self):
        """Test that PrimConv correctly converts data to float."""
        out = _out.PrimConv(
            name='F1',
            out_cls=float,
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['3.14']
            }
        )
        result = out(msg)['meta']['F1']
        assert result == 3.14

    def test_prim_read_handles_streamed_data_with_partial_delta(self):
        """Test that PrimConv handles streamed data with partial delta."""
        out = _out.PrimConv(
            name='F1',
            out_cls=str,
        )
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['Hel']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['lo']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store)['meta']['F1']
        assert result == 'Hello'

    def test_prim_read_handles_bool_case_insensitivity(self):
        """Test that PrimConv correctly handles case-insensitive boolean values."""
        out = _out.PrimConv(
            name='F1',
            out_cls=bool,
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['TrUe']
            }
        )
        result = out(msg)['meta']['F1']
        assert result is True

    def test_prim_read_handles_multiple_data_entries(self):
        """Test that PrimConv processes only the first data entry."""
        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['42', '100']
            }
        )
        result = out(msg)['meta']['F1']
        assert result == 42

    def test_prim_read_template_output(self):
        """Test that PrimConv generates the correct template string."""
        out = _out.PrimConv(
            name='F1',
            out_cls=str,
        )
        template = out.template()
        assert template == '<str>'

    def test_prim_read_example_output(self):
        """Test that PrimConv generates the correct example string."""
        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        example = out.example(123)
        assert example == '123'

    def test_read_reads_in_data(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        msg1 = Msg(
            role='user',
            meta={
                'data': ['1']
            }
        )

        result = out(msg1)['meta']['F1']
        assert result == 1

    def test_template_contains_key(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        temp = out.template()
        assert 'int' in temp

    def test_prim_read_reads_bool(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=bool,
        )

        msg1 = Msg(
            role='user',
            meta={
                'data': ['TRUE']
            }
        )
        result = out.__call__(msg1).m['F1']
        assert result is True

    def test_prim_read_reads_bool_with_stream(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=bool,
        )

        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['TR']
            }, is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['UE']
            }, is_last=True
        )
        store = {}
        out.forward(msg1, delta_store=store)
        result = out.forward(msg2, delta_store=store).m['F1']
        assert result is True


class TestKVConv(object):

    def test_kvconv_handles_valid_key_value_pairs(self):
        """Test KVConv processes valid key-value pairs correctly."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1',
                'key2': 'description2'
            }
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['key1::value1', 'key2::value2']
            }
        )
        result = out(msg).m['F1']
        assert result['key1'] == 'value1'
        assert result['key2'] == 'value2'

    def test_kvconv_raises_error_on_invalid_format(self):
        """Test KVConv raises an error for invalid key-value format."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1'
            }
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['key1:value1']  # Invalid separator
            }
        )
        with pytest.raises(RuntimeError):
            out(msg)

    def test_kvconv_handles_empty_data(self):
        """Test KVConv handles empty data gracefully."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1'
            }
        )
        msg = Msg(
            role='user',
            meta={
                'data': []
            }
        )
        result = out(msg).m['F1']
        assert result == {}

    def test_kvconv_handles_partial_streamed_data(self):
        """Test KVConv processes partial streamed data correctly."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1',
                'key2': 'description2'
            }
        )
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['key1::val']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['ue1::2', 'key2::value2']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store).m['F1']
        assert 'key1' not in result
        assert result['ue1'] == '2'
        assert result['key2'] == 'value2'

    def test_kvconv_template_output(self):
        """Test KVConv generates the correct template string."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1',
                'key2': 'description2'
            }
        )
        template = out.template()
        assert 'key1::description1' in template
        assert 'key2::description2' in template

    def test_kvconv_example_output(self):
        """Test KVConv generates the correct example string."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1',
                'key2': 'description2'
            }
        )
        example = out.example({'key1': 'example1', 'key2': 'example2'})
        assert 'key1::example1' in example
        assert 'key2::example2' in example

    def test_kvconv_handles_duplicate_keys(self):
        """Test KVConv processes duplicate keys by overwriting with the latest value."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1'
            }
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['key1::value1', 'key1::value2']
            }
        )
        result = out(msg).m['F1']
        assert result['key1'] == 'value2'

    def test_kvconv_handles_custom_separator(self):
        """Test KVConv processes key-value pairs with a custom separator."""
        out = text_proc.KVConv(
            name='F1',
            sep='=',
            key_descr={
                'key1': 'description1',
                'key2': 'description2'
            }
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['key1=value1', 'key2=value2']
            }
        )
        result = out(msg).m['F1']
        assert result['key1'] == 'value1'
        assert result['key2'] == 'value2'

    def test_kvconv_handles_missing_keys(self):
        """Test KVConv handles missing keys gracefully."""
        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'key1': 'description1',
                'key2': 'description2'
            }
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['key1::value1']
            }
        )
        result = out(msg).m['F1']
        assert result['key1'] == 'value1'
        assert 'key2' not in result

    def test_out_reads_in_the_class(self):

        k = ['x::1']
        k += ['y::4']

        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )

        msg1 = Msg(
            role='user',
            meta={
                'data': k
            }
        )
        result = out(msg1).m['F1']
        assert result['x'] == '1'
        assert result['y'] == '4'

    def test_template_contains_key(self):

        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        temp = out.template()
        assert 'x::' in temp
        assert 'y::' in temp

    def test_out_reads_in_with_delta(self):

        k = ['x::1']
        k += ['y::4']

        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        delta_store = {}
        ress = {}
        for i, t in enumerate(k):
            msg1 = StreamMsg(
                role='user',
                meta={
                    'data': [t]
                }, is_streamed=True,
                is_last=i == len(k)-1
            )
            res = out(msg1, delta_store).m['F1']
            if res is not utils.UNDEFINED:
                ress.update(res)

        assert ress['x'] == '1'
        assert ress['y'] == '4'


class TestNullOutConv(object):

    def test_nulloutconv_handles_string_data(self):
        """Test NullOutConv processes string data correctly."""
        out = text_proc.NullOutConv(name='F1')
        msg = Msg(
            role='user',
            meta={
                'data': ['hello']
            }
        )
        result = out(msg).m['F1']
        assert result == 'hello'

    def test_nulloutconv_handles_empty_data(self):
        """Test NullOutConv handles empty data gracefully."""
        out = text_proc.NullOutConv(name='F1')
        msg = Msg(
            role='user',
            meta={
                'data': []
            }
        )
        result = out(msg).m['F1']
        assert result == ''

    def test_nulloutconv_handles_streamed_data(self):
        """Test NullOutConv processes streamed data correctly."""
        out = text_proc.NullOutConv(name='F1')
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['part1']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['part2']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store).m['F1']
        assert result == 'part2'

    def test_nulloutconv_example_output(self):
        """Test NullOutConv generates the correct example output."""
        out = text_proc.NullOutConv(name='F1')
        example = out.example(123)
        assert example == '123'

    def test_nulloutconv_template_output(self):
        """Test NullOutConv generates an empty template."""
        out = text_proc.NullOutConv(name='F1')
        template = out.template()
        assert template == ''
    
    def test_nulloutconv_handles_partial_streamed_data(self):
        """Test NullOutConv processes partial streamed data correctly."""
        out = text_proc.NullOutConv(name='F1')
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['partial']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['stream']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store).m['F1']
        assert result == 'stream'

    def test_nulloutconv_handles_large_data(self):
        """Test NullOutConv handles large data gracefully."""
        out = text_proc.NullOutConv(name='F1')
        large_data = 'a' * 10000
        msg = Msg(
            role='user',
            meta={
                'data': [large_data]
            }
        )
        result = out(msg).m['F1']
        assert result == large_data


class TestJSONRead(object):

    def test_out_creates_out_class(self):

        out = text_proc.JSONConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )

        simple = SimpleStruct2(x='hi', y=1)
        d = model_to_text(simple)
        msg1 = Msg(
            role='user',
            meta={
                'data': [d]
            }
        )
        simple2 = out(msg1).m['F1']
        assert simple.x == simple2['x']

    def test_out_template(self):

        out = text_proc.JSONConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple2 = out.template()
        assert 'x' in simple2

    def test_out_reads_in_with_delta(self):

        simple = SimpleStruct2(x='hi', y=1)
        out = text_proc.JSONConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        delta_store = {}
        ress = []
        data = model_to_text(simple)
        msg1 = Msg(
            role='user',
            meta={
                'data': [data]
            }
        )
        cur = out(msg1, delta_store).m['F1']
        assert cur['x'] == 'hi'
        assert cur['y'] == 1


class TestIndexConv(object):

    def test_indexconv_handles_empty_data(self):
        """Test IndexConv handles empty data gracefully."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        msg = Msg(
            role='user',
            meta={
                'data': []
            }
        )
        result = out(msg).m['F1']
        assert result == []

    def test_indexconv_raises_error_on_invalid_format(self):
        """Test IndexConv raises an error for invalid key-value format."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['1:1']  # Invalid separator
            }
        )
        with pytest.raises(RuntimeError):
            out(msg)

    def test_indexconv_handles_partial_streamed_data(self):
        """Test IndexConv processes partial streamed data correctly."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['1::1']
            },
            is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': ['2::4']
            },
            is_last=True
        )
        delta_store = {}
        out(msg1, delta_store=delta_store)
        result = out(msg2, delta_store=delta_store).m['F1']
        assert result[0] == '4'

    def test_indexconv_handles_duplicate_indices(self):
        """Test IndexConv processes duplicate indices by overwriting with the latest value."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['1::1', '1::2']
            }
        )
        result = out(msg).m['F1']
        assert result[0] == '2'

    def test_indexconv_handles_custom_separator(self):
        """Test IndexConv processes key-value pairs with a custom separator."""
        out = text_proc.IndexConv(
            name='F1',
            sep='=',
            key_descr='the number of people'
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['1=1', '2=4']
            }
        )
        result = out(msg).m['F1']
        assert result[0] == '1'
        assert result[1] == '4'

    def test_indexconv_handles_missing_indices(self):
        """Test IndexConv handles missing indices gracefully."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        msg = Msg(
            role='user',
            meta={
                'data': ['1::1']
            }
        )
        result = out(msg).m['F1']
        assert result[0] == '1'
        assert len(result) == 1

    def test_indexconv_template_output(self):
        """Test IndexConv generates the correct template string."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        template = out.template()
        assert '1::' in template
        assert 'N::' in template

    def test_indexconv_example_output(self):
        """Test IndexConv generates the correct example string."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        example = out.example(['Alice', 'Bob'])
        assert '0::Alice' in example
        assert '1::Bob' in example

    def test_indexconv_handles_invalid_separator_in_stream(self):
        """Test IndexConv raises an error for invalid separator in streamed data."""
        out = text_proc.IndexConv(
            name='F1',
            sep='::',
            key_descr='the number of people'
        )
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': ['1:1']
            },
            is_last=False
        )
        delta_store = {}
        with pytest.raises(RuntimeError):
            out(msg1, delta_store=delta_store)

    def test_indexconv_handles_large_data(self):
        """Test IndexConv handles a large number of key-value pairs."""
        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        data = [f'{i + 1}::{i * 2}' for i in range(1000)]
        msg = Msg(
            role='user',
            meta={
                'data': data
            }
        )
        result = out(msg).m['F1']
        for i in range(1000):
            assert result[i] == str(i * 2)

    def test_out_reads_in_the_class(self):

        k = ['1::1']
        k += ['2::4']

        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )

        msg1 = Msg(
            role='user',
            meta={
                'data': k
            }
        )
        result = out(msg1).m['F1']
        assert result[0] == '1'
        assert result[1] == '4'

    def test_template_contains_key(self):

        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        temp = out.template()
        assert '1::' in temp
        assert 'N::' in temp
    
    def test_out_reads_in_the_class_with_delta(self):

        k = ['1::1']
        k += ['2::4']

        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )

        delta_store = {}
        ress = []
        for i, t in enumerate(k):

            msg1 = StreamMsg(
                role='user',
                meta={
                    'data': [t]
                },
                is_last=i == len(k) - 1
            )
            cur = out(msg1, delta_store).m['F1']
            if cur is not utils.UNDEFINED:
                ress.extend(cur)
        
        assert ress[0] == '1'
        assert ress[1] == '4'
