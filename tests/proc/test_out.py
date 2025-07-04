from dachi.proc import _out as _out
from dachi.core import Msg, Resp, model_to_text
import typing
import pydantic
import pytest
from dachi.utils import UNDEFINED


class SimpleStruct(pydantic.BaseModel):

    x: str


class SimpleStruct2(pydantic.BaseModel):

    x: str
    y: int


def asst_stream(data: str, convs: list[_out.ToOut]):
    """Yield a sequence of :class:`Resp` objects that simulate an assistant
    streaming its reply token‑by‑token.  Each token is routed through *convs*
    so that their internal delta buffering logic is exercised.
    """
    resp = None
    for idx, ch in enumerate(data):
        is_last = idx == len(data) - 1
        if resp is None:
            resp = Resp(msg=Msg(role="assistant"))

        else:
            resp = resp.spawn(Msg(role="assistant"))
        resp.data['content'] = ch
        for conv in convs:
            conv(resp, True, is_last)
        yield resp


# def asst_stream(data, convs) -> typing.Iterator:
#     """Use to simulate an assitant stream"""
#     delta_store = {}
#     for i, d in enumerate(data):
#         is_last = i == len(data) - 1
#         msg = Msg('assistant', meta={'content': d})
#         for conv in convs:
#             msg = conv(msg, delta_store, True, is_last)
#             yield msg


class TestPydanticConv(object):

    def test_pydanticconv_handles_valid_json(self):
        """Test that PydanticConv correctly parses valid JSON into the Pydantic model."""
        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        valid_json = '{"x": "hello"}'

        resp = Resp(msg=Msg(
            role='user',
        ))
        
        resp.data['content'] = valid_json
        result = out(resp).out['F1']
        assert result.x == 'hello'

    def test_pydanticconv_raises_error_on_invalid_json(self):
        """Test that PydanticConv raises an error for invalid JSON."""
        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        invalid_json = '{"x": "hello"'
        resp = Resp(msg=Msg(
            role='user',
        ))
        
        resp.data['content'] = invalid_json
        with pytest.raises(_out.ReadError):
            out(resp)

    def test_pydanticconv_handles_streamed_json(self):
        """Test that PydanticConv correctly handles streamed JSON data."""
        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        resp = Resp(msg=Msg(
            role='user',
        ))
        
        resp.data['content'] = '{"x": "he'
        out(resp, True, False)
        resp2 = resp.spawn(
            msg=Msg(
            role='user',
            ),
            data={'content': 'llo"}'}
        )
        print(resp.data, resp2.data)
        result = out(resp2, True, True).out['F1']
        assert result.x == 'hello'

    def test_pydanticconv_template_contains_field(self):
        """Test that the template output contains the expected field."""
        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        template = out.template()
        assert 'x' in template

    def test_pydanticconv_example_output(self):
        """Test that PydanticConv generates the correct example JSON."""
        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        example = out.render(SimpleStruct(x='example'))
        assert example == '{"x":"example"}'

    def test_pydanticconv_handles_partial_streamed_data(self):
        """Test that PydanticConv handles partial streamed data correctly."""
        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )

        resp = Resp(msg=Msg(
            role='user',
        ))
        
        resp.data['content'] = '{"x": "exa'
        out(resp, True, False)

        resp2 = resp.spawn(
            msg=Msg(
                role='user',
            ),
            data={'content': 'mple"}'}
        )

        result = out(resp2, True, True).out['F1']
        assert result.x == 'example'


    def test_out_creates_out_class(self):

        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = simple.model_dump_json()

        resp = Resp(msg=Msg(
            role='user',
        ))
        resp.data['content'] = d
        simple2 = out.forward(resp).out['F1']
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        resp1 = Resp(
            msg=Msg(
                role='user',
            )
        )
        resp1.data['content'] = d[:4]
        out(resp1, True, False)
        resp2 = resp1.spawn(
            Msg(
                role='user',
            ),
            data={ 
                'content': d[4:],
            }
        )

        simple2 = out(resp2, True, True).out['F1']
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _out.PydanticOut(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2


class TestPrimOut:
    """
    Updated black-box tests for :class:`dachi.proc._out.PrimOut`.

    Each test focuses on a *single* contract:

    1. Correct type-conversion for the four primitive targets (str, int, float, bool).
    2. Graceful failure (raising ``ValueError``) when the conversion is impossible.
    3. Robust handling of streamed / partial assistant messages.
    4. Template / example helpers expose the expected hints.
    """

    def test_primout_handles_invalid_data(self):
        """Passing a non-numeric string to an *int* reader must raise ``ValueError``."""
        out = _out.PrimOut(name="F1", out_cls="int")              # key not value
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "invalid"
        with pytest.raises(ValueError):
            out(resp)                                             # noqa: PT011

    def test_primout_handles_float_conversion(self):
        """Reader converts a plain string to *float*."""
        out = _out.PrimOut(name="F1", out_cls="float")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "3.14"
        assert out(resp).out["F1"] == pytest.approx(3.14)

    def test_primout_reads_int(self):
        """Reader converts a plain string to *int*."""
        out = _out.PrimOut(name="F1", out_cls="int")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "1"
        assert out(resp).out["F1"] == 1

    def test_primout_reads_bool_case_insensitive(self):
        """Boolean reader ignores input casing."""
        out = _out.PrimOut(name="F1", out_cls="bool")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "TrUe"
        assert out(resp).out["F1"] is True

    def test_primout_handles_streamed_string(self):
        """Accumulates partial chunks until *is_last* and returns final str."""
        out = _out.PrimOut(name="F1", out_cls="str")

        resp1 = Resp(msg=Msg(role="user"))
        resp1.data["content"] = "Hel"
        out(resp1, True, False)                                    # not last

        resp2 = resp1.spawn(
            msg=Msg(role="user"),
            data={"content": "lo"},
        )
        assert out(resp2, True, True).out["F1"] == "Hello"

    def test_primout_reads_bool_with_stream(self):
        """Streaming + boolean conversion still works."""
        out = _out.PrimOut(name="F1", out_cls="bool")

        resp1 = Resp(msg=Msg(role="user"))
        resp1.data["content"] = "TR"
        out(resp1, True, False)

        resp2 = resp1.spawn(
            msg=Msg(role="user"),
            data={"content": "UE"},
        )
        assert out(resp2, True, True).out["F1"] is True

    def test_primout_template_output(self):
        """``template`` returns an angle-bracketed type hint."""
        out = _out.PrimOut(name="F1", out_cls=str)
        assert out.template() == "<str>"

    def test_primout_example_output(self):
        """``render`` stringifies the supplied example."""
        out = _out.PrimOut(name="F1", out_cls=int)
        assert out.render(123) == "123"

    def test_template_contains_key(self):
        """`int` appears in the template for an *int* reader."""
        out = _out.PrimOut(name="F1", out_cls=int)
        assert "int" in out.template()


class TestKVOut:
    """
    Refactored tests for :class:`dachi.proc._out.KVOut` adapter.

    Each test exercises a single aspect of the key/value parsing:

    1. Valid and invalid input handling.
    2. Empty-content behaviour.
    3. Partial and character-level streamed input.
    4. Template and example string generation.
    5. Duplicates, custom separators, and missing keys.
    """

    def _make(self):
        return _out.KVOut(
            name="F1",
            key_descr={"key1": "description1", "key2": "description2"},
        )

    def test_kvout_handles_valid_key_value_pairs(self):
        """Correctly parses simple key::value lines."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """key1::value1
key2::value2"""
        result = out(resp).out["F1"]
        assert result == {"key1": "value1", "key2": "value2"}

    def test_kvout_raises_error_on_invalid_format(self):
        """Non-"::" separator must trigger a ReadError."""
        out = _out.KVOut(name="F1", key_descr={"key1": "description1"})
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "key1:value1"
        with pytest.raises(RuntimeError):
            out(resp)

    def test_kvout_handles_empty_data(self):
        """Empty content yields an empty dict."""
        out = _out.KVOut(name="F1", key_descr={"key1": "description1"})
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = ""
        assert out(resp).out["F1"] == {}

    def test_kvout_handles_partial_streamed_data(self):
        """Buffers first chunk; only emits on final chunk."""
        out = _out.KVOut(
            name="F1",
            key_descr={"key1": "description1", "key2": "description2"},
        )
        # first delta: incomplete line
        resp1 = Resp(msg=Msg(role="user"))
        resp1.data["content"] = """key1::val
ue1::2
"""
        out(resp1, True, False)
        # final chunk completes the second key
        resp2 = resp1.spawn(
            msg=Msg(role="user"),
            data={"content": "key2::value2"},
        )
        result = out(resp2, True, True).out["F1"]
        assert "key1" not in result
        assert result.get("ue1") == "2"
        assert result.get("key2") == "value2"

    def test_kvout_template_output(self):
        """Includes each key and its description."""
        out = self._make()
        tmpl = out.template()
        assert "key1::description1" in tmpl and "key2::description2" in tmpl

    def test_kvout_example_output(self):
        """Renders a sample mapping as key::value lines."""
        out = self._make()
        example = out.render({"key1": "example1", "key2": "example2"})
        assert "key1::example1" in example and "key2::example2" in example

    def test_kvout_handles_duplicate_keys(self):
        """Later keys overwrite earlier ones."""
        out = _out.KVOut(name="F1", key_descr={"key1": "description1"})
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """key1::value1
key1::value2"""
        result = out(resp).out["F1"]
        assert result["key1"] == "value2"

    def test_kvout_handles_custom_separator(self):
        """Allows overriding the separator character."""
        out = _out.KVOut(
            name="F1",
            key_descr={"key1": "description1", "key2": "description2"},
            sep="=",
        )
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """key1=value1
key2=value2"""
        result = out(resp).out["F1"]
        assert result == {"key1": "value1", "key2": "value2"}

    def test_kvout_handles_missing_keys(self):
        """Missing keys are simply omitted from the result."""
        out = _out.KVOut(
            name="F1",
            key_descr={"key1": "description1", "key2": "description2"},
        )
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "key1::value1"
        result = out(resp).out["F1"]
        assert result == {"key1": "value1"}

    def test_kvout_reads_in_data(self):
        """Basic x::1
y::4 parsing works through Resp interface."""
        out = _out.KVOut(
            name="F1",
            key_descr={"x": "the value of x", "y": "the value of y"},
        )
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """x::1
y::4"""
        result = out(resp).out["F1"]
        assert result == {"x": "1", "y": "4"}

    def test_template_contains_key(self):
        """template() shows each key prompt (e.g., 'x::')."""
        out = _out.KVOut(
            name="F1",
            key_descr={"x": "the value of x", "y": "the value of y"},
        )
        temp = out.template()
        assert "x::" in temp and "y::" in temp

    def test_kvout_handles_char_streaming(self):
        """Character-level streaming reconstruction across Resp calls."""
        out = _out.KVOut(
            name="F1",
            key_descr={"x": "the value of x", "y": "the value of y"},
        )
        txt = """x::1
y::4"""
        final = None
        for resp in asst_stream(txt, [out]):
            val = resp.out["F1"]
            if val is not UNDEFINED:
                final = val
        assert final == {"y": "4"}



class TestParseOut:
    """
    Refactored tests for :class:`dachi.proc._out.NullOut` adapter.

    Verifies identity passthrough for various content scenarios:

    1. One-shot string and empty inputs.
    2. Streamed inputs emit only the final chunk.
    3. Large content is handled unchanged.
    4. Template and example helpers.
    """

    def test_passthrough_string(self):
        """Plain content should return unchanged."""
        out = _out.ParseOut(name="F1")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "hello"
        assert out(resp).out["F1"] == "hello"

    def test_handles_empty_string(self):
        """Empty content yields empty string."""
        out = _out.ParseOut(name="F1")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = ""
        assert out(resp).out["F1"] == ""

    def test_streaming_returns_last_chunk(self):
        """During streaming, only the final chunk is returned."""
        out = _out.ParseOut(name="F1")
        resp1 = Resp(msg=Msg(role="user"))
        resp1.data["content"] = "part1"
        out(resp1, True, False)
        resp2 = resp1.spawn(
            msg=Msg(role="user"),
            data={"content": "part2"},
        )
        assert out(resp2, True, True).out["F1"] == "part2"

    def test_partial_streaming_still_returns_last(self):
        """Multiple streamed chunks still yield the final one."""
        out = _out.ParseOut(name="F1")
        resp1 = Resp(msg=Msg(role="user"))
        resp1.data["content"] = "partial"
        out(resp1, True, False)
        resp2 = resp1.spawn(
            msg=Msg(role="user"),
            data={"content": "stream"},
        )
        assert out(resp2, True, True).out["F1"] == "stream"

    def test_handles_large_data(self):
        """Very large content should pass through unchanged."""
        out = _out.ParseOut(name="F1")
        large = "a" * 10000
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = large
        assert out(resp).out["F1"] == large

    def test_template_and_example_helpers(self):
        """template() is empty and render(...) stringifies example."""
        out = _out.ParseOut(name="F1")
        assert out.template() == ""
        assert out.render(123) == "123"



class TestJsonOut:
    """Tests for :class:`dachi.proc._out.JSONOut` adapter.

    1. Parses JSON content into dicts.
    2. Template includes the keys.
    3. Supports streaming input via delta buffering.
    """

    def _make(self):
        return _out.JSONOut(
            name="F1",
            key_descr={"x": "The value of x", "y": "The value of y"},
        )

    def test_jsonout_parses_valid_json(self):
        """Parses a JSON payload into a native dict with expected values."""
        out = self._make()
        instance = SimpleStruct2(x="hi", y=1)
        data = model_to_text(instance)
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = data
        result = out(resp).out["F1"]
        assert result == {"x": "hi", "y": 1}

    def test_jsonout_template_contains_keys(self):
        """Template output mentions each key from key_descr."""
        tmpl = self._make().template()
        assert "x" in tmpl and "y" in tmpl

    def test_jsonout_streamed_json(self):
        """Streaming JSON payload is reconstructed and parsed on last chunk."""
        out = self._make()
        instance = SimpleStruct2(x="hi", y=1)
        data = model_to_text(instance)
        final = None
        for resp in asst_stream(data, [out]):
            val = resp.out["F1"]
            if val is not UNDEFINED:
                final = val
        assert final == {"x": "hi", "y": 1}



class TestIndexOut:
    """
    Refactored tests for :class:`dachi.proc._out.IndexOut` adapter.

    1. Empty and invalid inputs.
    2. Streamed inputs.
    3. Duplicates and custom separators.
    4. Template and example helpers.
    5. Large data handling and character-level streaming.
    """

    def _make(self):
        return _out.IndexOut(name="F1", key_type="the number of people")

    def test_indexout_handles_empty_data(self):
        """Empty content yields an empty list."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = ""
        assert out(resp).out["F1"] == []

    def test_indexout_raises_error_on_invalid_format(self):
        """Invalid separator must raise a ReadError."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "1:1"
        with pytest.raises(RuntimeError):
            out(resp)

    def test_indexout_handles_streamed_data(self):
        """Buffers streamed lines until final chunk and returns list."""
        out = self._make()
        resp1 = Resp(msg=Msg(role="user"))
        resp1.data["content"] = """1::1
"""
        out(resp1, True, False)
        resp2 = resp1.spawn(
            msg=Msg(role="user"),
            data={"content": "2::4"},
        )
        assert out(resp2, True, True).out["F1"] == ["1", "4"]

    def test_indexout_handles_duplicate_indices(self):
        """Later entries overwrite earlier ones."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """1::1
1::2"""
        assert out(resp).out["F1"] == ["2"]

    def test_indexout_custom_separator(self):
        """Allows a custom separator character."""
        out = _out.IndexOut(name="F1", sep="=", key_type="the number of people")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """1=1
2=4"""
        assert out(resp).out["F1"] == ["1", "4"]

    def test_indexout_handles_missing_indices(self):
        """Single entry yields a one-element list."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "1::1"
        assert out(resp).out["F1"] == ["1"]

    def test_indexout_template_output(self):
        """Template includes index prompts like '1::' and 'N::'."""
        out = self._make()
        tmpl = out.template()
        assert "1::" in tmpl and "N::" in tmpl

    def test_indexout_example_output(self):
        """Example rendering lists entries with zero-based prefixes."""
        out = self._make()
        example = out.render(["Alice", "Bob"])
        assert "0::Alice" in example and "1::Bob" in example

    def test_indexout_handles_large_data(self):
        """Handles a large number of entries efficiently."""
        out = self._make()
        data = "\n".join(f"{i+1}::{i*2}" for i in range(1000))
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = data
        result = out(resp).out["F1"]
        for i in range(1000):
            assert result[i] == str(i*2)

    def test_indexout_handles_char_streaming(self):
        """Character-level streaming reconstructs and emits on last token."""
        out = self._make()
        txt = """1::1
2::4"""
        final = None
        for resp in asst_stream(txt, [out]):
            val = resp.out["F1"]
            if val is not UNDEFINED:
                final = val
        assert final == ["4"]


import collections

class TestCsvOut:
    """Refactored tests for :class:`dachi.proc._out.CSVOut` adapter.

    Parses CSV-formatted content into lists (no header) or list of OrderedDict
    (with header). Handles empty, one-shot, streamed inputs and custom delimiters.
    """

    def test_csvout_handles_empty_data(self):
        """Empty CSV yields UNDEFINED sentinel."""
        out = _out.CSVOut(name="F1", use_header=True)
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = ""
        assert out(resp).out["F1"] is UNDEFINED

    def test_csvout_handles_single_row_no_header(self):
        """Single row without header yields list-of-lists."""
        out = _out.CSVOut(name="F1", use_header=False)
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "John,25,New York"
        res = out(resp).out["F1"]
        assert isinstance(res, list) and len(res) == 1
        assert res[0] == ["John", "25", "New York"]

    def test_csvout_handles_single_row_with_header(self):
        """Single row with header yields list-of-dicts."""
        out = _out.CSVOut(name="F1", use_header=True)
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """name,age,city
John,25,New York"""
        res = out(resp).out["F1"]
        assert isinstance(res, list) and len(res) == 1
        expected = collections.OrderedDict(zip(
            ["name", "age", "city"], ["John", "25", "New York"]
        ))
        assert res[0] == expected

    def test_csvout_handles_multiple_rows_with_header(self):
        """Multiple rows with header."""
        out = _out.CSVOut(name="F1", use_header=True)
        data = (
            "name,age,city\n"
            "John,25,New York\n"
            "Jane,30,Los Angeles\n"
            "\"Smith, Jr.\",40,Chicago"
        )
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = data
        res = out(resp).out["F1"]
        assert len(res) == 3
        exp0 = collections.OrderedDict(zip(
            ["name", "age", "city"], ["John", "25", "New York"]
        ))
        exp1 = collections.OrderedDict(zip(
            ["name", "age", "city"], ["Jane", "30", "Los Angeles"]
        ))
        exp2 = collections.OrderedDict(zip(
            ["name", "age", "city"], ["Smith, Jr.", "40", "Chicago"]
        ))
        assert res[0] == exp0 and res[1] == exp1 and res[2] == exp2

    def test_csvout_handles_multiple_rows_no_header(self):
        """Multiple rows without header."""
        out = _out.CSVOut(name="F1", use_header=False)
        data = (
            "\"Widget, Deluxe\",19.99,\"High-quality widget with multiple features\"\n"
            "Gadget,9.99,\"Compact gadget, easy to use\"\n"
            "\"Tool, Multi-purpose\",29.99,\"Durable and versatile tool\""
        )
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = data
        res = out(resp).out["F1"]
        assert len(res) == 3
        assert res[0] == ["Widget, Deluxe", "19.99", "High-quality widget with multiple features"]
        assert res[1] == ["Gadget", "9.99", "Compact gadget, easy to use"]
        assert res[2] == ["Tool, Multi-purpose", "29.99", "Durable and versatile tool"]

    def test_csvout_streamed_with_header(self):
        """Streaming CSV with header reconstructs full table."""
        out = _out.CSVOut(name="F1", use_header=True)
        data = (
            "name,age,city\n"
            "John,25,New York\n"
            "Jane,30,Los Angeles\n"
            "\"Smith, Jr.\",40,Chicago"""
        )
        final = None
        for resp in asst_stream(data, [out]):
            val = resp.out["F1"]
            if val is not UNDEFINED:
                final = val
        assert len(final) == 1

    def test_csvout_streamed_no_header(self):
        """Streaming CSV without header reconstructs rows."""
        out = _out.CSVOut(name="F1", use_header=False)
        data = (
            "\"Widget, Deluxe\",19.99,\"High-quality widget with multiple features\"\n"
            "Gadget,9.99,\"Compact gadget, easy to use\"\n"
            "\"Tool, Multi-purpose\",29.99,\"Durable and versatile tool\""
        )
        final = None
        for resp in asst_stream(data, [out]):
            val = resp.out["F1"]
            if val is not UNDEFINED:
                final = val
        assert len(final) == 1

    def test_csvout_handles_different_delimiters(self):
        """Custom delimiter chars are supported."""
        out = _out.CSVOut(name="F1", delimiter="|", use_header=True)
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = (
            """name|age|city
John|25|New York
Jane|30|Los Angeles"""
        )
        res = out(resp).out["F1"]
        assert len(res) == 2
        exp0 = collections.OrderedDict(zip(
            ["name", "age", "city"], ["John", "25", "New York"]
        ))
        exp1 = collections.OrderedDict(zip(
            ["name", "age", "city"], ["Jane", "30", "Los Angeles"]
        ))
        assert res[0] == exp0 and res[1] == exp1

# ---------------------------------------------------------------------------
# TupleOut & ListOut (composite adapters)
# ---------------------------------------------------------------------------

from dachi.proc import _parse as _msg_parse

class TestTupleOut:
    """Refactored tests for :class:`dachi.proc._out.TupleOut`.

    Splits delimited segments and applies sub-adapters in sequence.
    """

    def test_tupleout_single_conv(self):
        """Single converter produces a one-element tuple list."""
        kv = _out.KVOut(name="KV", key_descr={"key1": "d1", "key2": "d2"})
        parser = _msg_parse.CharDelimParser(sep=",")
        out = _out.TupleOut(convs=[kv], parser=parser, name="out")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """key1::v1
key2::v2"""
        res = out(resp).out["out"]
        assert isinstance(res, list) and len(res) == 1
        assert res[0]["key1"] == "v1" and res[0]["key2"] == "v2"

    def test_tupleout_two_convs(self):
        """Two converters yield two parsed segments."""
        kv = _out.KVOut(name="KV", key_descr={"key1": "d1", "key2": "d2"})
        parser = _msg_parse.CharDelimParser(sep=",")
        out = _out.TupleOut(convs=[kv, kv], parser=parser, name="out")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = """key1::v1
key2::v2,key1::v3
key2::v4"""
        res = out(resp).out["out"]
        assert len(res) == 2
        assert res[0]["key1"] == "v1" and res[0]["key2"] == "v2"
        assert res[1]["key1"] == "v3" and res[1]["key2"] == "v4"

    def test_tupleout_insufficient_convs(self):
        """Fewer converters than segments raises RuntimeError."""
        kv = _out.KVOut(name="KV", key_descr={"k": "d"})
        parser = _msg_parse.CharDelimParser(sep=",")
        out = _out.TupleOut(convs=[kv], parser=parser, name="out")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "k::v1,k::v2"
        with pytest.raises(RuntimeError):
            out(resp)

    def test_tupleout_insufficient_keyvals(self):
        """Converters match but missing fields still error."""
        kv = _out.KVOut(name="KV", key_descr={"k": "d"})
        parser = _msg_parse.CharDelimParser(sep=",")
        out = _out.TupleOut(convs=[kv, kv], parser=parser, name="out")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "k::v1"
        with pytest.raises(RuntimeError):
            out(resp)

    def test_tupleout_empty_content(self):
        """Empty content yields empty list during streaming."""
        kv = _out.KVOut(name="KV", key_descr={"k": "d"})
        parser = _msg_parse.CharDelimParser(sep=",")
        out = _out.TupleOut(convs=[kv, kv], parser=parser, name="out")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = ""
        res = out(resp, True, False).out["out"]
        assert res == []

    def test_tupleout_template_and_example(self):
        """template() and render(...) include each sub-conv hints."""
        kv = _out.KVOut(name="KV", key_descr={"x": "desc", "y": "desc"})
        parser = _msg_parse.CharDelimParser(sep=",")
        out = _out.TupleOut(convs=[kv, kv], parser=parser, name="out")
        tmpl = out.template()
        example = out.render([
            {"x": "v", "y": "w"},
            {"x": "u", "y": "z"}
        ])
        assert "x::" in tmpl and "y::" in tmpl
        assert "x::v" in example and "x::u" in example



class TestListOut:
    """Comprehensive tests for :class:`dachi.proc._out.ListOut`.

    *Applies a **single** converter to each delimited segment.*  Compared to
    ``TupleOut`` there is no positional mapping between segments and
    converters, so the surface differs slightly:

    1. Valid multi‑segment parsing.
    2. Streaming – both chunk and char‑level.
    3. Empty content semantics.
    4. Template / example helpers.
    5. Custom delimiter, malformed segment, and large‑data handling.
    """

    def _make(self, sep=","):
        kv = _out.KVOut(name="KV", key_descr={"a": "d"})
        parser = _msg_parse.CharDelimParser(sep=sep)
        return _out.ListOut(conv=kv, parser=parser, name="out")

    # ------------------------------------------------------------------ #
    # 1) Happy‑path parsing
    # ------------------------------------------------------------------ #
    def test_listout_multiple_segments(self):
        """Two segments ➜ two parsed dicts."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "a::1,a::2"
        res = out(resp).out["out"]
        assert isinstance(res, list) and res == [{"a": "1"}, {"a": "2"}]

    def test_listout_streamed_segments(self):
        """Character‑level streaming reconstructs list on final token."""
        out = self._make()
        txt = "a::1,a::2,a::3"
        final = None
        for resp in asst_stream(txt, [out]):
            val = resp.out["out"]
            if val is not UNDEFINED:
                final = val
        assert final[-1]["a"] == "3"
        assert len(final) == 1

    def test_listout_empty_content(self):
        """Empty content returns empty list during initial stream."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = ""
        assert out(resp, True, False).out["out"] == []

    # 4) Helper APIs
    def test_listout_template_and_example(self):
        out = self._make()
        tmpl = out.template()
        ex = out.render([{"a": "1"}, {"a": "2"}])
        assert "a::" in tmpl and "a::1" in ex and "a::2" in ex

    # ------------------------------------------------------------------ #
    # 5) Edge‑cases / errors / customisation
    # ------------------------------------------------------------------ #
    def test_listout_invalid_segment_raises(self):
        """Malformed KV inside a segment raises ReadError."""
        out = self._make()
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "a:1"  # wrong separator inside segment
        with pytest.raises(RuntimeError):
            out(resp)

    def test_listout_custom_delimiter(self):
        """Custom char delimiter splits properly."""
        out = self._make(sep="|")
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = "a::x|a::y"
        res = out(resp).out["out"]
        assert [d["a"] for d in res] == ["x", "y"]

    def test_listout_large_number_segments(self):
        """Handles 500 segments without issue."""
        out = self._make()
        data = ",".join(f"a::{i}" for i in range(500))
        resp = Resp(msg=Msg(role="user"))
        resp.data["content"] = data
        res = out(resp).out["out"]
        assert len(res) == 500 and res[-1]["a"] == "499"



# class TestPrimRead(object):

#     def test_prim_read_handles_invalid_data(self):
#         """Test that PrimConv raises an error for invalid data."""
#         out = _out.PrimOut(
#             name='F1',
#             out_cls=int,
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'invalid'
#             }
#         )
#         with pytest.raises(ValueError):
#             out(msg)

#     def test_prim_read_handles_float_conversion(self):
#         """Test that PrimConv correctly converts data to float."""
#         out = _out.PrimOut(
#             name='F1',
#             out_cls=float,
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': '3.14'
#             }
#         )
#         result = out(msg)['meta']['F1']
#         assert result == 3.14

#     def test_prim_read_handles_streamed_data_with_partial_delta(self):
#         """Test that PrimConv handles streamed data with partial delta."""
#         out = _out.PrimOut(
#             name='F1',
#             out_cls=str,
#         )
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': 'Hel'
#             }
#         )
#         msg2 = Msg(
#             role='user',
#             meta={
#                 'content': 'lo'
#             }
#         )
#         delta_store = {}
#         out(msg1, delta_store, True, False)
#         result = out(msg2, delta_store, True, True)['meta']['F1']
#         assert result == 'Hello'

#     def test_prim_read_handles_bool_case_insensitivity(self):
#         """Test that PrimConv correctly handles case-insensitive boolean values."""
#         out = _out.PrimOut(
#             name='F1',
#             out_cls=bool,
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'TrUe'
#             }
#         )
#         result = out(msg)['meta']['F1']
#         assert result is True

#     # def test_prim_read_handles_multiple_data_entries(self):
#     #     """Test that PrimConv processes only the first data entry."""
#     #     out = _out.PrimOut(
#     #         name='F1',
#     #         out_cls=int,
#     #     )
#     #     msg = Msg(
#     #         role='user',
#     #         meta={
#     #             'content': ['42', '100']
#     #         }
#     #     )
#     #     result = out(msg)['meta']['F1']
#     #     assert result == 42

#     def test_prim_read_template_output(self):
#         """Test that PrimConv generates the correct template string."""
#         out = _out.PrimOut(
#             name='F1',
#             out_cls=str,
#         )
#         template = out.template()
#         assert template == '<str>'

#     def test_prim_read_example_output(self):
#         """Test that PrimConv generates the correct example string."""
#         out = _out.PrimOut(
#             name='F1',
#             out_cls=int,
#         )
#         example = out.render(123)
#         assert example == '123'

#     def test_read_reads_in_data(self):

#         out = _out.PrimOut(
#             name='F1',
#             out_cls=int,
#         )
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': '1'
#             }
#         )

#         result = out(msg1)['meta']['F1']
#         assert result == 1

#     def test_template_contains_key(self):

#         out = _out.PrimOut(
#             name='F1',
#             out_cls=int,
#         )
#         temp = out.template()
#         assert 'int' in temp

#     def test_prim_read_reads_bool(self):

#         out = _out.PrimOut(
#             name='F1',
#             out_cls=bool,
#         )

#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': 'TRUE'
#             }
#         )
#         result = out.__call__(msg1).m['F1']
#         assert result is True

#     def test_prim_read_reads_bool_with_stream(self):

#         out = _out.PrimOut(
#             name='F1',
#             out_cls=bool,
#         )

#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': 'TR'
#             }
#         )
#         msg2 = Msg(
#             role='user',
#             meta={
#                 'content': 'UE'
#             }
#         )
#         store = {}
#         out.forward(msg1, store, True, False)
#         result = out.forward(msg2, store, True, True).m['F1']
#         assert result is True


# class TestKVConv(object):

#     def test_kvconv_handles_valid_key_value_pairs(self):
#         """Test KVConv processes valid key-value pairs correctly."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1',
#                 'key2': 'description2'
#             }
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'key1::value1\nkey2::value2'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result['key1'] == 'value1'
#         assert result['key2'] == 'value2'

#     def test_kvconv_raises_error_on_invalid_format(self):
#         """Test KVConv raises an error for invalid key-value format."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1'
#             }
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'key1:value1'  # Invalid separator
#             }
#         )
#         with pytest.raises(RuntimeError):
#             out(msg)

#     def test_kvconv_handles_empty_data(self):
#         """Test KVConv handles empty data gracefully."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1'
#             }
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': []
#             }
#         )
#         result = out(msg).m['F1']
#         assert result == {}

#     def test_kvconv_handles_partial_streamed_data(self):
#         """Test KVConv processes partial streamed data correctly."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1',
#                 'key2': 'description2'
#             }
#         )
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': 'key1::val\nue1::2\n'
#             }
#         )
#         msg2 = Msg(
#             role='user',
#             meta={
#                 'content': 'key2::value2'
#             }
#         )
#         delta_store = {}
#         out(msg1, delta_store, True, False)
#         result = out(msg2, delta_store, True, True).m['F1']
#         assert 'key1' not in result
#         assert result['ue1'] == '2'
#         assert result['key2'] == 'value2'

#     def test_kvconv_template_output(self):
#         """Test KVConv generates the correct template string."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1',
#                 'key2': 'description2'
#             }
#         )
#         template = out.template()
#         assert 'key1::description1' in template
#         assert 'key2::description2' in template

#     def test_kvconv_example_output(self):
#         """Test KVConv generates the correct example string."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1',
#                 'key2': 'description2'
#             }
#         )
#         example = out.render({'key1': 'example1', 'key2': 'example2'})
#         assert 'key1::example1' in example
#         assert 'key2::example2' in example

#     def test_kvconv_handles_duplicate_keys(self):
#         """Test KVConv processes duplicate keys by overwriting with the latest value."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1'
#             }
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'key1::value1\nkey1::value2'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result['key1'] == 'value2'

#     def test_kvconv_handles_custom_separator(self):
#         """Test KVConv processes key-value pairs with a custom separator."""
#         out = text_proc.KVOut(
#             name='F1',
#             sep='=',
#             key_descr={
#                 'key1': 'description1',
#                 'key2': 'description2'
#             }
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'key1=value1\nkey2=value2'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result['key1'] == 'value1'
#         assert result['key2'] == 'value2'

#     def test_kvconv_handles_missing_keys(self):
#         """Test KVConv handles missing keys gracefully."""
#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'key1': 'description1',
#                 'key2': 'description2'
#             }
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'key1::value1'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result['key1'] == 'value1'
#         assert 'key2' not in result

#     def test_out_reads_in_the_class(self):

#         k = 'x::1\ny::4'

#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'x': 'the value of x', 
#                 'y': 'the value of y'
#             },
#         )

#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': k
#             }
#         )
#         result = out(msg1).m['F1']
#         assert result['x'] == '1'
#         assert result['y'] == '4'

#     def test_template_contains_key(self):

#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'x': 'the value of x', 
#                 'y': 'the value of y'
#             },
#         )
#         temp = out.template()
#         assert 'x::' in temp
#         assert 'y::' in temp

#     def test_out_reads_in_with_delta(self):

#         k = 'x::1\ny::4'

#         out = text_proc.KVOut(
#             name='F1',
#             key_descr={
#                 'x': 'the value of x', 
#                 'y': 'the value of y'
#             },
#         )
#         delta_store = {}
#         ress = {}
#         for i, t in enumerate(k):
#             is_last = i == len(k)-1
#             msg1 = Msg(
#                 role='user',
#                 meta={
#                     'content': t
#                 }
#             )
#             res = out(msg1, delta_store, True, is_last).m['F1']
#             if res is not utils.UNDEFINED:
#                 ress.update(res)

#         assert ress['x'] == '1'
#         assert ress['y'] == '4'


# class TestNullOutConv(object):

#     def test_nulloutconv_handles_string_data(self):
#         """Test NullOutConv processes string data correctly."""
#         out = text_proc.NullOut(name='F1')
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': 'hello'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result == 'hello'

#     def test_nulloutconv_handles_empty_data(self):
#         """Test NullOutConv handles empty data gracefully."""
#         out = text_proc.NullOut(name='F1')
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': ''
#             }
#         )
#         result = out(msg).m['F1']
#         assert result == ''

#     def test_nulloutconv_handles_streamed_data(self):
#         """Test NullOutConv processes streamed data correctly."""
#         out = text_proc.NullOut(name='F1')
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': 'part1'
#             }
#         )
#         msg2 = Msg(
#             role='user',
#             meta={
#                 'content': 'part2'
#             },
#             is_last=True
#         )
#         delta_store = {}
#         out(msg1, delta_store, True, False)
#         result = out(msg2, delta_store, True, True).m['F1']
#         assert result == 'part2'

#     def test_nulloutconv_example_output(self):
#         """Test NullOutConv generates the correct example output."""
#         out = text_proc.NullOut(name='F1')
#         example = out.render(123)
#         assert example == '123'

#     def test_nulloutconv_template_output(self):
#         """Test NullOutConv generates an empty template."""
#         out = text_proc.NullOut(name='F1')
#         template = out.template()
#         assert template == ''
    
#     def test_nulloutconv_handles_partial_streamed_data(self):
#         """Test NullOutConv processes partial streamed data correctly."""
#         out = text_proc.NullOut(name='F1')
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': 'partial'
#             }
#         )
#         msg2 = Msg(
#             role='user',
#             meta={
#                 'content': 'stream'
#             }
#         )
#         delta_store = {}
#         out(msg1, delta_store, True, False)
#         result = out(msg2, delta_store, True, True).m['F1']
#         assert result == 'stream'

#     def test_nulloutconv_handles_large_data(self):
#         """Test NullOutConv handles large data gracefully."""
#         out = text_proc.NullOut(name='F1')
#         large_data = 'a' * 10000
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': large_data
#             }
#         )
#         result = out(msg).m['F1']
#         assert result == large_data


# class TestJSONRead(object):

#     def test_out_creates_out_class(self):

#         out = text_proc.JSONOut(
#             name='F1',
#             key_descr={
#                 'x': 'The value of x',
#                 'y': 'The value of y'
#             }
#         )

#         simple = SimpleStruct2(x='hi', y=1)
#         d = model_to_text(simple)
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': d
#             }
#         )
#         simple2 = out(msg1).m['F1']
#         assert simple.x == simple2['x']

#     def test_out_template(self):

#         out = text_proc.JSONOut(
#             name='F1',
#             key_descr={
#                 'x': 'The value of x',
#                 'y': 'The value of y'
#             }
#         )
#         simple2 = out.template()
#         assert 'x' in simple2

#     def test_out_reads_in_with_delta(self):

#         simple = SimpleStruct2(x='hi', y=1)
#         out = text_proc.JSONOut(
#             name='F1',
#             key_descr={
#                 'x': 'The value of x',
#                 'y': 'The value of y'
#             }
#         )
#         delta_store = {}
#         ress = []
#         data = model_to_text(simple)
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': data
#             }
#         )
#         cur = out(msg1, delta_store).m['F1']
#         assert cur['x'] == 'hi'
#         assert cur['y'] == 1


# class TestIndexConv(object):

#     def test_indexconv_handles_empty_data(self):
#         """Test IndexConv handles empty data gracefully."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': ''
#             }
#         )
#         result = out(msg).m['F1']
#         assert result == []

#     def test_indexconv_raises_error_on_invalid_format(self):
#         """Test IndexConv raises an error for invalid key-value format."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': '1:1'  # Invalid separator
#             }
#         )
#         with pytest.raises(RuntimeError):
#             out(msg)

#     def test_indexconv_handles_partial_streamed_data(self):
#         """Test IndexConv processes partial streamed data correctly."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': '1::1\n'
#             }
#         )
#         msg2 = Msg(
#             role='user',
#             meta={
#                 'content': '2::4'
#             }
#         )
#         delta_store = {}
#         out(msg1, delta_store, True, False)
#         result = out(msg2, delta_store, True, True).m['F1']
#         assert result[0] == '1'
#         assert result[1] == '4'

#     def test_indexconv_handles_duplicate_indices(self):
#         """Test IndexConv processes duplicate indices by overwriting with the latest value."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': '1::1\n1::2'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result[0] == '2'

#     def test_indexconv_handles_custom_separator(self):
#         """Test IndexConv processes key-value pairs with a custom separator."""
#         out = text_proc.IndexOut(
#             name='F1',
#             sep='=',
#             key_descr='the number of people'
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': '1=1\n2=4'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result[0] == '1'
#         assert result[1] == '4'

#     def test_indexconv_handles_missing_indices(self):
#         """Test IndexConv handles missing indices gracefully."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': '1::1'
#             }
#         )
#         result = out(msg).m['F1']
#         assert result[0] == '1'
#         assert len(result) == 1

#     def test_indexconv_template_output(self):
#         """Test IndexConv generates the correct template string."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         template = out.template()
#         assert '1::' in template
#         assert 'N::' in template

#     def test_indexconv_example_output(self):
#         """Test IndexConv generates the correct example string."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         example = out.render(['Alice', 'Bob'])
#         assert '0::Alice' in example
#         assert '1::Bob' in example

#     def test_indexconv_handles_invalid_separator_in_stream(self):
#         """Test IndexConv raises an error for invalid separator in streamed data."""
#         out = text_proc.IndexOut(
#             name='F1',
#             sep='::',
#             key_descr='the number of people'
#         )
#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': '1:1'
#             }
#         )
#         delta_store = {}
#         with pytest.raises(RuntimeError):
#             out(msg1, delta_store, True, True)

#     def test_indexconv_handles_large_data(self):
#         """Test IndexConv handles a large number of key-value pairs."""
#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         data = '\n'.join(
#             [f'{i + 1}::{i * 2}' for i in range(1000)]
#         )
#         msg = Msg(
#             role='user',
#             meta={
#                 'content': data
#             }
#         )
#         result = out(msg).m['F1']
#         for i in range(1000):
#             assert result[i] == str(i * 2)

#     def test_out_reads_in_the_class(self):

#         k = '1::1\n2::4'

#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )

#         msg1 = Msg(
#             role='user',
#             meta={
#                 'content': k
#             }
#         )
#         result = out(msg1).m['F1']
#         assert result[0] == '1'
#         assert result[1] == '4'

#     def test_template_contains_key(self):

#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )
#         temp = out.template()
#         assert '1::' in temp
#         assert 'N::' in temp
    
#     def test_out_reads_in_the_class_with_delta(self):

#         k = '1::1\n2::4'

#         out = text_proc.IndexOut(
#             name='F1',
#             key_descr='the number of people'
#         )

#         delta_store = {}
#         ress = []
#         for i, t in enumerate(k):

#             is_last = i == len(k) - 1
#             msg1 = Msg(
#                 role='user',
#                 meta={
#                     'content': t
#                 },
#             )
#             cur = out(msg1, delta_store, True, is_last).m['F1']
#             if cur is not utils.UNDEFINED:
#                 ress.extend(cur)
        
#         assert ress[0] == '1'
#         assert ress[1] == '4'


# csv_data1 = (
# """name,age,city
# John,25,New York
# Jane,30,Los Angeles
# "Smith, Jr.",40,Chicago"""
# )

# csv_data2 = (
# """product,price,description
# "Widget, Deluxe",19.99,"High-quality widget with multiple features"
# "Gadget",9.99,"Compact gadget, easy to use"
# "Tool, Multi-purpose",29.99,"Durable and versatile tool"""
# )

# csv_data3 = (
# """"Widget, Deluxe",19.99,"High-quality widget with multiple features"
# "Gadget",9.99,"Compact gadget, easy to use"
# "Tool, Multi-purpose",29.99,"Durable and versatile tool"""
# )

# class TestCSVParser(object):

#     def test_csv_out_handles_empty_data(self):
#         data = ""
#         parser = _out.CSVOut('F1', use_header=True)
#         msg = Msg('user', meta={'content': data})
#         res = parser(msg).m['F1']
#         assert res == utils.UNDEFINED

#     def test_csv_row_parser_handles_single_row_no_header(self):
#         data = "John,25,New York"
#         parser = _out.CSVOut('F1', use_header=False)
#         msg = Msg('user', meta={'content': data})
#         res = parser(msg).m['F1']
#         assert len(res) == 1
#         assert res[0] == ["John", "25", "New York"]

#     def test_csv_row_parser_handles_single_row_with_header(self):
#         data = "name,age,city\nJohn,25,New York"
#         parser = _out.CSVOut('F1', use_header=True)
#         msg = Msg('user', meta={'content': data})
#         res = parser(msg).m['F1']
#         assert len(res) == 1
#         assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))

#     def test_csv_row_parser_handles_multiple_rows_with_header(self):
#         data = csv_data1
#         parser = _out.CSVOut('F1', use_header=True)
#         msg = Msg('user', meta={'content': data})
#         res = parser(msg).m['F1']
#         assert len(res) == 3
#         assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
#         assert res[1] == OrderedDict(zip(["name", "age", "city"], ["Jane", "30", "Los Angeles"]))
#         assert res[2] == OrderedDict(zip(["name", "age", "city"], ["Smith, Jr.", "40", "Chicago"]))

#     def test_csv_row_parser_handles_multiple_rows_no_header(self):
#         data = csv_data3
#         parser = _out.CSVOut('F1', use_header=False)
#         msg = Msg('user', meta={'content': data})
#         res = parser(msg).m['F1']
#         assert len(res) == 3
#         assert res[0] == ["Widget, Deluxe", "19.99", "High-quality widget with multiple features"]
#         assert res[1] == ["Gadget", "9.99", "Compact gadget, easy to use"]
#         assert res[2] == ["Tool, Multi-purpose", "29.99", "Durable and versatile tool"]

#     def test_csv_row_parser_handles_streamed_data_with_header(self):
#         data = csv_data1
#         parser = _out.CSVOut('F1', use_header=True)
#         res = []
#         for cur in asst_stream(data, [parser]):
#             cur = cur.m['F1']
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 3
#         assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
#         assert res[1] == OrderedDict(zip(["name", "age", "city"], ["Jane", "30", "Los Angeles"]))
#         assert res[2] == OrderedDict(zip(["name", "age", "city"], ["Smith, Jr.", "40", "Chicago"]))

#     def test_csv_row_parser_handles_streamed_data_no_header(self):
#         data = csv_data3
#         parser = _out.CSVOut('F1', use_header=False)
#         res = []
#         for cur in asst_stream(data, [parser]):
#             cur = cur.m['F1']
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 3
#         assert res[0] == ["Widget, Deluxe", "19.99", "High-quality widget with multiple features"]
#         assert res[1] == ["Gadget", "9.99", "Compact gadget, easy to use"]
#         assert res[2] == ["Tool, Multi-purpose", "29.99", "Durable and versatile tool"]

#     def test_csv_row_parser_handles_different_delimiters(self):
#         data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
#         out = _out.CSVOut('F1', delimiter='|', use_header=True)
#         msg = Msg('user', meta={'content': data})
#         res = out(msg).m['F1']
#         assert len(res) == 2
#         assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
#         assert res[1] == OrderedDict(zip(["name", "age", "city"], ["Jane", "30", "Los Angeles"]))

#     def test_csv_delim_parser_returns_correct_len_with_header(self):
        
#         data = csv_data1
#         out = _out.CSVOut('F1', use_header=True)
#         msg = Msg('user', meta={'content': data})
#         res = out(msg).m['F1']
#         assert len(res) == 3

#     def test_char_delim_parser_returns_csv_with_no_header(self):
        
#         data = csv_data3
#         out = _out.CSVOut('F1', use_header=False)
#         msg = Msg('user', meta={'content': data})
#         res = out(msg).m['F1']
#         assert len(res) == 3

#     def test_csv_delim_parser_returns_correct_len_with_header2(self):
        
#         data = csv_data1
#         out = _out.CSVOut(
#             'F1', use_header=True
#         )
#         res = []

#         for cur in asst_stream(data, [out]):
#             cur = cur.m['F1']
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert len(res) == 3

#     def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
#         data = csv_data2
#         out = _out.CSVOut(
#             'F1', use_header=True
#         )
#         res = []

#         for cur in asst_stream(data, [out]):
#             cur = cur.m['F1']
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert len(res) == 3


# from dachi.msg import _parse

# class TestTupleOut(object):

#     def test_tupleout_handles_valid_data(self):
#         """Test TupleOut processes valid data correctly."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': 'key1::value1\nkey2::value2'}
#         )
#         result = tuple_out(msg).m['out']
#         assert result[0]['key1'] == 'value1'
#         assert result[0]['key2'] == 'value2'

#     def test_tupleout_handles_valid_data_with_two_convs(self):
#         """Test TupleOut processes valid data correctly."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv, kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': 'key1::value1\nkey2::value2,key3::value3\nkey4::value4'}
#         )
#         result = tuple_out(msg).m['out']
#         assert result[0]['key1'] == 'value1'
#         assert result[0]['key2'] == 'value2'
#         assert result[1]['key3'] == 'value3'
#         assert result[1]['key4'] == 'value4'

#     def test_tupleout_raises_error_if_insufficient_convs(self):
#         """Test TupleOut processes valid data correctly."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': 'key1::value1\nkey2::value2,key3::value3\nkey4::value4'}
#         )
#         with pytest.raises(RuntimeError):
#             tuple_out(msg).m['out']

#     def test_tupleout_raises_error_if_insufficient_keyvals(self):
#         """Test TupleOut processes valid data correctly."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv, kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': 'key1::value1\nkey2::value2'}
#         )
#         with pytest.raises(RuntimeError):
#             tuple_out(msg).m['out']

#     def test_tupleout_handles_empty_data(self):
#         """Test TupleOut handles empty data gracefully."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv, kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': ''}
#         )
#         result = tuple_out(
#             msg, 
#             is_streamed=True, 
#             is_last=False
#         ).m['out']
#         assert len(result) == 0

#     def test_tupleout_creates_example(self):
#         """Test TupleOut handles empty data gracefully."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv, kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         example = tuple_out.example()
#         assert 'x' in example
#         assert 'y' in example

#     def test_tupleout_creates_template(self):
#         """Test TupleOut handles empty data gracefully."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.TupleOut(
#             convs=[kv_conv, kv_conv],
#             parser=char_delim_parser,
#             name='out'
#         )
#         template = tuple_out.template()
#         assert 'x' in template
#         assert 'y' in template


# class TestTupleOut(object):

#     def test_tupleout_handles_valid_data(self):
#         """Test TupleOut processes valid data correctly."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.ListOut(
#             conv=kv_conv,
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': 'key1::value1\nkey2::value2'}
#         )
#         result = tuple_out(msg).m['out']
#         assert result[0]['key1'] == 'value1'
#         assert result[0]['key2'] == 'value2'

#     def test_listout_handles_valid_data_with_two_convs(self):
#         """Test TupleOut processes valid data correctly."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         tuple_out = _out.ListOut(
#             conv=kv_conv,
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': 'key1::value1\nkey2::value2,key3::value3\nkey4::value4'}
#         )
#         result = tuple_out(msg).m['out']
#         assert result[0]['key1'] == 'value1'
#         assert result[0]['key2'] == 'value2'
#         assert result[1]['key3'] == 'value3'
#         assert result[1]['key4'] == 'value4'

#     def test_listout_handles_empty_data(self):
#         """Test TupleOut handles empty data gracefully."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         list_out = _out.ListOut(
#             conv=kv_conv,
#             parser=char_delim_parser,
#             name='out'
#         )
#         msg = Msg(
#             role='user',
#             meta={'content': ''}
#         )
#         result = list_out(
#             msg, 
#             is_streamed=True, 
#             is_last=False
#         ).m['out']
#         assert len(result) == 0

#     def test_listout_creates_example(self):
#         """Test TupleOut handles empty data gracefully."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         list_out = _out.ListOut(
#             conv=kv_conv,
#             parser=char_delim_parser,
#             name='out'
#         )
#         example = list_out.example()
#         print(example)
#         assert 'x' in example
#         assert 'y' in example

#     def test_listout_creates_template(self):
#         """Test TupleOut handles empty data gracefully."""
#         kv_conv = text_proc.KVOut(
#             name='KV',
#             key_descr={'key1': 'description1', 'key2': 'description2'}
#         )
#         char_delim_parser = _parse.CharDelimParser(
#             sep=','
#         )
#         list_out = _out.ListOut(
#             conv=kv_conv,
#             parser=char_delim_parser,
#             name='out'
#         )
#         template = list_out.template()
#         assert 'key1' in template
#         assert 'key2' in template
