from dachi.core import ModuleList
from dachi.proc._resp import PrimOut, KVOut, IndexOut, CSVOut, ToOut, JSONListOut, JSONValsOut, StructOut, TextOut, TupleOut
from dachi.proc._parser import LineParser
from dachi.proc._process import Process
from dachi import utils
import json
import pytest
import pydantic


# Test helper classes that use the new ToOut API
class EchoOut(ToOut):
    """Pass-through processor for testing."""
    
    def forward(self, resp: str | None):
        return str(resp) if resp is not None else utils.UNDEFINED
    
    def delta(self, resp: str | None, delta_store, is_last=True):
        return str(resp) if resp is not None else ""
    
    def render(self, data):
        return str(data)
    
    def template(self):
        return "<echo>"
    
    def example(self):
        return "example"


class ConcatOut(ToOut):
    """Concatenates chunks across streaming."""
    
    def forward(self, resp: str | None):
        return str(resp) if resp is not None else utils.UNDEFINED
    
    def delta(self, chunk: str | None, delta_store, is_last=True):
        chunk_str = str(chunk) if chunk is not None else ""
        buf = delta_store.get("buf", "") + chunk_str
        if is_last:
            return buf
        delta_store["buf"] = buf
        return utils.UNDEFINED
    
    def render(self, data):
        return str(data)
    
    def template(self):
        return "<concat>"
    
    def example(self):
        return "concatenated"


# Tests for the new ToOut API
class TestToOutBasic:
    """Test basic ToOut functionality with forward() method."""
    
    def test_toprim_forward_bool_true(self):
        proc = PrimOut(out_cls='bool')
        result = proc.forward("true")
        assert result == True
        
    def test_toprim_forward_bool_false(self):
        proc = PrimOut(out_cls='bool')
        result = proc.forward("false")
        assert result == False
        
    def test_toprim_forward_int(self):
        proc = PrimOut(out_cls='int')
        result = proc.forward("42")
        assert result == 42
        
    def test_toprim_forward_float(self):
        proc = PrimOut(out_cls='float')
        result = proc.forward("3.14")
        assert result == 3.14
        
    def test_toprim_forward_str(self):
        proc = PrimOut(out_cls='str')
        result = proc.forward("hello")
        assert result == "hello"
        
    def test_kvout_forward(self):
        proc = KVOut(sep='::')
        result = proc.forward("name::John\nage::25")
        assert result == {'name': 'John', 'age': '25'}
        
    def test_indexout_forward(self):
        proc = IndexOut(sep='::')
        result = proc.forward("1::First\n2::Second\n3::Third")
        assert result == ['First', 'Second', 'Third']
        
    def test_csvout_forward_with_header(self):
        proc = CSVOut()
        result = proc.forward("name,age\nJohn,25\nJane,30")
        expected = [{'name': 'John', 'age': '25'}, {'name': 'Jane', 'age': '30'}]
        assert result == expected
        
    def test_csvout_forward_no_header(self):
        proc = CSVOut(use_header=False)
        result = proc.forward("John,25\nJane,30")
        expected = [['John', '25'], ['Jane', '30']]
        assert result == expected


class TestToOutStreaming:
    """Test streaming functionality with delta() method."""
    
    def test_toprim_delta_streaming(self):
        proc = PrimOut(out_cls='int')
        delta_store = {}
        
        # First chunk
        result1 = proc.delta("4", delta_store, is_last=False)
        assert result1 == utils.UNDEFINED
        
        # Final chunk
        result2 = proc.delta("2", delta_store, is_last=True)
        assert result2 == 42
        
    def test_concat_streaming(self):
        proc = ConcatOut()
        delta_store = {}
        
        # First chunk
        result1 = proc.delta("Hel", delta_store, is_last=False)
        assert result1 == utils.UNDEFINED
        
        # Second chunk
        result2 = proc.delta("lo", delta_store, is_last=False)
        assert result2 == utils.UNDEFINED
        
        # Final chunk
        result3 = proc.delta("!", delta_store, is_last=True)
        assert result3 == "Hello!"



class TestPrimOutExtended:
    """Extended tests for PrimOut with various data types."""
    
    def test_toprim_bool_true_variations(self):
        """Test various true boolean representations."""
        proc = PrimOut(out_cls='bool')
        for true_val in ['true', 'True', 'TRUE', 'y', 'yes', '1', 't']:
            assert proc.forward(true_val) is True, f"Failed for: {true_val}"
            
    def test_toprim_bool_false_variations(self):
        """Test various false boolean representations."""
        proc = PrimOut(out_cls='bool')
        for false_val in ['false', 'False', 'FALSE', 'n', 'no', '0', 'f', 'other']:
            assert proc.forward(false_val) is False, f"Failed for: {false_val}"
    
    def test_toprim_streaming_accumulation(self):
        """Test that PrimOut accumulates streaming data correctly."""
        proc = PrimOut(out_cls='int')
        delta_store = {}
        
        # Stream individual digits
        assert proc.delta("1", delta_store, is_last=False) == utils.UNDEFINED
        assert proc.delta("2", delta_store, is_last=False) == utils.UNDEFINED  
        result = proc.delta("3", delta_store, is_last=True)
        assert result == 123


class TestKVOutExtended:
    """Extended tests for KVOut key-value parsing."""
    
    def test_kvout_multiline_values(self):
        """Test key-value parsing with complex values."""
        proc = KVOut(sep='::')
        text = "name::John Doe\ndescription::A person who likes testing\nage::25"
        result = proc.forward(text)
        # Should handle the multiline gracefully
        assert result == {'name': 'John Doe', 'description': 'A person who likes testing', 'age': '25'}
        
    def test_kvout_different_separators(self):
        """Test different separator characters."""
        separators = ['::', ':', '=', '|']
        for sep in separators:
            proc = KVOut(sep=sep)
            text = f"key1{sep}value1\nkey2{sep}value2"
            result = proc.forward(text)
            assert result == {'key1': 'value1', 'key2': 'value2'}
            
    def test_kvout_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        proc = KVOut(sep='::')
        text = "  name  ::  John  \n  age  ::  25  "
        result = proc.forward(text)
        # KVOut strips whitespace from keys and values
        assert result == {'name': 'John', 'age': '25'}


class TestIndexOutExtended:
    """Extended tests for IndexOut indexed parsing."""
    
    def test_indexout_out_of_order(self):
        """Test parsing indices provided out of order."""
        proc = IndexOut(sep='::')
        text = "3::Third\n1::First\n2::Second"
        result = proc.forward(text)
        assert result == ['First', 'Second', 'Third']
        
    def test_indexout_zero_based_vs_one_based(self):
        """Test that 1-based indices are converted to 0-based."""
        proc = IndexOut(sep='::')
        text = "1::First\n2::Second"
        result = proc.forward(text)
        assert result[0] == 'First'
        assert result[1] == 'Second'
        
    def test_indexout_large_gaps(self):
        """Test handling of large gaps in indices."""
        proc = IndexOut(sep='::')
        text = "1::First\n10::Tenth"
        result = proc.forward(text)
        assert len(result) == 10
        assert result[0] == 'First'
        assert result[9] == 'Tenth'
        assert all(x is None for x in result[1:9])


class TestCSVOutExtended:
    """Extended tests for CSVOut CSV parsing."""
    
    def test_csvout_quoted_fields(self):
        """Test CSV parsing with quoted fields containing commas."""
        proc = CSVOut()
        csv_text = 'name,description\n"John, Jr.","A person, who likes commas"\nJane,Simple'
        result = proc.forward(csv_text)
        assert len(result) == 2
        assert result[0]['name'] == 'John, Jr.'
        assert result[0]['description'] == 'A person, who likes commas'
        
    def test_csvout_different_delimiters(self):
        """Test CSV parsing with various delimiters."""
        delimiters = [',', ';', '|', '\t']
        for delim in delimiters:
            proc = CSVOut(delimiter=delim, use_header=False)
            text = f"John{delim}25{delim}Boston\nJane{delim}30{delim}Seattle"
            result = proc.forward(text)
            assert len(result) == 2
            assert result[0] == ['John', '25', 'Boston']
            
    def test_csvout_malformed_data(self):
        """Test CSV parsing with malformed data."""
        proc = CSVOut(use_header=False)
        # Missing fields in second row
        text = "John,25,Boston\nJane,30\nBob,35,Chicago"
        result = proc.forward(text)
        assert len(result) == 3
        assert len(result[1]) == 2  # Jane row has only 2 fields
        
    def test_csvout_empty_and_whitespace(self):
        """Test CSV parsing with empty fields and whitespace."""
        proc = CSVOut(use_header=False)
        text = "John,,Boston\n ,25, \n\n"
        result = proc.forward(text)
        # Should handle empty fields and whitespace
        assert len(result) >= 2


class TestJSONListOut:
    """Tests for JSONListOut class."""
    
    def test_jsonlistout_forward_basic(self):
        """Test basic JSON array parsing."""
        proc = JSONListOut()
        text = '[{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]'
        result = proc.forward(text)
        expected = [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]
        assert result == expected
        
    def test_jsonlistout_forward_empty(self):
        """Test empty JSON array."""
        proc = JSONListOut()
        result = proc.forward('[]')
        assert result == []
        
    def test_jsonlistout_forward_single_item(self):
        """Test JSON array with single item."""
        proc = JSONListOut()
        result = proc.forward('[{"id": 1, "name": "test"}]')
        assert result == [{"id": 1, "name": "test"}]
        
    def test_jsonlistout_delta_streaming(self):
        """Test streaming JSON array parsing."""
        proc = JSONListOut()
        delta_store = {}
        
        # Stream JSON array in chunks
        chunks = ['[{"name":', ' "John"}, {"name":', ' "Jane"}]']
        results = []
        
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = proc.delta(chunk, delta_store, is_last)
            if result != utils.UNDEFINED:
                results.extend(result if isinstance(result, list) else [result])
        
        expected = [{"name": "John"}, {"name": "Jane"}]
        assert results == expected


class TestJSONValsOut:
    """Tests for JSONValsOut class."""
    
    def test_jsonvalsout_forward_basic(self):
        """Test basic JSON object key-value parsing."""
        proc = JSONValsOut()
        text = '{"name": "John", "age": 25, "city": "Boston"}'
        result = proc.forward(text)
        expected = [("name", "John"), ("age", 25), ("city", "Boston")]
        assert result == expected
        
    def test_jsonvalsout_forward_empty(self):
        """Test empty JSON object."""
        proc = JSONValsOut()
        result = proc.forward('{}')
        assert result == []
        
    def test_jsonvalsout_forward_single_key(self):
        """Test JSON object with single key."""
        proc = JSONValsOut()
        result = proc.forward('{"status": "success"}')
        assert result == [("status", "success")]
        
    def test_jsonvalsout_delta_streaming(self):
        """Test streaming JSON object parsing."""
        proc = JSONValsOut()
        delta_store = {}
        
        # Stream JSON object in chunks  
        chunks = ['{"name":', ' "John", ', '"age": 25}']
        results = []
        
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            result = proc.delta(chunk, delta_store, is_last)
            if result != utils.UNDEFINED and result is not None:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
        
        # Should get key-value tuples
        assert len(results) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results if isinstance(item, tuple))


class TestStructOut:
    """Tests for StructOut structured data processing."""
    
    def test_structout_forward_basic_json(self):
        """Test basic JSON parsing."""
        proc = StructOut()
        result = proc.forward('{"name": "John", "age": 25}')
        assert result == {"name": "John", "age": 25}
    
    def test_structout_forward_with_pydantic_model(self):
        """Test structured output with Pydantic model validation."""
        class Person(pydantic.BaseModel):
            name: str
            age: int
        
        proc = StructOut(struct=Person)
        result = proc.forward('{"name": "John", "age": 25}')
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 25
    
    def test_structout_forward_invalid_json(self):
        """Test error handling for invalid JSON."""
        proc = StructOut()
        with pytest.raises(RuntimeError, match="Failed to parse JSON"):
            proc.forward('{"invalid": json')


class TestTextOut:
    """Tests for TextOut text processing."""
    
    def test_textout_forward_basic(self):
        """Test basic text extraction."""
        proc = TextOut()
        result = proc.forward('Hello world')
        assert result == 'Hello world'
    
    def test_textout_forward_none_text(self):
        """Test handling of None text."""
        proc = TextOut()
        result = proc.forward(None)
        assert result == utils.UNDEFINED
    
    def test_textout_delta_streaming(self):
        """Test TextOut streaming behavior."""
        text_out = TextOut()
        delta_store = {}
        
        # Test first chunk
        result1 = text_out.delta('Hello', delta_store, is_last=False)
        assert result1 == 'Hello'  # Should have the chunk text
        
        # Test second chunk  
        result2 = text_out.delta(' world', delta_store, is_last=True)
        assert result2 == ' world'  # Should have the chunk text


class TestTupleOut:
    """Tests for TupleOut tuple processing."""

    def test_tupleout_forward_basic(self):
        """Test basic tuple processing with ToOut-based processors."""

        # Processors that inherit from ToOut, take strings and return objects
        class StrProcessor(ToOut):
            def forward(self, s: str) -> str:
                return s.strip()
            def delta(self, s: str, delta_store: dict, is_last: bool = False):
                return s.strip()
            def template(self) -> str:
                return "string"
            def example(self) -> str:
                return "example"
            def render(self, data) -> str:
                return str(data)

        class IntProcessor(ToOut):
            def forward(self, s: str) -> int:
                return int(s.strip())
            def delta(self, s: str, delta_store: dict, is_last: bool = False):
                return int(s.strip())
            def template(self) -> str:
                return "integer"
            def example(self) -> str:
                return "42"
            def render(self, data) -> str:
                return str(data)

        processors = ModuleList(vals=[StrProcessor(), IntProcessor()])
        proc = TupleOut(processors=processors, parser=LineParser())
        result = proc.forward('hello\n42')
        assert result == ('hello', 42)

    def test_tupleout_rejects_process_instead_of_toout(self):
        """Test that TupleOut rejects Process instances (must be ToOut)."""

        class StrProcessor(Process):
            def forward(self, s: str) -> str:
                return s.strip()

        processors = ModuleList(vals=[StrProcessor()])
        with pytest.raises(pydantic.ValidationError):
            TupleOut(processors=processors, parser=LineParser())

    def test_tupleout_forward_mismatch_count(self):
        """Test error when processor count doesn't match parsed parts."""

        class StrProcessor(ToOut):
            def forward(self, s: str) -> str:
                return s.strip()
            def delta(self, s: str, delta_store: dict, is_last: bool = False):
                return s.strip()
            def template(self) -> str:
                return "string"
            def example(self) -> str:
                return "example"
            def render(self, data) -> str:
                return str(data)

        processors = ModuleList(vals=[StrProcessor()])  # Only 1 processor
        proc = TupleOut(processors=processors, parser=LineParser())
        with pytest.raises(RuntimeError, match="Expected 1 parts, got 2"):
            proc.forward('hello\n42')  # 2 parts


# TODO: MIGRATION NOTES
"""
TESTS CONSOLIDATED FROM test_out.py and test_parse.py

This file now contains tests for all response processing functionality that was
previously spread across multiple files:

1. ToOut classes (formerly in test_out.py):
   - PrimOut, KVOut, IndexOut, CSVOut
   - Both forward() and delta() methods
   - Streaming behavior

2. Parser classes (formerly in test_parse.py):
   - CSVRowParser, LineParser, CharDelimParser
   - These need more comprehensive test coverage

3. Deprecated functionality:
   - RespProc.run() method - removed
   - Field-based processing (name/from_) - removed
   - Removed classes: TextConv, StructConv, ParsedConv, etc.

The new unified API focuses on:
- forward(resp) for complete responses  
- delta(resp, delta_store, is_last) for streaming
- No automatic field management (handled by _ai.py)
- Cleaner, simpler interface

Next steps:
- Add comprehensive parser tests from test_parse.py
- Test integration with _ai.py when available
- Test template() and render() methods
- Test error conditions with new API
"""


if __name__ == "__main__":
    pytest.main([__file__])