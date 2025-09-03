from dachi.core import Msg, Resp
from dachi.proc._resp import PrimOut, KVOut, IndexOut, CSVOut, ToOut, JSONListOut, JSONValsOut
from dachi import utils
import json
import pytest
import pydantic


# Test helper classes that use the new ToOut API
class EchoOut(ToOut):
    """Pass-through processor for testing."""
    
    def forward(self, resp):
        return str(resp.text) if hasattr(resp, 'text') else str(resp)
    
    def delta(self, resp, delta_store, is_last=True):
        return str(resp)
    
    def render(self, data):
        return str(data)
    
    def template(self):
        return "<echo>"
    
    def example(self):
        return "example"


class ConcatOut(ToOut):
    """Concatenates chunks across streaming."""
    
    def forward(self, resp):
        return str(resp.text) if hasattr(resp, 'text') else str(resp)
    
    def delta(self, chunk, delta_store, is_last=True):
        buf = delta_store.get("buf", "") + str(chunk)
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
        resp = Resp(msg=Msg(role="assistant", text="true"))
        result = proc.forward(resp)
        assert result == True
        
    def test_toprim_forward_bool_false(self):
        proc = PrimOut(out_cls='bool')
        resp = Resp(msg=Msg(role="assistant", text="false"))
        result = proc.forward(resp)
        assert result == False
        
    def test_toprim_forward_int(self):
        proc = PrimOut(out_cls='int')
        resp = Resp(msg=Msg(role="assistant", text="42"))
        result = proc.forward(resp)
        assert result == 42
        
    def test_toprim_forward_float(self):
        proc = PrimOut(out_cls='float')
        resp = Resp(msg=Msg(role="assistant", text="3.14"))
        result = proc.forward(resp)
        assert result == 3.14
        
    def test_toprim_forward_str(self):
        proc = PrimOut(out_cls='str')
        resp = Resp(msg=Msg(role="assistant", text="hello"))
        result = proc.forward(resp)
        assert result == "hello"
        
    def test_kvout_forward(self):
        proc = KVOut(sep='::')
        resp = Resp(msg=Msg(role="assistant", text="name::John\nage::25"))
        result = proc.forward(resp)
        assert result == {'name': 'John', 'age': '25'}
        
    def test_indexout_forward(self):
        proc = IndexOut(sep='::')
        resp = Resp(msg=Msg(role="assistant", text="1::First\n2::Second\n3::Third"))
        result = proc.forward(resp)
        assert result == ['First', 'Second', 'Third']
        
    def test_csvout_forward_with_header(self):
        proc = CSVOut()
        resp = Resp(msg=Msg(role="assistant", text="name,age\nJohn,25\nJane,30"))
        result = proc.forward(resp)
        expected = [{'name': 'John', 'age': '25'}, {'name': 'Jane', 'age': '30'}]
        assert result == expected
        
    def test_csvout_forward_no_header(self):
        proc = CSVOut(use_header=False)
        resp = Resp(msg=Msg(role="assistant", text="John,25\nJane,30"))
        result = proc.forward(resp)
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
            resp = Resp(msg=Msg(role='assistant', text=true_val))
            assert proc.forward(resp) is True, f"Failed for: {true_val}"
            
    def test_toprim_bool_false_variations(self):
        """Test various false boolean representations."""
        proc = PrimOut(out_cls='bool')
        for false_val in ['false', 'False', 'FALSE', 'n', 'no', '0', 'f', 'other']:
            resp = Resp(msg=Msg(role='assistant', text=false_val))
            assert proc.forward(resp) is False, f"Failed for: {false_val}"
    
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
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        # Should handle the multiline gracefully
        assert result == {'name': 'John Doe', 'description': 'A person who likes testing', 'age': '25'}
        
    def test_kvout_different_separators(self):
        """Test different separator characters."""
        separators = ['::', ':', '=', '|']
        for sep in separators:
            proc = KVOut(sep=sep)
            text = f"key1{sep}value1\nkey2{sep}value2"
            resp = Resp(msg=Msg(role='assistant', text=text))
            result = proc.forward(resp)
            assert result == {'key1': 'value1', 'key2': 'value2'}
            
    def test_kvout_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        proc = KVOut(sep='::')
        text = "  name  ::  John  \n  age  ::  25  "
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        # KVOut strips whitespace from keys and values
        assert result == {'name': 'John', 'age': '25'}


class TestIndexOutExtended:
    """Extended tests for IndexOut indexed parsing."""
    
    def test_indexout_out_of_order(self):
        """Test parsing indices provided out of order."""
        proc = IndexOut(sep='::')
        text = "3::Third\n1::First\n2::Second"
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        assert result == ['First', 'Second', 'Third']
        
    def test_indexout_zero_based_vs_one_based(self):
        """Test that 1-based indices are converted to 0-based."""
        proc = IndexOut(sep='::')
        text = "1::First\n2::Second"
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        assert result[0] == 'First'
        assert result[1] == 'Second'
        
    def test_indexout_large_gaps(self):
        """Test handling of large gaps in indices."""
        proc = IndexOut(sep='::')
        text = "1::First\n10::Tenth"
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
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
        resp = Resp(msg=Msg(role='assistant', text=csv_text))
        result = proc.forward(resp)
        assert len(result) == 2
        assert result[0]['name'] == 'John, Jr.'
        assert result[0]['description'] == 'A person, who likes commas'
        
    def test_csvout_different_delimiters(self):
        """Test CSV parsing with various delimiters."""
        delimiters = [',', ';', '|', '\t']
        for delim in delimiters:
            proc = CSVOut(delimiter=delim, use_header=False)
            text = f"John{delim}25{delim}Boston\nJane{delim}30{delim}Seattle"
            resp = Resp(msg=Msg(role='assistant', text=text))
            result = proc.forward(resp)
            assert len(result) == 2
            assert result[0] == ['John', '25', 'Boston']
            
    def test_csvout_malformed_data(self):
        """Test CSV parsing with malformed data."""
        proc = CSVOut(use_header=False)
        # Missing fields in second row
        text = "John,25,Boston\nJane,30\nBob,35,Chicago"
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        assert len(result) == 3
        assert len(result[1]) == 2  # Jane row has only 2 fields
        
    def test_csvout_empty_and_whitespace(self):
        """Test CSV parsing with empty fields and whitespace."""
        proc = CSVOut(use_header=False)
        text = "John,,Boston\n ,25, \n\n"
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        # Should handle empty fields and whitespace
        assert len(result) >= 2


class TestJSONListOut:
    """Tests for JSONListOut class."""
    
    def test_jsonlistout_forward_basic(self):
        """Test basic JSON array parsing."""
        proc = JSONListOut()
        text = '[{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]'
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        expected = [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]
        assert result == expected
        
    def test_jsonlistout_forward_empty(self):
        """Test empty JSON array."""
        proc = JSONListOut()
        resp = Resp(msg=Msg(role='assistant', text='[]'))
        result = proc.forward(resp)
        assert result == []
        
    def test_jsonlistout_forward_single_item(self):
        """Test JSON array with single item."""
        proc = JSONListOut()
        resp = Resp(msg=Msg(role='assistant', text='[{"id": 1, "name": "test"}]'))
        result = proc.forward(resp)
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
        resp = Resp(msg=Msg(role='assistant', text=text))
        result = proc.forward(resp)
        expected = [("name", "John"), ("age", 25), ("city", "Boston")]
        assert result == expected
        
    def test_jsonvalsout_forward_empty(self):
        """Test empty JSON object."""
        proc = JSONValsOut()
        resp = Resp(msg=Msg(role='assistant', text='{}'))
        result = proc.forward(resp)
        assert result == []
        
    def test_jsonvalsout_forward_single_key(self):
        """Test JSON object with single key."""
        proc = JSONValsOut()
        resp = Resp(msg=Msg(role='assistant', text='{"status": "success"}'))
        result = proc.forward(resp)
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


# TODO: Parser tests from test_parse.py need to be added here
# The following Parser classes still exist in _resp.py and need testing:
# - CSVRowParser
# - LineParser  
# - CharDelimParser
#
# These were previously tested in test_parse.py but since all functionality
# is now consolidated in _resp.py, their tests should be moved here.

class TestParsersIntegration:
    """Tests for Parser classes that are used by ToOut classes."""
    
    def test_csv_parser_basic(self):
        """Basic test for CSVRowParser - more comprehensive tests needed."""
        # Import the parser from _resp since it's no longer in _parse
        from dachi.proc._resp import CSVRowParser
        from dachi.core import Resp, Msg
        parser = CSVRowParser(use_header=True)
        resp = Resp(msg=Msg(role="assistant", text="name,age\nJohn,25"))
        result = parser.forward(resp)
        # This is a placeholder - full test suite from test_parse.py should be migrated
        assert len(result) == 1
    
    def test_line_parser_basic(self):
        """Basic test for LineParser."""
        from dachi.proc._resp import LineParser
        parser = LineParser()
        result = parser.forward("line1\nline2\nline3")
        # This is a placeholder - more comprehensive tests needed
        assert isinstance(result, list)


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