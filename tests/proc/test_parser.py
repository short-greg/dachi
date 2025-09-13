from dachi.proc._parser import CSVRowParser, LineParser, CharDelimParser
from dachi import utils
from collections import OrderedDict

class TestCSVRowParser:
    """Tests for CSVRowParser class."""
    
    def test_forward_with_header_returns_dicts(self):
        """Test CSV parsing with header returns list of dicts."""
        parser = CSVRowParser(use_header=True)
        text = "name,age,city\nJohn,25,Boston\nJane,30,Seattle"
        result = parser.forward(text)
        assert len(result) == 2
        assert result[0]['name'] == 'John'
        assert result[0]['age'] == '25'
        assert result[0]['city'] == 'Boston'
        assert result[1]['name'] == 'Jane'
        assert result[1]['age'] == '30'
        assert result[1]['city'] == 'Seattle'
    
    def test_forward_no_header_returns_lists(self):
        """Test CSV parsing without header returns list of lists."""
        parser = CSVRowParser(use_header=False)
        text = "John,25,Boston\nJane,30,Seattle"
        result = parser.forward(text)
        assert len(result) == 2
        assert result[0] == ['John', '25', 'Boston']
        assert result[1] == ['Jane', '30', 'Seattle']
    
    def test_forward_empty_returns_empty_list(self):
        """Test CSV parsing with empty input returns empty list."""
        parser = CSVRowParser(use_header=True)
        assert parser.forward(None) == []
        assert parser.forward("") == []
    
    def test_delta_streaming_returns_new_rows(self):
        """Test CSV streaming returns new rows as they complete."""
        parser = CSVRowParser(use_header=True, delimiter=',')
        delta_store = {}
        
        # Stream header + partial first row
        result1 = parser.delta("name,age\nJo", delta_store, is_last=False)
        assert result1 == utils.UNDEFINED
        
        # Complete first row + start second
        result2 = parser.delta("hn,25\nJa", delta_store, is_last=False) 
        assert len(result2) == 1
        assert result2[0]['name'] == 'John'
        assert result2[0]['age'] == '25'
        
        # Complete second row
        result3 = parser.delta("ne,30", delta_store, is_last=True)
        assert len(result3) == 1
        assert result3[0]['name'] == 'Jane'
        assert result3[0]['age'] == '30'


class TestLineParser:
    """Tests for LineParser class."""
    
    def test_forward_basic_returns_lines(self):
        """Test basic line parsing returns list of lines."""
        parser = LineParser()
        text = "line1\nline2\nline3"
        result = parser.forward(text)
        assert result == ['line1', 'line2', 'line3']
    
    def test_forward_continuations_joins_lines(self):
        """Test line parsing with backslash continuations joins lines."""
        parser = LineParser()
        text = "line1\\\nline2\nline3\\\nline4"
        result = parser.forward(text)
        assert result == ['line1line2', 'line3line4']
    
    def test_forward_empty_returns_empty_list(self):
        """Test line parsing with empty input returns empty list."""
        parser = LineParser()
        assert parser.forward(None) == []
        assert parser.forward("") == []
    
    def test_delta_streaming_returns_completed_lines(self):
        """Test line streaming returns completed lines."""
        parser = LineParser()
        delta_store = {}
        
        # Stream partial line
        result1 = parser.delta("Hello wo", delta_store, is_last=False)
        assert result1 == utils.UNDEFINED
        
        # Complete first line + start second
        result2 = parser.delta("rld\nSecond li", delta_store, is_last=False)
        assert result2 == ['Hello world']
        
        # Complete second line
        result3 = parser.delta("ne", delta_store, is_last=True)
        assert result3 == ['Second line']


class TestCharDelimParser:
    """Tests for CharDelimParser class."""
    
    def test_forward_basic_returns_split_values(self):
        """Test basic character-delimited parsing returns split values."""
        parser = CharDelimParser(sep=',')
        text = "value1,value2,value3"
        result = parser.forward(text)
        assert result == ['value1', 'value2', 'value3']
    
    def test_forward_different_separators_work(self):
        """Test parsing with different separators works."""
        separators = ['|', ';', ':', '\t']
        for sep in separators:
            parser = CharDelimParser(sep=sep)
            text = f"val1{sep}val2{sep}val3"
            result = parser.forward(text)
            assert result == ['val1', 'val2', 'val3'], f"Failed for separator: {sep}"
    
    def test_forward_empty_returns_empty_list(self):
        """Test parsing with empty input returns empty list."""
        parser = CharDelimParser(sep=',')
        assert parser.forward(None) == []
        assert parser.forward("") == []
    
    def test_delta_streaming_returns_completed_values(self):
        """Test character-delimited streaming returns completed values."""
        parser = CharDelimParser(sep=',')
        delta_store = {}
        
        # Stream partial values
        result1 = parser.delta("val1,val2,par", delta_store, is_last=False)
        assert result1 == ['val1', 'val2']
        
        # Complete partial value + new value
        result2 = parser.delta("tial,val3", delta_store, is_last=False)
        assert result2 == ['partial']
        
        # Final value
        result3 = parser.delta(",final", delta_store, is_last=True)
        assert result3 == ['val3', 'final']
