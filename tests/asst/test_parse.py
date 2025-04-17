from dachi.asst import _parse
from dachi.msg import Msg, StreamMsg
import typing
from dachi import utils

data = (
"""x: y
z: e"""
)

data2 = (
"""t: y\
xy
z: e"""
)


def asst_stream(data, convs) -> typing.Iterator:
    """Use to simulate an assitant stream"""
    delta_store = {}
    for i, d in enumerate(data):
        is_last = i == len(data) - 1
        msg = StreamMsg('assistant', meta={'content': d}, is_last=is_last)
        for conv in convs:
            msg = conv(msg, delta_store)
            yield msg


class TestLineParser(object):

    def test_line_parse_parses_when_no_newline_symbols_with_correct_num_lines(self):
        
        line_parser = _parse.LineParser(
            'F1', 'content' 
        )
        msg = Msg(
            role='user', meta={'content': data}
        )
        lines = line_parser(msg).m['F1']
        assert len(lines) == 2

    def test_line_parse_parses_when_no_newline_symbols_with_correct_val(self):
        
        msg = Msg(
            role='user', meta={'content': data}
        )
        line_parser = _parse.LineParser('F1')
        lines = line_parser(msg).m['F1']
        assert lines[1] == "z: e"

    def test_line_parse_parses_when_newline_symbols_with_correct_count(self):
        
        msg = Msg(
            role='user', meta={'content': data}
        )
        line_parser = _parse.LineParser('F1')
        msg = line_parser(msg)
        
        assert len(msg.m['F1']) == 2

    def test_line_parse_parses_when_newline_symbols_with_correct_count(self):
        target = (
"""t: y\
xy"""
        )
        msg = Msg(
            role='user', meta={'content': data2}
        )
        line_parser = _parse.LineParser('F1')
        lines = line_parser(msg).m['F1']
        assert lines[0] == target

    def test_line_parse_delta_correctly(self):
        target = (
"""t: y\
xy"""
        )
        res = []
        data = (
            """x: y
            z: e"""
        )

        parser = _parse.LineParser('F1')
        
        for cur in asst_stream(data, [parser]):
            if cur.m['F1'] != utils.UNDEFINED:
                res.extend(cur.m['F1'])

        assert len(res) == 2

    def test_line_parse_delta_correctly_to_value(self):
        t1 = "x: y"
        t2 = "z: e"
        res = []

        parser = _parse.LineParser('F1')

        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == t1
        assert res[1] == t2

    def test_line_parse_delta_correctly_length_with_continuation(self):
        t1 = (
"""t: y\
xy"""
        )
        t2 = "z: e"
        res = []

        parser = _parse.LineParser('F1')

        for cur in asst_stream(data2, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 2

    def test_line_parse_delta_correctly_to_value_with_continuation2(self):
        t1 = (
"""t: y\
xy"""
        )
        t2 = "z: e"
        res = []

        parser = _parse.LineParser('F1')

        for cur in asst_stream(data2, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == t1
        assert res[1] == t2


class TestFullParser(object):

    def test_full_parser_accumulates_data_until_end_of_stream(self):
        data = ["Hello", " ", "World"]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "Hello World"

    def test_full_parser_handles_empty_stream(self):
        data = []
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 0

    def test_full_parser_handles_single_chunk_stream(self):
        data = ["SingleChunk"]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "SingleChunk"

    def test_full_parser_handles_multiple_chunks_with_special_characters(self):
        data = ["Hello,", " ", "this", " ", "is", " ", "a", " ", "test!"]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "Hello, this is a test!"

    def test_full_parser_clears_delta_store_after_stream_end(self):
        data = ["Part1", "Part2"]
        parser = _parse.FullParser('F1')
        delta_store = {}
        for cur in asst_stream(data, [parser]):
            parser.delta(cur.m['content'], delta_store, streamed=True, is_last=False)
        assert 'val' in delta_store
        parser.delta(data[-1], delta_store, streamed=True, is_last=True)
        assert 'val' not in delta_store

    def test_full_parser_render_converts_data_to_string(self):
        data = ["Rendered", " ", "Output"]
        parser = _parse.FullParser('F1')
        res = parser.render(data)
        assert res == "['Rendered', ' ', 'Output']"

    def test_full_parser_handles_stream_with_numbers(self):
        data = ["123", "456", "789"]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "123456789"

    def test_full_parser_handles_stream_with_mixed_data_types(self):
        data = ["Hello", "123", "World"]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "Hello123World"

    def test_full_parser_handles_stream_with_unicode_characters(self):
        data = ["こんにちは", " ", "世界"]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "こんにちは 世界"

    def test_full_parser_handles_stream_with_empty_strings(self):
        data = ["", "Hello", "", "World", ""]
        parser = _parse.FullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 1
        assert res[0] == "HelloWorld"

    def test_full_parser_only_returns_last_element(self):
        
        data = "2.0"
        full_parser = _parse.FullParser('F1')
        msg = Msg('user', meta={'content': data})
        res = full_parser(msg)
        res = res.m['F1']
        assert res == ["2.0"]

    def test_full_parser_only_returns_last_element_with_stream(self):
        
        data = "2.0"
        parser = _parse.FullParser('F1')

        res = []

        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == "2.0"


class TestCharDelimParser(object):

    def test_char_delim_parser_handles_empty_string(self):
        data = ""
        parser = _parse.CharDelimParser(name='F1', sep=',')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert res == []

    def test_char_delim_parser_handles_single_character(self):
        data = "a"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 1
        assert res[0] == "a"

    def test_char_delim_parser_handles_no_delimiters(self):
        data = "abc"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 1
        assert res[0] == "abc"

    def test_char_delim_parser_handles_multiple_delimiters(self):
        data = "a,b,c"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_trailing_delimiter(self):
        data = "a,b,c,"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_leading_delimiter(self):
        data = ",a,b,c"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 4
        assert res[0] == ""
        assert res[1] == "a"
        assert res[2] == "b"
        assert res[3] == "c"

    def test_char_delim_parser_handles_custom_delimiter(self):
        data = "a|b|c"
        parser = _parse.CharDelimParser(name='F1', sep='|')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data(self):
        data = "a,b,c"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data_with_trailing_delimiter(self):
        data = "a,b,c,"
        parser = _parse.CharDelimParser(name='F1', sep=',')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data_with_partial_chunk(self):
        data = ["a,b", ",c"]
        parser = _parse.CharDelimParser(name='F1', sep=',')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data_with_custom_delimiter(self):
        data = "a|b|c"
        parser = _parse.CharDelimParser(name='F1', sep='|')
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_rendering(self):
        data = ["a", "b", "c"]
        parser = _parse.CharDelimParser(name='F1', sep=',')
        rendered = parser.render(data)
        assert rendered == "a,b,c"

    def test_char_delim_parser_handles_rendering_with_custom_delimiter(self):
        data = ["a", "b", "c"]
        parser = _parse.CharDelimParser(name='F1', sep='|')
        rendered = parser.render(data)
        assert rendered == "a|b|c"

    def test_char_delim_parser_returns_len_2(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser(name='F1', sep='.')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 2

    def test_char_delim_parser_returns_correct_values(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser(name='F1', sep='.')
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert res[0] == "2"
        assert res[1] == "0"

    def test_char_delim_parser_returns_correct_values_with_delta(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser(name='F1', sep='.')
        res = []

        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == "2"
        assert res[1] == "0"

csv_data1 = (
"""name,age,city
John,25,New York
Jane,30,Los Angeles
"Smith, Jr.",40,Chicago"""
)

csv_data2 = (
"""product,price,description
"Widget, Deluxe",19.99,"High-quality widget with multiple features"
"Gadget",9.99,"Compact gadget, easy to use"
"Tool, Multi-purpose",29.99,"Durable and versatile tool"""
)

csv_data3 = (
""""Widget, Deluxe",19.99,"High-quality widget with multiple features"
"Gadget",9.99,"Compact gadget, easy to use"
"Tool, Multi-purpose",29.99,"Durable and versatile tool"""
)

from collections import OrderedDict

class TestCSVParser(object):

    
    def test_csv_row_parser_handles_empty_data(self):
        data = ""
        parser = _parse.CSVRowParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert res == utils.UNDEFINED

    def test_csv_row_parser_handles_single_row_no_header(self):
        data = "John,25,New York"
        parser = _parse.CSVRowParser('F1', use_header=False)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 1
        assert res[0] == ["John", "25", "New York"]

    def test_csv_row_parser_handles_single_row_with_header(self):
        data = "name,age,city\nJohn,25,New York"
        parser = _parse.CSVRowParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 1
        assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))

    def test_csv_row_parser_handles_multiple_rows_with_header(self):
        data = csv_data1
        parser = _parse.CSVRowParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
        assert res[1] == OrderedDict(zip(["name", "age", "city"], ["Jane", "30", "Los Angeles"]))
        assert res[2] == OrderedDict(zip(["name", "age", "city"], ["Smith, Jr.", "40", "Chicago"]))

    def test_csv_row_parser_handles_multiple_rows_no_header(self):
        data = csv_data3
        parser = _parse.CSVRowParser('F1', use_header=False)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == ["Widget, Deluxe", "19.99", "High-quality widget with multiple features"]
        assert res[1] == ["Gadget", "9.99", "Compact gadget, easy to use"]
        assert res[2] == ["Tool, Multi-purpose", "29.99", "Durable and versatile tool"]

    def test_csv_row_parser_handles_streamed_data_with_header(self):
        data = csv_data1
        parser = _parse.CSVRowParser('F1', use_header=True)
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
        assert res[1] == OrderedDict(zip(["name", "age", "city"], ["Jane", "30", "Los Angeles"]))
        assert res[2] == OrderedDict(zip(["name", "age", "city"], ["Smith, Jr.", "40", "Chicago"]))

    def test_csv_row_parser_handles_streamed_data_no_header(self):
        data = csv_data3
        parser = _parse.CSVRowParser('F1', use_header=False)
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == ["Widget, Deluxe", "19.99", "High-quality widget with multiple features"]
        assert res[1] == ["Gadget", "9.99", "Compact gadget, easy to use"]
        assert res[2] == ["Tool, Multi-purpose", "29.99", "Durable and versatile tool"]

    def test_csv_row_parser_handles_different_delimiters(self):
        data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
        parser = _parse.CSVRowParser('F1', delimiter='|', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 2
        assert res[0] == OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
        assert res[1] == OrderedDict(zip(["name", "age", "city"], ["Jane", "30", "Los Angeles"]))

    def test_csv_delim_parser_returns_correct_len_with_header(self):
        
        data = csv_data1
        parser = _parse.CSVRowParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3

    def test_char_delim_parser_returns_csv_with_no_header(self):
        
        data = csv_data3
        parser = _parse.CSVRowParser('F1', use_header=False)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3

    def test_csv_delim_parser_returns_correct_len_with_header2(self):
        
        data = csv_data1
        parser = _parse.CSVRowParser(
            'F1', use_header=True
        )
        res = []

        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 3

    def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
        data = csv_data2
        parser = _parse.CSVRowParser(
            'F1', use_header=True
        )
        res = []

        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 3


class TestCSVCellParser(object):

    def test_csv_cell_parser_handles_empty_data(self):
        data = ""
        parser = _parse.CSVCellParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert res == utils.UNDEFINED

    def test_csv_cell_parser_handles_single_row_with_header(self):
        data = "name,age,city\nJohn,25,New York"
        parser = _parse.CSVCellParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")

    def test_csv_cell_parser_handles_single_row_without_header(self):
        data = "John,25,New York"
        parser = _parse.CSVCellParser('F1', use_header=False)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 3
        assert res[0] == (0, "John")
        assert res[1] == (0, "25")
        assert res[2] == (0, "New York")

    def test_csv_cell_parser_handles_multiple_rows_with_header(self):
        data = "name,age,city\nJohn,25,New York\nJane,30,Los Angeles"
        parser = _parse.CSVCellParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 6
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")
        assert res[3] == (1, "name", "Jane")
        assert res[4] == (1, "age", "30")
        assert res[5] == (1, "city", "Los Angeles")

    def test_csv_cell_parser_handles_multiple_rows_without_header(self):
        data = "John,25,New York\nJane,30,Los Angeles"
        parser = _parse.CSVCellParser('F1', use_header=False)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 6
        assert res[0] == (0, "John")
        assert res[1] == (0, "25")
        assert res[2] == (0, "New York")
        assert res[3] == (1, "Jane")
        assert res[4] == (1, "30")
        assert res[5] == (1, "Los Angeles")

    def test_csv_cell_parser_handles_different_delimiters(self):
        data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
        parser = _parse.CSVCellParser('F1', delimiter='|', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 6
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")
        assert res[3] == (1, "name", "Jane")
        assert res[4] == (1, "age", "30")
        assert res[5] == (1, "city", "Los Angeles")

    def test_csv_cell_parser_handles_streamed_data_with_header(self):
        data = csv_data1
        parser = _parse.CSVCellParser('F1', use_header=True)
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 9
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")
        assert res[3] == (1, "name", "Jane")
        assert res[4] == (1, "age", "30")
        assert res[5] == (1, "city", "Los Angeles")
        assert res[6] == (2, "name", "Smith, Jr.")
        assert res[7] == (2, "age", "40")
        assert res[8] == (2, "city", "Chicago")

    def test_csv_cell_parser_handles_streamed_data_without_header(self):
        data = csv_data3
        parser = _parse.CSVCellParser('F1', use_header=False)
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 9
        assert res[0] == (0, "Widget, Deluxe")
        assert res[1] == (0, "19.99")
        assert res[2] == (0, "High-quality widget with multiple features")
        assert res[3] == (1, "Gadget")
        assert res[4] == (1, "9.99")
        assert res[5] == (1, "Compact gadget, easy to use")
        assert res[6] == (2, "Tool, Multi-purpose")
        assert res[7] == (2, "29.99")
        assert res[8] == (2, "Durable and versatile tool")

    def test_csv_cell_parser(self):
        
        data = csv_data1
        parser = _parse.CSVCellParser('F1', use_header=True)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 9

    def test_csv_cell_parser_without_header(self):
        
        data = csv_data3
        parser = _parse.CSVCellParser('F1', use_header=False)
        msg = Msg('user', meta={'content': data})
        res = parser(msg).m['F1']
        assert len(res) == 9

    def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
        data = csv_data1
        parser = _parse.CSVCellParser(
            'F1', use_header=True
        )
        res = []

        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 9

    def test_csv_delim_parser_returns_correct_len_with_no_header(self):
        
        data = csv_data3
        parser = _parse.CSVCellParser(
            'F1', use_header=False
        )
        res = []
        for cur in asst_stream(data, [parser]):
            cur = cur.m['F1']
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 9


class TestNullParser(object):

    def test_null_parser_returns_same_value(self):
        
        data = "2.0"
        null_parser = _parse.NullParser('F1')
        msg = Msg('assistant', meta={'content': data})
        res = null_parser(msg).m['F1']
        assert res == ["2.0"]

    def test_null_parser_only_returns_last_element_when_continuing(self):
        
        data = "2.0"
        parser = _parse.NullParser('F1')
        res = []
        for cur in asst_stream(data, [parser]):
            res.extend(cur.m['F1'])
            
        assert res == ["2", ".", "0"]
