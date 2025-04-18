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


def asst_stream(data, parser) -> typing.Iterator:
    """Use to simulate an assitant stream"""
    delta_store = {}
    for i, d in enumerate(data):
        is_last = i == len(data) - 1
        # msg = StreamMsg('assistant', meta={'content': d}, is_last=is_last)
        msg = parser(d, delta_store, True, is_last)
        yield msg


class TestLineParser(object):

    def test_line_parse_parses_when_no_newline_symbols_with_correct_num_lines(self):
        
        line_parser = _parse.LineParser()
        lines = line_parser(data)
        assert len(lines) == 2

    def test_line_parse_parses_when_no_newline_symbols_with_correct_val(self):
        
        line_parser = _parse.LineParser()
        lines = line_parser(data)
        assert lines[1] == "z: e"

    def test_line_parse_parses_when_newline_symbols_with_correct_count(self):
        
        line_parser = _parse.LineParser()
        lines = line_parser(data)
        
        assert len(lines) == 2

    def test_line_parse_parses_when_newline_symbols_with_correct_count(self):
        target = (
"""t: y\
xy"""
        )
        line_parser = _parse.LineParser()
        lines = line_parser(data2)
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

        parser = _parse.LineParser()
        
        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 2

    def test_line_parse_delta_correctly_to_value(self):
        t1 = "x: y"
        t2 = "z: e"
        res = []

        parser = _parse.LineParser()

        for cur in asst_stream(data, parser):
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

        parser = _parse.LineParser()

        for cur in asst_stream(data2, parser):
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

        parser = _parse.LineParser()

        for cur in asst_stream(data2, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == t1
        assert res[1] == t2


class TestCharDelimParser(object):

    def test_char_delim_parser_handles_empty_string(self):
        data = ""
        parser = _parse.CharDelimParser()
        res = parser(data)
        assert res == []

    def test_char_delim_parser_handles_single_character(self):
        data = "a"
        parser = _parse.CharDelimParser(sep=',')
        res = parser(data)
        assert len(res) == 1
        assert res[0] == "a"

    def test_char_delim_parser_handles_no_delimiters(self):
        data = "abc"
        parser = _parse.CharDelimParser(sep=',')
        res = parser(data)
        assert len(res) == 1
        assert res[0] == "abc"

    def test_char_delim_parser_handles_multiple_delimiters(self):
        data = "a,b,c"
        parser = _parse.CharDelimParser(sep=',')
        res = parser(data)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_trailing_delimiter(self):
        data = "a,b,c,"
        parser = _parse.CharDelimParser(sep=',')
        res = parser(data)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_leading_delimiter(self):
        data = ",a,b,c"
        parser = _parse.CharDelimParser(sep=',')
        res = parser(data)
        assert len(res) == 4
        assert res[0] == ""
        assert res[1] == "a"
        assert res[2] == "b"
        assert res[3] == "c"

    def test_char_delim_parser_handles_custom_delimiter(self):
        data = "a|b|c"
        parser = _parse.CharDelimParser(sep='|')
        res = parser(data)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data(self):
        data = "a,b,c"
        parser = _parse.CharDelimParser(sep=',')
        res = []
        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data_with_trailing_delimiter(self):
        data = "a,b,c,"
        parser = _parse.CharDelimParser(sep=',')
        res = []
        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        print(res)
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data_with_partial_chunk(self):
        data = ["a,b", ",c"]
        parser = _parse.CharDelimParser(sep=',')
        res = []
        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_streamed_data_with_custom_delimiter(self):
        data = "a|b|c"
        parser = _parse.CharDelimParser(sep='|')
        res = []
        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)
        assert len(res) == 3
        assert res[0] == "a"
        assert res[1] == "b"
        assert res[2] == "c"

    def test_char_delim_parser_handles_rendering(self):
        data = ["a", "b", "c"]
        parser = _parse.CharDelimParser(sep=',')
        rendered = parser.render(data)
        assert rendered == "a,b,c"

    def test_char_delim_parser_handles_rendering_with_custom_delimiter(self):
        data = ["a", "b", "c"]
        parser = _parse.CharDelimParser(sep='|')
        rendered = parser.render(data)
        assert rendered == "a|b|c"

    def test_char_delim_parser_returns_len_2(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser(sep='.')
        res = parser(data)
        assert len(res) == 2

    def test_char_delim_parser_returns_correct_values(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser(sep='.')
        res = parser(data)
        assert res[0] == "2"
        assert res[1] == "0"

    def test_char_delim_parser_returns_correct_values_with_delta(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser(sep='.')
        res = []

        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == "2"
        assert res[1] == "0"


class TestCSVCellParser(object):

    def test_csv_cell_parser_handles_empty_data(self):
        data = ""
        parser = _parse.CSVCellParser(use_header=True)
        res = parser(data)
        assert res == utils.UNDEFINED

    def test_csv_cell_parser_handles_single_row_with_header(self):
        data = "name,age,city\nJohn,25,New York"
        parser = _parse.CSVCellParser(use_header=True)
        res = parser(data)
        assert len(res) == 3
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")

    def test_csv_cell_parser_handles_single_row_without_header(self):
        data = "John,25,New York"
        parser = _parse.CSVCellParser(use_header=False)
        res = parser(data)
        assert len(res) == 3
        assert res[0] == (0, "John")
        assert res[1] == (0, "25")
        assert res[2] == (0, "New York")

    def test_csv_cell_parser_handles_multiple_rows_with_header(self):
        data = "name,age,city\nJohn,25,New York\nJane,30,Los Angeles"
        parser = _parse.CSVCellParser(use_header=True)
        res = parser(data)
        assert len(res) == 6
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")
        assert res[3] == (1, "name", "Jane")
        assert res[4] == (1, "age", "30")
        assert res[5] == (1, "city", "Los Angeles")

    def test_csv_cell_parser_handles_multiple_rows_without_header(self):
        data = "John,25,New York\nJane,30,Los Angeles"
        parser = _parse.CSVCellParser(use_header=False)
        res = parser(data)
        assert len(res) == 6
        assert res[0] == (0, "John")
        assert res[1] == (0, "25")
        assert res[2] == (0, "New York")
        assert res[3] == (1, "Jane")
        assert res[4] == (1, "30")
        assert res[5] == (1, "Los Angeles")

    def test_csv_cell_parser_handles_different_delimiters(self):
        data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
        parser = _parse.CSVCellParser(delimiter='|', use_header=True)
        res = parser(data)
        assert len(res) == 6
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")
        assert res[3] == (1, "name", "Jane")
        assert res[4] == (1, "age", "30")
        assert res[5] == (1, "city", "Los Angeles")

    def test_csv_cell_parser_handles_streamed_data_with_header(self):
        data = csv_data1
        parser = _parse.CSVCellParser(use_header=True)
        res = []
        for cur in asst_stream(data, parser):
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
        parser = _parse.CSVCellParser(use_header=False)
        res = []
        for cur in asst_stream(data, parser):
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
        parser = _parse.CSVCellParser(use_header=True)
        res = parser(data)
        assert len(res) == 9

    def test_csv_cell_parser_without_header(self):
        
        data = csv_data3
        parser = _parse.CSVCellParser(use_header=False)
        res = parser(data)
        assert len(res) == 9

    def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
        data = csv_data1
        parser = _parse.CSVCellParser(
            use_header=True
        )
        res = []

        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 9

    def test_csv_delim_parser_returns_correct_len_with_no_header(self):
        
        data = csv_data3
        parser = _parse.CSVCellParser(
            use_header=False
        )
        res = []
        for cur in asst_stream(data, parser):
            if cur != utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 9
