import typing as t
import pytest
import csv
from collections import OrderedDict

from dachi.proc import _parse as _msg_parse   # CSVRowParser lives here
from dachi.utils import UNDEFINED            # sentinel used by all parsers


# --------------------------------------------------------------------------- #
# Test data fixtures (mirrors the original CSVOut test strings)
# --------------------------------------------------------------------------- #
csv_data1 = (
    "name,age,city\n"
    "John,25,New York\n"
    "Jane,30,Los Angeles\n"
    "\"Smith, Jr.\",40,Chicago"
)

csv_data2 = (
    "name,age,city\n"
    "Alpha,10,Boston\n"
    "Beta,20,\"Los\nAngeles\"\n"
    "Gamma,30,Paris"
)

csv_data3 = (
    "\"Widget, Deluxe\",19.99,\"High-quality widget with multiple features\"\n"
    "Gadget,9.99,\"Compact gadget, easy to use\"\n"
    "\"Tool, Multi-purpose\",29.99,\"Durable and versatile tool\""
)


# --------------------------------------------------------------------------- #
# Helper: stream data character-by-character through the parser.
# --------------------------------------------------------------------------- #
def _stream_parser(data: str, parser: _msg_parse.CSVRowParser, *, use_header: bool):
    """Yield parser outputs while feeding data char-wise."""
    delta: dict = {}
    results: list = []
    for i, ch in enumerate(data):
        out = parser.forward(
            ch,
            delta_store=delta,
            streamed=True,
            is_last=(i == len(data) - 1),
        )
        if out is not UNDEFINED:
            results.extend(out)
    return results


class TestCSVRowParser:
    """Unit tests for :class:`dachi.msg._parse.CSVRowParser`."""

    def test_csv_parser_handles_empty_data(self):
        parser = _msg_parse.CSVRowParser(use_header=True)
        res = parser.forward("")
        assert res is UNDEFINED

    def test_csv_parser_single_row_no_header(self):
        parser = _msg_parse.CSVRowParser(use_header=False)
        res = parser.forward("John,25,New York")
        assert res == [["John", "25", "New York"]]

    def test_csv_parser_single_row_with_header(self):
        parser = _msg_parse.CSVRowParser(use_header=True)
        data = "name,age,city\nJohn,25,New York"
        res = parser.forward(data)
        expect = OrderedDict(zip(["name", "age", "city"], ["John", "25", "New York"]))
        assert res == [expect]

    def test_csv_parser_multiple_rows_with_header(self):
        parser = _msg_parse.CSVRowParser(use_header=True)
        res = parser.forward(csv_data1)
        assert len(res) == 3
        assert res[2]["name"] == "Smith, Jr." and res[2]["city"] == "Chicago"

    def test_csv_parser_multiple_rows_no_header(self):
        parser = _msg_parse.CSVRowParser(use_header=False)
        res = parser.forward(csv_data3)
        assert len(res) == 3
        assert res[0][0] == "Widget, Deluxe" and res[2][2] == "Durable and versatile tool"

    def test_csv_parser_streamed_with_header(self):
        parser = _msg_parse.CSVRowParser(use_header=True)
        res = _stream_parser(csv_data1, parser, use_header=True)
        assert len(res) == 3
        assert res[1]["age"] == "30"

    def test_csv_parser_streamed_no_header(self):
        parser = _msg_parse.CSVRowParser(use_header=False)
        res = _stream_parser(csv_data3, parser, use_header=False)
        assert len(res) == 3
        assert res[1][0] == "Gadget"

    def test_csv_parser_different_delimiter(self):
        data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
        parser = _msg_parse.CSVRowParser(delimiter="|", use_header=True)
        res = parser.forward(data)
        assert len(res) == 2 and res[0]["city"] == "New York"

    def test_csv_parser_row_count_header(self):
        parser = _msg_parse.CSVRowParser(use_header=True)
        res = parser.forward(csv_data1)
        assert len(res) == 3

    def test_csv_parser_row_count_no_header(self):
        parser = _msg_parse.CSVRowParser(use_header=False)
        res = parser.forward(csv_data3)
        assert len(res) == 3

    def test_csv_parser_handles_embedded_newline(self):
        parser = _msg_parse.CSVRowParser(use_header=True)
        res = parser.forward(csv_data2)
        # The embedded newline should not split "Los\nAngeles"
        assert res[1]["city"] == "Los\nAngeles"


class TestCSVCellParser:
    """Refactored tests for :class:`dachi.msg._parse.CSVCellParser`.

    The parser breaks a CSV document down to **cells**:
      • With *use_header=True* each cell is a tuple *(row_idx, column_name, value)*  
      • With *use_header=False* each cell is *(row_idx, value)*
    """

    def test_csv_cell_parser_handles_empty_data(self):
        parser = _msg_parse.CSVCellParser(use_header=True)
        assert parser.forward("") is UNDEFINED

    def test_csv_cell_parser_single_row_with_header(self):
        data = "name,age,city\nJohn,25,New York"
        parser = _msg_parse.CSVCellParser(use_header=True)
        res = parser.forward(data)
        assert len(res) == 3
        assert res[0] == (0, "name", "John")
        assert res[1] == (0, "age", "25")
        assert res[2] == (0, "city", "New York")

    def test_csv_cell_parser_single_row_without_header(self):
        data = "John,25,New York"
        parser = _msg_parse.CSVCellParser(use_header=False)
        res = parser.forward(data)
        assert len(res) == 3
        assert res[0] == (0, "John")
        assert res[1] == (0, "25")
        assert res[2] == (0, "New York")

    def test_csv_cell_parser_multiple_rows_with_header(self):
        data = "name,age,city\nJohn,25,New York\nJane,30,Los Angeles"
        parser = _msg_parse.CSVCellParser(use_header=True)
        res = parser.forward(data)
        assert len(res) == 6
        assert res[3] == (1, "name", "Jane")
        assert res[4] == (1, "age", "30")
        assert res[5] == (1, "city", "Los Angeles")

    def test_csv_cell_parser_multiple_rows_without_header(self):
        data = "John,25,New York\nJane,30,Los Angeles"
        parser = _msg_parse.CSVCellParser(use_header=False)
        res = parser.forward(data)
        assert len(res) == 6
        assert res[3] == (1, "Jane")
        assert res[4] == (1, "30")
        assert res[5] == (1, "Los Angeles")

    def test_csv_cell_parser_different_delimiters(self):
        data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
        parser = _msg_parse.CSVCellParser(delimiter="|", use_header=True)
        res = parser.forward(data)
        assert len(res) == 6
        assert res[0] == (0, "name", "John")
        assert res[5] == (1, "city", "Los Angeles")

    def test_csv_cell_parser_streamed_data_with_header(self):
        parser = _msg_parse.CSVCellParser(use_header=True)
        res = _stream_parser(csv_data1, parser, use_header=True)
        assert len(res) == 9
        assert res[6] == (2, "name", "Smith, Jr.")
        assert res[8] == (2, "city", "Chicago")

    def test_csv_cell_parser_streamed_data_without_header(self):
        parser = _msg_parse.CSVCellParser(use_header=False)
        res = _stream_parser(csv_data3, parser, use_header=False)
        assert len(res) == 9
        assert res[0] == (0, "Widget, Deluxe")
        assert res[8] == (2, "Durable and versatile tool")



class TestCharDelimParser:
    """Refactored tests for :class:`dachi.msg._parse.CharDelimParser`.

    Splits a string on a *single-character* delimiter and supports
    incremental streaming plus round-trip rendering.
    """

    def test_char_delim_parser_handles_empty_string(self):
        parser = _msg_parse.CharDelimParser()
        assert parser.forward("") == []

    def test_char_delim_parser_single_character(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = parser.forward("a")
        assert res == ["a"]

    def test_char_delim_parser_no_delimiters(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = parser.forward("abc")
        assert res == ["abc"]

    def test_char_delim_parser_multiple_delimiters(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = parser.forward("a,b,c")
        assert res == ["a", "b", "c"]

    def test_char_delim_parser_trailing_delimiter(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = parser.forward("a,b,c,")
        assert res == ["a", "b", "c"]  # trailing empty token ignored

    def test_char_delim_parser_leading_delimiter(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = parser.forward(",a,b,c")
        assert res == ["", "a", "b", "c"]

    def test_char_delim_parser_custom_delimiter(self):
        parser = _msg_parse.CharDelimParser(sep="|")
        res = parser.forward("a|b|c")
        assert res == ["a", "b", "c"]

    def test_char_delim_parser_streamed_data(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = _stream_parser("a,b,c", parser, use_header=False)
        assert res == ["a", "b", "c"]

    def test_char_delim_parser_streamed_with_trailing_delim(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        res = _stream_parser("a,b,c,", parser, use_header=False)
        assert res == ["a", "b", "c"]

    def test_char_delim_parser_streamed_partial_chunks(self):
        parts = ["a,b", ",c"]
        parser = _msg_parse.CharDelimParser(sep=",")
        collected = []
        delta = {}
        for i, chunk in enumerate(parts):
            out = parser.forward(
                chunk,
                delta_store=delta,
                streamed=True,
                is_last=(i == len(parts) - 1),
            )
            if out is not UNDEFINED:
                collected.extend(out)
        assert collected == ["a", "b", "c"]

    def test_char_delim_parser_streamed_custom_delimiter(self):
        parser = _msg_parse.CharDelimParser(sep="|")
        res = _stream_parser("a|b|c", parser, use_header=False)
        assert res == ["a", "b", "c"]

    def test_char_delim_parser_render_default(self):
        parser = _msg_parse.CharDelimParser(sep=",")
        assert parser.render(["a", "b", "c"]) == "a,b,c"

    def test_char_delim_parser_render_custom_delimiter(self):
        parser = _msg_parse.CharDelimParser(sep="|")
        assert parser.render(["a", "b", "c"]) == "a|b|c"

    def test_char_delim_parser_dot_delimiter_len(self):
        parser = _msg_parse.CharDelimParser(sep=".")
        res = parser.forward("2.0")
        assert len(res) == 2

    def test_char_delim_parser_dot_delimiter_values(self):
        parser = _msg_parse.CharDelimParser(sep=".")
        res = parser.forward("2.0")
        assert res == ["2", "0"]

    def test_char_delim_parser_dot_delimiter_streaming(self):
        parser = _msg_parse.CharDelimParser(sep=".")
        res = _stream_parser("2.0", parser, use_header=False)
        assert res == ["2", "0"]



class TestLineParser:
    """Refactored tests for :class:`dachi.msg._parse.LineParser`.

    Splits text into logical *lines*:

    • A back-slash (``\``) at the physical end-of-line means the content
      continues on the next physical line.

    • Streaming is fully incremental: parser returns new logical lines
      whenever they are completed; otherwise it yields ``UNDEFINED``.
    """

    data  = "x: y\nz: e"
    data2 = "t: y\\\nxy\nz: e"          # first logical line = "t: yxy"

    def test_line_parser_two_lines(self):
        parser = _msg_parse.LineParser()
        lines = parser.forward(self.data)
        assert len(lines) == 2

    def test_line_parser_correct_second_line(self):
        parser = _msg_parse.LineParser()
        lines = parser.forward(self.data)
        assert lines[1] == "z: e"

    def test_line_parser_continuation_merging(self):
        parser = _msg_parse.LineParser()
        lines = parser.forward(self.data2)
        assert len(lines) == 2
        assert lines[0] == "t: yxy"
        assert lines[1] == "z: e"

    def test_line_parser_streaming_simple(self):
        parser = _msg_parse.LineParser()
        lines = _stream_parser(self.data, parser, use_header=False)
        assert lines == ["x: y", "z: e"]

    def test_line_parser_streaming_with_continuation(self):
        parser = _msg_parse.LineParser()
        lines = _stream_parser(self.data2, parser, use_header=False)
        print(lines)
        assert len(lines) == 2
        assert lines[0] == "t: yxy"
        assert lines[1] == "z: e"



# from dachi.msg import _parse
# import typing
# from dachi import utils

# data = (
# """x: y
# z: e"""
# )

# data2 = (
# """t: y\
# xy
# z: e"""
# )

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


# def asst_stream(data, parser) -> typing.Iterator:
#     """Use to simulate an assitant stream"""
#     delta_store = {}
#     for i, d in enumerate(data):
#         is_last = i == len(data) - 1
#         # msg = StreamMsg('assistant', meta={'content': d}, is_last=is_last)
#         msg = parser(d, delta_store, True, is_last)
#         yield msg


# class TestLineParser(object):

#     def test_line_parse_parses_when_no_newline_symbols_with_correct_num_lines(self):
        
#         line_parser = _parse.LineParser()
#         lines = line_parser(data)
#         assert len(lines) == 2

#     def test_line_parse_parses_when_no_newline_symbols_with_correct_val(self):
        
#         line_parser = _parse.LineParser()
#         lines = line_parser(data)
#         assert lines[1] == "z: e"

#     def test_line_parse_parses_when_newline_symbols_with_correct_count(self):
        
#         line_parser = _parse.LineParser()
#         lines = line_parser(data)
        
#         assert len(lines) == 2

#     def test_line_parse_parses_when_newline_symbols_with_correct_count(self):
#         target = (
# """t: y\
# xy"""
#         )
#         line_parser = _parse.LineParser()
#         lines = line_parser(data2)
#         assert lines[0] == target

#     def test_line_parse_delta_correctly(self):
#         target = (
# """t: y\
# xy"""
#         )
#         res = []
#         data = (
#             """x: y
#             z: e"""
#         )

#         parser = _parse.LineParser()
        
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert len(res) == 2

#     def test_line_parse_delta_correctly_to_value(self):
#         t1 = "x: y"
#         t2 = "z: e"
#         res = []

#         parser = _parse.LineParser()

#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert res[0] == t1
#         assert res[1] == t2

#     def test_line_parse_delta_correctly_length_with_continuation(self):
#         t1 = (
# """t: y\
# xy"""
#         )
#         t2 = "z: e"
#         res = []

#         parser = _parse.LineParser()

#         for cur in asst_stream(data2, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert len(res) == 2

#     def test_line_parse_delta_correctly_to_value_with_continuation2(self):
#         t1 = (
# """t: y\
# xy"""
#         )
#         t2 = "z: e"
#         res = []

#         parser = _parse.LineParser()

#         for cur in asst_stream(data2, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert res[0] == t1
#         assert res[1] == t2


# class TestCharDelimParser(object):

#     def test_char_delim_parser_handles_empty_string(self):
#         data = ""
#         parser = _parse.CharDelimParser()
#         res = parser(data)
#         assert res == []

#     def test_char_delim_parser_handles_single_character(self):
#         data = "a"
#         parser = _parse.CharDelimParser(sep=',')
#         res = parser(data)
#         assert len(res) == 1
#         assert res[0] == "a"

#     def test_char_delim_parser_handles_no_delimiters(self):
#         data = "abc"
#         parser = _parse.CharDelimParser(sep=',')
#         res = parser(data)
#         assert len(res) == 1
#         assert res[0] == "abc"

#     def test_char_delim_parser_handles_multiple_delimiters(self):
#         data = "a,b,c"
#         parser = _parse.CharDelimParser(sep=',')
#         res = parser(data)
#         assert len(res) == 3
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_trailing_delimiter(self):
#         data = "a,b,c,"
#         parser = _parse.CharDelimParser(sep=',')
#         res = parser(data)
#         assert len(res) == 3
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_leading_delimiter(self):
#         data = ",a,b,c"
#         parser = _parse.CharDelimParser(sep=',')
#         res = parser(data)
#         assert len(res) == 4
#         assert res[0] == ""
#         assert res[1] == "a"
#         assert res[2] == "b"
#         assert res[3] == "c"

#     def test_char_delim_parser_handles_custom_delimiter(self):
#         data = "a|b|c"
#         parser = _parse.CharDelimParser(sep='|')
#         res = parser(data)
#         assert len(res) == 3
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_streamed_data(self):
#         data = "a,b,c"
#         parser = _parse.CharDelimParser(sep=',')
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 3
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_streamed_data_with_trailing_delimiter(self):
#         data = "a,b,c,"
#         parser = _parse.CharDelimParser(sep=',')
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 3
#         print(res)
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_streamed_data_with_partial_chunk(self):
#         data = ["a,b", ",c"]
#         parser = _parse.CharDelimParser(sep=',')
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 3
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_streamed_data_with_custom_delimiter(self):
#         data = "a|b|c"
#         parser = _parse.CharDelimParser(sep='|')
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 3
#         assert res[0] == "a"
#         assert res[1] == "b"
#         assert res[2] == "c"

#     def test_char_delim_parser_handles_rendering(self):
#         data = ["a", "b", "c"]
#         parser = _parse.CharDelimParser(sep=',')
#         rendered = parser.render(data)
#         assert rendered == "a,b,c"

#     def test_char_delim_parser_handles_rendering_with_custom_delimiter(self):
#         data = ["a", "b", "c"]
#         parser = _parse.CharDelimParser(sep='|')
#         rendered = parser.render(data)
#         assert rendered == "a|b|c"

#     def test_char_delim_parser_returns_len_2(self):
        
#         data = "2.0"
#         parser = _parse.CharDelimParser(sep='.')
#         res = parser(data)
#         assert len(res) == 2

#     def test_char_delim_parser_returns_correct_values(self):
        
#         data = "2.0"
#         parser = _parse.CharDelimParser(sep='.')
#         res = parser(data)
#         assert res[0] == "2"
#         assert res[1] == "0"

#     def test_char_delim_parser_returns_correct_values_with_delta(self):
        
#         data = "2.0"
#         parser = _parse.CharDelimParser(sep='.')
#         res = []

#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert res[0] == "2"
#         assert res[1] == "0"


# class TestCSVCellParser(object):

#     def test_csv_cell_parser_handles_empty_data(self):
#         data = ""
#         parser = _parse.CSVCellParser(use_header=True)
#         res = parser(data)
#         assert res == utils.UNDEFINED

#     def test_csv_cell_parser_handles_single_row_with_header(self):
#         data = "name,age,city\nJohn,25,New York"
#         parser = _parse.CSVCellParser(use_header=True)
#         res = parser(data)
#         assert len(res) == 3
#         assert res[0] == (0, "name", "John")
#         assert res[1] == (0, "age", "25")
#         assert res[2] == (0, "city", "New York")

#     def test_csv_cell_parser_handles_single_row_without_header(self):
#         data = "John,25,New York"
#         parser = _parse.CSVCellParser(use_header=False)
#         res = parser(data)
#         assert len(res) == 3
#         assert res[0] == (0, "John")
#         assert res[1] == (0, "25")
#         assert res[2] == (0, "New York")

#     def test_csv_cell_parser_handles_multiple_rows_with_header(self):
#         data = "name,age,city\nJohn,25,New York\nJane,30,Los Angeles"
#         parser = _parse.CSVCellParser(use_header=True)
#         res = parser(data)
#         assert len(res) == 6
#         assert res[0] == (0, "name", "John")
#         assert res[1] == (0, "age", "25")
#         assert res[2] == (0, "city", "New York")
#         assert res[3] == (1, "name", "Jane")
#         assert res[4] == (1, "age", "30")
#         assert res[5] == (1, "city", "Los Angeles")

#     def test_csv_cell_parser_handles_multiple_rows_without_header(self):
#         data = "John,25,New York\nJane,30,Los Angeles"
#         parser = _parse.CSVCellParser(use_header=False)
#         res = parser(data)
#         assert len(res) == 6
#         assert res[0] == (0, "John")
#         assert res[1] == (0, "25")
#         assert res[2] == (0, "New York")
#         assert res[3] == (1, "Jane")
#         assert res[4] == (1, "30")
#         assert res[5] == (1, "Los Angeles")

#     def test_csv_cell_parser_handles_different_delimiters(self):
#         data = "name|age|city\nJohn|25|New York\nJane|30|Los Angeles"
#         parser = _parse.CSVCellParser(delimiter='|', use_header=True)
#         res = parser(data)
#         assert len(res) == 6
#         assert res[0] == (0, "name", "John")
#         assert res[1] == (0, "age", "25")
#         assert res[2] == (0, "city", "New York")
#         assert res[3] == (1, "name", "Jane")
#         assert res[4] == (1, "age", "30")
#         assert res[5] == (1, "city", "Los Angeles")

#     def test_csv_cell_parser_handles_streamed_data_with_header(self):
#         data = csv_data1
#         parser = _parse.CSVCellParser(use_header=True)
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 9
#         assert res[0] == (0, "name", "John")
#         assert res[1] == (0, "age", "25")
#         assert res[2] == (0, "city", "New York")
#         assert res[3] == (1, "name", "Jane")
#         assert res[4] == (1, "age", "30")
#         assert res[5] == (1, "city", "Los Angeles")
#         assert res[6] == (2, "name", "Smith, Jr.")
#         assert res[7] == (2, "age", "40")
#         assert res[8] == (2, "city", "Chicago")

#     def test_csv_cell_parser_handles_streamed_data_without_header(self):
#         data = csv_data3
#         parser = _parse.CSVCellParser(use_header=False)
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)
#         assert len(res) == 9
#         assert res[0] == (0, "Widget, Deluxe")
#         assert res[1] == (0, "19.99")
#         assert res[2] == (0, "High-quality widget with multiple features")
#         assert res[3] == (1, "Gadget")
#         assert res[4] == (1, "9.99")
#         assert res[5] == (1, "Compact gadget, easy to use")
#         assert res[6] == (2, "Tool, Multi-purpose")
#         assert res[7] == (2, "29.99")
#         assert res[8] == (2, "Durable and versatile tool")

#     def test_csv_cell_parser(self):
        
#         data = csv_data1
#         parser = _parse.CSVCellParser(use_header=True)
#         res = parser(data)
#         assert len(res) == 9

#     def test_csv_cell_parser_without_header(self):
        
#         data = csv_data3
#         parser = _parse.CSVCellParser(use_header=False)
#         res = parser(data)
#         assert len(res) == 9

#     def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
#         data = csv_data1
#         parser = _parse.CSVCellParser(
#             use_header=True
#         )
#         res = []

#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert len(res) == 9

#     def test_csv_delim_parser_returns_correct_len_with_no_header(self):
        
#         data = csv_data3
#         parser = _parse.CSVCellParser(
#             use_header=False
#         )
#         res = []
#         for cur in asst_stream(data, parser):
#             if cur != utils.UNDEFINED:
#                 res.extend(cur)

#         assert len(res) == 9
