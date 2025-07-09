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
