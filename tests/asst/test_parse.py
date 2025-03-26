from dachi.asst import _parse
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
        lines = line_parser(data2)
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

        line_parser = _parse.LineParser()

        delta_store = {}
        for i, d in enumerate(data):
            # if d =='\n':
            #    print('Newline')
            cur = line_parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 2

    def test_line_parse_delta_correctly_to_value(self):
        t1 = "x: y"
        t2 = "z: e"
        res = []

        line_parser = _parse.LineParser()

        delta_store = {}
        for i, d in enumerate(data):
            # if d =='\n':
            #    print('Newline')
            cur = line_parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
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

        line_parser = _parse.LineParser()

        delta_store = {}
        for i, d in enumerate(data2):
            cur = line_parser.delta(
                d, delta_store, i == (len(data2) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 2

    def test_line_parse_delta_correctly_to_value_with_continuation(self):
        t1 = (
"""t: y\
xy"""
        )
        t2 = "z: e"
        res = []

        line_parser = _parse.LineParser()

        delta_store = {}
        for i, d in enumerate(data2):
            cur = line_parser.delta(
                d, delta_store, i == (len(data2) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert res[0] == t1
        assert res[1] == t2


class TestFullParser(object):

    def test_full_parser_only_returns_last_element(self):
        
        data = "2.0"
        full_parser = _parse.FullParser()
        res = full_parser(data)
        assert res == "2.0"

    def test_full_parser_only_returns_last_element_when_continuing(self):
        
        data = "2.0"
        full_parser = _parse.FullParser()

        res = []
        delta_store = {}
        for i, d in enumerate(data):
            cur = full_parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        res = full_parser(data)
        assert res == "2.0"


class TestFullParser(object):

    def test_full_parser_only_returns_last_element(self):
        
        data = "2.0"
        full_parser = _parse.FullParser()
        res = full_parser(data)
        assert res == "2.0"

    def test_full_parser_only_returns_last_element_when_continuing(self):
        
        data = "2.0"
        full_parser = _parse.FullParser()

        res = []
        delta_store = {}
        for i, d in enumerate(data):
            cur = full_parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.append(cur)

        assert res[0] == "2.0"


class TestNullParser(object):

    def test_null_parser_returns_same_value(self):
        
        data = "2.0"
        null_parser = _parse.NullParser()
        res = null_parser(data)
        assert res == "2.0"

    def test_null_parser_only_returns_last_element_when_continuing(self):
        
        data = "2.0"
        null_parser = _parse.NullParser()

        res = []
        delta_store = {}
        for i, d in enumerate(data):
            cur = null_parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.append(cur)

        assert res == ["2", ".", "0"]

        
class TestCharDelimParser(object):

    def test_char_delim_parser_returns_len_2(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser('.')
        res = parser(data)
        assert len(res) == 2

    def test_char_delim_parser_returns_correct_values(self):
        
        data = "2.0"
        parser = _parse.CharDelimParser('.')
        res = parser(data)
        assert res[0] == "2"
        assert res[1] == "0"

    def test_char_delim_parser_returns_correct_values_with_delta(self):
        
        data = "2.0"
        delta_store = {}
        parser = _parse.CharDelimParser('.')
        res = []
        for i, d in enumerate(data):
            cur = parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
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


class TestCSVParser(object):

    def test_csv_delim_parser_returns_correct_len_with_header(self):
        
        data = csv_data1
        parser = _parse.CSVRowParser(use_header=True)
        res = parser(data)
        assert len(res) == 3

    def test_char_delim_parser_returns_csv_with_no_header(self):
        
        data = csv_data3
        parser = _parse.CSVRowParser(use_header=False)
        res = parser(data)
        assert len(res) == 3

    def test_csv_delim_parser_returns_correct_len_with_header(self):
        
        data = csv_data1
        parser = _parse.CSVRowParser(
            use_header=True
        )
        delta_store = {}
        res = []
        for i, d in enumerate(data):
            cur = parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 3

    def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
        data = csv_data2
        parser = _parse.CSVRowParser(
            use_header=True
        )
        delta_store = {}
        res = []
        for i, d in enumerate(data):
            cur = parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 3


class TestCSVCellParser(object):

    def test_csv_cell_parser(self):
        
        data = csv_data1
        parser = _parse.CSVCellParser(use_header=True)
        res = parser(data)
        print(res[0:4])
        assert len(res) == 9

    def test_csv_cell_parser_without_header(self):
        
        data = csv_data3
        parser = _parse.CSVCellParser(use_header=False)
        res = parser(data)
        print(res[0:4])
        assert len(res) == 9

    def test_csv_delim_parser_returns_correct_len_with_newline(self):
        
        data = csv_data1
        parser = _parse.CSVCellParser(
            use_header=True
        )
        delta_store = {}
        res = []
        for i, d in enumerate(data):
            cur = parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 9

    def test_csv_delim_parser_returns_correct_len_with_no_header(self):
        
        data = csv_data3
        parser = _parse.CSVCellParser(
            use_header=False
        )
        delta_store = {}
        res = []
        for i, d in enumerate(data):
            cur = parser.delta(
                d, delta_store, i == (len(data) - 1)
            )
            if cur is not utils.UNDEFINED:
                res.extend(cur)

        assert len(res) == 9

