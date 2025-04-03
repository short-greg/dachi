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
        lines = line_parser(msg)
        assert len(lines) == 2

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


class TestCSVParser(object):

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



# class TestNullParser(object):

#     def test_null_parser_returns_same_value(self):
        
#         data = "2.0"
#         null_parser = _parse.NullParser()
#         res = null_parser(data)
#         assert res == ["2.0"]

#     def test_null_parser_only_returns_last_element_when_continuing(self):
        
#         data = "2.0"
#         parser = _parse.NullParser()

#         res = []
#         for cur in parser.stream(asst_stream(data)):
#             res.extend(cur)
            
#         assert res == ["2", ".", "0"]
