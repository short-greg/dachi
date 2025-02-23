from dachi.proc import model_to_text
from dachi.adapt import _text_proc as text_proc
from .._structs import SimpleStruct2
from dachi.proc import END_TOK


class TestCSVRead(object):

    def test_out_reads_in_the_class(self):

        csv = 'x,y\n'
        csv += '1,2\n'
        csv += '3,4\n'

        out = text_proc.CSVConv(
            name='F1',
            indexed=False,
        )

        result = out(csv)
        assert result[0]['x'] == 1
        assert result[0]['y'] == 2
        assert len(result) == 2

    def test_out_reads_in_with_delta(self):

        csv = 'x,y\n'
        csv += '1,2\n'
        csv += '3,4\n'

        out = text_proc.CSVConv(
            name='F1',
            indexed=False,
        )
        delta_store = {}
        ress = []
        for t in csv:
            ress.append(out.delta(t, delta_store))
        ress.append(out.delta(END_TOK, delta_store))
        result = [res for res in ress if res is not None]
        print(result)
        assert result[0][0]['x'] == 1
        assert result[0][0]['y'] == 2
        assert len(result) == 2

    def test_template_outputs_a_valid_template(self):

        out = text_proc.CSVConv(
            name='F1',
            indexed=True,
            cols=[('x', 'X value', 'int'), ('y', 'Y value', 'str')]
        )
        temp = out.template()
        assert 'N,' in temp
        assert 'X value <int>' in temp

    def test_template_outputs_a_valid_template_for_not_indexed(self):

        out = text_proc.CSVConv(
            name='F1',
            indexed=False,
            cols=[('x', 'X value', 'int'), ('y', 'Y value', 'str')]
        )

        temp = out.template()
        assert 'N,' not in temp
        assert 'X value <int>' in temp


class TestKVRead(object):

    def test_out_reads_in_the_class(self):

        k = 'x::1\n'
        k += 'y::4\n'

        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )

        result = out(k)
        assert result['x'] == '1'
        assert result['y'] == '4'

    def test_template_contains_key(self):


        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        temp = out.template()
        assert 'x::' in temp
        assert 'y::' in temp

    def test_out_reads_in_with_delta(self):

        k = 'x::1\n'
        k += 'y::4\n'

        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        delta_store = {}
        ress = []
        for t in k:
            ress.append(out.delta(t, delta_store))
        ress.append(out.delta(END_TOK, delta_store))
        result = [res for res in ress if res is not None]
        
        assert result[0]['x'] == '1'
        assert result[0]['y'] == '4'


class TestJSONRead(object):

    def test_out_creates_out_class(self):

        out = text_proc.JSONConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple = SimpleStruct2(x='hi', y=1)
        d = model_to_text(simple)
        simple2 = out(d)
        assert simple.x == simple2['x']

    def test_out_template(self):

        out = text_proc.JSONConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple2 = out.template()
        assert 'x' in simple2

    def test_out_reads_in_with_delta(self):

        k = 'x::1\n'
        k += 'y::4\n'

        simple = SimpleStruct2(x='hi', y=1)
        out = text_proc.JSONConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        delta_store = {}
        ress = []
        for t in model_to_text(simple):
            ress.append(out.delta(t, delta_store))
        ress.append(out.delta(END_TOK, delta_store))
        result = [res for res in ress if res is not None]
        
        assert result[0]['x'] == 'hi'
        assert result[0]['y'] == 1


class TestYAMLRead(object):

    def test_out_creates_out_class(self):

        out = text_proc.YAMLConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple = SimpleStruct2(x='hi', y=1)
        d = model_to_text(simple)
        simple2 = out.__call__(d)
        assert simple.x == simple2['x']

    def test_out_template(self):

        out = text_proc.YAMLConv(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple2 = out.template()
        assert 'x' in simple2


class TestIndexRead(object):

    def test_out_reads_in_the_class(self):

        k = '1::1\n'
        k += '2::4\n'

        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )

        result = out.__call__(k)
        assert result[0] == '1'
        assert result[1] == '4'

    def test_template_contains_key(self):

        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )
        temp = out.template()
        assert '1::' in temp
        assert 'N::' in temp
