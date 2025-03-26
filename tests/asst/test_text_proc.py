from dachi.msg import model_to_text, END_TOK
from dachi.asst import _out as text_proc
from .._structs import SimpleStruct2
from dachi import utils


class TestKVRead(object):

    def test_out_reads_in_the_class(self):

        k = ['x::1']
        k += ['y::4']

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

        k = ['x::1']
        k += ['y::4']

        out = text_proc.KVConv(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        delta_store = {}
        ress = {}
        for i, t in enumerate(k):
            is_last = len(k) - 1 == i
            cur = out.delta([t], delta_store, is_last)
            if cur is not utils.UNDEFINED:
                ress.update(cur)
        # ress.append(out.delta(END_TOK, delta_store))
        # result = [res for res in ress if res is not None]
        
        assert ress['x'] == '1'
        assert ress['y'] == '4'


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
        data = model_to_text(simple)
        for i, t in enumerate(data):
            is_last = i == len(data) - 1
            cur = out.delta(t, delta_store, is_last)
            if cur is not utils.UNDEFINED:
                ress.append(cur)

        assert ress[0]['x'] == 'hi'
        assert ress[0]['y'] == 1


class TestIndexRead(object):

    def test_out_reads_in_the_class(self):

        k = ['1::1']
        k += ['2::4']

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
    
    def test_out_reads_in_the_class_with_delta(self):

        k = ['1::1']
        k += ['2::4']

        out = text_proc.IndexConv(
            name='F1',
            key_descr='the number of people'
        )

        delta_store = {}
        ress = []
        for i, t in enumerate(k):
            is_last = len(k) - 1 == i
            cur = out.delta([t], delta_store, is_last)
            if cur is not utils.UNDEFINED:
                ress.extend(cur)

        assert ress[0] == '1'
        assert ress[1] == '4'
