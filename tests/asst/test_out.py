from ..utils.test_core import SimpleStruct
from dachi.asst import _out as _out
from dachi.msg import model_to_text


from dachi.msg import model_to_text, END_TOK
from dachi.asst import _out as text_proc
from .._structs import SimpleStruct2
from dachi import utils


class TestStructRead:

    def test_out_creates_out_class(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        simple2 = out.__call__([d])
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        simple2 = out.__call__([d])
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2

    def test_read_reads_in_the_class(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        s = model_to_text(SimpleStruct(x='2'))
        simple2 = out([s])
        assert simple2.x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')

        assert out([model_to_text(simple)]).x == '2'


class TestPrimRead(object):

    def test_read_reads_in_data(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )

        result = out.__call__('1')
        assert result == 1

    def test_template_contains_key(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        temp = out.template()
        assert 'int' in temp

    def test_prim_read_reads_bool(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=bool,
        )
        result = out.__call__(['TRUE'])
        assert result is True

    def test_prim_read_reads_bool_correctly(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=bool,
        )
        result = out.__call__(['false'])
        assert result is False


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
            cur = out.delta([t], delta_store)
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
        simple2 = out([d])
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
        cur = out.delta([data], delta_store)
        assert cur['x'] == 'hi'
        assert cur['y'] == 1


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
            cur = out.delta([t], delta_store)
            if cur is not utils.UNDEFINED:
                ress.extend(cur)
        
        assert ress[0] == '1'
        assert ress[1] == '4'
