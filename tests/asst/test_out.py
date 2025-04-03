from ..utils.test_core import SimpleStruct
from dachi.asst import _out as _out
from dachi.msg import model_to_text


from dachi.msg import model_to_text, END_TOK, Msg, StreamMsg
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
        msg = Msg(
            role='user',
            meta={
                'data': d,
            }
        )
        simple2 = out.forward(msg)['meta']['F1']
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        msg1 = StreamMsg(
            role='user',
            meta={
                'data': d[:4],
            }, is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': d[4:],
            }, is_last=True
        )
        delta = {}
        out(msg1, delta_store=delta)
        simple2 = out(msg2, delta_store=delta)['meta']['F1']
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _out.PydanticConv(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2


class TestPrimRead(object):

    def test_read_reads_in_data(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=int,
        )
        msg1 = Msg(
            role='user',
            meta={
                'data': '1'
            }
        )

        result = out(msg1)['meta']['F1']
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

        msg1 = Msg(
            role='user',
            meta={
                'data': 'TRUE'
            }
        )
        result = out.__call__(msg1).m['F1']
        assert result is True

    def test_prim_read_reads_bool_with_stream(self):

        out = _out.PrimConv(
            name='F1',
            out_cls=bool,
        )

        msg1 = StreamMsg(
            role='user',
            meta={
                'data': 'TR'
            }, is_last=False
        )
        msg2 = StreamMsg(
            role='user',
            meta={
                'data': 'UE'
            }, is_last=True
        )
        store = {}
        out.forward(msg1, delta_store=store)
        result = out.forward(msg2, delta_store=store).m['F1']
        assert result is True


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

        msg1 = Msg(
            role='user',
            meta={
                'data': k
            }
        )
        result = out(msg1).m['F1']
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
            msg1 = StreamMsg(
                role='user',
                meta={
                    'data': [t]
                }, is_streamed=True,
                is_last=i == len(k)-1
            )
            res = out(msg1, delta_store).m['F1']
            if res is not utils.UNDEFINED:
                ress.update(res)

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
        msg1 = Msg(
            role='user',
            meta={
                'data': d
            }
        )
        simple2 = out(msg1).m['F1']
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
        msg1 = Msg(
            role='user',
            meta={
                'data': data
            }
        )
        cur = out(msg1, delta_store).m['F1']
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

        msg1 = Msg(
            role='user',
            meta={
                'data': k
            }
        )
        result = out(msg1).m['F1']
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

            msg1 = StreamMsg(
                role='user',
                meta={
                    'data': [t]
                },
                is_last=i == len(k) - 1
            )
            cur = out(msg1, delta_store).m['F1']
            if cur is not utils.UNDEFINED:
                ress.extend(cur)
        
        assert ress[0] == '1'
        assert ress[1] == '4'
