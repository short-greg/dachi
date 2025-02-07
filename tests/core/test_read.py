from .test_core import SimpleStruct, SimpleStruct2
from dachi import _core
from dachi.utils import model_to_text


class TestMultiRead(object):

    def test_out_writes_in_the_class(self):

        struct_list = {'data': [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ], 'i': 2}

        out = _core.MultiTextProc(
            name='Multi',
            outs=[_core.PydanticProc(
                name='F1',
                out_cls=SimpleStruct
            ), _core.PydanticProc(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.example(struct_list)
        assert 'x' in text
        assert 'F2' in text

    def test_out_reads_in_the_class(self):

        struct_list = {'data': [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ], 'i': 2}

        out = _core.MultiTextProc(
            name='Multi',
            outs=[_core.PydanticProc(
                name='F1',
                out_cls=SimpleStruct
            ), _core.PydanticProc(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.example(struct_list)
        print('Before')
        print(text)
        print('After')
        structs = out(text)
        assert structs['data'][0].x == struct_list['data'][0].x

    def test_out_stream_read_in_the_class(self):

        struct_list = {'data': [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ], 'i': 2}

        out = _core.MultiTextProc(
            name='Multi',
            outs=[_core.PydanticProc(
                name='F1',
                out_cls=SimpleStruct
            ), _core.PydanticProc(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.example(struct_list)
        structs = out.__call__(text)
        assert structs['data'][0].x == struct_list['data'][0].x
        # assert failed_on is None


class TestStructRead:

    def test_out_creates_out_class(self):

        out = _core.PydanticProc(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        simple2 = out.__call__(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _core.PydanticProc(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        simple2 = out.__call__(d)
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _core.PydanticProc(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2

    def test_read_reads_in_the_class(self):

        out = _core.PydanticProc(
            name='F1',
            out_cls=SimpleStruct
        )
        s = model_to_text(SimpleStruct(x='2'))
        simple2 = out(s)
        assert simple2.x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.PydanticProc(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')

        assert out(model_to_text(simple)).x == '2'


class TestPrimRead(object):

    def test_read_reads_in_data(self):

        out = _core.PrimProc(
            name='F1',
            out_cls=int,
        )

        result = out.__call__('1')
        assert result == 1

    def test_template_contains_key(self):

        out = _core.PrimProc(
            name='F1',
            out_cls=int,
        )
        temp = out.template()
        assert 'int' in temp

    def test_prim_read_reads_bool(self):

        out = _core.PrimProc(
            name='F1',
            out_cls=bool,
        )
        result = out.__call__('TRUE')
        assert result is True

    def test_prim_read_reads_bool_correctly(self):

        out = _core.PrimProc(
            name='F1',
            out_cls=bool,
        )
        result = out.__call__('false')
        assert result is False
