from dachi._core import _io as _core
from .test_core import SimpleStruct


class TestListOut(object):

    def test_out_reads_in_the_class(self):

        struct_list = _core.StructList(
            structs=[
                SimpleStruct(x='2'),
                SimpleStruct(x='3')
            ]
        )

        out = _core.StructListRead(
            name='F1',
            out_cls=SimpleStruct
        )

        assert out.read(struct_list.to_text())[0].x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')

        assert out.read(simple.to_text()).x == '2'


class TestStructListRead(object):

    def test_out_reads_in_the_class(self):

        struct_list = _core.StructList(
            structs=[
                SimpleStruct(x='2'),
                SimpleStruct(x='3')
            ]
        )

        out = _core.StructListRead(
            name='F1',
            out_cls=SimpleStruct
        )

        assert out.read(struct_list.to_text())[0].x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')
        assert out.read(simple.to_text()).x == '2'


class TestMultiOut(object):

    def test_out_writes_in_the_class(self):

        struct_list = {'data': [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ], 'i': 2}

        out = _core.MultiRead(
            name='Multi',
            outs=[_core.StructRead(
                name='F1',
                out_cls=SimpleStruct
            ), _core.StructRead(
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

        out = _core.MultiRead(
            name='Multi',
            outs=[_core.StructRead(
                name='F1',
                out_cls=SimpleStruct
            ), _core.StructRead(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.example(struct_list)
        structs = out.read(text)
        assert structs['data'][0].x == struct_list['data'][0].x

    def test_out_stream_read_in_the_class(self):

        struct_list = {'data': [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ], 'i': 2}

        out = _core.MultiRead(
            name='Multi',
            outs=[_core.StructRead(
                name='F1',
                out_cls=SimpleStruct
            ), _core.StructRead(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.example(struct_list)
        structs = out.read(text)
        assert structs['data'][0].x == struct_list['data'][0].x
        # assert failed_on is None


class TestOut:

    def test_out_creates_out_class(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = simple.to_text()
        simple2 = out.read(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = simple.to_text()
        simple2 = out.read(d)
        assert simple.x == simple2.x
    
    def test_out_template(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )
        simple2 = out.template()
        assert 'x' in simple2

    def test_read_reads_in_the_class(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )
        s = SimpleStruct(x='2').to_text()
        simple2 = out.read(s)
        assert simple2.x == '2'
