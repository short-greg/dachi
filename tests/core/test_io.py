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
        assert out.stream_read(simple.to_text()).x == '2'


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
        assert out.stream_read(simple.to_text()).x == '2'


class TestMultiOut(object):

    def test_out_writes_in_the_class(self):

        struct_list = [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ]

        out = _core.MultiRead(
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

        struct_list = [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ]

        out = _core.MultiRead(
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
        assert structs[0].x == struct_list[0].x

    def test_out_stream_read_in_the_class(self):

        struct_list = [
            SimpleStruct(x='2'),
            SimpleStruct(x='3')
        ]

        out = _core.MultiRead(
            outs=[_core.StructRead(
                name='F1',
                out_cls=SimpleStruct
            ), _core.StructRead(
                name='F2',
                out_cls=SimpleStruct
            )]
        )

        text = out.example(struct_list)
        structs, failed_on = out.stream_read(text)
        assert structs[0].x == struct_list[0].x
        assert failed_on is None
