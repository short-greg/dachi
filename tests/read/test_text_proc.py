from dachi.utils import model_to_text
from dachi._core import _read as _core
from .._structs import SimpleStruct, SimpleStruct2


# class TestStructListRead(object):

#     def test_out_reads_in_the_class(self):

#         struct_list = _core.DataList(
#             data=[
#                 SimpleStruct(x='2'),
#                 SimpleStruct(x='3')
#             ]
#         )

#         out = _core.StructListRead(
#             name='F1',
#             out_cls=SimpleStruct
#         )

#         assert out.read(model_to_text(struct_list))[0].x == '2'


#     def test_out_reads_in_the_class(self):

#         struct_list = _core.DataList(
#             data=[
#                 SimpleStruct(x='2'),
#                 SimpleStruct(x='3')
#             ]
#         )

#         out = _core.StructListRead(
#             name='F1',
#             out_cls=SimpleStruct
#         )

#         assert out.read(model_to_text(struct_list))[0].x == '2'

#     # def test_out_reads_in_the_class_with_str(self):

#     #     out = _core.StructRead(
#     #         name='F1',
#     #         out_cls=SimpleStruct
#     #     )

#     #     simple = SimpleStruct(x='2')
#     #     assert out.read(model_to_text(simple)).x == '2'


class TestCSVRead(object):

    def test_out_reads_in_the_class(self):

        csv = 'x,y\n'
        csv += '1,2\n'
        csv += '3,4\n'

        out = _core.CSVProc(
            name='F1',
            indexed=False,
        )

        result = out.__call__(csv)
        print(result[0])
        assert result[0]['x'] == 1
        assert result[0]['y'] == 2
        assert len(result) == 2

    def test_template_outputs_a_valid_template(self):

        out = _core.CSVProc(
            name='F1',
            indexed=True,
            cols=[('x', 'X value', 'int'), ('y', 'Y value', 'str')]
        )

        temp = out.template()
        assert 'N,' in temp
        assert 'X value <int>' in temp

    def test_template_outputs_a_valid_template_for_not_indexed(self):

        out = _core.CSVProc(
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

        out = _core.KVProc(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )

        result = out.__call__(k)
        assert result['x'] == '1'
        assert result['y'] == '4'

    def test_template_contains_key(self):


        out = _core.KVProc(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        temp = out.template()
        assert 'x::' in temp
        assert 'y::' in temp



class TestJSONRead(object):

    def test_out_creates_out_class(self):

        out = _core.JSONProc(
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

        out = _core.JSONProc(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple2 = out.template()
        assert 'x' in simple2


class TestYAMLRead(object):

    def test_out_creates_out_class(self):

        out = _core.YAMLRead(
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

        out = _core.YAMLRead(
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

        out = _core.IndexProc(
            name='F1',
            key_descr='the number of people'
        )

        result = out.__call__(k)
        assert result[0] == '1'
        assert result[1] == '4'

    def test_template_contains_key(self):

        out = _core.IndexProc(
            name='F1',
            key_descr='the number of people'
        )
        temp = out.template()
        assert '1::' in temp
        assert 'N::' in temp

