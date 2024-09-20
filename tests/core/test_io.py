from dachi._core import _read as _core, model_to_text
from .test_core import SimpleStruct, SimpleStruct2


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

        assert out.read(model_to_text(struct_list))[0].x == '2'


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

        assert out.read(model_to_text(struct_list))[0].x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')
        assert out.read(model_to_text(simple)).x == '2'


class TestMultiRead(object):

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
        print('Text: ', text)
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


class TestStructRead:

    def test_out_creates_out_class(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
            # name='Simple', signature='...',
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
        simple2 = out.read(d)
        assert simple.x == simple2.x

    def test_out_creates_out_class_with_string(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )
        simple = SimpleStruct(x='hi')
        d = model_to_text(simple)
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
        s = model_to_text(SimpleStruct(x='2'))
        simple2 = out.read(s)
        assert simple2.x == '2'

    def test_out_reads_in_the_class_with_str(self):

        out = _core.StructRead(
            name='F1',
            out_cls=SimpleStruct
        )

        simple = SimpleStruct(x='2')

        assert out.read(model_to_text(simple)).x == '2'


class TestCSVRead(object):

    def test_out_reads_in_the_class(self):

        csv = 'x,y\n'
        csv += '1,2\n'
        csv += '3,4\n'

        out = _core.CSVRead(
            name='F1',
            indexed=False,
        )

        result = out.read(csv)
        print(result[0])
        assert result[0]['x'] == 1
        assert result[0]['y'] == 2
        assert len(result) == 2

    def test_template_outputs_a_valid_template(self):

        out = _core.CSVRead(
            name='F1',
            indexed=False,
            cols=[('x', 'X value', 'int'), ('y', 'Y value', 'str')]
        )

        temp = out.template()
        assert 'N,' in temp
        assert 'X value <int>' in temp


class TestKVRead(object):

    def test_out_reads_in_the_class(self):

        k = 'x::1\n'
        k += 'y::4\n'

        out = _core.KVRead(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )

        result = out.read(k)
        assert result['x'] == '1'
        assert result['y'] == '4'

    def test_template_contains_key(self):


        out = _core.KVRead(
            name='F1',
            key_descr={
                'x': 'the value of x', 
                'y': 'the value of y'
            },
        )
        temp = out.template()
        assert 'x::' in temp
        assert 'y::' in temp


class TestPrimRead(object):

    def test_read_reads_in_data(self):

        out = _core.PrimRead(
            name='F1',
            out_cls=int,
        )

        result = out.read('1')
        assert result == 1

    def test_template_contains_key(self):

        out = _core.PrimRead(
            name='F1',
            out_cls=int,
        )
        temp = out.template()
        assert 'int' in temp


class TestJSONRead(object):

    def test_out_creates_out_class(self):

        out = _core.JSONRead(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple = SimpleStruct2(x='hi', y=1)
        d = model_to_text(simple)
        simple2 = out.read(d)
        assert simple.x == simple2['x']

    def test_out_template(self):

        out = _core.JSONRead(
            name='F1',
            key_descr={
                'x': 'The value of x',
                'y': 'The value of y'
            }
        )
        simple2 = out.template()
        assert 'x' in simple2
