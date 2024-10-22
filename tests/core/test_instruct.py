from dachi._core import _instruct as core
from .test_core import SimpleStruct
from .test_ai import DummyAIModel


class TestSignatureF:

    def test_inserts_into_docstring(self):

        @core.signaturefunc()
        def signaturep(x: str) -> SimpleStruct:
            """Output the value of x
            
            x: {x}

            Args:
                x (str): The input

            Returns:
                SimpleStruct: The value of x
            """
            pass

        result = signaturep.i(2)

        assert 'x: 2' in result.text

    def test_inserts_into_docstring_with_method(self):

        class X(object):
            @core.signaturefunc()
            def signaturep(self, x: str) -> SimpleStruct:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    SimpleStruct: The value of x
                """
                pass

        x = X()
        result = x.signaturep.i(2)

        assert 'x: 2' in result.text

    def test_signature_executes_model(self):

        class X(object):
            
            @core.signaturefunc(engine=DummyAIModel())
            def signaturep(self, x: str) -> SimpleStruct:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    SimpleStruct: The value of x
                """
                pass

        x = X()
        result = x.signaturep(2)

        assert result == 'Great!'

    def test_signature_streams_the_output(self):

        class X(object):

            @core.signaturefunc(engine=DummyAIModel())
            def signaturep(self, x: str) -> SimpleStruct:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    SimpleStruct: The value of x
                """
                pass

        x = X()
        for d, dx in x.signaturep.stream_forward(2):
            pass

        assert d == 'Great!'
        assert dx == '!'

    def test_signature_uses_the_correct_model(self):

        class X(object):

            def __init__(self, model):
                super().__init__()
                self.model = model
            
            @core.signaturefunc(engine='model')
            def signaturep(self, x: str) -> SimpleStruct:
                """Output the value of x
                
                x: {x}

                Args:
                    x (str): The input

                Returns:
                    SimpleStruct: The value of x
                """
                pass

        x = X(DummyAIModel('Awesome'))
        x2 = X(DummyAIModel('Fabulous'))
        result = x.signaturep(2)
        result2 = x2.signaturep(2)

        assert result == 'Awesome'
        assert result2 == 'Fabulous'


class TestInstructF:

    def test_instruct(self):

        @core.instructfunc(engine=DummyAIModel())
        def instructrep(x: str) -> str:
            """Output the value of x
            
            x: {x}

            Args:
                x (str): The input

            Returns:
                SimpleStruct: The value of x
            """
            return core.Cue(f'Do {x}')

        result = instructrep.i(2)

        assert 'Do 2' == result.text

    def test_inserts_into_docstring_with_method(self):

        class X(object):

            @core.instructfunc(engine=DummyAIModel())
            def instructrep(self, x: str) -> str:
                """
                """
                return core.Cue(f'Do {x}')

        x = X()
        result = x.instructrep.i(2)

        assert 'Do 2' == result.text

    def test_x_has_different_instance_for_instruct_rep(self):

        class X(object):

            @core.instructfunc(engine=DummyAIModel())
            def instructrep(self, x: str) -> str:
                """
                """
                return core.Cue(f'Do {x}')

        x = X()
        x2 = X()
        assert x.instructrep is not x2.instructrep

    def test_x_has_same_value_for_instruct_rep(self):

        class X(object):

            @core.instructfunc(engine=DummyAIModel())
            def instructrep(self, x: str) -> str:
                """
                """
                return core.Cue(f'Do {x}')

        x = X()
        assert x.instructrep is x.instructrep
