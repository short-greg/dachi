import pytest
from dachi.proc import Process, AsyncProcess, ProcessCall, Ref
from typing import List


class AddNumbers(Process):
    """Test process that adds two numbers"""

    def forward(self, x: int, y: int) -> int:
        return x + y


class MultiplyNumbers(AsyncProcess):
    """Test async process that multiplies two numbers"""

    async def aforward(self, a: float, b: float) -> float:
        return a * b


class ConcatStrings(Process):
    """Test process that concatenates strings"""

    def forward(self, s1: str, s2: str) -> str:
        return s1 + s2


class UntypedProcess(Process):
    """Process without type annotations - should fail in restricted_schema"""

    def forward(self, x, y):
        return x + y


class TestProcessCall:

    def test_process_call_creation(self):
        add = AddNumbers()
        pc = ProcessCall(process=add, args={'x': 5, 'y': 3})

        assert pc.process is add
        assert pc.args == {'x': 5, 'y': 3}

    def test_process_call_args_defaults_to_empty_dict(self):
        add = AddNumbers()
        pc = ProcessCall(process=add)

        assert pc.args == {}

    def test_is_async_returns_false_for_process(self):
        add = AddNumbers()
        pc = ProcessCall(process=add, args={})

        assert pc.is_async() is False

    def test_is_async_returns_true_for_async_process(self):
        multiply = MultiplyNumbers()
        pc = ProcessCall(process=multiply, args={})

        assert pc.is_async() is True

    def test_process_call_with_reft_args(self):
        add = AddNumbers()
        pc = ProcessCall(
            process=add,
            args={'x': Ref(name='input'), 'y': 10}
        )

        assert isinstance(pc.args['x'], Ref)
        assert pc.args['x'].name == 'input'
        assert pc.args['y'] == 10


class TestProcessCallSerialization:

    def test_process_call_creates_spec(self):
        add = AddNumbers()
        pc = ProcessCall(process=add, args={'x': 5, 'y': 3})

        spec = pc.spec()
        assert spec is not None
        assert hasattr(spec, 'process')
        assert hasattr(spec, 'args')

    def test_process_call_spec_contains_process_spec(self):
        add = AddNumbers()
        pc = ProcessCall(process=add, args={'x': 5, 'y': 3})

        spec = pc.spec()
        # The process should be converted to its spec
        assert spec.process.kind == 'AddNumbers'

    def test_process_call_spec_preserves_args(self):
        add = AddNumbers()
        pc = ProcessCall(
            process=add,
            args={'x': 5, 'y': Ref(name='input')}
        )

        spec = pc.spec()
        assert spec.args['x'] == 5
        assert isinstance(spec.args['y'], Ref)
        assert spec.args['y'].name == 'input'
