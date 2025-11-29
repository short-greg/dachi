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
        pc = add.forward_process_call(x=5, y=3)

        assert pc.process is add
        assert pc.args.x == 5
        assert pc.args.y == 3

    def test_process_call_has_args_attribute(self):
        add = AddNumbers()
        pc = add.forward_process_call(x=0, y=0)

        assert hasattr(pc, 'args')
        assert hasattr(pc.args, 'x')
        assert hasattr(pc.args, 'y')

    def test_process_call_has_process_attribute(self):
        add = AddNumbers()
        pc = add.forward_process_call(x=0, y=0)

        assert hasattr(pc, 'process')
        assert isinstance(pc.process, Process)

    def test_async_process_call_has_process_attribute(self):
        multiply = MultiplyNumbers()
        pc = multiply.aforward_process_call(a=0.0, b=0.0)

        assert hasattr(pc, 'process')
        assert isinstance(pc.process, AsyncProcess)

    def test_process_call_with_reft_args(self):
        add = AddNumbers()
        pc = add.forward_process_call(x=Ref(name='input'), y=10, _ref=True)

        assert isinstance(pc.args.x, Ref)
        assert pc.args.x.name == 'input'
        assert pc.args.y == 10


class TestProcessCallAttributes:

    def test_process_call_has_required_attributes(self):
        add = AddNumbers()
        pc = add.forward_process_call(x=5, y=3)

        assert hasattr(pc, 'process')
        assert hasattr(pc, 'args')
        assert pc.process is add
        assert pc.args.x == 5
        assert pc.args.y == 3

    def test_process_call_is_module_instance(self):
        add = AddNumbers()
        pc = add.forward_process_call(x=5, y=3)

        from dachi.core import Module
        assert isinstance(pc, Module)

    def test_ref_process_call_preserves_refs(self):
        add = AddNumbers()
        pc = add.forward_process_call(x=5, y=Ref(name='input'), _ref=True)

        assert pc.args.x == 5
        assert isinstance(pc.args.y, Ref)
        assert pc.args.y.name == 'input'


class TestProcessCallSerialization:

    def test_to_spec_preserves_process_and_args(self):
        """to_spec() preserves process and args structure."""
        add = AddNumbers()
        pc = add.forward_process_call(x=10, y=20)

        spec = pc.to_spec()

        assert "KIND" in spec
        assert "process" in spec
        assert "args" in spec
        assert spec["args"]["x"] == 10
        assert spec["args"]["y"] == 20

    def test_to_spec_includes_kind_field(self):
        """to_spec() includes KIND field for registry lookup."""
        add = AddNumbers()
        pc = add.forward_process_call(x=5, y=3)

        spec = pc.to_spec()

        assert "KIND" in spec
        assert "ProcessCall" in spec["KIND"]

    def test_to_spec_preserves_ref_args(self):
        """to_spec() preserves Ref objects in args."""
        add = AddNumbers()
        pc = add.forward_process_call(x=Ref(name='input_x'), y=100, _ref=True)

        spec = pc.to_spec()

        assert spec["args"]["x"]["name"] == 'input_x'
        assert spec["args"]["y"] == 100

    def test_to_spec_with_async_process(self):
        """to_spec() works with AsyncProcess."""
        multiply = MultiplyNumbers()
        pc = multiply.aforward_process_call(a=3.14, b=2.71)

        spec = pc.to_spec()

        assert "AsyncProcessCall" in spec["KIND"]
        assert spec["args"]["a"] == 3.14
        assert spec["args"]["b"] == 2.71

    def test_to_spec_preserves_string_args(self):
        """to_spec() preserves string argument types."""
        concat = ConcatStrings()
        pc = concat.forward_process_call(s1="hello", s2="world")

        spec = pc.to_spec()

        assert spec["args"]["s1"] == "hello"
        assert spec["args"]["s2"] == "world"

    def test_to_spec_preserves_int_args(self):
        """to_spec() preserves integer argument types."""
        add = AddNumbers()
        pc = add.forward_process_call(x=42, y=99)

        spec = pc.to_spec()

        assert spec["args"]["x"] == 42
        assert spec["args"]["y"] == 99
        assert isinstance(spec["args"]["x"], int)
        assert isinstance(spec["args"]["y"], int)

