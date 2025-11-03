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


class TestProcessCallRestrictedSchema:

    def test_restricted_schema_with_no_processes_returns_base_schema(self):
        schema = ProcessCall.restricted_schema(processes=None)
        base_schema = ProcessCall.schema()

        assert schema == base_schema

    def test_restricted_schema_restricts_process_field(self):
        schema = ProcessCall.restricted_schema(
            processes=[AddNumbers, MultiplyNumbers]
        )

        assert 'properties' in schema
        assert 'process' in schema['properties']

    def test_restricted_schema_restricts_args_field(self):
        schema = ProcessCall.restricted_schema(
            processes=[AddNumbers, ConcatStrings]
        )

        args_schema = schema['properties']['args']
        assert 'additionalProperties' in args_schema

        additional_props = args_schema['additionalProperties']
        assert 'oneOf' in additional_props

    def test_restricted_schema_includes_input_types(self):
        schema = ProcessCall.restricted_schema(
            processes=[AddNumbers, ConcatStrings]
        )

        args_additional = schema['properties']['args']['additionalProperties']
        type_schemas = args_additional['oneOf']

        type_strings = []
        for ts in type_schemas:
            if 'type' in ts:
                type_strings.append(ts['type'])

        assert 'integer' in type_strings
        assert 'string' in type_strings

    def test_restricted_schema_always_includes_reft(self):
        schema = ProcessCall.restricted_schema(
            processes=[AddNumbers]
        )

        args_additional = schema['properties']['args']['additionalProperties']

        has_reft = False
        for ts in args_additional['oneOf']:
            if 'properties' in ts and 'name' in ts.get('properties', {}):
                has_reft = True
                break

        assert has_reft, "RefT should always be included in args types"

    def test_restricted_schema_raises_for_untyped_process(self):
        with pytest.raises(TypeError, match="must have a type annotation"):
            ProcessCall.restricted_schema(
                processes=[UntypedProcess]
            )

    def test_restricted_schema_extracts_float_from_async_process(self):
        schema = ProcessCall.restricted_schema(
            processes=[MultiplyNumbers]
        )

        args_additional = schema['properties']['args']['additionalProperties']
        type_schemas = args_additional['oneOf']

        has_number = False
        for ts in type_schemas:
            if ts.get('type') == 'number':
                has_number = True
                break

        assert has_number, "Should extract float (number) type from MultiplyNumbers"
