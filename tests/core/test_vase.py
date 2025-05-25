"""
pytest -everything for Dachi core
$ pytest -q
"""

import typing
import pytest
from itertools import permutations

# ------------------------------------------------------------------
#  import your real classes
# ------------------------------------------------------------------
from dachi.core import (
    BaseProcess, BaseStruct, Attr, Param,
    BuildContext, Ref,
)

# ------------------------------------------------------------------
#  Helper factories
# ------------------------------------------------------------------
class Leaf(BaseProcess):
    def __init__(self, val: int):
        super().__init__()
        self.val = val


class Mid(BaseProcess):
    def __init__(self, left: Leaf, right: Leaf | None = None):
        super().__init__()
        self.left  = left
        self.right = right or left          # aliasing on purpose


class Root(BaseProcess):
    spec_style = "flat"                     # override default
    def __init__(self, mids: list[Mid], tag: str = "root"):
        super().__init__()
        self.mids = mids
        self.tag  = tag


class DataPacket(BaseStruct):
    uid: int = Attr()
    weight: float = Param(0.0)


# ------------------------------------------------------------------
#  1. schema generation corner-cases
# ------------------------------------------------------------------
@pytest.mark.parametrize("proc_cls, expected_fields", [
    (Leaf, ["val"]),
    (Mid,  ["left", "right"]),
    (Root, ["mids", "tag"]),
])
def test_schema_fields(proc_cls, expected_fields):
    schema = proc_cls.to_schema()
    assert list(schema.model_fields) == expected_fields + ["style"]  # style always last


def test_schema_union_variants():
    ann = Mid.to_schema().model_fields["left"].annotation
    assert typing.get_origin(ann) is typing.Union
    # must contain: runtime class, spec class, Ref
    args = set(typing.get_args(ann))
    assert Leaf in args and Ref in args and Leaf.to_schema() in args


# ------------------------------------------------------------------
#  2. invalid signatures
# ------------------------------------------------------------------
def test_varargs_forbidden():
    with pytest.raises(TypeError, match="variable-length"):

        class Bad(BaseProcess):
            def __init__(self, *vals: int):
                super().__init__()


def test_missing_typehint_forbidden():
    with pytest.raises(TypeError, match="Missing type hint"):

        class Bad2(BaseProcess):
            def __init__(self, x):
                super().__init__()
                self.x = x


# ------------------------------------------------------------------
#  3. structured spec round-trip
# ------------------------------------------------------------------
def test_structured_roundtrip():
    leaf = Leaf(3)
    clone = Leaf.from_spec(leaf.to_spec())
    assert clone.val == 3


# ------------------------------------------------------------------
#  4. deep / aliasing flat round-trip
# ------------------------------------------------------------------
def test_flat_roundtrip_with_aliasing():
    l1, l2 = Leaf(1), Leaf(2)
    mid1, mid2 = Mid(l1), Mid(l1)           # same leaf shared
    root = Root([mid1, mid2])

    ctx = root.to_flat_spec()
    rebuilt = Root.from_flat_spec(ctx, ctx.obj2id[root])

    # shared instance should stay shared
    assert rebuilt.mids[0].left is rebuilt.mids[1].left
    assert len(ctx.specs) == 3              # Leaf, Mid, Root (aliasing deduped)


# ------------------------------------------------------------------
#  5. BuildContext & dependency order
# ------------------------------------------------------------------
def test_dependency_topological_order():
    l = Leaf(4)
    root = Root([Mid(l)])

    topo = root.dependencies()
    # First element must be Leaf or Mid depending on alias depth
    assert isinstance(topo[0], Leaf)
    assert isinstance(topo[-1], Root)


# ------------------------------------------------------------------
#  6. cycle detection
# ------------------------------------------------------------------
def test_cycle_detection():
    class Cyclic(BaseProcess):
        def __init__(self):
            super().__init__()
            self.me = self                   # forms a direct cycle

    c = Cyclic()
    with pytest.raises(ValueError, match="Cyclic"):
        c.to_flat_spec()


# ------------------------------------------------------------------
#  7. state_dict / load_state_dict
# ------------------------------------------------------------------
def test_param_state_roundtrip():
    class Learner(BaseProcess):
        def __init__(self):
            super().__init__()
            self.bias = Param(0.0)
            self.register_attr("bias", self.bias)

    learner = Learner()
    learner.bias.value = 1.23
    state = learner.state_dict()
    learner.bias.value = 0.0                # wipe
    learner.load_state_dict(state)
    assert learner.bias.value == 1.23


# ------------------------------------------------------------------
#  8. BaseStruct param discovery
# ------------------------------------------------------------------
def test_basestruct_param_registration():
    packet = DataPacket(uid=5, weight=2.5)
    params = list(packet.parameters())
    assert len(params) == 1 and isinstance(params[0], Param)
    assert params[0].value == 2.5


# ------------------------------------------------------------------
#  9. BuildContext robustness to massive duplicates
# ------------------------------------------------------------------
def test_buildcontext_dedup_massive():
    leaf = Leaf(99)
    root = Root([Mid(leaf) for _ in range(50)])

    ctx = root.to_flat_spec()
    # Only Leaf, Mid, Root payloads expected (no matter how many aliases)
    assert len(ctx.specs) == 3


# ------------------------------------------------------------------
#  10. invalid Ref resolution
# ------------------------------------------------------------------
def test_invalid_ref_resolution():
    ctx = BuildContext()
    with pytest.raises(IndexError):
        ctx.resolve(Ref(id=0, target_id=999))   # never registered


