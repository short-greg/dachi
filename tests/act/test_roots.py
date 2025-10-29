import pytest
from dachi.act._bt._core import TaskStatus
from dachi.act._bt._roots import BT
#, StateMachine, statefunc
from .utils import ImmediateAction, SetStorageAction
from dachi.act import _states as S
from tests.act.utils import _AsyncBoolProc, _BoolProc


# @pytest.mark.asyncio
# class TestRoot:
#     async def test_root_without_child(self):
#         assert await BT().tick() is TaskStatus.SUCCESS

#     async def test_root_delegates(self):
#         r = BT(root=ImmediateAction(status_val=TaskStatus.FAILURE))
#         assert await r.tick() is TaskStatus.FAILURE

#     async def test_root_reset_propagates(self):
#         child = ImmediateAction(status_val=TaskStatus.SUCCESS)
#         r = BT(root=child)
#         await r.tick(); r.reset()
#         assert child.status is TaskStatus.READY



# @pytest.mark.asyncio
# class TestStateMachine:
#     """Edge-cases: no init state / single terminal state."""

#     async def test_no_init_is_success(self) -> None:
#         class SM(StateMachine):  # no states defined
#             pass

#         sm = SM()
#         # `_cur_state` is None and `_init_state` is None → SUCCESS immediately
#         assert await sm.tick() is TaskStatus.SUCCESS

#     async def test_single_state_success(self) -> None:
#         class SM(StateMachine):
#             @_state
#             async def only(self) -> TaskStatus:      # noqa: D401
#                 return TaskStatus.SUCCESS

#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("only")
#                 self._states["only"] = S.BranchState(f=_AsyncBoolProc(val=True))
#                 self._transitions.set({
#                     "only": {TaskStatus.SUCCESS: TaskStatus.SUCCESS},
#                 })

#         sm = SM()
#         assert await sm.tick() is TaskStatus.SUCCESS

#     async def test_two_step_chain_to_success(self) -> None:
#         class SM(StateMachine):
#             @_state
#             async def A(self) -> str:                 # noqa: D401
#                 return "B"

#             @_state
#             async def B(self) -> TaskStatus:          # noqa: D401
#                 return TaskStatus.SUCCESS

#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("A")
#                 self._states["A"] = "A"
#                 self._states["B"] = "B"

#                 self._transitions.set({
#                     "A": {"B": "B"},
#                     "B": {TaskStatus.SUCCESS: TaskStatus.SUCCESS},
#                 })

#         sm = SM()
#         assert await sm.tick() is TaskStatus.RUNNING     # ran A
#         assert await sm.tick() is TaskStatus.SUCCESS     # ran B
#         # further ticks must keep returning SUCCESS and not re-enter states
#         assert await sm.tick() is TaskStatus.SUCCESS

#     async def test_failure_path(self) -> None:
#         class SM(StateMachine):
#             @_state
#             async def entry(self) -> TaskStatus:         # noqa: D401
#                 return TaskStatus.FAILURE

#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("entry")
#                 self._states["entry"] = S.BranchState(f=_AsyncBoolProc(val=False))
#                 self._transitions.set({
#                     "entry": {TaskStatus.FAILURE: TaskStatus.FAILURE},
#                 })

#         sm = SM()
#         assert await sm.tick() is TaskStatus.FAILURE

#     async def test_reset(self) -> None:
#         class SM(StateMachine):
#             @_state
#             async def first(self) -> str:                # noqa: D401
#                 return "last"

#             @_state
#             async def last(self) -> TaskStatus:          # noqa: D401
#                 return TaskStatus.SUCCESS

#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("first")
#                 self._states["first"] = "first"
#                 self._states["last"] = "last"
#                 self._transitions.set({
#                     "first": {"last": "last"},
#                     "last": {TaskStatus.SUCCESS: TaskStatus.SUCCESS},
#                 })

#         sm = SM()
#         # reach terminal SUCCESS
#         await sm.tick(); await sm.tick()
#         assert sm.status is TaskStatus.SUCCESS
#         # reset and rerun
#         sm.reset()
#         assert sm.status is TaskStatus.READY 
#         await sm.tick(); await sm.tick()
#         assert sm.status is TaskStatus.SUCCESS

#     async def test_missing_transition_raises(self) -> None:
#         class SM(StateMachine):
#             @_state
#             async def foo(self) -> str:                 # noqa: D401
#                 return "bar"   # "bar" not in transitions map

#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("foo")
#                 self._transitions.set({"foo": {}})

#         with pytest.raises(KeyError):
#             await SM().tick()

#     async def test_undefined_state_name_raises(self) -> None:
#         class SM(StateMachine):
#             @_state
#             async def foo(self) -> str:                 # noqa: D401
#                 return "missing"

#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("foo")
#                 self._states["foo"] = "foo"
#                 self._transitions.set({"foo": {"missing": "missing"}})

#         sm = SM()
#         # first tick runs foo and returns "missing"
#         await sm.tick()
#         # second tick tries to execute undefined state -> KeyError
#         with pytest.raises(KeyError):
#             await sm.tick()



# class TestStateFuncDecorator:
#     """`statefunc` must simply flag the target callable with *_is_state*."""

#     def test_flag_added(self):
#         @statefunc
#         def dummy():
#             pass

#         assert getattr(dummy, "_is_state", False) is True


# def _state(fn): fn._is_state = True; return fn                   # decorator helper


# @pytest.mark.asyncio
# class TestStateMachineExtras:
#     """Edge and mixed-node scenarios."""

#     async def test_cur_state_initialised_on_first_tick(self):
#         class SM(StateMachine):
#             @_state
#             async def leaf(self): return TaskStatus.SUCCESS
#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("leaf")
#                 self._states["leaf"] = "leaf"
#                 self._transitions.set({"leaf": {TaskStatus.SUCCESS: TaskStatus.SUCCESS}})
#         sm = SM()
#         assert sm._cur_state.data is None
#         await sm.tick()
#         assert sm._cur_state.data is TaskStatus.SUCCESS

#     async def test_running_stays_in_state(self):
#         class SM(StateMachine):
#             @_state
#             async def loop(self): return TaskStatus.RUNNING
#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("loop")
#                 self._states["loop"] = "loop"
#                 self._transitions.set({"loop": {TaskStatus.RUNNING: "loop"}})
#         sm = SM()
#         assert await sm.tick() is TaskStatus.RUNNING
#         assert sm._cur_state.data == "loop"

#     async def test_fail_path_terminal(self):
#         class SM(StateMachine):
#             @_state
#             async def entry(self): return TaskStatus.FAILURE
#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("entry")
#                 self._states["entry"] = "entry"
#                 self._transitions.set({"entry": {TaskStatus.FAILURE: TaskStatus.FAILURE}})
#         sm = SM()
#         assert await sm.tick() is TaskStatus.FAILURE

#     async def test_mixed_callable_and_state_objects(self):
#         class SM(StateMachine):
#             @_state
#             async def check(self): return TaskStatus.SUCCESS
#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("check")
#                 # mix: string alias + BranchState instance
#                 self._states["check"] = "check"
#                 self._states["branch"] = S.BranchState(f=_BoolProc(True))
#                 self._transitions.set({
#                     "check": {TaskStatus.SUCCESS: "branch"},
#                     "branch": {TaskStatus.SUCCESS: TaskStatus.SUCCESS}
#                 })
#         sm = SM()
#         assert await sm.tick() is TaskStatus.RUNNING              # ran check
#         assert await sm.tick() is TaskStatus.SUCCESS              # ran BranchState

#     async def test_reset_restores_init_state(self):
#         class SM(StateMachine):
#             @_state
#             async def one(self): return TaskStatus.SUCCESS
#             def __post_init__(self):
#                 super().__post_init__()
#                 self._init_state.set("one")
#                 self._states["one"] = "one"
#                 self._transitions.set({"one": {TaskStatus.SUCCESS: TaskStatus.SUCCESS}})
#         sm = SM()
#         await sm.tick()
#         assert sm.status is TaskStatus.SUCCESS
#         sm.reset()
#         assert sm.status is TaskStatus.READY
#         assert sm._cur_state.data == "one"


class TestBTRestrictedSchema:
    """Test BT.restricted_schema() - Pattern C (Single Field)"""

    def test_restricted_schema_returns_unrestricted_when_tasks_none(self):
        """Test that tasks=None returns unrestricted schema"""
        bt = BT(root=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = bt.restricted_schema(tasks=None)
        unrestricted = bt.schema()

        # Should be identical
        assert restricted == unrestricted

    def test_restricted_schema_updates_root_field_with_variants(self):
        """Test that root field is restricted to specified variants"""
        bt = BT(root=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Restrict to only ImmediateAction and SetStorageAction
        restricted = bt.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction]
        )

        # Check that schema was updated
        assert "$defs" in restricted
        assert "Allowed_TaskSpec" in restricted["$defs"]

        # Check that Allowed_TaskSpec contains our variants
        allowed_union = restricted["$defs"]["Allowed_TaskSpec"]
        assert "oneOf" in allowed_union
        refs = allowed_union["oneOf"]
        assert len(refs) == 2

        # Extract spec names from refs
        spec_names = {ref["$ref"].split("/")[-1] for ref in refs}
        assert any("ImmediateActionSpec" in name for name in spec_names)
        assert any("SetStorageActionSpec" in name for name in spec_names)

    def test_restricted_schema_uses_shared_profile_by_default(self):
        """Test that default profile is 'shared' (uses $defs/Allowed_*)"""
        bt = BT(root=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = bt.restricted_schema(tasks=[ImmediateAction])

        # Should use shared union in $defs
        assert "Allowed_TaskSpec" in restricted["$defs"]

        # root field should reference the shared union
        root_schema = restricted["properties"]["root"]
        assert root_schema == {"$ref": "#/$defs/Allowed_TaskSpec"}

    def test_restricted_schema_inline_profile_creates_oneof(self):
        """Test that _profile='inline' creates inline oneOf"""
        bt = BT(root=ImmediateAction(status_val=TaskStatus.SUCCESS))
        restricted = bt.restricted_schema(
            tasks=[ImmediateAction, SetStorageAction],
            _profile="inline"
        )

        # Should still have defs for the individual tasks
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)

        # But root field should have inline oneOf (no Allowed_TaskSpec)
        root_schema = restricted["properties"]["root"]
        assert "oneOf" in root_schema

    def test_restricted_schema_with_task_spec_class(self):
        """Test that TaskSpec classes work as variants"""
        bt = BT(root=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Get the spec class
        spec_class = ImmediateAction.schema_model()

        restricted = bt.restricted_schema(tasks=[spec_class])

        # Should work and include the task
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)

    def test_restricted_schema_with_mixed_formats(self):
        """Test that mixed variant formats work together"""
        bt = BT(root=ImmediateAction(status_val=TaskStatus.SUCCESS))

        # Mix: Task class, TaskSpec class, and schema dict
        action_spec = SetStorageAction.schema_model()
        immediate_schema = ImmediateAction.schema()

        restricted = bt.restricted_schema(
            tasks=[
                ImmediateAction,  # Task class
                action_spec,       # Spec class
                immediate_schema   # Schema dict (will be duplicate)
            ]
        )

        # Should deduplicate and work correctly
        defs_keys = restricted["$defs"].keys()
        assert any("ImmediateActionSpec" in key for key in defs_keys)
        assert any("SetStorageActionSpec" in key for key in defs_keys)
