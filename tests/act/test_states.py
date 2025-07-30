# tests/act/test_states.py


import pytest
from typing import Any
import typing as t

from dachi.act import _states as S
from dachi.act._core import Task, TaskStatus
from dachi.proc import Process, AsyncProcess



class _SyncBoolProc(Process):
    """Synchronous Process that returns the bool supplied at init."""
    def __init__(self, val: Any) -> None:
        self._val = val

    def forward(self) -> Any:  # noqa: D401
        return self._val


class _AsyncBoolProc(AsyncProcess):
    """AsyncProcess counterpart of _SyncBoolProc."""
    val: t.Any

    async def aforward(self) -> Any:  # noqa: D401
        return self.val


class _ImmediateTask(Task):
    """Task that immediately finishes with a fixed status."""
    def __init__(self, status: TaskStatus) -> None:
        super().__init__()
        self._status_to_return = status

    async def tick(self) -> TaskStatus:  # noqa: D401
        return self._status_to_return


@pytest.mark.asyncio
class TestBranchState:
    """`BranchState.update` maps (a) truthiness to TaskStatus and
    (b) propagates exceptions unchanged."""

    # ---- truthiness mapping (sync) ----
    @pytest.mark.parametrize("value, expected", [
        (True,  TaskStatus.SUCCESS),
        (1,     TaskStatus.SUCCESS),      # non-bool truthy
        (False, TaskStatus.FAILURE),
        (0,     TaskStatus.FAILURE),      # non-bool falsy
    ])
    async def test_sync_process(self, value: Any, expected: TaskStatus) -> None:
        st = S.BranchState(f=_SyncBoolProc(value))
        assert await st.update() is expected

    @pytest.mark.parametrize("value, expected", [
        (True,  TaskStatus.SUCCESS),
        (False, TaskStatus.FAILURE),
    ])
    async def test_async_process(self, value: bool, expected: TaskStatus) -> None:
        st = S.BranchState(f=_AsyncBoolProc(val=value))
        assert await st.update() is expected

    # ---- exception bubbling ----
    async def test_exception_propagation(self) -> None:
        class _Boom(Process):
            def forward(self):  # noqa: D401
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await S.BranchState(f=_Boom()).update()

@pytest.mark.asyncio
class TestTaskStatePassThrough:
    """`TaskState.update` must return *exactly* what the wrapped task returns."""

    @pytest.mark.parametrize("status", list(TaskStatus))
    async def test_pass_through(self, status: TaskStatus) -> None:
        ts = S.TaskState(task=_ImmediateTask(status))
        assert await ts.update() is status


@pytest.mark.asyncio
class TestTaskStateBehaviour:
    """Extra behavioural guarantees for TaskState."""

    async def test_tick_called_once(self) -> None:
        counter = {"n": 0}

        class _CountingTask(Task):
            async def tick(self_inner):  # noqa: D401
                counter["n"] += 1
                return TaskStatus.SUCCESS

        ts = S.TaskState(task=_CountingTask())
        await ts.update()
        assert counter["n"] == 1

    async def test_exception_propagation(self) -> None:
        class _Boom(Task):
            async def tick(self):  # noqa: D401
                raise ValueError("kaboom")

        with pytest.raises(ValueError):
            await S.TaskState(task=_Boom()).update()


def _state(fn):
    fn._is_state = True      # noqa: SLF001
    return fn


@pytest.mark.asyncio
class TestStateMachineTrivial:
    """Edge-cases: no init state / single terminal state."""

    async def test_no_init_is_success(self) -> None:
        class SM(S.StateMachine):  # no states defined
            pass

        sm = SM()
        # `_cur_state` is None and `_init_state` is None → SUCCESS immediately
        assert await sm.tick() is TaskStatus.SUCCESS

    async def test_single_state_success(self) -> None:
        class SM(S.StateMachine):
            @_state
            async def only(self) -> TaskStatus:      # noqa: D401
                return TaskStatus.SUCCESS

            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("only")
                self._states["only"] = S.BranchState(f=_AsyncBoolProc(val=True))
                self._transitions.set({
                    "only": {TaskStatus.SUCCESS: TaskStatus.SUCCESS},
                })

        sm = SM()
        assert await sm.tick() is TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestStateMachineMultistep:
    """Literal-to-literal transitions, failure path, reset behaviour."""

    async def test_two_step_chain_to_success(self) -> None:
        class SM(S.StateMachine):
            @_state
            async def A(self) -> str:                 # noqa: D401
                return "B"

            @_state
            async def B(self) -> TaskStatus:          # noqa: D401
                return TaskStatus.SUCCESS

            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("A")
                self._states["A"] = "A"
                self._states["B"] = "B"

                self._transitions.set({
                    "A": {"B": "B"},
                    "B": {TaskStatus.SUCCESS: TaskStatus.SUCCESS},
                })

        sm = SM()
        assert await sm.tick() is TaskStatus.RUNNING     # ran A
        assert await sm.tick() is TaskStatus.SUCCESS     # ran B
        # further ticks must keep returning SUCCESS and not re-enter states
        assert await sm.tick() is TaskStatus.SUCCESS

    async def test_failure_path(self) -> None:
        class SM(S.StateMachine):
            @_state
            async def entry(self) -> TaskStatus:         # noqa: D401
                return TaskStatus.FAILURE

            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("entry")
                self._states["entry"] = S.BranchState(f=_AsyncBoolProc(val=False))
                self._transitions.set({
                    "entry": {TaskStatus.FAILURE: TaskStatus.FAILURE},
                })

        sm = SM()
        assert await sm.tick() is TaskStatus.FAILURE

    async def test_reset(self) -> None:
        class SM(S.StateMachine):
            @_state
            async def first(self) -> str:                # noqa: D401
                return "last"

            @_state
            async def last(self) -> TaskStatus:          # noqa: D401
                return TaskStatus.SUCCESS

            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("first")
                self._states["first"] = "first"
                self._states["last"] = "last"
                self._transitions.set({
                    "first": {"last": "last"},
                    "last": {TaskStatus.SUCCESS: TaskStatus.SUCCESS},
                })

        sm = SM()
        # reach terminal SUCCESS
        await sm.tick(); await sm.tick()
        assert sm.status is TaskStatus.SUCCESS
        # reset and rerun
        sm.reset()
        assert sm.status is TaskStatus.READY 
        await sm.tick(); await sm.tick()
        assert sm.status is TaskStatus.SUCCESS


@pytest.mark.asyncio
class TestStateMachineErrorPaths:
    """Robustness checks: missing transitions / undefined states."""

    async def test_missing_transition_raises(self) -> None:
        class SM(S.StateMachine):
            @_state
            async def foo(self) -> str:                 # noqa: D401
                return "bar"   # "bar" not in transitions map

            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("foo")
                self._transitions.set({"foo": {}})

        with pytest.raises(KeyError):
            await SM().tick()

    async def test_undefined_state_name_raises(self) -> None:
        class SM(S.StateMachine):
            @_state
            async def foo(self) -> str:                 # noqa: D401
                return "missing"

            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("foo")
                self._states["foo"] = "foo"
                self._transitions.set({"foo": {"missing": "missing"}})

        sm = SM()
        # first tick runs foo and returns "missing"
        await sm.tick()
        # second tick tries to execute undefined state -> KeyError
        with pytest.raises(KeyError):
            await sm.tick()


# @pytest.mark.asyncio
# class TestStateMachine:
    
#     async def test_success_path(self):

#         def state_b():
#             return TaskStatus.SUCCESS
#         def state_a():
#             return state_b
#         sm = behavior.StateMachine(init_state=state_a)
#         assert await sm.tick() is TaskStatus.RUNNING
#         assert await sm.tick() is TaskStatus.SUCCESS

#     async def test_immediate_failure(self):
#         sm = behavior.StateMachine(init_state=lambda: TaskStatus.FAILURE)
#         assert await sm.tick() is TaskStatus.FAILURE

#     async def test_reset(self):
#         sm = behavior.StateMachine(init_state=lambda: TaskStatus.SUCCESS)
#         await sm.tick(); sm.reset()
#         assert sm.status is TaskStatus.READY
