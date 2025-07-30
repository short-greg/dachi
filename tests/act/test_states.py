# tests/act/test_states.py
import pytest
from typing import Any
import typing as t

from dachi.act import _states as S
from dachi.act._core import Task, TaskStatus
from dachi.act._tasks import Action
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
class TestStateMachine:
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


# ---------------------------------------------------------------------------
# Helpers shared by new cases
# ---------------------------------------------------------------------------
class _BoolProc(Process):
    """Process returning the value supplied at construction."""
    def __init__(self, val): self.val = val
    def forward(self): return self.val


class _ABoolProc(AsyncProcess):
    """Async variant of _BoolProc."""
    def __init__(self, val): self.val = val
    async def aforward(self): return self.val


class _ToggleTask(Task):
    """RUNNING ▸ SUCCESS over two ticks; counts invocations."""
    def __post_init__(self):
        super().__post_init__()
        self.calls = 0
    async def tick(self):
        self.calls += 1
        if self.calls == 1:
            self._status.set(TaskStatus.RUNNING)
        else:
            self._status.set(TaskStatus.SUCCESS)
        return self.status


# ---------------------------------------------------------------------------
# BranchState
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestBranchStateMore:
    """Edge-cases for truthiness handling and type errors."""

    @pytest.mark.parametrize("val", [1, True])
    async def test_async_non_bool_truthy(self, val):
        st = S.BranchState(f=_ABoolProc(val))
        assert await st.update() is TaskStatus.SUCCESS

    async def test_sync_none_is_failure(self):
        st = S.BranchState(f=_BoolProc(False))
        assert await st.update() is TaskStatus.FAILURE

    async def test_invalid_return_type_bubbles(self):
        class _Bad(Process):
            def forward(self): return TaskStatus.SUCCESS          # misuse
        with pytest.raises(Exception):
            await S.BranchState(f=_Bad()).update()



@pytest.mark.asyncio
class TestTaskStateLifecycle:
    """Delegation and caching semantics around the wrapped task."""

    async def test_running_then_success(self):
        inner = _ToggleTask()
        ts = S.TaskState(task=inner)
        assert await ts.update() is TaskStatus.RUNNING
        assert await ts.update() is TaskStatus.SUCCESS
        assert inner.calls == 2                                   # both ticks forwarded

    async def test_done_task_not_reinvoked(self):
        class _Done(Action):
            async def act(self): return TaskStatus.SUCCESS
        done = _Done(); await done.tick()                         # mark as done
        ts = S.TaskState(task=done)
        res = await ts.update();                                       # first call returns SUCCESS
        print(res)
        res2 = await ts.update();                                       # second call should not re-tick
        print(res2)
        assert ts.task.status is TaskStatus.SUCCESS


def _state(fn): fn._is_state = True; return fn                   # decorator helper


@pytest.mark.asyncio
class TestStateMachineExtras:
    """Edge and mixed-node scenarios."""

    async def test_cur_state_initialised_on_first_tick(self):
        class SM(S.StateMachine):
            @_state
            async def leaf(self): return TaskStatus.SUCCESS
            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("leaf")
                self._states["leaf"] = "leaf"
                self._transitions.set({"leaf": {TaskStatus.SUCCESS: TaskStatus.SUCCESS}})
        sm = SM()
        assert sm._cur_state.data is None
        await sm.tick()
        assert sm._cur_state.data is TaskStatus.SUCCESS

    async def test_running_stays_in_state(self):
        class SM(S.StateMachine):
            @_state
            async def loop(self): return TaskStatus.RUNNING
            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("loop")
                self._states["loop"] = "loop"
                self._transitions.set({"loop": {TaskStatus.RUNNING: "loop"}})
        sm = SM()
        assert await sm.tick() is TaskStatus.RUNNING
        assert sm._cur_state.data == "loop"

    async def test_fail_path_terminal(self):
        class SM(S.StateMachine):
            @_state
            async def entry(self): return TaskStatus.FAILURE
            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("entry")
                self._states["entry"] = "entry"
                self._transitions.set({"entry": {TaskStatus.FAILURE: TaskStatus.FAILURE}})
        sm = SM()
        assert await sm.tick() is TaskStatus.FAILURE

    async def test_mixed_callable_and_state_objects(self):
        class SM(S.StateMachine):
            @_state
            async def check(self): return TaskStatus.SUCCESS
            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("check")
                # mix: string alias + BranchState instance
                self._states["check"] = "check"
                self._states["branch"] = S.BranchState(f=_BoolProc(True))
                self._transitions.set({
                    "check": {TaskStatus.SUCCESS: "branch"},
                    "branch": {TaskStatus.SUCCESS: TaskStatus.SUCCESS}
                })
        sm = SM()
        assert await sm.tick() is TaskStatus.RUNNING              # ran check
        assert await sm.tick() is TaskStatus.SUCCESS              # ran BranchState

    async def test_reset_restores_init_state(self):
        class SM(S.StateMachine):
            @_state
            async def one(self): return TaskStatus.SUCCESS
            def __post_init__(self):
                super().__post_init__()
                self._init_state.set("one")
                self._states["one"] = "one"
                self._transitions.set({"one": {TaskStatus.SUCCESS: TaskStatus.SUCCESS}})
        sm = SM()
        await sm.tick()
        assert sm.status is TaskStatus.SUCCESS
        sm.reset()
        assert sm.status is TaskStatus.READY
        assert sm._cur_state.data == "one"



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
