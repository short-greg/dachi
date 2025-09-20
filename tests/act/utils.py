from dachi.core import InitVar, Attr, ModuleList
from dachi.act._bt._core import TaskStatus, Task

from typing import Any
from dachi.act._bt._leafs import Action, Condition, WaitCondition
from dachi.proc import Process, AsyncProcess


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


class _BoolProc(Process):
    """Process returning the value supplied at construction."""
    def __init__(self, val): self.val = val
    def forward(self): return self.val


class _ABoolProc(AsyncProcess):
    """Async variant of _BoolProc."""
    def __init__(self, val): self.val = val
    async def aforward(self): return self.val


class AlwaysTrueCond(Condition):
    async def condition(self):
        return True

class AlwaysFalseCond(Condition):
    async def condition(self):
        return False


class ImmediateAction(Action):
    """A task that immediately returns a preset *status_val*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._ret = status_val

    async def act(self) -> TaskStatus:
        return self._ret


class ATask(Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def act(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(Action):
    """Action whose success/failure depends on *value*."""

    value: InitVar[int] = 4

    def __post_init__(self, value: int):
        # TODO: enforce post init is called
        super().__post_init__()
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        print('Acting!')
        return TaskStatus.FAILURE if self.value.data < 0 else TaskStatus.SUCCESS


class SampleCondition(Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def condition(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    # __store__ = ["value"]
    value: InitVar[int] = 4

    def __post_init__(self, value: int=4):
        super().__post_init__()
        self._count = 0
        self.value = Attr[int](value)

    async def act(self) -> TaskStatus:  # noqa: D401
        if self.value.data == 0:
            return TaskStatus.FAILURE
        self._count += 1
        if self._count == 2:
            return TaskStatus.SUCCESS
        if self._count < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING


class FlagWaitCond(WaitCondition):
    """WaitCondition whose outcome is controlled via the *flag* attribute."""

    flag: bool = True

    async def condition(self) -> bool:
        return self.flag



class _SyncBoolProc(Process):
    """Synchronous Process that returns the bool supplied at init."""
    def __init__(self, val: Any) -> None:
        self._val = val

    def forward(self) -> Any:  # noqa: D401
        return self._val


class _AsyncBoolProc(AsyncProcess):
    """AsyncProcess counterpart of _SyncBoolProc."""
    val: Any

    async def aforward(self) -> Any:  # noqa: D401
        return self.val


class _ImmediateTask(Task):
    """Task that immediately finishes with a fixed status."""
    def __init__(self, status: TaskStatus) -> None:
        super().__init__()
        self._status_to_return = status

    async def tick(self) -> TaskStatus:  # noqa: D401
        return self._status_to_return


