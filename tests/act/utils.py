from dachi.core import Runtime, PrivateRuntime
from dachi.act.comm import Scope
from dachi.act.bt._core import TaskStatus, Task

from typing import Any
from dachi.act.bt._leafs import Action, Condition, WaitCondition
from dachi.proc import Process, AsyncProcess


class _ToggleTask(Task):
    """RUNNING ▸ SUCCESS over two ticks; counts invocations."""

    _calls: Runtime[int] = PrivateRuntime(0)

    async def tick(self, ctx):
        self._calls.set(self._calls.get() + 1)
        if self._calls.get() == 1:
            self._status.set(TaskStatus.RUNNING)
        else:
            self._status.set(TaskStatus.SUCCESS)
        return self.status


class _BoolProc(Process):
    """Process returning the value supplied at construction."""

    val: Any

    def forward(self): return self.val


class _ABoolProc(AsyncProcess):
    """Async variant of _BoolProc."""
    val: Any

    async def aforward(self): return self.val


class AlwaysTrueCond(Condition):
    async def execute(self):
        return True

class AlwaysFalseCond(Condition):
    async def execute(self):
        return False


class ImmediateAction(Action):
    """A task that immediately returns a preset *status_val*."""

    status_val: TaskStatus = TaskStatus.SUCCESS
    async def execute(self) -> TaskStatus:
        return self.status_val


class ATask(Action):
    """Always succeeds – used to stub out generic actions."""
    x: int = 1

    async def execute(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.SUCCESS


class SetStorageAction(Action):
    """Action whose success/failure depends on *value*."""

    value: int = 4

    async def execute(self) -> TaskStatus:  # noqa: D401
        return TaskStatus.FAILURE if self.value < 0 else TaskStatus.SUCCESS


class SampleCondition(Condition):
    """Condition – true if *x* is non‑negative."""

    x: int = 1

    async def execute(self) -> bool:  # noqa: D401
        return self.x >= 0


class SetStorageActionCounter(Action):
    """Counts invocations – succeeds on the 2nd tick unless *value* == 0."""

    # __store__ = ["value"]
    value: int = 4
    _count: Runtime[int] = PrivateRuntime(0)
    
    async def execute(self) -> TaskStatus:  # noqa: D401
        if self.value == 0:
            return TaskStatus.FAILURE
        self._count.set(self._count.get() + 1)
        if self._count.get() == 2:
            return TaskStatus.SUCCESS
        if self._count.get() < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING


class FlagWaitCond(WaitCondition):
    """WaitCondition whose outcome is controlled via the *flag* attribute."""

    flag: bool = True

    async def execute(self) -> bool:
        return self.flag



class _SyncBoolProc(Process):
    """Synchronous Process that returns the bool supplied at init."""

    val: Any

    def forward(self) -> Any:  # noqa: D401
        return self.val


class _AsyncBoolProc(AsyncProcess):
    """AsyncProcess counterpart of _SyncBoolProc."""
    val: Any

    async def aforward(self) -> Any:  # noqa: D401
        return self.val


class _ImmediateTask(Task):
    """Task that immediately finishes with a fixed status."""
    status_to_return: TaskStatus


    async def tick(self, ctx) -> TaskStatus:  # noqa: D401
        return self.status_to_return

def create_test_ctx():
    """Helper to create test context"""
    scope = Scope()
    return scope.ctx()


