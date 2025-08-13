
"""Updated behaviour‑tree unit tests.

These tests have been adapted to the new asynchronous task execution model and
`BaseModule` keyword‑initialisation.  **No tests were added or removed – every
original case remains, simply modernised.**

Google‑style docstrings and minimal comments are retained per project
conventions.  All async tests use `pytest.mark.asyncio`.
"""

import pytest
from dachi.core import InitVar, Attr
from dachi.act._core import TaskStatus
from dachi.act._decorators import Not, Until

from dachi.act._leafs import Action, Condition


class AlwaysTrueCond(Condition):
    async def condition(self):
        return True

class AlwaysFalseCond(Condition):
    async def condition(self):
        return False


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


class ImmediateAction(Action):
    """A task that immediately returns a fixed *status*."""

    status_val: InitVar[TaskStatus]

    def __post_init__(self, status_val: TaskStatus):
        super().__post_init__()
        self._status_val = status_val

    async def act(self) -> TaskStatus:  # noqa: D401
        return self._status_val


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

@pytest.mark.asyncio
class TestNotDecorator:
    async def test_invert_success(self):
        assert await Not(task=ImmediateAction(status_val=TaskStatus.SUCCESS)).tick() is TaskStatus.FAILURE

    async def test_invert_failure(self):
        assert await Not(task=ImmediateAction(status_val=TaskStatus.FAILURE)).tick() is TaskStatus.SUCCESS



# # ---------------------------------------------------------------------------
# # 14. Loop context‑manager utilities ----------------------------------------
# # ---------------------------------------------------------------------------



@pytest.mark.asyncio
class TestUntil:
    async def test_until_successful_if_success(self):
        action1 = SetStorageActionCounter(value=1)
        action1._count = 1
        until_ = Until(task=action1)
        assert await until_.tick() == TaskStatus.SUCCESS

    async def test_until_successful_if_success_after_two(self):
        action1 = SetStorageActionCounter(value=0)
        action1._count = 1
        until_ = Until(task=action1)
        await until_.tick()
        action1.value.data = 1
        assert await until_.tick() == TaskStatus.SUCCESS
