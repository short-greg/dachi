# 1st party
import typing as t
from dachi.core import Runtime, PrivateRuntime
from dachi.act.comm import Scope, Ctx
from ._core import TASK
from ._serial import SequenceTask, SelectorTask
from ._parallel import MultiTask
# local
from ._core import Task, TaskStatus


class BT(Task, t.Generic[TASK]):
    """The root task for a behavior tree
    """
    root: TASK | None = None
    bindings: t.Dict[str, str] | None = None
    _scope: Runtime[Scope] = PrivateRuntime(default_factory=Scope)
    ROOT_IDX: int = 0
    
    async def tick(self, ctx: Ctx | None=None) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status after tick
        """
        if self._adapted is None:
            return TaskStatus.SUCCESS
        
        if ctx is None:
            scope = self.scope.data
        else:
            scope = self.scope.data.bind(ctx, self.bindings)
        
        status = await self.root.tick(scope.ctx(0))
        self._status.set(status)
        return status

    def reset(self):
        super().reset()
        if self.root is not None:
            self.root.reset()

    def set_base_ctx(self, key: str, value: t.Any, _skip_serialize: bool=False) -> None:
        """Set a value in the chart's scope context.

        Args:
            key: Context key
            value: Value to set
        """
        ctx = self._scope.get().ctx()

        return ctx.set(key, value, skip_serialize=_skip_serialize)

    def get_base_ctx(self, key: str, default=None) -> t.Any:
        """Get a value from the chart's scope context.

        Args:
            key: Context key

        Returns:
            The value associated with the key
        """
        ctx = self._scope.get().ctx()
        return ctx.get(key, default)

    @property
    def scope(self) -> Scope:
        """Get the scope of the behavior tree.

        Returns:
            Scope: The scope
        """
        return self._scope.get()


DeepBT = BT[
    SequenceTask[TASK] | MultiTask[TASK] | SelectorTask[TASK] | TASK
]
