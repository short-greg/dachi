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
        
        status = await self.root.tick(scope.ctx())
        self._status.set(status)
        return status

    def reset(self):
        super().reset()
        if self.root is not None:
            self.root.reset()


DeepBT = BT[
    SequenceTask[TASK] | MultiTask[TASK] | SelectorTask[TASK] | TASK
]
