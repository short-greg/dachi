# 1st party
from abc import abstractmethod
import typing as t

# local
from ._core import Task, TaskStatus, CompositeTask, LeafTask, RestrictedTaskSchemaMixin, TASK, LEAF
from dachi.core._base import filter_class_variants


class Decorator(CompositeTask, t.Generic[TASK]):
    """A task that decorates another task
    """

    task: TASK

    def update_loop(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task

        Returns:
            ModuleList: The sub-tasks
        """
        yield self.task

    def sub_tasks(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task

        Returns:
            ModuleList: The sub-tasks
        """
        yield self.task

    async def update_status(self) -> TaskStatus:
        if self.status.is_done:
            return self.status
        self._status.set(await self.decorate(
            self.task.status
        ))
        return self.status

    @abstractmethod
    async def decorate(self, status: TaskStatus, reset: bool=False) -> bool:
        pass

    async def tick(self, ctx) -> TaskStatus:
        """Decorate the tick for the decorated task

        Args:
            ctx: Context for data flow and input resolution

        Returns:
            TaskStatus: The decorated status
        """
        res = await self.task.tick(ctx=ctx)
        await self.update_status()
        return self.status


class Until(Decorator):
    """Loop until a condition is met
    """

    target_status: TaskStatus = TaskStatus.SUCCESS

    async def decorate(
        self, 
        status: TaskStatus
    ) -> TaskStatus:
        """Continue running unless the result is a success

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            return status
        if status.is_done:
            self.task.reset()
        return TaskStatus.RUNNING


class AsLongAs(Decorator):
    """Loop while a condition is met
    """
    target_status: TaskStatus = TaskStatus.SUCCESS

    async def decorate(
        self, status: TaskStatus
    ) -> TaskStatus:
        """Continue running unless the result is a failure

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            if status.is_done:
                self.task.reset()
        elif status.is_done:
            return status
        return TaskStatus.RUNNING


class Not(Decorator):
    """Invert the result
    """

    async def decorate(
        self, 
        status: TaskStatus
    ) -> TaskStatus:
        """Return Success if status is a Failure or Failure if it is a SUCCESS

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        return status.invert()


class BoundTask(Task, t.Generic[LEAF]):
    """Bind will map variables in the context to
    the inputs of the decorated task
    """

    leaf: LEAF
    bindings: t.Dict[str, str]

    async def tick(self, ctx) -> TaskStatus:
        """Tick the task and bind outputs to context

        Args:
            ctx: Context for data flow and input resolution

        Returns:
            Union[TaskStatus, Tuple[TaskStatus, dict]]: The status and outputs
        """
        if self.status.is_done:
            return self.status
        ctx = ctx.bind(self.bindings)
        result = await self.leaf.tick(ctx)
        self._status.set(result)
        return self.status
