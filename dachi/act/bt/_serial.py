# 1st party
import typing as t

# local
from ._core import Task, TaskStatus, CompositeTask, TASK
from dachi.core import Runtime, ModuleList, Runtime, PrivateRuntime
from ._leafs import CONDITION

from pydantic import Field
import pydantic
from typing import Literal


class SerialTask(CompositeTask[TASK], t.Generic[TASK]):
    """A task consisting of other tasks executed one 
    after the other
    """
    _cascaded: Runtime[bool] = PrivateRuntime(default=False)

    @property
    def cascaded(self) -> bool:
        return self._cascaded.data

    def cascade(self, cascaded: bool = True):
        """Set whether the task is cascaded or not

        Args:
            cascaded (bool, optional): Whether the task is cascaded or not. Defaults to True.
        """
        self._cascaded.data = cascaded


class SequenceTask(SerialTask[TASK]):
    """Create a sequence of tasks to execute
    """
    tasks: ModuleList[TASK] = Field(
        default_factory=list,
        description="The tasks to run in sequence"
    )
    _idx: Runtime[int] = PrivateRuntime(default=0)

    @pydantic.field_validator('tasks', mode='before')
    def task_validator(cls, v):
        return ModuleList[TASK](vals=[*v])

    def sub_tasks(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task
        
        Returns:
            ModuleList: The sub-tasks
        """
        if self.tasks is not None:
            yield from self.tasks

    async def update_status(self) -> TaskStatus:
        """Update the status of the task based on the current task status
        
        Returns:
            TaskStatus: The status of the task
        """
        if self.status.is_done:
            return self.status
        
        status = self.tasks[self._idx.data].status
        if status == TaskStatus.SUCCESS:
            self._idx.data += 1
            if self._idx.data >= len(self.tasks):
                self._status.set(TaskStatus.SUCCESS)
            else:
                self._status.set(TaskStatus.RUNNING)
        elif status == TaskStatus.FAILURE:
            self._status.set(TaskStatus.FAILURE)
        else:
            self._status.set(TaskStatus.RUNNING)
            
        return self.status

    async def tick(self, ctx) -> TaskStatus:
        """Update the task with context support

        Args:
            ctx: Context for data flow and input resolution

        Returns:
            TaskStatus: The status
        """
        if self.status.is_done:
            return self.status

        # Handle empty task list
        if len(self.tasks) == 0:
            self._status.set(TaskStatus.SUCCESS)
            return self.status

        if self.cascaded:
            for i, task in enumerate(self.tasks[self._idx.data:], self._idx.data):
                child_ctx = ctx.child(i)
                status = await task.tick(ctx=child_ctx)
                
                await self.update_status()
                if task.status.running or self.status.is_done:
                    break
        else:
            task = self.tasks[self._idx.data]
            child_ctx = ctx.child(self._idx.data)
            status = await task.tick(ctx=child_ctx)
            await self.update_status()
        
        return self.status
    
    def reset(self):
        
        super().reset()
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()

        self._idx.data = 0

    def update_loop(self) -> t.Iterator[TASK]:
        """Get the sub-tasks of the composite task

        Yields:
            t.Iterator[Task]: The sub-tasks
        """
        yield from self.tasks


class SelectorTask(SerialTask[TASK]):
    """Create a set of tasks to select from
    """
    tasks: ModuleList[TASK] = Field(
        default_factory=ModuleList,
        description="The tasks to select from"
    )
    _idx: Runtime[int] = PrivateRuntime(default=0)

    @pydantic.field_validator('tasks', mode='before')
    def task_validator(cls, v):
        return ModuleList[TASK](vals=[*v])

    def sub_tasks(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task
        
        Returns:
            ModuleList: The sub-tasks
        """
        if self.tasks is not None:
            yield from self.tasks

    async def update_status(self) -> TaskStatus:
        """Update the status of the task based on the current task status
        
        Returns:
            TaskStatus: The status of the task
        """
        if self.status.is_done:
            return self.status
        
        status = self.tasks[self._idx.data].status
        if status == TaskStatus.SUCCESS:
            self._status.set(TaskStatus.SUCCESS)
        elif status == TaskStatus.FAILURE:
            self._idx.data += 1
            if self._idx.data >= len(self.tasks):
                self._status.set(TaskStatus.FAILURE)
            else:
                self._status.set(TaskStatus.RUNNING)
        else:
            self._status.set(TaskStatus.RUNNING)

        return self.status

    async def tick(self, ctx) -> TaskStatus:
        """Update the task with context support
        
        Args:
            ctx: Context for data flow and input resolution
            
        Returns:
            TaskStatus: The status
        """
        if self.status.is_done:
            return self.status
        
        # Handle empty task list
        if len(self.tasks) == 0:
            self._status.set(TaskStatus.FAILURE)
            return self.status
        
        if self.cascaded:
            while self._idx.data < len(self.tasks) and not self.status.is_done:
                task = self.tasks[self._idx.data]
                child_ctx = ctx.child(self._idx.data)
                res = await task.tick(ctx=child_ctx)
                
                if isinstance(res, tuple):
                    status, outputs = res
                    child_ctx.update(outputs)
                
                await self.update_status()
                if task.status.running or self.status.is_done:
                    return self.status
        else:
            task = self.tasks[self._idx.data]
            child_ctx = ctx.child(self._idx.data)
            
            res = await task.tick(ctx=child_ctx)
            
            if isinstance(res, tuple):
                status, outputs = res
                child_ctx.update(outputs)
            
            await self.update_status()
        
        return self.status
    
    def reset(self):
        super().reset()
        self._idx.data = 0
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()

    def update_loop(self) -> t.Iterator[TASK]:
        """Get the sub-tasks of the composite task

        Yields:
            t.Iterator[Task]: The sub-tasks
        """
        yield from self.tasks


FallbackTask = SelectorTask


class PreemptCond(SerialTask[TASK], t.Generic[TASK, CONDITION]):
    """Use to have a condition applied with
    each tick in order to stop the execution
    of other tasks
    """
    cond: CONDITION | None = None
    task: TASK | None = None

    _cascaded: Literal[True] = True

    def update_loop(self) -> t.Iterator[TASK]:
        """Get the sub-tasks of the composite task

        Yields:
            t.Iterator[Task]: The sub-tasks
        """
        yield self.cond
        if self.cond.status.success:
            yield self.task

    def sub_tasks(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task

        Returns:
            ModuleList: The sub-tasks
        """
        yield self.cond
        yield self.task

    async def update_status(self) -> TaskStatus:
        """Update the status of the task based on the condition and task status

        Returns:
            TaskStatus: The status of the task
        """
        if self.cond.status.success:
            self._status.set(self.task.status)
        else:
            self._status.set(TaskStatus.FAILURE)
        return self.status

    async def tick(self, ctx) -> TaskStatus:
        """

        Args:
            reset (bool, optional): . Defaults to False.

        Returns:
            TaskStatus: 
        """
        if isinstance(self.cond, ModuleList):
            status = TaskStatus.SUCCESS
            for cond in self.cond:
                cond.reset()
                status = await cond.tick(ctx) & status
        else:
            status = await self.cond.tick(ctx)
        
        if status.failure:
            self._status.set(
                TaskStatus.FAILURE
            )
        
        else:
            self._status.set(
                await self.task.tick(ctx)
            )
        return self.status
    
    def reset(self):
        self.cond.reset()
        self.task.reset()
