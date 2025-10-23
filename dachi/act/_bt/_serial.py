# 1st party
from abc import abstractmethod
import typing as t
from dataclasses import InitVar

# local
from ._core import Task, TaskStatus, Composite
from dachi.core import Attr, ModuleList
from ._leafs import Condition


class Serial(Composite):
    """A task consisting of other tasks executed one 
    after the other
    """

    def __post_init__(self):
        super().__post_init__()
        self._cascaded = Attr[bool](data=False)

    @property
    def cascaded(self) -> bool:
        return self._cascaded.data

    def cascade(self, cascaded: bool = True):
        """Set whether the task is cascaded or not

        Args:
            cascaded (bool, optional): Whether the task is cascaded or not. Defaults to True.
        """
        self._cascaded.data = cascaded


class Sequence(Serial):
    """Create a sequence of tasks to execute
    """

    tasks: ModuleList[Task] | None = None

    def __post_init__(self):
        """Initialize the Sequence. A Sequence will execute its tasks in order. If one fails then it will Fail. If cascaded is True, it will try to execute all tasks in order until one fails

        Raises:
            ValueError: If tasks is not a list of Task objects.
        """
        super().__post_init__()
        if self.tasks is None:
            self.tasks = ModuleList(items=[])
        elif isinstance(self.tasks, t.List):
            self.tasks = ModuleList(items=self.tasks)
        if self.tasks is not None and not isinstance(self.tasks, ModuleList):
            raise ValueError(
                f"Tasks must be of type ModuleList not {type(self.tasks)}"
            )
        self._idx = Attr[int](data=0)

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


class Selector(Serial):
    """Create a set of tasks to select from
    """
    tasks: ModuleList[Task] | None = None

    def __post_init__(self):
        """Initialize the selector. A Selector will try to execute
        each task in order until one succeeds. If none succeed, the
        selector fails. If cascaded is True, it will try to execute
        all tasks in order until one succeeds or all fail.

        Raises:
            ValueError: If tasks is not a list of Task objects.
        """
        super().__post_init__()
        if self.tasks is None:
            self.tasks = ModuleList(items=[])
        elif isinstance(self.tasks, t.List):
            self.tasks = ModuleList(items=self.tasks)
        if self.tasks is not None and not isinstance(self.tasks, ModuleList):
            raise ValueError(
                f"Tasks must be of type ModuleList not {type(self.tasks)}"
            )
        self._idx = Attr[int](data=0)

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


Fallback = Selector


class PreemptCond(Serial):
    """Use to have a condition applied with
    each tick in order to stop the execution
    of other tasks
    """
    cond: Condition | ModuleList
    task: Task

    def __post_init__(self):
        """Initialize the PreemptCond. A PreemptCond will first evaluate the condition. If the condition succeeds, it will execute the task. If the condition fails, it will fail the PreemptCond.

        Raises:
            ValueError: If cond is not a Condition or task is not a Task.
        """
        super().__post_init__()
        if isinstance(self.cond, t.List):
            self.cond = ModuleList(items=self.cond)
        
        self.cascade(cascaded=True)
        if not isinstance(self.cond, Condition):
            raise ValueError(
                f"Condition must be of type Condition not {type(self.cond)}"
            )
        if not isinstance(self.task, Task):
            raise ValueError(
                f"Task must be of type Task not {type(self.task)}"
            )
        self._status.set(TaskStatus.RUNNING)

    def update_loop(self) -> t.Iterator[Task]:
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
