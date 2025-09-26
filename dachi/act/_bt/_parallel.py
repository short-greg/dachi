# 1st party
import typing as t
import time
import asyncio

# local
from ._core import Task, TaskStatus, Composite
from dachi.core import Attr, ModuleList


class Parallel(Composite):
    """A task that runs multiple tasks in parallel
    """
    def update_loop(self) -> t.Iterator[Task]:
        yield from self.sub_tasks()


class Multi(Parallel):
    """A composite task for running multiple tasks in parallel
    """

    tasks: ModuleList
    fails_on: int=1
    succeeds_on: int=-1
    success_priority: bool = True
    preempt: bool = False
    auto_run: bool = True

    def __post_init__(self):
        """Create a parallel task
        
        Args:   
            tasks (t.List[Task], optional): The tasks to run in parallel. Defaults to None.
            fails_on (int, optional): The number of tasks that must fail for the parallel task to fail. Defaults to 1.
            succeeds_on (int, optional): The number of tasks that must succeed for the parallel task to succeed. Defaults to -1.
            success_priority (bool, optional): Whether to prioritize success over failure. Defaults to True.
            preempt (bool, optional): Whether to preempt the task if a condition is met. Defaults to False.
            auto_run (bool, optional): Whether to automatically run the task. Defaults to True.
        """
        super().__post_init__()
        self._status = Attr[TaskStatus](data=TaskStatus.READY)
        self.validate()

    def sub_tasks(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task
        
        Returns:
            ModuleList: The sub-tasks
        """
        if self.tasks is not None:
            yield from self.tasks

    def validate(self):
        """Validate the number of tasks required to succeed and fail
        Raises:
            ValueError: If the number of tasks is less than the number of fails or succeeds
            ValueError: If the number of fails or succeeds is less than 0
        """
        if self.fails_on < 0:
            fails_on = len(self.tasks) + self.fails_on + 1
        else:
            fails_on = self.fails_on
        if self.succeeds_on < 0:
            succeeds_on = len(self.tasks) + self.succeeds_on + 1
        else:
            succeeds_on = self.succeeds_on

        if (
            fails_on + succeeds_on - 1
        ) > len(self.tasks):
            raise ValueError(
                'The number of tasks required to succeed or fail is greater than the number of tasks'
            )
        if fails_on <= 0 or succeeds_on <= 0:
            raise ValueError(
                'The number of fails or succeeds '
                'must be greater than 0'
            )
        
    async def run(self, task: Task, ctx) -> TaskStatus:
        """Run the task until it is done

        Args:
            task (Task): The task to run

        Returns:
            TaskStatus: The status of the task after running
        """
        if task.status.is_done:
            return task.status
        
        status = await task.tick(ctx)
        return status
    
    async def update_status(self):
        
        statuses = [
            task.status for task in self.tasks
        ]
        failures = sum(
            1 if status.failure else 0 for status in statuses
        )
        successes = sum(
            1 if status.success else 0 for status in statuses
        )

        # Handle negative threshold values
        if self.fails_on < 0:
            fails_on = len(self.tasks) + self.fails_on + 1
        else:
            fails_on = self.fails_on
        if self.succeeds_on < 0:
            succeeds_on = len(self.tasks) + self.succeeds_on + 1
        else:
            succeeds_on = self.succeeds_on

        # Check if all tasks are done
        all_done = (failures + successes) >= len(statuses)
        
        # Check termination conditions 
        if (
            successes >= succeeds_on and
            failures >= fails_on
        ):
            self._status.set(TaskStatus.from_bool(
                self.success_priority
            ))
        elif successes >= succeeds_on:
            self._status.set(TaskStatus.SUCCESS)
        elif failures >= fails_on and (self.preempt or all_done):
            self._status.set(TaskStatus.FAILURE)
        elif all_done:
            # All tasks done but no termination condition met
            self._status.set(TaskStatus.FAILURE)
        else:
            self._status.set(TaskStatus.RUNNING)
        return self.status
    
    async def tick(self, ctx) -> TaskStatus:
        """Execute the Parallel task
        It runs all tasks in parallel and waits for the response. Once all tasks are done, it checks the results and sets the status according to the values set by
        fails_on and succeeds_on.

        Returns:
            TaskStatus: The status of the task after running
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:
            
            for i, task in enumerate(self.tasks):
                if task.status.is_done:
                    tasks.append(task.status)
                else:
                    tasks.append(
                        tg.create_task(
                            self.run(task, ctx.child(i))
                        )
                    )

        return await self.update_status()

    def reset(self):
        super().reset()
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()


