# 1st party
import typing as t
import asyncio
import pydantic

# local
from ._core import Task, TaskStatus, CompositeTask, TASK
from dachi.core import ModuleList, PrivateRuntime, Runtime


class ParallelTask(CompositeTask[TASK]):
    """A task that runs multiple tasks in parallel
    """
    def update_loop(self) -> t.Iterator[TASK]:
        yield from self.sub_tasks()


class MultiTask(ParallelTask[TASK], t.Generic[TASK]):
    """A composite task for running multiple tasks in parallel
    """

    tasks: ModuleList[TASK] = pydantic.Field(
        default_factory=ModuleList,
        description="The tasks to run in parallel"
    )
    fails_on: int=1
    succeeds_on: int=-1
    success_priority: bool = True
    preempt: bool = False
    auto_run: bool = True

    _status: Runtime[TaskStatus] = PrivateRuntime(TaskStatus.READY)

    # @pydantic.field_validator('tasks', mode='before')
    # def validate_tasks(cls, v):
    #     """Validate and convert tasks to ModuleList

    #     Args:
    #         v: The tasks input (list or ModuleList)

    #     Returns:
    #         ModuleList: The tasks as a ModuleList with correct generic type
    #     """
    #     return ModuleList[TASK](vals=[*v])
    
    @pydantic.field_validator('tasks', mode='before')
    def validate_regions(cls, v):
        """Validate and convert regions to ModuleList

        Args:
            v: The regions input (list, ModuleList)

        Returns:
            ModuleList[TASK]: The regions as a ModuleList
        """
        # Accept any ModuleList regardless of type parameter
        # Accept ModuleList and convert

        # get the annotation args for the generic for ModuleList 
        
        base_state = cls.model_fields['tasks'].annotation.__pydantic_generic_metadata__['args'][0]

        if isinstance(v, list):
            converted = ModuleList[base_state](vals=v)
            return converted
        if isinstance(v, ModuleList):
            converted = ModuleList[base_state](vals=v.vals)
            return converted

        return v

    def sub_tasks(self) -> t.Iterator[TASK]:
        """Get the sub-tasks of the composite task
        
        Returns:
            ModuleList: The sub-tasks
        """
        if self.tasks is not None:
            yield from self.tasks

    @pydantic.model_validator(mode='after')
    def validate_thresholds(self):
        """Validate that the thresholds are not zero and within bounds

        Raises:
            ValueError: If the thresholds are invalid

        Returns:
            Self: The validated instance
        """
        fails_on = self.fails_on
        if fails_on < 0:
            fails_on = len(self.tasks) + fails_on + 1

        succeeds_on = self.succeeds_on
        if succeeds_on < 0:
            succeeds_on = len(self.tasks) + succeeds_on + 1

        if (fails_on + succeeds_on - 1) > len(self.tasks):
            raise ValueError(
                'The number of tasks required to succeed or fail is greater than the number of tasks'
            )
        if fails_on <= 0 or succeeds_on <= 0:
            raise ValueError(
                'The number of fails or succeeds must be greater than 0'
            )
        return self

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
