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

    # @classmethod
    # def restricted_schema(
    #     cls, 
    #     *, 
    #     tasks=None, 
    #     _profile="shared", 
    #     _seen=None, 
    #     **kwargs
    # ):
    #     """
    #     Generate restricted schema for Decorator.

    #     Pattern C: Single Field - processes task variants for the task field.

    #     Args:
    #         tasks: List of allowed task variants for task field
    #         _profile: "shared" (default) or "inline"
    #         _seen: Cycle detection dict
    #         **kwargs: Passed to nested restricted_schema() calls

    #     Returns:
    #         Restricted schema dict with task field limited to specified variants
    #     """
    #     # If no tasks provided, return unrestricted schema
    #     if tasks is None:
    #         return cls.schema()

    #     # Process task variants (handles RestrictedTaskSchemaMixin recursion)
    #     task_schemas = cls._schema_process_variants(
    #         tasks,
    #         restricted_schema_cls=RestrictedTaskSchemaMixin,
    #         _seen=_seen,
    #         tasks=tasks,
    #         **kwargs
    #     )

    #     # Update schema's task field (single Task)
    #     schema = cls.schema()
    #     return cls._schema_update_single_field(
    #         schema,
    #         field_name="task",
    #         placeholder_name="TaskSpec",
    #         variant_schemas=task_schemas,
    #         profile=_profile
    #     )


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

    @classmethod
    def restricted_schema(cls, *, tasks=None, _profile="shared", _seen=None, **kwargs):
        """
        Generate restricted schema for BoundTask.

        Pattern C with Leaf filter - processes task variants for the leaf field,
        but only allows Leaf subclasses.

        Args:
            tasks: List of allowed Leaf variants for leaf field
            _profile: "shared" (default) or "inline"
            _seen: Cycle detection dict
            **kwargs: Passed to nested restricted_schema() calls

        Returns:
            Restricted schema dict with leaf field limited to specified Leaf variants
        """
        # If no tasks provided, return unrestricted schema
        if tasks is None:
            return cls.schema()

        # Filter to only Leaf subclasses
        leaf_variants = list(filter_class_variants(LeafTask, tasks))

        # If no valid Leaf variants, return unrestricted schema
        if not leaf_variants:
            return cls.schema()

        # Process leaf variants (handles RestrictedTaskSchemaMixin recursion)
        leaf_schemas = cls._schema_process_variants(
            leaf_variants,
            restricted_schema_cls=RestrictedTaskSchemaMixin,
            _seen=_seen,
            tasks=tasks,
            **kwargs
        )

        # Update schema's leaf field (single Leaf)
        schema = cls.schema()
        return cls._schema_update_single_field(
            schema,
            field_name="leaf",
            placeholder_name="LeafSpec",
            variant_schemas=leaf_schemas,
            profile=_profile
        )

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
