# 1st party
import typing as t
from dataclasses import InitVar
from dachi.core import Ctx, Scope

# local
# from dachi.core import AdaptModule
from ._core import Task, TaskStatus
from dachi.core._base import RestrictedSchemaMixin


class BT(Task, RestrictedSchemaMixin):
    """The root task for a behavior tree
    """
    root: InitVar[Task | None] = None
    bindings: t.Dict[str, str] | None = None

    def __post_init__(self, root: Task):
        """Create a behavior tree task
        Args:
            root (Task | None): The root task for the behavior tree
        """
        super().__post_init__()
        Task.__post_init__(self)
        self.root = root
        self.scope = Scope()
        self.__method_tasks__ = {}
        for name in dir(self):
            # Get the attribute from the instance
            attr = getattr(self, name)
            # Check if it's a bound method of this instance
            if (
                callable(attr)
                and hasattr(attr, "__self__")
                and attr.__self__ is self
                and getattr(attr, "is_task", False) is True
            ):
                self.__method_tasks__[name] = attr
    
    def task(
        self, 
        name: str, 
        *args, 
        **kwargs
    ) -> Task:
        """Get a task by name

        Args:
            name (str): The name of the task
            *args: The arguments to pass to the task
            **kwargs: The keyword arguments to pass to the task

        Returns:
            Task: The task with the given name
        """
        if name not in self.__method_tasks__:
            raise ValueError(f"Task {name} not found")
        return self.__method_tasks__[name].create(
            name=name, 
            args=args, 
            kwargs=kwargs,
            f=self.__method_tasks__[name],
        )
    
    async def tick(self, ctx: Ctx | None=None) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status after tick
        """
        if self._adapted is None:
            return TaskStatus.SUCCESS
        
        if ctx is None:
            scope = self.scope
        else:
            scope = self.scope.bind(ctx, self.bindings)
        
        status = await self.root.tick(scope.ctx())
        self._status.set(status)
        return status

    def reset(self):
        super().reset()
        if self.root is not None:
            self.root.reset()

    def restricted_schema(self, *, _profile = "shared", _seen = None, tasks: Task | None = None, **kwargs) -> t.Dict:

        options = []
        if self.root is None:
            raise RuntimeError(
                "BT root task is not set "
            )

        for task in tasks:
            if isinstance(task, RestrictedSchemaMixin):
                options.append(task.restricted_schema(
                    _profile=_profile,
                    _seen=_seen,
                    **kwargs
                ))
            else:
                options.append(
                    task.schema_dict()
                )

        

    


# TODO: Remove this at a later time. Later we will add a State Chart
# class StateMachine(AdaptModule, Task):
#     """StateMachine is a task composed of multiple tasks in a directed graph
#     """

#     def __post_init__(self):
#         """
#         Initialize the state machine with an empty set of states and transitions.
#         """
#         super().__post_init__()
#         Task.__post_init__(self)
#         self.adapted: ModuleDict = ModuleDict(items={})
#         END_STATUS = t.Literal[TaskStatus.SUCCESS | TaskStatus.FAILURE]
#         self._transitions = Attr[t.Dict[
#             str, t.Dict[str | END_STATUS, str | END_STATUS]
#         ]](data={})
#         self._init_state = Attr[str | END_STATUS | None](data=None)
#         self._cur_state = Attr[str | END_STATUS | None](data=None)
#         self._states = ModuleDict(items={})
#         self.__states__ = {
#             name: method
#             for name, method in self.__class__.__dict__.items()
#             if callable(method) and getattr(method, "_is_state", False)
#         }
    
#     async def tick(self) -> TaskStatus:
#         """Update the state machine
#         """
#         if self.status.is_done:
#             return self.status

#         if self._cur_state.data is None:
#             self._cur_state.set(self._init_state.data)

#         if self._cur_state.data is None:
#             return TaskStatus.SUCCESS
        
#         if self._cur_state.data in {TaskStatus.SUCCESS, TaskStatus.FAILURE}:
#             return self._cur_state
        
#         state = self._states[self._cur_state.data]
#         if isinstance(state, str):
#             res = await self.__states__[state](self)
#         else:
#             res = await state.update()
#         if res == TaskStatus.RUNNING:
#             self._status.set(
#                 TaskStatus.RUNNING
#             )
#             return res
#         new_state = self._transitions.data[self._cur_state.data][res]
#         self._cur_state.set(new_state)

#         if self._cur_state.data in {TaskStatus.SUCCESS, TaskStatus.FAILURE}:
#             self._status.set(
#                 self._cur_state.data
#             )
#             return self._cur_state.data
        
#         self._status.set(
#             TaskStatus.RUNNING
#         )
#         return TaskStatus.RUNNING

#     def reset(self):
#         """Reset the state machine
#         """
#         super().reset()
#         self._cur_state.set(self._init_state.data)

#     @classmethod
#     def schema(
#         cls,
#         mapping: t.Mapping[type[BaseModule], t.Iterable[type[BaseModule]]] | None = None,
#     ):
#         if mapping is None:
#             return super().schema()
#         return cls._restricted_schema(mapping)


# def statefunc(func):
#     """Decorator to mark a function as a state for StateMachine."""
#     func._is_state = True
#     return func

