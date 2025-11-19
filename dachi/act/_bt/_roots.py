# 1st party
import typing as t
from dataclasses import InitVar
from dachi.core import Ctx, Scope, Runtime, PrivateRuntime
from ._core import TASK
from ._serial import Sequence, Selector
from ._parallel import MultiTask
# local
from ._core import Task, TaskStatus


class BT(Task, t.Generic[TASK]):
    """The root task for a behavior tree
    """
    root: TASK | None = None
    bindings: t.Dict[str, str] | None = None
    _scope: Runtime[Scope] = PrivateRuntime(default_factory=Scope)

    # # TDOO: Find out what "method task" is
    # def task(
    #     self, 
    #     name: str, 
    #     *args, 
    #     **kwargs
    # ) -> Task:
    #     """Get a task by name

    #     Args:
    #         name (str): The name of the task
    #         *args: The arguments to pass to the task
    #         **kwargs: The keyword arguments to pass to the task

    #     Returns:
    #         Task: The task with the given name
    #     """
    #     if name not in self.__method_tasks__:
    #         raise ValueError(f"Task {name} not found")
    #     return self.__method_tasks__[name].create(
    #         name=name, 
    #         args=args, 
    #         kwargs=kwargs,
    #         f=self.__method_tasks__[name],
    #     )
    
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
    Sequence[TASK] | MultiTask[TASK] | Selector[TASK] | TASK
]


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

