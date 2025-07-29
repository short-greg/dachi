import typing as t
from typing import Iterable
from ._core import Task, State, TaskStatus
from ..proc import Process, AsyncProcess
from dachi.core import ModuleDict, AdaptModule, BaseModule, Attr



class StateMachine(AdaptModule, Task):
    """StateMachine is a task composed of multiple tasks in a directed graph
    """

    def __post_init__(self):
        super().__post_init__()
        Task.__post_init__(self)
        self.adapted: ModuleDict = ModuleDict[State](data={})
        END_STATUS = t.Literal[TaskStatus.SUCCESS | TaskStatus.FAILURE]
        self._transitions = Attr[t.Dict[
            str, t.Dict[str | END_STATUS, str | END_STATUS]
        ]](data={})
        self._init_state = Attr[str | END_STATUS | None](data=None)
        self._cur_state = Attr[str | END_STATUS | None](data=None)
        self.__states__ = {
            name: method
            for name, method in self.__cls__.__dict__.items()
            if callable(method) and getattr(method, "_is_state", False)
        }
    
    async def tick(self) -> TaskStatus:
        """Update the state machine
        """
        if self.status.is_done:
            return self.status

        if self._cur_state is None:
            self._cur_state = self._init_state

        if self._cur_state is None:
            return TaskStatus.SUCCESS
        
        if self._cur_state in {TaskStatus.SUCCESS, TaskStatus.FAILURE}:
            return self._cur_state
        
        state = self._states[self._cur_state]
        if isinstance(state, str):
            res = await self.__states__[state]()
        else:
            res = await state.update()
        if res == TaskStatus.RUNNING:
            self._status.set(
                TaskStatus.RUNNING
            )
            return res
        
        new_state = self._transitions[self._cur_state][res]
        self._cur_state.set(new_state)

        if self._cur_state in {TaskStatus.SUCCESS, TaskStatus.FAILURE}:
            self._status.set(
                self._cur_state
            )
            return self._cur_state
        
        self._status.set(
            TaskStatus.RUNNING
        )
        return TaskStatus.RUNNING

    def reset(self):
        """Reset the state machine
        """
        super().reset()
        self._cur_state = self.init_state

    @classmethod
    def schema(
        cls,
        mapping: t.Mapping[type[BaseModule], Iterable[type[BaseModule]]] | None = None,
    ):
        if mapping is None:
            return super().schema()
        return cls._restricted_schema(mapping)


class BranchState(State):
    """Branch state has two branches, one for success and one for failure. It wraps a process that returns a boolean value
    """

    f: Process | AsyncProcess

    async def update(self) -> t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE]:
        """ Update the state by executing the wrapped process and returning the status based on its result.

        Returns:
            t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE]: The status of the branch state, either SUCCESS or FAILURE.
            If the wrapped process returns True, it will return SUCCESS, otherwise it will return FAILURE.
            If the wrapped process is an AsyncProcess, it will await the process before returning the status
        """

        if isinstance(self.f, AsyncProcess):
            if await self.f():
                return TaskStatus.SUCCESS
        else:
            if self.f():
                return TaskStatus.SUCCESS
        return TaskStatus.FAILURE


class TaskState(State):
    """Wraps a behavior tree task in a state
    """

    task: Task

    async def update(self) -> t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.RUNNING]:
        """ Update the state by executing the wrapped task and returning the status based on its result.

        Args:
            reset (bool, optional): Whether to reset the state. Defaults to False.

        Returns:
            t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.RUNNING]: The status of the task.
        """
        return await self.task.tick()


            # if filter_type is None or isinstance(obj, filter_type):
            #     fn(obj)
