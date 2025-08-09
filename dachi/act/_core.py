# 1st party
from enum import Enum
from abc import abstractmethod
import typing as t

# local
from ..core import Attr, BaseModule
from ..proc import Process
# TODO: Add in Action (For GOAP)
from abc import ABC
import typing as t


class TaskStatus(Enum):
    """Status of a Behavior Tree task
    """
    # The task is currently in progress
    RUNNING = 'running'
    # The task is currently waiting for a response
    WAITING = 'waiting'
    # The task was successful
    SUCCESS = 'success'
    # The task failed
    FAILURE = 'failure'
    # The task has not been executed
    READY = 'ready'

    @property
    def is_done(self) -> bool:
        """Get whether the task has succeeded or failed

        Returns:
            bool: Whether the task is "done"
        """
        return self == TaskStatus.FAILURE or self == TaskStatus.SUCCESS
    
    @property
    def in_progress(self) -> bool:
        """Get whether the task is in progress

        Returns:
            bool: whether the task is still running
        """
        return self == TaskStatus.RUNNING or self == TaskStatus.WAITING
    
    @property
    def ready(self) -> bool:
        """Get whether the task is ready to be executed

        Returns:
            bool: Whether the task is ready to be executed
        """
        return self == TaskStatus.READY
    
    @property
    def failure(self) -> bool:
        """Get whether the task has failed

        Returns:
            bool: whether the task has failed
        """
        return self == TaskStatus.FAILURE
    
    @property
    def success(self) -> bool:
        """Get whether the task 

        Returns:
            bool: Whether the task has succeeded
        """
        return self == TaskStatus.SUCCESS
    
    @property
    def running(self) -> bool:
        """Get whether the task is running

        Returns:
            bool: Whether the task is running or not
        """
        return self == TaskStatus.RUNNING
    
    @property
    def waiting(self) -> bool:
        """Get whether the task is waiting

        Returns:
            bool: Whether the task is waiting or not
        """
        return self == TaskStatus.WAITING

    @classmethod
    def from_bool(cls, val: bool) -> 'TaskStatus':
        """Convert a boolean to a TaskStatus

        Args:
            val (bool): The value to convert

        Returns:
            TaskStatus: The status
        """
        return TaskStatus.SUCCESS if val is True else TaskStatus.FAILURE
    
    def __or__(self, other: 'TaskStatus') -> 'TaskStatus':
        """Compute the union of two statuses

        Returns:
            SangoStatus: The resulting status. Returns success if one status
            is success.
        """
        if self == other:
            return self
        
        if (
            (self.success or other.success)
        ):
            return TaskStatus.SUCCESS
        if self.running or other.running:
            return TaskStatus.RUNNING
        
        if self.waiting or other.waiting:
            return TaskStatus.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.failure and other.failure):
            return self
        
        raise ValueError(f'Invalid combination of statuses {self} and {other}')

    def __and__(self, other: 'TaskStatus') -> 'TaskStatus':
        """Compute the intersection of two statuses

        Returns:
            SangoStatus: The resulting status. Returns failure if one status
            is failure.
        """

        if self == other:
            return self
        
        if (
            (self.failure or other.failure)
        ):
            return TaskStatus.FAILURE
        if self.running or other.running:
            return TaskStatus.RUNNING
        
        if self.waiting or other.waiting:
            return TaskStatus.WAITING
        
        if (self.success and not other.success):
            return other
        if (not self.success and other.success):
            return self
        raise ValueError(f'Invalid combination of statuses {self} and {other}')

    def invert(self) -> 'TaskStatus':

        if self.success:
            return TaskStatus.FAILURE
        if self.failure:
            return TaskStatus.SUCCESS
        return self


WAITING = TaskStatus.WAITING
READY = TaskStatus.READY
SUCCESS = TaskStatus.SUCCESS
FAILURE = TaskStatus.FAILURE
RUNNING = TaskStatus.RUNNING


class Task(BaseModule):
    """The base class for a task in the behavior tree
    """
    SUCCESS: t.ClassVar[TaskStatus] = TaskStatus.SUCCESS
    FAILURE: t.ClassVar[TaskStatus] = TaskStatus.FAILURE
    READY: t.ClassVar[TaskStatus] = TaskStatus.READY
    RUNNING: t.ClassVar[TaskStatus] = TaskStatus.RUNNING

    def __post_init__(self):
        """Initialize the task
        """
        super().__post_init__()
        self._status = Attr(data=self.READY)
        self._id = id(self)

    @abstractmethod    
    async def tick(self) -> TaskStatus:
        raise NotImplementedError

    def sync_tick(self) -> TaskStatus:
        raise NotImplementedError

    async def __call__(self) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The status of the task after execution
        """
        return await self.tick()
    
    @property
    def status(self) -> TaskStatus:
        """The status

        Returns:
            TaskStatus: 
        """
        return self._status.data

    @property
    def id(self):
        """Get the id for the task

        Returns:
            The id of the task 
        """

        return self._id
    
    def reset(self):

        self._status.set(TaskStatus.READY)


class Composite(Task):
    """A task that is composed of other tasks
    """

    @abstractmethod
    def update_loop(self) -> t.Iterator[Task]:
        """Get the current sub-task of the composite task

        Yields:
            Task: Each current sub-task
        """
        pass

    @abstractmethod
    def sub_tasks(self) -> t.Iterator[Task]:
        """Get the sub-tasks of the composite task

        Yields:
            Task: Each current sub-task
        """
        pass

    @abstractmethod
    async def update_status(self) -> TaskStatus:
        pass


class Leaf(Task):
    """A task that is composed of other tasks
    """
    pass


# class FuncTask(Task):
#     """A task that executes a function
#     """
#     name: str
#     args: t.List[t.Any]
#     kwargs: t.Dict[str, t.Any]

#     def __post_init__(self):
#         super().__post_init__()
#         self.obj = None

#     async def func_tick(self) -> TaskStatus:
#         """Execute the function

#         Returns:
#             TaskStatus: The status after executing the function
#         """
#         pass

#     async def tick(self) -> TaskStatus:
#         """Execute the task

#         Returns:
#             TaskStatus: The status after executing the task
#         """
#         if self.status.is_done:
#             return self.status
        
#         if self.obj is None:
#             raise ValueError(
#                 "Task object is not set. "
#                 "Please set the object before calling tick."
#             )

#         status = await self.func_tick()
#         self._status.set(status)
#         return status
    
#     def reset(self):
#         """Reset the task
#         """
#         super().reset()
#         self._task = None

    # TODO: Think how to handle specifications
    
    

    # def from_spec(self, spec: t.Dict[str, t.Any]) -> 'FuncTask':
    #     """Create a FuncTask from a specification

    #     Args:
    #         spec (dict): The specification for the task

    #     Returns:
    #         FuncTask: The created task
    #     """
    #     raise RuntimeError(
    #         "FuncTask cannot be created from a specification. "
    #     )
    
    # def spec(self, *, to_dict = False):
    #     """Get the specification for the task

    #     Args:
    #         to_dict (bool): Whether to return the specification as a dict

    #     Returns:
    #         dict: The specification for the task
    #     """
    #     raise RuntimeError(
    #         "FuncTask cannot be converted to a specification. "
    #         "Check your object "
    #     )


class FTask(Task):
    """A task that executes a function
    """
    name: str
    args: t.List[t.Any]
    kwargs: t.Dict[str, t.Any]

    def __post_init__(self):
        """Initialize the FTask"""
        super().__post_init__()
        self.obj = None
        self._task = None

    async def tick(self) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The status after executing the task
        """

        if self.status.is_done:
            return self.status
        
        if self.obj is None:
            raise ValueError(
                "Task object is not set. "
                "Please set the object before calling tick."
            )

        status = await self.func_tick()
        self._status.set(status)
        return status
    
    def reset(self):
        """Reset the task
        """
        super().reset()
        self._task = None



class ToStatus(Process):
    """Use to convert a value to a status
    """

    @abstractmethod
    def __call__(self, val) -> TaskStatus:
        """Convert the value to a status

        Args:
            val: The value to convert

        Returns:
            TaskStatus: The status
        """
        pass


TOSTATUS = ToStatus | t.Callable[[t.Any], TaskStatus]


def from_bool(status: bool) -> TaskStatus:
    """functional code to map from a boolean

    Args:
        status (bool): The status in boolean form

    Returns:
        TaskStatus: The task status
    """
    return TaskStatus.from_bool(status)


class State(BaseModule):
    """Use State creating a state machine
    SubClasses must return a literal 
            string or type TaskStatus
    """
    @abstractmethod
    async def update(self) -> None:
        """Update the state

        Returns:
            # SubClasses must return a literal 
            of type TaskStatus or 
            None
        """
        pass

    async def __call__(self, reset: bool=False):
        return await self.update(reset)


STATE_CALL = State | t.Callable[[], State | TaskStatus]

class Router(Process, ABC):
    """Use to route a value to a Task
    """

    @abstractmethod
    def forward(self, val) -> TaskStatus | State:
        pass


ROUTE = Router | t.Callable[[t.Any], TaskStatus | State]

