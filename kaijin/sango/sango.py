from enum import Enum
from abc import abstractmethod, ABC, abstractproperty
import typing


class Request(Enum):
    
    INPUT = 'input'


class Status(Enum):

    RUNNING = 'running'
    WAITING = 'waiting'
    SUCCESS = 'success'
    FAILURE = 'failure'
    READY = 'ready'

    def is_done(self) -> bool:
        return self == Status.FAILURE or self == Status.SUCCESS


class Control(object):

    def __init__(self):
        pass

        


class Comm(object):

    def send(self, request: Request, name: str, data, f=None):
        pass

    def receive(self, request: Request, name: str, f):
        pass

    def get(self, key: str) -> typing.Any:
        pass

    def unset(self, key: str) -> typing.Any:
        pass

    def set(self, key: str, value: typing.Any) -> typing.Any:
        pass

    def clear(self):
        pass


class Task(ABC):

    def __init__(self, name: str, comm: Comm) -> None:
        super().__init__()
        self._name = name
        self._comm = comm
        self._status = Status.READY

    @property
    def name(self) -> str:

        return self._name
    
    @property
    def status(self) -> Status:
        return self._status

    @property
    def comm(self) -> Comm:
        return self._comm

    @abstractmethod    
    def tick(self) -> Status:
        pass

    @abstractmethod
    def clone(self) -> 'Task':
        pass

    def state_dict(self) -> typing.Dict:

        return {
            'name': self._name
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        self._name = state_dict['name']
        self._status = state_dict['status']


class CompositeExecutor(object):

    @abstractmethod
    def reset(self, items: list=None):
        """Reset the planner to the start state

        Args:
            items (list, optional): Updated items to set the planner to. Defaults to None.
        """
        pass

    @abstractproperty
    def idx(self):
        pass
    
    @idx.setter
    def idx(self, idx):
        pass

    @abstractmethod
    def end(self) -> bool:
        """Advance the planner

        Returns:
            bool
        """
        pass

    @abstractmethod
    def adv(self) -> bool:
        """Advance the planner

        Returns:
            bool
        """
        pass
    
    @abstractproperty
    def cur(self) -> Task:
        """
        Returns:
            Task
        """
        pass

    @abstractmethod
    def rev(self) -> bool:
        """Reverse the planner

        Returns:
            bool
        """
        pass

    @abstractmethod
    def clone(self):
        """Clone the linear planner

        Returns:
            LinearPlanner
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass


class Composite(Task):
    """Task composed of subtasks
    """

    def __init__(
        self, tasks: typing.List[Task], name: str='', executor: CompositeExecutor=None
    ):
        super().__init__(name)
        self._executor = executor or LinearExecutor(tasks)
        self._tasks = tasks

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return [*self._tasks]

    @abstractmethod
    def subtick(self) -> Status:
        """Tick each subtask. Implement when implementing a new Composite task"""
        raise NotImplementedError

    def tick(self):
        # TODO: pass in Comm or not?
        # I think it is better
        # 

        status = self.subtick()
        self._status = status
        return status
    
    def reset(self):
        super().reset()
        self._executor.reset()
        for task in self._tasks:
            task.reset()

    def state_dict(self) -> typing.Dict:

        return {
            'tasks': [
                task.state_dict()
                for task in self._tasks
            ],
            'executor': self._executor.state_dict(),
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        super().load_state_dict(state_dict)
        for task, task_state_dict in zip(self._tasks, state_dict['tasks']):
            task.load_state_dict(task_state_dict)
        self._executor.load_state_dict(state_dict['executor'])

    # def iterate(self, filter: Filter=None, deep: bool=True):
    #     filter = filter or NullFilter()    
    #     for task in self._tasks:
    #         if filter.check(task):
    #             yield task
    #             if deep:
    #                 for subtask in task.iterate(filter, deep):
    #                     yield subtask


class Sequence(Composite):

    def __init__(self, name: str, comm: Comm, tasks: typing.List[Task]=None):
        super().__init__(name, comm)
        self._tasks = tasks or []
        self._order = []
        self._order = LinearExecutor()

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def tick(self) -> Status:
        
        exeuctor: CompositeExecutor = self._comm.get('executor')
        if exeuctor is None:
            self._comm.set('executor', self._executor(self._tasks))

        status = exeuctor.cur.tick()
        if status == Status.FAILURE:
            return Status.FAILURE
        
        if status == Status.SUCCESS:
            exeuctor.adv()
        return exeuctor.cur.STATUS

    def clone(self) -> 'Task':
        return Sequence(
            self.name, [task.clone() for task in self._tasks]
        )


class LinearExecutor(CompositeExecutor):

    def __init__(self, items: typing.List[Task]):
        """initializer

        Args:
            items (typing.List[Task])
        """
        self._items = items
        self._idx = 0

    def reset(self, items: list=None):
        """Reset the planner to the start state

        Args:
            items (list, optional): Updated items to set the planner to. Defaults to None.
        """
        self._idx = 0
        self._items = items # coalesce(items, self._items)
    
    @property
    def idx(self):
        return self._idx
    
    @idx.setter
    def idx(self, idx):
        if not (0 <= idx <= len(self._items)):
            raise IndexError(f"Index {idx} is out of range in {len(self._items)}")
        self._idx = idx

    def end(self) -> bool:
        """Advance the planner

        Returns:
            bool
        """
        if self._idx == len(self._items):
            return True
        return False
    
    def adv(self) -> bool:
        """Advance the planner

        Returns:
            bool
        """
        if self._idx == len(self._items):
            return False
        self._idx += 1
        return True
    
    @property
    def cur(self) -> Task:
        """
        Returns:
            Task
        """
        if self._idx == len(self._items):
            return None
        return self._items[self._idx]

    def rev(self) -> bool:
        """Reverse the planner

        Returns:
            bool
        """
        if self._idx == 0:
            return False
        self._idx -= 1
        return True

    def clone(self):
        """Clone the linear planner

        Returns:
            LinearPlanner
        """
        return LinearExecutor([*self._items])
    
    def __len__(self) -> int:
        return len(self._items)


class Action(Task):

    @abstractmethod
    def act(self):
        raise NotImplementedError

    def tick(self):

        if self._status.is_done():
            return self._status
        self._status = self.act()
        return self._status


class Condition(Task):

    @abstractmethod
    def condition(self) -> bool:
        pass

    def tick(self) -> Status:
        
        if self.condition():
            return Status.SUCCESS
        return Status.FAILURE
