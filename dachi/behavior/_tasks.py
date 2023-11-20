from abc import abstractmethod
import typing

from ._cooordination import Terminal, Message
from ._status import Status
from ._base import Behavior

# How to deal with waiting
# How to continue running


class Task(Behavior):

    def _tick_wrapper(self, f) -> typing.Callable[[Terminal], Status]:

        def _tick(terminal: Terminal, *args, **kwargs) -> Status:
            if not terminal.initialized:
                self.__init_terminal__(terminal)
                terminal.initialize()

            return f(terminal, *args, **kwargs)
        return _tick

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.tick = self._tick_wrapper(self.tick)

    def __init_terminal__(self, terminal: Terminal):
        terminal.storage.add('status', Status.READY)

    @property
    def name(self) -> str:

        return self._name

    @abstractmethod    
    def tick(self, terminal: Terminal) -> Status:
        pass

    def clone(self) -> 'Task':
        return self.__class__(self.name)

    def state_dict(self) -> typing.Dict:

        return {
            'id': self._id,
            'name': self._name
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        self._id = state_dict['id']
        self._name = state_dict['name']

    def __call__(self, terminal: Terminal) -> Status:
    
        return self.tick(terminal)

    def reset(self, terminal):

        terminal.reset()
        self.__init_terminal__(terminal)

    def receive(self, message: Message):
        pass

    @property
    def id(self):
        return self._id


class Sango(Behavior):

    def __init__(self, name: str, root: 'Task'=None):

        super().__init__(name)
        self._root = root

    def tick(self, terminal: Terminal) -> Status:

        if self._root is None:
            return Status.SUCCESS
    
        return self._root.tick(terminal.child(self._root))

    @property
    def root(self):
        return self._root
    
    @root.setter
    def root(self, root: 'Task'):
        self._root = root

    def state_dict(self) -> typing.Dict:

        return {
            'root': self._root.state_dict() if self._root is not None else None,
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        super().load_state_dict(state_dict)
        if self._root is not None:
            self._root.load_state_dict(state_dict['root'])
        else:
            if state_dict['root'] is not None:
                raise ValueError('Cannot load state dict because root is none')


class Composite(Task):
    """Task composed of subtasks
    """
    def __init__(
        self, tasks: typing.List[Task], name: str=''
    ):
        super().__init__(name)
        self._tasks = tasks

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.add('idx', 0)

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return [*self._tasks]

    @abstractmethod
    def subtick(self, terminal: Terminal) -> Status:
        """Tick each subtask. Implement when implementing a new Composite task"""
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> Status:
        status = self.subtick(terminal)
        terminal.storage['status'] = status
        return status
    
    def reset(self):
        super().reset()
        for task in self._tasks:
            task.reset()

    def state_dict(self) -> typing.Dict:

        return {
            'tasks': [
                task.state_dict()
                for task in self._tasks
            ],
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict: typing.Dict):

        super().load_state_dict(state_dict)
        for task, task_state_dict in zip(self._tasks, state_dict['tasks']):
            task.load_state_dict(task_state_dict)


class Sequence(Composite):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> Status:
        
        idx = terminal.storage['idx']
        if terminal.storage['status'].is_done:
            return terminal.storage['status']
    
        task = self._tasks[idx]
        status = task.tick(terminal.child(task))

        if status == Status.FAILURE:
            return Status.FAILURE
        
        if status == Status.SUCCESS:
            terminal.storage['idx'] += 1
            status = Status.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = Status.SUCCESS

        return status

    def clone(self) -> 'Sequence':
        return Sequence(
            self.name, [task.clone() for task in self._tasks]
        )


class Fallback(Composite):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> Status:
        
        idx = terminal.storage['idx']
        if terminal.storage['status'].is_done:
            return terminal.storage['status']
    
        task = self._tasks[idx]
        status = task.tick(terminal.child(task))

        print(status)
        if status == Status.SUCCESS:
            return Status.SUCCESS
        
        if status == Status.FAILURE:
            terminal.storage['idx'] += 1
            status = Status.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = Status.FAILURE

        return status

    def clone(self) -> 'Fallback':
        return Fallback(
            self.name, [task.clone() for task in self._tasks]
        )


class Parallel(Composite):

    def __init__(self, name: str, tasks: typing.List[Task], runner=None, fails_on: int=1, succeeds_on: int=None, success_priority: bool=True):
        super().__init__(name, tasks)

        self._use_default_runner = runner is None
        self._runner = runner or self._default_runner
        self.set_condition(fails_on, succeeds_on)
        self._success_priority = success_priority

    def add(self, task: Task):
        self._tasks.append(task)

    def set_condition(self, fails_on: int, succeeds_on: int):
        self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        self._succeeds_on = succeeds_on if succeeds_on is not None else len(self._tasks)

        if (self._fails_on + self._succeeds_on - 1) >= len(self._tasks):
            raise ValueError('')
        if self._fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('')
    
    def _default_runner(self, tasks: typing.List[Task], terminal: Terminal) -> typing.List[Status]:
        
        statuses = []
        for task in tasks:
            statuses.append(task.tick(terminal.child(task)))

        return statuses

    def _accumulate(self, statuses: typing.List[Status]) -> Status:
        
        successes = 0
        failures = 0
        waiting = 0
        dones = 0
        for status in statuses:
            failures += status.failure
            successes += status.success

            dones += status.is_done
            waiting += status.waiting

        has_failed = failures >= self._fails_on
        has_succeeded = successes >= self._succeeds_on
        if self._success_priority:
            if has_succeeded:
                return Status.SUCCESS
            if has_failed:
                return Status.FAILURE

        if has_failed:
            return Status.FAILURE
        if has_succeeded:
            return Status.SUCCESS
        if waiting == (len(statuses) - dones):
            return Status.WAITING
        # failures + successes - 1
        return Status.RUNNING

    def subtick(self, terminal: Terminal) -> Status:

        statuses = self._runner(self._tasks, terminal)
        return self._accumulate(statuses)
            
    def clone(self) -> 'Parallel':
        return Parallel(
            self.name, [task.clone() for task in self._tasks],
            self._runner if not self._use_default_runner else None,
            fails_on=self._fails_on, succeeds_on=self._succeeds_on,
            success_priority=self._success_priority
        )


class Action(Task):

    @abstractmethod
    def act(self, terminal: Terminal):
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> Status:

        if terminal.storage['status'].is_done:
            return terminal.storage['status']
        status = self.act(terminal)
        terminal.storage[status] = status
        return status


class Condition(Task):

    @abstractmethod
    def condition(self, terminal: Terminal) -> bool:
        pass

    def tick(self, terminal: Terminal) -> Status:
        
        if self.condition(terminal):
            return Status.SUCCESS
        return Status.FAILURE
