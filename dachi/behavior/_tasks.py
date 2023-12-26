from abc import abstractmethod
import typing


from ..comm import Receiver, Terminal, Signal
from ._status import SangoStatus
from dataclasses import dataclass


@dataclass
class TaskMessage:

    name: str
    data: typing.Any


class Task(Receiver):

    SUCCESS = SangoStatus.SUCCESS
    FAILURE = SangoStatus.FAILURE
    RUNNING = SangoStatus.RUNNING

    def _tick_wrapper(self, f) -> typing.Callable[[Terminal], SangoStatus]:
        """Wraps the tick function so that the terminal will be initialized
        if not initialized

        Args:
            f (): The tick function

        Returns:
            typing.Callable[[Terminal], SangoStatus]: The wrapped tick function
        """
        def _tick(terminal: Terminal, *args, **kwargs) -> SangoStatus:
            if not terminal.initialized:
                self.__init_terminal__(terminal)
                terminal.initialize()

            if terminal.storage['status'].is_done():
                return terminal.storage['status']
            return f(terminal, *args, **kwargs)
        return _tick

    def __init__(self, name: str) -> None:
        """Create the task

        Args:
            name (str): The name of the task
        """
        super().__init__(name)
        self.tick = self._tick_wrapper(self.tick)

    def __init_terminal__(self, terminal: Terminal):
        """Initialize the terminal

        Args:
            terminal (Terminal): The terminal
        """
        terminal.storage['status'] = SangoStatus.READY

    @abstractmethod    
    def tick(self, terminal: Terminal) -> SangoStatus:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> 'Task':
        raise NotImplementedError

    def state_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The state dict for the task
        """
        return {
            'id': self._id,
            'name': self._name
        }
    
    def load_state_dict(self, state_dict: typing.Dict):
        """Load the state of the task 

        Args:
            state_dict (typing.Dict): The state
        """
        self._id = state_dict['id']
        self._name = state_dict['name']

    def __call__(self, terminal: Terminal) -> SangoStatus:
        """

        Args:
            terminal (Terminal): _description_

        Returns:
            SangoStatus: _description_
        """
        return self.tick(terminal)

    def reset(self, terminal):
        """Reset the terminal

        Args:
            terminal (_type_): The reset terminal and initialized
        """
        terminal.reset()
        self.__init_terminal__(terminal)

    def reset_status(self, terminal: Terminal):
        """Reset the status of the task

        Args:
            terminal (Terminal): The terminal
        """
        terminal.storage['status'] = SangoStatus.READY

    def receive(self, message: Signal):
        pass

    @property
    def id(self):
        return self._id


class Sango(Task):

    def __init__(self, name: str, root: 'Task'=None):
        """Create a tree to store the tasks

        Args:
            name (str): The name of the tree
            root (Task, optional): The root task. Defaults to None.
        """
        super().__init__(name)
        self._root = root

    def tick(self, terminal: Terminal) -> SangoStatus:
        """Update the task

        Args:
            terminal (Terminal): The terminal storing the task

        Returns:
            SangoStatus: The status after tick
        """
        if self._root is None:
            return SangoStatus.SUCCESS

        return self._root.tick(terminal.child(self._root))

    @property
    def root(self) -> Task:
        """
        Returns:
            Task: The root task
        """
        return self._root
    
    @root.setter
    def root(self, root: 'Task'):
        """
        Args:
            root (Task): The root task
        """
        self._root = root


    def state_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The state dict for the task
        """
        return {
            'root': self._root.state_dict() if self._root is not None else None,
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict: typing.Dict):
        """Load the state of the task 

        Args:
            state_dict (typing.Dict): The state
        """
        super().load_state_dict(state_dict)
        if self._root is not None:
            self._root.load_state_dict(state_dict['root'])
        else:
            if state_dict['root'] is not None:
                raise ValueError('Cannot load state dict because root is none')

    def reset_status(self, terminal: Terminal):
        """Reset the status of the task

        Args:
            terminal (Terminal): The terminal
        """
        super().reset_status(terminal)
        if self._root is not None:
            self._root.reset_status(terminal.child(self._root))

    def clone(self) -> 'Sango':
        """Clone the task

        Returns:
            Sango: The 
        """
        return Sango(self.name, self.root)


class Serial(Task):
    """Task composed of subtasks
    """
    def __init__(
        self, tasks: typing.List[Task], name: str=''
    ):
        super().__init__(name)
        self._tasks = tasks

    def __init_terminal__(self, terminal: Terminal):
        """Initialize the terminal

        Args:
            terminal (Terminal): 
        """
        super().__init_terminal__(terminal)
        terminal.storage.get_or_set('idx', 0)

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return [*self._tasks]

    @abstractmethod
    def subtick(self, terminal: Terminal) -> SangoStatus:
        """Tick each subtask. Implement when implementing a new Composite task"""
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> SangoStatus:
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

    def reset_status(self, terminal: Terminal):

        super().reset_status(terminal)
        for task in self.tasks:
            task.reset_status(terminal.child(task))


class Sequence(Serial):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> SangoStatus:
        
        idx = terminal.storage['idx']
        if terminal.storage['status'].is_done:
            return terminal.storage['status']
    
        task = self._tasks[idx]
        status = task.tick(terminal.child(task))

        if status == SangoStatus.FAILURE:
            return SangoStatus.FAILURE
        
        if status == SangoStatus.SUCCESS:
            terminal.storage['idx'] += 1
            status = SangoStatus.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = SangoStatus.SUCCESS

        return status

    def clone(self) -> 'Sequence':
        return Sequence(
            self.name, [task.clone() for task in self._tasks]
        )

    def reset_status(self, terminal: Terminal):
        super().reset_status(terminal)
        terminal.storage['idx'] = 0


class Sequence(Serial):

    def add(self, task: Task) -> 'Sequence':
        self._tasks.append(task)

    def subtick(self, terminal: Terminal) -> SangoStatus:
        
        idx = terminal.storage['idx']
        if terminal.storage['status'].is_done:
            return terminal.storage['status']
    
        task = self._tasks[idx]
        status = task.tick(terminal.child(task))

        if status == SangoStatus.FAILURE:
            return SangoStatus.FAILURE
        
        if status == SangoStatus.SUCCESS:
            terminal.storage['idx'] += 1
            status = SangoStatus.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = SangoStatus.SUCCESS

        return status

    def clone(self) -> 'Sequence':
        return Sequence(
            self.name, [task.clone() for task in self._tasks]
        )

    def reset_status(self, terminal: Terminal):
        super().reset_status(terminal)
        terminal.storage['idx'] = 0


class Selector(Serial):

    def add(self, task: Task) -> 'Sequence':
        """Add task to the selector

        Args:
            task (Task): 

        Returns:
            Sequence: The sequence added to
        """
        self._tasks.append(task)
        return self
    
    def subtick(self, terminal: Terminal) -> SangoStatus:
        
        idx = terminal.storage['idx']
        if terminal.storage['status'].is_done:
            return terminal.storage['status']
    
        task = self._tasks[idx]
        status = task.tick(terminal.child(task))

        if status == SangoStatus.SUCCESS:
            return SangoStatus.SUCCESS
        
        if status == SangoStatus.FAILURE:
            terminal.storage['idx'] += 1
            status = SangoStatus.RUNNING
        if terminal.storage['idx'] >= len(self._tasks):
            status = SangoStatus.FAILURE

        return status

    def clone(self) -> 'Selector':
        return Selector(
            self.name, [task.clone() for task in self._tasks]
        )

    def reset_status(self, terminal: Terminal):
        super().reset_status(terminal)
        terminal.storage['idx'] = 0
        

class Parallel(Task):
    """A composite task for running multiple tasks in parallel
    """

    def __init__(self, name: str, tasks: typing.List[Task], runner=None, fails_on: int=None, succeeds_on: int=None, success_priority: bool=True):
        super().__init__(name)
        self._tasks = tasks
        self._use_default_runner = runner is None
        self._runner = runner or self._default_runner
        self.set_condition(fails_on, succeeds_on)
        self._success_priority = success_priority

    def add(self, task: Task):
        self._tasks.append(task)

    def set_condition(self, fails_on: int, succeeds_on: int):
        """Set the number of falures or successes it takes to end

        Args:
            fails_on (int): The number of failures it takes to fail
            succeeds_on (int): The number of successes it takes to succeed

        Raises:
            ValueError: If teh number of successes or failures is invalid
        """
        self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        self._succeeds_on = succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)

        if (self._fails_on + self._succeeds_on - 1) > len(self._tasks):
            raise ValueError('')
        if self._fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('')
    
    def _default_runner(self, tasks: typing.List[Task], terminal: Terminal) -> typing.List[SangoStatus]:
        """Run the paralel

        Args:
            tasks (typing.List[Task]): The tasks to run
            terminal (Terminal): The terminal for the task

        Returns:
            typing.List[SangoStatus]: _description_
        """
        statuses = []
        for task in tasks:
            statuses.append(task.tick(terminal.child(task)))

        return statuses

    def _accumulate(self, statuses: typing.List[SangoStatus]) -> SangoStatus:
        
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
                return SangoStatus.SUCCESS
            if has_failed:
                return SangoStatus.FAILURE

        if has_failed:
            return SangoStatus.FAILURE
        if has_succeeded:
            return SangoStatus.SUCCESS
        if waiting == (len(statuses) - dones):
            return SangoStatus.WAITING
        # failures + successes - 1
        return SangoStatus.RUNNING

    def subtick(self, terminal: Terminal) -> SangoStatus:

        statuses = self._runner(self._tasks, terminal)
        return self._accumulate(statuses)

    @property
    def fails_on(self) -> int:
        return self._fails_on

    @property
    def succeeds_on(self) -> int:
        return self._succeeds_on        
    
    def clone(self) -> 'Parallel':
        return Parallel(
            self.name, [task.clone() for task in self._tasks],
            self._runner if not self._use_default_runner else None,
            fails_on=self._fails_on, succeeds_on=self._succeeds_on,
            success_priority=self._success_priority
        )

    def reset_status(self, terminal: Terminal):

        super().reset_status(terminal)
        for task in self.tasks:
            task.reset_status(terminal.child(task))


class Action(Task):

    @abstractmethod
    def act(self, terminal: Terminal) -> SangoStatus:
        raise NotImplementedError

    def tick(self, terminal: Terminal) -> SangoStatus:

        if terminal.storage['status'].is_done:
            return terminal.storage['status']
        status = self.act(terminal)
        terminal.storage[status] = status
        return status


class Condition(Task):
    """Check whether a condition is satisfied before
    running a task.
    """

    @abstractmethod
    def condition(self, terminal: Terminal) -> bool:
        pass

    def tick(self, terminal: Terminal) -> SangoStatus:
        """Check the condition

        Args:
            terminal (Terminal): The terminal 

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        if self.condition(terminal):
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE


class Decorator(Task):

    # name should retrieve the name of the decorated
    def __init__(self, task: Task) -> None:
        super().__init__('')
        self._task = task

    @abstractmethod
    def decorate(self, status: SangoStatus) -> bool:
        pass

    @property
    def task(self) -> Task:
        return self._task

    def tick(self, terminal: Terminal) -> SangoStatus:
        
        status: SangoStatus = terminal.storage.get('status')
        if status.is_done:
            return status

        status = terminal.storage['status'] = self.decorate(
            self.task.tick(terminal.child(self.task))
        )
        return status

    def reset_status(self, terminal: Terminal):

        super().reset_status(terminal)
        self._task.reset_status(terminal.child(self._task))


class Until(Decorator):
    """Loop until a condition is met
    """

    def decorate(self, terminal: Terminal, status: SangoStatus) -> SangoStatus:
        if status.success:
            return SangoStatus.SUCCESS
        if status.failure:
            return SangoStatus.RUNNING
        return status


class While(Decorator):
    """Loop while a condition is met
    """

    def decorate(self, terminal: Terminal, status: SangoStatus) -> SangoStatus:
        if status.failure:
            return SangoStatus.FAILURE
        if status.success:
            return SangoStatus.RUNNING
        return status


class Not(Decorator):
    """Invert the result
    """

    def decorate(self, terminal: Terminal, status: SangoStatus) -> SangoStatus:
        if status.failure:
            return SangoStatus.SUCCESS
        if status.success:
            return SangoStatus.FAILURE
        return status


class CheckReady(Condition):

    def __init__(self, name: str, field_name: str):
        """Check if a field has been prepared

        Args:
            name (str): The name of the task
            field_name (str): The name of the field to check
        """
        super().__init__(name)
        self.field_name = field_name

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.storage[self.field_name] is not None


class Check(Condition):

    def __init__(self, name: str, f):
        """Check if a field has been prepared

        Args:
            name (str): The name of the task
            f (typing.Callable): The function to call
        """
        super().__init__(name)
        self.f = f

    def condition(self, terminal: Terminal) -> bool:
        
        return self.f(terminal)


class CheckTrue(Condition):

    def __init__(self, name: str, field_name: str):
        """Check if a field is true

        Args:
            name (str): The name of the task
            field_name (str): The name of the field to check
        """
        super().__init__(name)
        self.field_name = field_name

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.storage[self.field_name]
