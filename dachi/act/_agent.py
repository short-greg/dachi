from ._tasks import TaskStatus, Task
import threading
import typing

import time
from enum import Enum
from . import Root


class AgentStatus(Enum):

    READY = "ready"
    PAUSED = "paused"
    RUNNING = "running"
    STOPPED = "stopped"
    WAITING = "waiting"


class Agent(object):
    """Agent class
    """

    def __init__(
        self, sango: Root=None, interval: float=None
    ):
        """

        Args:
            sango (Sango, optional): The default sango to use if act() is not overridden. Defaults to None.
            interval (float, optional): The interval to repeat actions at. Defaults to None.
        """
        super().__init__()
        self._task_status = None
        self._status = None
        self._sango = sango
        self._interval = interval
        self._queued = []

    def act(self) -> TaskStatus:
        """Execute the agent

        Returns:
            TaskStatus: The resulting status
        """
        if self._sango is None:
            return TaskStatus.SUCCESS
        return self._sango.tick()

    def pause(self):
        self._status = AgentStatus.PAUSED

    def reset(self):
        self._status = AgentStatus.READY

    def stop(self):
        self._status = AgentStatus.STOPPED

    def run(self, continue_on_fail: bool) -> typing.Iterator[TaskStatus]:
        """Execute the agent at the specified interval
        """
        while self._status != AgentStatus.STOPPED:
            self._task_status = self.act()
            if self._interval is not None:
                time.sleep(self._interval)
            if (
                (not continue_on_fail and self._task_status == TaskStatus.FAILURE)
            ):
                self._status = AgentStatus.STOPPED
            yield self._task_status
