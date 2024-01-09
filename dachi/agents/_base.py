from abc import ABC, abstractmethod
from ..behavior import SangoStatus

import time
from enum import Enum

class AgentStatus(Enum):

    READY = "ready"
    PAUSED = "paused"
    RUNNING = "running"
    STOPPED = "stopped"
    WAITING = "waiting"

    @classmethod
    def from_status(cls, status: SangoStatus) -> 'AgentStatus':

        if status == SangoStatus.FAILURE or status == SangoStatus.SUCCESS:
            return AgentStatus.STOPPED
        if status == SangoStatus.READY:
            return AgentStatus.READY
        if status == SangoStatus.WAITING:
            return AgentStatus.WAITING
        return AgentStatus.RUNNING


class Agent(ABC):
    """Agent c;lass
    """

    def __init__(self, interval: float=None):
        """

        Args:
            server (Server, optional): The server the agent uses. Defaults to None.
            interval (float, optional): The interval to repeat actions at. Defaults to None.
        """
        super().__init__()
        self._status = AgentStatus.READY
        self._interval = interval

    @abstractmethod
    def act(self) -> AgentStatus:
        pass

    def pause(self):
        self._status = AgentStatus.PAUSED

    def reset(self):
        self._status = AgentStatus.READY

    def stop(self):
        self._status = AgentStatus.STOPPED

    def start(self):
        """Execute the agent at the specified interval
        """
        
        while self._status != AgentStatus.STOPPED:
            # possible race condition here
            # status should not be set in "act"
            status = self.act()
            if self._status == AgentStatus.RUNNING:
                self._status = status
            if self._interval is not None:
                time.sleep(self._interval)
