from abc import ABC, abstractmethod
from ..behavior import Server

import time
from enum import Enum

class AgentStatus(Enum):

    READY = "ready"
    PAUSED = "paused"
    RUNNING = "running"
    STOPPED = "stopped"
    WAITING = "waiting"


class Agent(ABC):

    def __init__(self, server: Server=None, interval: float=None):
        super().__init__()
        self._server = server or Server()
        self._status = AgentStatus.READY
        self._interval = interval

    @property
    def server(self) -> Server:
        return self._server

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
        
        while self._status != AgentStatus.STOPPED:
            # possible race condition here
            # status should not be set in "act"
            status = self.act()
            if self._status == AgentStatus.RUNNING:
                self._status = status
            if self._interval is not None:
                time.sleep(self._interval)
