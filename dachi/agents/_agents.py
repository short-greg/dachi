from ..behavior import Sango
from ..graph import Tako
from ..behavior import Server
from ._base import AgentStatus, Agent


class SangoAgent(Agent):

    def __init__(self, sango: Sango, interval: float=1./60):
        """

        Args:
            sango (Sango): 
            interval (float, optional): . Defaults to 1./60.
        """
        super().__init__(
            Server(), interval
        )
        self._sango = sango

    def act(self) -> AgentStatus:
        if self._status == AgentStatus.RUNNING:
            status = self._sango.tick()
            return AgentStatus.from_sango(status)
        return self._status
