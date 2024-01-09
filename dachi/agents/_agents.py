from ..behavior import Sango
from ._base import AgentStatus, Agent


class SangoAgent(Agent):
    """An agent that makes 
    """

    def __init__(self, sango: Sango, interval: float=1./60):
        """
        Args:
            sango (Sango): The behavior tree
            interval (float, optional): The interval to repeat actions at. Defaults to 1./60.
        """
        super().__init__(
            interval
        )
        self._sango = sango

    def act(self) -> AgentStatus:
        """
        Returns:
            AgentStatus: The current status of the agent
        """
        if self._status == AgentStatus.RUNNING:
            status = self._sango.tick()
            return AgentStatus.from_sango(status)
        return self._status
