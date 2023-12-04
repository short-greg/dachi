from abc import ABC

from dachi.agents import Agent, AgentStatus
from dachi.behavior import Server
from dachi import behavior


class DummyTask(behavior.Action):

    pass


class LanguageTeacher(Agent):

    def __init__(self, server: Server=None, interval: float=None):
        super().__init__()
        self._server = server or Server()
        self._status = AgentStatus.READY
        self._interval = interval

        with behavior.sango('Teach') as language_teacher:
            with behavior.select(language_teacher) as teach:
                with behavior.sequence(teach) as llm_handler:
                    pass
                with behavior.sequence(teach) as response:
                    pass
                with behavior.sequence(teach) as message:
                    pass
                with behavior.sequence(teach) as plan:
                    pass
        self._behavior = language_teacher.build()
        self._terminal = self._server.register(self._behavior)

    def act(self) -> AgentStatus:
        
        sango_status = self.behavior.tick(self._terminal)
        return AgentStatus.from_status(sango_status)

