from abc import ABC

from dachi.agents import Agent, AgentStatus
from dachi.behavior import Server
from dachi import behavior
from .tasks import planner, prompter, interface
from .comm import IOHandler


class LanguageTeacher(Agent):

    def __init__(self, server: Server=None, interval: float=None):

        super().__init__()
        self._server = server or Server()
        self._status = AgentStatus.READY
        self._interval = interval
        self._io = IOHandler(self._server, 'Bot')

        llm_response_signal = 'llm_response'
        prompt_name = 'prompt'
        output_name = 'message'
        input_name = 'input_name'
        plan_name = 'plan'
        self._io.register_input(input_name)
        with behavior.sango('Language Teacher') as language_teacher:
            with behavior.select('Teach', language_teacher) as teach:
                teach.add(prompter.PromptLLM(prompt_name, llm_response_signal))
                with behavior.sequence('Output', teach) as message:
                    message.add(interface.OutputWaiting(output_name))
                    message.add(interface.OutputMessage(output_name, self._io))
                teach.add(planner.PlanLearning(input_name, plan_name, output_name))
        self._behavior = language_teacher.build()
        self._terminal = self._server.register(self._behavior)

    @property
    def io(self):
        return self._io

    def act(self) -> AgentStatus:
        
        sango_status = self._behavior.tick(self._terminal)
        return AgentStatus.from_status(sango_status)

    def reset(self):
        self._behavior.reset_status(self._terminal)
