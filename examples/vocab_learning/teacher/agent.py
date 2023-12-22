from abc import ABC

from dachi.agents import Agent, AgentStatus
from dachi.comm import Server
from .queries import UIQuery, LLMQuery
from dachi import behavior
from .tasks import lesson, planner, base
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

        # llm_name 
        # ui_output_name
        # 1) needs to create a plan
        # 2) 
        # plan_request = '' # A request to create a plan
        # plan_llm_request = '' # 
        # ui_output_request = ''
        # with fallback as teach
        #     with sequence as lesson
        #         Check(lambda terminal: terminal.storage['plan'] is not None)
        #         with fallback as section:
        #             Quiz()
        #             Explain()
        #         UpdateLesson() # sets the plan to none
        #     PlanLesson()

        # UICallback

        self._io.register_input(input_name)
        with behavior.sango('Language Teacher', server) as language_teacher:
            with behavior.select('Teach', language_teacher) as teach:
                teach.add(lesson.Complete('completed'))
                with behavior.sequence('Quiz', teach) as message:
                    # # Consider something like this to make it clearer
                    # message << (
                    #     behavior.CheckReady('plan'),
                    #     lesson.StartLesson('plan', 'convo'),
                    # )
                    message.add(behavior.CheckReady('plan'))
                    message.add(lesson.StartLesson('plan', 'convo'))
                    message.add(lesson.QuizUser('ai', 'convo', LLMQuery(server)))
                    message.add(lesson.UserConversationResponse('convo', 'ai', 'user', UIQuery(...)))
                with behavior.select('Plan', language_teacher) as plan:
                    message.add(planner.StartPlanning('plan_convo'))
                    plan.add(base.UserConversationResponse('plan_convo', 'plan_ai', 'planuser', UIQuery(...)))
                    plan.add(planner.CreatePlan('plan', 'ai', 'plan_convo', LLMQuery(server)))
                # teach.add(planner.PlanGenerator(...))
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
