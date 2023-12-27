from abc import ABC

from dachi.agents import Agent, AgentStatus
from dachi.comm import Server, gen_refs
from .queries import UIQuery, LLMQuery
from dachi import behavior
from .tasks import lesson, planner, base
from .comm import IOHandler


class LanguageTeacher(Agent):

    def __init__(self, server: Server=None, interval: float=None):

        super().__init__()
        self._server = server or Server()
        self._root_terminal = server.terminal()

        self._status = AgentStatus.READY
        self._interval = interval
        self._io = IOHandler(self._server, 'Bot')

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

        # set teh ser
        # Make naming optional

        llm_query = LLMQuery()
        ui_query = UIQuery(self.io.backend_callback)
        plan, convo, request, ai_message = (
            gen_refs(['plan', 'convo', 'request', 'ai_message'])
        )
        quiz_prompt = lesson.QUIZ_PROMPT
        plan_prompt = plan.PLAN_PROMPT
        with behavior.sango() as language_teacher:
            with behavior.select(language_teacher) as teach:
                teach.add(lesson.Complete())
                with behavior.sequence(teach) as message:
                    # # Consider something like this to make it clearer
                    # message << (
                    #     behavior.CheckReady('plan'),
                    #     lesson.StartLesson('plan', 'convo'),
                    # )

                    message.add([
                        behavior.CheckReady(plan),
                        base.PreparePrompt(convo, quiz_prompt, components={'plan': plan}),
                        base.AIConvMessage(convo, request, llm_query),
                        base.Display(),
                        lesson.ProcessAIMessage(request, ai_message, convo),
                        base.UIConvMessage(convo, ui_query)
                    ])
                with behavior.select(language_teacher) as plan:
                    plan.add([
                        base.Display(),
                        base.UIConvMessage(convo, ui_query),
                        base.PreparePrompt(convo, quiz_prompt, components={'plan': plan}),
                        planner.CreatePlan('plan', 'ai', 'plan_convo', LLMQuery(server))
                    ])
                # teach.add(planner.PlanGenerator(...))
        self._behavior = language_teacher.build()
        self._terminal = self._server.register(self._behavior)

    @property
    def io(self):
        return self._io

    def act(self) -> AgentStatus:
        
        sango_status = self._behavior.tick(self._root_terminal)
        return AgentStatus.from_status(sango_status)

    def reset(self):
        self._behavior.reset_status(self._terminal)
