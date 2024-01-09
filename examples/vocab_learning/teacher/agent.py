from abc import ABC

from dachi.agents import Agent, AgentStatus
from dachi.comm import Server, refer
from .queries import UIQuery, LLMQuery
from dachi import behavior
from .tasks import lesson, planner, base
from .comm import IOHandler

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

# self._io.register_input(input_name)

# set teh ser
# Make naming optional


class LanguageTeacher(Agent):

    def __init__(self, server: Server=None, interval: float=None):

        super().__init__()
        self._server = server or Server()
        self._root_terminal = server.terminal()

        self._status = AgentStatus.READY
        self._interval = interval
        self._io = IOHandler(self._server, 'Bot')

        llm_query = LLMQuery()
        ui_query = UIQuery(self.io.backend_callback)

        # refs = RefGroup([])

        convo, request, ai_message, = (
            refer(
                ['convo', 'request', 'ai_message'])
        )

        plan, plan_message, plan_conv, plan_request = refer(
            ['plan', 'plan_message', 'plan_conv', 'plan_request']
        )
        quiz_prompt = lesson.QUIZ_PROMPT
        plan_prompt = plan.PLAN_PROMPT
        with behavior.sango() as language_teacher:
            with behavior.select(language_teacher) as teach:
                # can make these two trees
                with behavior.sequence(teach) as message:
                    message.add([
                        behavior.CheckReady(plan),
                        base.PreparePrompt(
                            convo, quiz_prompt, components={'plan': plan}
                        ),
                        base.AIConvMessage(convo, request, llm_query),
                        lesson.ProcessAIMessage(request, ai_message, convo),
                        base.Display(ai_message),
                        base.UIConvMessage(convo, ui_query)
                    ])
                with behavior.select(language_teacher) as plan:
                    plan.add([
                        base.Display(plan_message, self.io),
                        base.UIConvMessage(plan_conv, ui_query),
                        base.PreparePrompt(plan_conv, plan_prompt),
                        base.AIConvMessage(plan_conv, plan_request, llm_query),
                        planner.CreateAIPlan(plan_request, plan, plan_conv)
                    ])
        self._behavior = language_teacher.build()
        self._terminal = self._server.register(self._behavior)

    @property
    def io(self):
        return self._io

    def act(self) -> AgentStatus:
        
        sango_status = self._behavior.tick(self._root_terminal)
        return AgentStatus.from_status(sango_status)

    def reset(self):
        self._behavior.reset_status(self._root_terminal)
