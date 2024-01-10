from abc import ABC

from dachi.agents import Agent, AgentStatus
from .queries import UIQuery, LLMQuery
from dachi import behavior, struct
from .tasks import lesson, planner, base
from .comm import UIInterface


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

    def __init__(self, ui_interface: UIInterface, interval: float=None):

        super().__init__()

        self._status = AgentStatus.READY
        self._interval = interval
        self._plan_conv = planner.PlanConv()
        self._user_conv = lesson.QuizConv()

        llm_query = LLMQuery()
        ui_query = UIQuery(ui_interface)

        # Set the first messaeg of plan conv
        # self._user_conv.set_system(plan=self._plan_conv.plan)
        with behavior.sango() as language_teacher:
            with behavior.select(language_teacher) as teach:
                # can make these two trees
                with behavior.sequence(teach) as message:
                    message.add([
                        behavior.Check(self._plan_conv.r('plan'), lambda plan: plan is not None),
                        base.PreparePrompt(
                            self._user_conv, plan=self._plan_conv.r('plan')
                        ),
                        base.ConvMessage(self._user_conv, llm_query, 'assistant'),
                        base.DisplayAI(self._user_conv, ui_interface),
                        base.ConvMessage(self._user_conv, ui_query, 'user'),
                        behavior.Reset(self._plan_conv, self._user_conv.r('completed'))
                    ])
                with behavior.sequence(language_teacher) as plan:
                    plan.add([
                        base.DisplayAI(self._plan_conv, ui_interface),
                        base.ConvMessage(self._plan_conv, ui_query, 'user'),
                        base.ConvMessage(self._plan_conv, llm_query, 'assistant')
                    ])
        self._behavior = language_teacher
        self._terminal = self._server.register(self._behavior)

    @property
    def io(self):
        return self._io

    def act(self) -> AgentStatus:
        
        sango_status = self._behavior.tick(self._root_terminal)
        return AgentStatus.from_status(sango_status)

    def reset(self):
        self._behavior.reset_status(self._root_terminal)
