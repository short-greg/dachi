# from abc import ABC

# from dachi.agents import Agent, AgentStatus
# from ...tools import base
# from ...tools.queries import UIQuery, LLMQuery
# from dachi import behavior, storage
# from .tasks import lesson, planner
# from ...tools.comm import UI


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


# class LanguageTeacher(Agent):

#     def __init__(self, ui_interface: UI, interval: float=None):

#         super().__init__()

#         self._status = AgentStatus.READY
#         self._interval = interval
#         self._plan_conv = planner.PlanConv()
#         self._user_conv = lesson.QuizConv()

#         llm_query = LLMQuery()
#         ui_query = UIQuery(ui_interface)

#         # Set the first messaeg of plan conv
#         # self._user_conv.set_system(plan=self._plan_conv.plan)
#         with behavior.sango() as language_teacher:
#             with behavior.select(language_teacher) as teach:
#                 # can make these two trees
#                 with behavior.sequence(teach) as message:
#                     message.add_tasks([
#                         behavior.Check(self._plan_conv.r('plan'), lambda plan: plan is not None),
#                         base.PreparePrompt(
#                             self._user_conv, lesson.QUIZ_PROMPT, plan=self._plan_conv.r('plan')
#                         ),
#                         base.ConvMessage(self._user_conv, llm_query, 'assistant'),
#                         base.DisplayAI(self._user_conv, ui_interface),
#                         base.ConvMessage(self._user_conv, ui_query, 'user'),
#                         behavior.Not(
#                             behavior.Reset(self._plan_conv.d, self._user_conv.r('completed'))
#                         )
#                     ])
#                 with behavior.sequence(teach) as plan:
#                     plan.add_tasks([
#                         behavior.Reset(self._user_conv.d),
#                         base.DisplayAI(self._plan_conv, ui_interface),
#                         base.ConvMessage(self._plan_conv, ui_query, 'user'),
#                         # think how to improve this
#                         base.PreparePrompt(
#                             self._plan_conv, planner.PLAN_PROMPT
#                         ),
#                         base.ConvMessage(self._plan_conv, llm_query, 'assistant')
#                     ])
#         self._behavior = language_teacher

#     @property
#     def io(self):
#         return self._io

#     def act(self) -> AgentStatus:
        
#         sango_status = self._behavior.tick()
#         return AgentStatus.from_status(sango_status)

#     def reset(self):
#         self._behavior.reset()
