from abc import ABC

from dachi.agents import Agent, AgentStatus
from ..tools import base
from ..tools.queries import UIQuery, LLMQuery
from dachi import behavior, storage
from . import tasks
from ..tools.comm import UI


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


class StoryWriter(Agent):

    def __init__(self, ui_interface: UI, interval: float=None):

        super().__init__()

        self._status = AgentStatus.READY
        self._interval = interval
        self._assist_conv = tasks.StoryTellerConv()
        self._summary_conv = tasks.SummaryCompletion()

        llm_query = LLMQuery()
        ui_query = UIQuery(ui_interface)

        self._prompts = storage.DDict([tasks.QUESTION_PROMPT, tasks.ANSWER_PROMPT])        
        self._default = tasks.DEFAULT_PROMPT
        self._wrapper = storage.Wrapper(tasks.QUESTION_PROMPT)
        # Set the first messaeg of plan conv
        # self._user_conv.set_system(plan=self._plan_conv.plan)
        with behavior.sango() as story_writer:
            with behavior.select(story_writer) as teach:
                # can make these two trees
                with behavior.sequence(teach) as summarizer:
                    # Can't this be simplified.. If this is too complex it makes
                    # it hard to use
                    # CheckTrue
                    # RunPrompt <- I want it to be simple.. If it was just two 
                    #  items it would be easier.. Prompt, Conv, Display, 
                    summarizer.add_tasks([
                        # CheckTrue
                        # Converse
                        # Reset
                        behavior.CheckTrue(self._assist_conv.r('completed')),
                        behavior.Reset(self._summary_conv.d),
                        base.PreparePrompt(
                           self._summary_conv, tasks.SUMMARY_PROMPT, 
                           conversation=self._assist_conv.r('conv'),
                           summary=self._summary_conv.r('summary')
                        ),
                        base.ConvMessage(self._summary_conv, llm_query, 'assistant'),
                        base.DisplayAI(self._summary_conv, ui_interface),
                        behavior.Reset(self._assist_conv.d),
                        base.AdvPrompt(self._prompts, self._default, self._wrapper),
                    ])
                # teach.add(ConverseCond())
                # 
                with behavior.sequence(teach) as assistance:

                    # ConverseCond # Can easily create multiple "assistances" with this
                    # Spawn a thread which goes through each point
                    # function 1, function 2, etc... Should be relatively easy
                    assistance.add_tasks([
                        base.PreparePrompt(
                            self._assist_conv, self._wrapper.r('wrapped'), 
                            summary=self._summary_conv.r('summary')
                        ),
                        base.ConvMessage(self._assist_conv, llm_query, 'assistant'),
                        behavior.Not(behavior.CheckTrue(self._assist_conv.r('completed'))),
                        base.DisplayAI(self._assist_conv, ui_interface),
                        base.ConvMessage(self._assist_conv, ui_query, 'user'),
                    ])
        self._behavior = story_writer

    @property
    def io(self):
        return self._io

    def act(self) -> AgentStatus:
        
        sango_status = self._behavior.tick()
        return AgentStatus.from_status(sango_status)

    def reset(self):
        self._behavior.reset()
