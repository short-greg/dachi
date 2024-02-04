from abc import ABC

from dachi.agents import Agent, AgentStatus
from ..tools import base
from ..tools.queries import UIQuery, OpenAIQuery
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
        self._state = storage.DDict()
        self._state.set('conv_summary', 'Not available')
        # Work on this a little more
        self._define_question = storage.PromptConv(
            storage.PromptGen(tasks.QUESTION_PROMPT, summary=self._state.r('conv_summary'))
        )
        self._converse = storage.PromptConv(
            storage.PromptGen(tasks.DEFAULT_PROMPT, summary=self._state.r('conv_summary'))
        )
        self._summary_conv = storage.Completion(
            storage.PromptGen(
                tasks.SUMMARY_PROMPT, summary=self._state.r('conv_summary'), 
                conversation=self._define_question.f('as_turns')
            )
        )

        llm_query = OpenAIQuery()

        # self._prompts = storage.DList([tasks.QUESTION_PROMPT, tasks.ANSWER_PROMPT])        
        self._prompt = tasks.QUESTION_PROMPT
        self._default = tasks.DEFAULT_PROMPT

        # Set the first messaeg of plan conv
        with behavior.sango() as story_writer:
            with behavior.select(story_writer) as teach:
                with behavior.sequence(teach) as define_question:
                    # TODO: Wrap into one sequence
                    define_question.add(behavior.CheckFalse(self._state.r('question_defined')))
                    with behavior.select(define_question) as converse:
                        converse.add_tasks([
                            base.Converse(
                                self._define_question,  llm_query, ui_interface, 
                                tasks.ProcessComplete('question_defined', self._state)
                            ),
                            base.PromptCompleter(
                                self._summary_conv, llm_query, ui_interface,
                                post_processor=base.Transfer(
                                    self._summary_conv.r('response'), self._state, 'conv_summary')
                            ),
                        ])
                teach.add(
                    base.Converse(
                        self._converse, llm_query, ui_interface, 
                        tasks.ProcessComplete('question_defined', self._state)
                    )
                )
        self._behavior = story_writer

    @property
    def io(self):
        return self._io

    def act(self) -> AgentStatus:
        
        sango_status = self._behavior.tick()
        return AgentStatus.from_status(sango_status)

    def reset(self):
        self._behavior.reset()
