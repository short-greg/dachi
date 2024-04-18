from abc import ABC

from dachi.act._agent import Agent, AgentStatus
from dachi.depracated.storage import act, storage
from . import tasks
from dachi.comm import UI, OpenAIQuery


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

        # TODO: Make it possible to define the args in the prompt completer
        self._summary_conv = storage.Completion(
            storage.PromptGen(
                tasks.SUMMARY_PROMPT, summary=self._state.r('conv_summary'), 
                conversation=self._define_question.f('as_turns')
            )
        )

        llm_query = OpenAIQuery()      
        self._prompt = tasks.QUESTION_PROMPT
        self._default = tasks.DEFAULT_PROMPT

        with act.sango() as story_writer:
            with act.select(story_writer) as teach:
                # change "define question to 1 line
                with act.sequence(teach) as define_question:
                    define_question.add(act.CheckFalse(self._state.r('question_defined')))
                    with act.select(define_question) as converse:
                        converse.add_tasks([
                            act.Converse(
                                self._define_question,  llm_query, ui_interface, 
                                tasks.ProcessComplete('question_defined', self._state)
                            ),
                            act.PromptCompleter(
                                self._summary_conv, llm_query, ui_interface,
                                post_processor=storage.Transfer(
                                    self._summary_conv.r('response'), self._state, 'conv_summary')
                            ),
                        ])
                teach.add(
                    act.Converse(
                        self._converse, llm_query, ui_interface
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
