from examples.vocab_learning.teacher.tasks import lesson
from examples.vocab_learning.teacher.tasks import planner
from examples.tools import base
from dachi.comm import _requests as requests
from dachi import storage
from time import sleep
import threading
from dachi.behavior import SangoStatus
from dachi.storage import Prompt, D
from copy import deepcopy
import typing


class UI(base.UI):

    def __init__(self, user_message):
        super().__init__()
        self.called = False
        self.user_message = user_message
        self.bot_message = ''

    def request_message(self, callback: typing.Callable[[str], None]):
        
        callback(self.user_message)
    
    def post_message(self, speaker, message: str):
        self.bot_message = message


class DummyQuery(requests.Query):

    def __init__(self, response):
        super().__init__()
        self.called = False
        self.response = response
    
    def exec_post(self, request):

        self.respond(request)


class DummyLLMQuery(base.LLMQuery):

    def __init__(self, termperature=0.0):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.called = False
        self.temperature = termperature

    def prepare_response(self, request: base.Request):
        return 'Response from LLM!'


class DummyQueryDelay(requests.Query):

    def __init__(self, response, delay):
        super().__init__()
        self.called = False
        self.response = response
        self.delay = delay
    
    def delay_response(self, request):

        sleep(self.delay)
        self.respond(request)
    
    def exec_post(self, request):

        thread = threading.Thread(target=self.delay_response, args=[request])
        thread.start()


# class TestConvMessage:

#     def test_conv_message_acts(self):

#         conv = storage.Conv()
#         message = base.ConvMessage(
#             conv, DummyQuery('RESPONSE')
#         )
#         message.tick()
#         assert len(conv) == 1

#     def test_conv_message_does_not_act(self):

#         conv = storage.Conv()
#         message = base.ConvMessage(
#             conv, DummyQueryDelay('RESPONSE', 0.02)
#         )
#         status = message.tick()
        
#         assert status == SangoStatus.RUNNING

#     def test_conv_message_does_not_act_til_after_the_delay(self):

#         conv = storage.Conv()
#         query = DummyQueryDelay('RESPONSE', 0.01)
#         message = base.ConvMessage(
#             conv, query
#         )
#         status = message.tick()
#         sleep(0.02)
        
#         status = message.tick()
        
#         assert status == SangoStatus.SUCCESS


# class TestDisplayAI:

#     def test_display_ai_returns_message(self):

#         x = 2
#         conv = storage.Conv()
#         conv.add_turn('assistant', 'HELLO')
#         ui =  UI('HI my name is Roy')
#         display = base.DisplayAI(conv, ui)
#         display.tick()
#         assert ui.bot_message == 'HELLO'

#     def test_fails_on_second_display(self):

#         conv = storage.Conv()
#         conv.add_turn('assistant', 'HELLO')
#         ui =  UI('HI my name is Roy')
#         display = base.DisplayAI(conv, ui)
#         display.tick()
#         status = display.tick()
#         assert status == SangoStatus.FAILURE

#     def test_failure_if_resetting_after_second(self):

#         conv = storage.Conv()
#         conv.add_turn('assistant', 'HELLO')
#         ui =  UI('HI my name is Roy')
#         display = base.DisplayAI(conv, ui)
#         display.tick()
#         display.reset()
#         status = display.tick()
        
#         assert status == SangoStatus.FAILURE


# class TestPreparePrompt:

#     def test_prepare_prompt_updates_plan(self):

#         conv = planner.PlanConv()
#         prompt = base.PreparePrompt(conv, lesson.QUIZ_PROMPT, plan='Big Plan')
#         prompt.tick()
#         assert conv[0].text is not None

#     def test_prepare_prompt_updates_plan(self):

#         conv = planner.PlanConv()
#         prompt = base.PreparePrompt(conv, lesson.QUIZ_PROMPT, plan='Big Plan')
#         assert conv[0].text is None

#     def test_prepare_prompt_does_not_update_prompt_with_two_ticks(self):

#         conv = planner.PlanConv()
#         prompt = base.PreparePrompt(conv, lesson.QUIZ_PROMPT, plan=D('Big Plan'))
#         prompt.tick()
#         prompt.tick()
#         assert conv[0].text is not None


# class TestPlanConv:

#     def test_adds_system_message(self):

#         conv = planner.PlanConv()
#         prompt = planner.PLAN_PROMPT
#         prompt = prompt.format()
#         conv.set_system(prompt)
#         assert conv[0].text is not None

#     def test_plan_is_extracted(self):

#         conv = planner.PlanConv()
#         prompt = planner.PLAN_PROMPT
#         prompt = prompt.format()
#         conv.set_system(prompt)
#         conv.add_turn('assistant', '{"Plan": "Big plan"}')
#         assert conv.plan == 'Big plan'

#     def test_plan_is_not_extracted_if_error(self):

#         conv = planner.PlanConv()
#         prompt = planner.PLAN_PROMPT
#         prompt = prompt.format()
#         conv.set_system(prompt)
#         conv.add_turn('assistant', '{"Error": "Error"}')
#         assert conv.plan is None

#     def test_response_is_unknown_if_invalid_input(self):

#         conv = planner.PlanConv()
#         prompt = planner.PLAN_PROMPT
#         prompt = prompt.format()
#         conv.set_system(prompt)
#         conv.add_turn('assistant', '{"X": "Error"}')
#         assert conv.plan is None
#         assert conv.error is None


# class TestQuizConv:

#     def test_plan_set_if_system_set(self):

#         conv = lesson.QuizConv()
#         prompt = lesson.QUIZ_PROMPT
#         prompt = prompt.format(plan='PLAN')
#         conv.set_system(prompt)
#         assert conv[0].text is not None

#     def test_completed_if_completed_extracted(self):

#         conv = lesson.QuizConv()
#         prompt = lesson.QUIZ_PROMPT
#         prompt = prompt.format(plan='PLAN')
#         conv.add_turn(
#             "assistant", '{"Completed": "Completed"}'
#         )
#         assert conv.completed is True

#     def test_not_completed_if_not_extracted(self):

#         conv = lesson.QuizConv()
#         prompt = lesson.QUIZ_PROMPT
#         prompt = prompt.format(plan='PLAN')
#         conv.add_turn(
#             "assistant", '{"Message": "Completed"}'
#         )
#         assert conv.completed is False

#     def test_error_extracted_if_error(self):

#         conv = lesson.QuizConv()
#         prompt = lesson.QUIZ_PROMPT
#         prompt = prompt.format(plan='PLAN')
#         conv.add_turn(
#             "assistant", '{"Error": "Completed"}'
#         )
#         assert conv.completed is False
#         assert conv.error == 'Completed'


class TestConverse:

    PROMPT = (
    '''
    You are a chatbot. Please respond.
    '''
    )

    def test_converse_outputs_correct_value(self):

        prompt = base.Prompt([], self.PROMPT)
        prompt_gen = base.PromptGen(prompt)
        conv = base.ChatConv()
        llm_query = DummyLLMQuery(0.1)
        ui = UI('message')
        converse = base.Converse(
            prompt_gen, conv, llm_query, ui
        )
        converse.tick()
        assert len(conv) == 3


class TestMessage:

    PROMPT = (
    '''
    You are a chatbot. Please respond.
    '''
    )

    def test_converse_outputs_correct_value(self):

        prompt = base.Prompt([], self.PROMPT)
        prompt_gen = base.PromptGen(prompt)
        conv = base.ChatConv()
        ui = UI('message')
        llm_query = DummyLLMQuery(0.1)
        converse = base.Message(
            prompt_gen, conv, llm_query, ui
        )
        converse.tick()
        assert len(conv) == 2

