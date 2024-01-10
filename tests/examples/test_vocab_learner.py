from examples.vocab_learning.teacher.tasks import lesson
from examples.vocab_learning.teacher.tasks import planner
from examples.vocab_learning.teacher.tasks import base
from dachi.comm import _requests as requests
from dachi import struct
from time import sleep
import threading
from dachi.behavior import SangoStatus
from dachi.struct import Prompt
from copy import deepcopy
import typing


class UI(base.UIInterface):

    def __init__(self, user_message):
        super().__init__()
        self.called = False
        self.user_message = user_message
        self.bot_message = ''

    def request_message(self, callback: typing.Callable[[str], None]):
        
        callback(self.user_message)
    
    def post_message(self, speaker, message: str):
        print('POSTING!')
        self.bot_message = message


class DummyQuery(requests.Query):

    def __init__(self, response):
        super().__init__()
        self.called = False
        self.response = response

    def prepare_response(self, request):
        return self.response
    
    def prepare_post(self, request):

        self.respond(request)


class DummyQueryDelay(requests.Query):

    def __init__(self, response, delay):
        super().__init__()
        self.called = False
        self.response = response
        self.delay = delay

    def prepare_response(self, request):
        return self.response
    
    def delay_response(self, request):

        sleep(self.delay)
        print('Slept')
        self.respond(request)
    
    def prepare_post(self, request):

        thread = threading.Thread(target=self.delay_response, args=[request])
        thread.start()


class TestConvMessage:

    def test_conv_message_acts(self):

        conv = struct.Conv()
        message = base.ConvMessage(
            conv, DummyQuery('RESPONSE')
        )
        message.tick()
        assert len(conv) == 1

    def test_conv_message_does_not_act(self):

        conv = struct.Conv()
        message = base.ConvMessage(
            conv, DummyQueryDelay('RESPONSE', 0.02)
        )
        status = message.tick()
        
        assert status == SangoStatus.RUNNING

    def test_conv_message_does_not_act_til_after_the_delay(self):

        conv = struct.Conv()
        query = DummyQueryDelay('RESPONSE', 0.01)
        message = base.ConvMessage(
            conv, query
        )
        status = message.tick()
        sleep(0.02)
        
        status = message.tick()
        
        assert status == SangoStatus.SUCCESS


class TestDisplayAI:

    def test_display_ai_returns_message(self):

        x = 2
        conv = struct.Conv()
        conv.add_turn('assistant', 'HELLO')
        ui =  UI('HI my name is Roy')
        display = base.DisplayAI(conv, ui)
        display.tick()
        assert ui.bot_message == 'HELLO'

    def test_fails_on_second_display(self):

        conv = struct.Conv()
        conv.add_turn('assistant', 'HELLO')
        ui =  UI('HI my name is Roy')
        display = base.DisplayAI(conv, ui)
        display.tick()
        status = display.tick()
        assert status == SangoStatus.FAILURE

    def test_failure_if_resetting_after_second(self):

        conv = struct.Conv()
        conv.add_turn('assistant', 'HELLO')
        ui =  UI('HI my name is Roy')
        display = base.DisplayAI(conv, ui)
        display.tick()
        display.reset()
        status = display.tick()
        
        assert status == SangoStatus.FAILURE


class TestPlanConv:

    def test_adds_system_message(self):

        conv = planner.PlanConv()
        conv.set_system(target_vocabulary='X Y Z')
        assert conv[0].text is not None

    def test_plan_is_extracted(self):

        conv = planner.PlanConv()
        conv.set_system(target_vocabulary='X Y Z')
        conv.add_turn('assistant', '{"Plan": "Big plan"}')
        assert conv.plan == 'Big plan'

    def test_plan_is_not_extracted_if_error(self):

        conv = planner.PlanConv()
        conv.set_system(target_vocabulary='X Y Z')
        conv.add_turn('assistant', '{"Error": "Error"}')
        assert conv.plan is None

    def test_response_is_unknown_if_invalid_input(self):

        conv = planner.PlanConv()
        conv.set_system(target_vocabulary='X Y Z')
        conv.add_turn('assistant', '{"X": "Error"}')
        assert conv.plan is None
        assert conv.error == "Unknown response from LLM"


class TestQuizConv:

    def test_plan_set_if_system_set(self):

        conv = lesson.QuizConv()
        conv.set_system(plan='PLAN')
        assert conv[0].text is not None

    def test_completed_if_completed_extracted(self):

        conv = lesson.QuizConv()
        conv.set_system(plan='PLAN')
        conv.add_turn(
            "assistant", '{"Completed": "Completed"}'
        )
        assert conv.completed is True

    def test_not_completed_if_not_extracted(self):

        conv = lesson.QuizConv()
        conv.set_system(plan='PLAN')
        conv.add_turn(
            "assistant", '{"Message": "Completed"}'
        )
        assert conv.completed is False

    def test_error_extracted_if_error(self):

        conv = lesson.QuizConv()
        conv.set_system(plan='PLAN')
        conv.add_turn(
            "assistant", '{"Error": "Completed"}'
        )
        assert conv.completed is False
        assert conv.error == 'Completed'
