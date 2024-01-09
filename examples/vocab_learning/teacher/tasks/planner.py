from dachi.comm import Terminal, Ref
from dachi.behavior import Action, SangoStatus
from dachi.struct import Prompt, ConvBase
import json


PLAN_PROMPT = Prompt(
    ['target_vocabulary'],
    """ 
    You are language teacher who teaches Japanese vocabulary to native English learners of Japanese.
    First, create a plan in JSON based on TEMPLATE JSON for the words to teach the learner based on the TARGET VOCABULARY received from the learner.

    If the learner response is not a list of Japanese vocabulary, then return an error message JSON as shown in ERROR JSON. 

    TEMPLATE JSON
    {
        "Plan": {
        "<Japanese word>": {
                "Translation": "<English translation>",
                "Definition": "<Japanese definition>", 
            }
        },
        ...
    }

    ERROR JSON
    {"Error": "<Reason for error>"}

    TARGET VOCABULARY = {target_vocabulary}
""")


class QuizConv(object):
    pass


class CreateAIPlan(Action):
    
    def __init__(
        self, 
        ai_request: Ref, 
        plan: Ref, 
        plan_conv: Ref
    ):
        """

        Args:
            ai_request (Ref): 
            ai_message (Ref): 
            conversation (Ref): 
        """
        self.ai_request = ai_request
        self.plan = plan
        self.plan_conv = plan_conv

    def act(self, terminal: Terminal) -> SangoStatus:
        
        request = self.ai_request.get(terminal)

        plan_conv = self.plan_conv.get(terminal)
        response = json.loads(request.response)
        if 'error' in response and plan_conv is not None:
            # how to react to an error (?)
            self.plan_conv.add_turn('assistant', response['error'])
            return self.FAILURE
        if 'plan' in response and plan_conv is not None:
            self.plan.set(terminal, response['message'])
            plan_conv.clear()
            
            return self.SUCCESS
            
        return self.FAILURE
