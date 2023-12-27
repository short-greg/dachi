from dachi.comm import Terminal, Ref
from dachi.behavior import Action, SangoStatus
from dachi.gengo import Prompt, Conversation
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


class ProcessAIMessage(Action):
    
    def __init__(self, ai_request: Ref, plan: Ref, conversation: Conversation):
        """

        Args:
            ai_request (Ref): 
            ai_message (Ref): 
            conversation (Conversation): 
        """
        self.ai_request = ai_request
        self.plan = plan
        self.conversation = conversation

    def act(self, terminal: Terminal) -> SangoStatus:
        
        request = self.ai_request.get(terminal)

        conv = self.conversation.get(terminal)
        response = json.loads(request.response)
        if 'error' in response:
            # how to react to an error (?)
            self.ai_message.set(terminal, response['error'])
            conv.add_turn('assistant', response['message'])
            return self.FAILURE
        if 'plan' in response:
            self.plan.message.set(terminal, response['message'])
            self.conversation.clear()
            
            return self.SUCCESS
            
        return self.FAILURE
