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
"""


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


# class StartPlanning(PrepareConversation):

#     def prepare_conversation(self, terminal: Terminal):

#         convo = Conversation(['AI', 'System', 'User'])
#         convo.add_turn('System', get_prompt())
#         convo.add_turn('AI', "学習したい語彙を教えてください。")
#         terminal.cnetral.set(self.convo_name, convo)
#         return True


# class CreatePlan(ConversationAI):

#     def __init__(self, plan: str, ai_message: str, convo_var: str, query: LLMQuery) -> None:
#         super().__init__(ai_message, convo_var, query)
#         self.plan = plan

#     def process_response(self, terminal: Terminal):

#         response = terminal.cnetral.get(self.ai_message)
        
#         response = json.loads(response)
#         if 'Error' in response:
#             return False, response['Error']
#         if 'Plan' in response:
#             terminal.cnetral.get_or_set(self.plan, response['Plan'])
#             return True, response['Plan']
        
#         return False, 'Unknown error occurred'
