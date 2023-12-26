from dachi.comm import Terminal
from dachi.gengo import Prompt, Conversation
import json
from base import UIResponse, AIResponse, PreparePrompt
from examples.vocab_learning.teacher.queries import LLMQuery


def get_prompt():

    return Prompt(
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

        TARGET VOCABULARY = {target_vocabulary}

        ERROR JSON
        {"Message": "<Reason for error>"}

        """
    )


class PlanPrompt(PreparePrompt):

    pass



class VocabularyList(UIResponse):
    pass


class PlanQuiz(AIResponse):

    pass


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
