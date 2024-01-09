from dachi.behavior._status import SangoStatus
from dachi.comm import Terminal, Ref
from dachi.behavior import Action
from dachi.struct import Prompt, ConvBase
import json


QUIZ_PROMPT = Prompt(
    ["plan"],
    """
    # Instructions
    You are language teacher who teaches Japanese vocabulary to native English learners of Japanese.
    Teach the student vocabulary based on the PLAN Below. 

    Give the student a multiple choice quiz with each item having 4 options according to the plan. If the user
    gets an answer wrong, then give the user advice before giving the next quiz.
    The prompt for the quiz item is the word in Japanese. The four options are definitions
    in Japanese. Return a message to the user based on the user
    
    When the quiz is over, fill in COMPLETED TEMPLATE

    # PLAN
    {plan}

    # RESPONSE CHOICES - Choose from one of these

    - RESULT TEMPLATE (JSON)
    \{
        "Message": "<The prompt and four questions >"
    \}
    - COMPLETED TEMPLATE (JSON)
    \{
        "Completed": "<Evaluation of performance>"
    \}
    - ERROR TEMPLATE (JSON)
    \{'Error': '<Reason for error>'\}
    
    """
)


class PlanConv(object):
    pass


class ProcessAIMessage(Action):
    
    def __init__(
        self, ai_request: Ref, ai_message: Ref, 
        conv: Ref
    ):
        """

        Args:
            ai_request (Ref): 
            ai_message (Ref): 
            conversation (Conversation): 
        """
        self.ai_request = ai_request
        self.ai_message = ai_message
        self.conv = conv

    def act(self, terminal: Terminal) -> SangoStatus:
        
        request = self.ai_request.get(terminal)

        conv = self.conversation.get(terminal)
        response = json.loads(request.response)
        if 'completed' in response:
            conv.clear()
            self.ai_message.clear()
            return self.FAILURE
        if 'error' in response:
            # how to react to an error (?)
            self.ai_message.set(terminal, response['error'])
            conv.add_turn('assistant', response['message'])
            return self.SUCCESS
        if 'message' in response:
            self.ai_message.set(terminal, response['message'])
            conv.add_turn('assistant', response['message'])
            return self.SUCCESS
            
        return self.FAILURE

