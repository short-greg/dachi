from dachi.behavior import Terminal
from dachi.comm import Terminal
from dachi.gengo import Prompt, Conversation
from base import PrepareConversation, ChatConversationAI
import json


def get_prompt():

    return Prompt(
        ["plan"],
        """ 
        You are language teacher who teaches Japanese vocabulary to native English learners of Japanese.
        Teach the student vocabulary based on the PLAN Below. 

        Give the student a multiple choice quiz with each item having 4 options according to the plan. If the user
        gets an answer wrong, then give the user advice before giving the next quiz.
        The prompt for the quiz item is the word in Japanese. The four options are definitions
        in Japanese. Return a message to the user based on the user
        
        When the quiz is over, fill in COMPLETED TEMPLATE

        ===PLAN===
        {plan}

        ===RESULT TEMPLATE (JSON)===
        \{
            "Message": "<The prompt and four questions >"
        \}

        ===COMPLETED TEMPLATE (JSON)===
        \{
            "Completed": "<The prompt and four questions >"
        \}

        ===ERROR TEMPLATE (JSON)===
        \{'Error': '<Reason for error>'\}
        
        """
    )


class StartLesson(PrepareConversation):

    def prepare_conversation(self, terminal: Terminal):

        convo = Conversation(['AI', 'System', 'User'])
        convo.add_turn('System', get_prompt())
        convo.add_turn('AI', "学習したい語彙を教えてください。")
        terminal.shared.set(self.convo_name, convo)
        return True


class QuizUser(ChatConversationAI):

    def process_response(self, terminal: Terminal):
        
        response = json.loads(terminal.shared[self.user_var])
        if 'Error' in response:
            return False, response['Error']
        if 'Completed' in response:
            return True, ''
        if 'Item' in response:
            return False, response['Item']
        return True, 'Unknown error occurred'
