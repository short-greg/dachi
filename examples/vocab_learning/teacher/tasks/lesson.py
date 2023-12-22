from dachi.behavior._status import SangoStatus
from dachi.comm import Terminal
from dachi.behavior import Action
from dachi.gengo import Prompt, Conversation
from base import PrepareConversation, ConversationAI, UserConversationResponse
import json


def get_prompt(plan):

    return Prompt(
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
    ).format(plan=plan)


class StartLesson(PrepareConversation):

    def __init__(self, plan: str, convo_name: str) -> None:
        super().__init__(convo_name)
        self.plan = plan

    def prepare_conversation(self, terminal: Terminal):

        convo = Conversation(['AI', 'System', 'User'])
        plan = terminal.shared.get(self.plan)
        if plan is None:
            return False
        convo.add_turn(
            'system', plan
        )

        terminal.shared.add(self.convo_name, convo)
        return True


class QuizUser(ConversationAI):

    def process_response(self, terminal: Terminal):
        
        response = json.loads(terminal.shared[self.user_var])
        if 'Error' in response:
            return False, response['Error']
        if 'Completed' in response:
            return True, response['Completed']
        if 'Item' in response:
            return True, response['Item']
        return False, 'Unknown error occurred'


# class ProcessAnswer(UserConversationResponse):
#     pass

    # def process_response(self, terminal: Terminal):
        
    #     response = json.loads(terminal.shared[self.user_var])
    #     if 'Error' in response:
    #         return response['Error']
    #     if 'Completed' in response:
    #         return response['Completed']
    #     if 'Item' in response:
    #         return response['Item']
    #     return 'Unknown error occurred'


class Complete(Action):

    def __init__(self, completion: str, plan: str, convo: str):

        self.completion = completion
        self.plan = plan
        self.convo = convo

    def __init_terminal__(self, terminal: Terminal):
        
        super().__init_terminal__(terminal)
        terminal.shared.get_or_set('completed', False)

    def act(self, terminal: Terminal) -> SangoStatus:
        
        completed = terminal.shared.get('completed')
        if completed is True:
            plan = terminal.shared.get('plan')
            convo = terminal.shared.get('convo')
            if plan is not None:
                # This requires me to understand how it works
                # I don't like this
                terminal.shared['plan'] = None
            if convo is not None:
                convo.clear()
            terminal.shared['completed'] = False
            # I just want to return self.SUCCESS
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE