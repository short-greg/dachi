from dachi.struct import Prompt, Conv
import json

from dachi.struct._prompting import Conv


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


class QuizConv(Conv):
    
    def __init__(self, max_turns: int=None):

        # add introductory message
        super().__init__(
            ['system', 'assistant', 'user'], 
            max_turns, True
        )
        self.add_turn('system', None)
        self._plan = None

    def set_system(self, heading: str=None, plan=''):

        self[0].text = QUIZ_PROMPT.format(
            plan
        ).as_text(heading=heading)

    def add_turn(self, role: str, text: str) -> Conv:
        if role == 'assistant':
            result = json.loads(
                self.filter('assistant')[-1].text
            )
            if 'Message' in result:
                self._completed = False
                super().add_turn(role, result['Message'])
            if 'Error' in result:
                self._completed = False
                super().add_turn(role, result['Error'])
            if 'Completed' in result:
                self._completed = True
                super().add_turn(role, result['Completed'])
        else:
            super().add_turn(role, text)

    def reset(self):
        super().reset()
        self.add_turn('system', None)
        self._completed = False

    @property
    def completed(self) -> bool:
        return self._completed
