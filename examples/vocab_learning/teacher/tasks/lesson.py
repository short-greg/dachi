from dachi.depracated.storage import Prompt, Conv, PromptConv
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
    {{
        "Message": "<The prompt and four questions >"
    }}
    - COMPLETED TEMPLATE (JSON)
    {{
        "Completed": "<Evaluation of performance>"
    }}
    - ERROR TEMPLATE (JSON)
    {{'Error': '<Reason for error>'}}
    
    """
)


class QuizConv(PromptConv):
    
    def __init__(self, max_turns: int=None):

        # add introductory message
        super().__init__(
            max_turns
        )
        self._completed = False
        self._error = None

    def add_turn(self, role: str, text: str) -> Conv:
        if role == 'assistant':
            try:
                result = json.loads(text)
                if 'Message' in result:
                    self._completed = False
                    self._error = None
                    super().add_turn(role, result['Message'])
                elif 'Error' in result:
                    self._completed = False
                    self._error = result['Error']
                    super().add_turn(role, result['Error'])
                elif 'Completed' in result:
                    self._completed = True
                    self._error = None
                    super().add_turn(role, result['Completed'])
                else:
                    self._completed = True
                    self._error = "Unknown response"
                    super().add_turn(role, result)
            except json.JSONDecodeError:
                self._error = 'Could not decode JSON.'
                super().add_turn(role, text)

        else:
            self._error = None
            self._completed = False
            super().add_turn(role, text)

    @property
    def error(self) -> str:

        return self._error

    def reset(self):
        super().reset()
        self._completed = False

    @property
    def completed(self) -> bool:
        return self._completed
