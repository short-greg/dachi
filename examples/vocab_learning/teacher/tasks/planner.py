from dachi.depracated.storage import Prompt, Conv, PromptConv
import json


PLAN_PROMPT = Prompt(
    [],
    """ 
    You are language teacher who teaches Japanese vocabulary to native English learners of Japanese.
    First, create a plan in JSON based on TEMPLATE JSON for the words to teach the learner based on the TARGET VOCABULARY received from the learner.

    If the learner response is not a list of Japanese vocabulary, then return an error message JSON as shown in ERROR JSON. 

    TEMPLATE JSON
    {{
        "Plan": {{
        "<Japanese word>": {{
                "Translation": "<English translation>",
                "Definition": "<Japanese definition>", 
            }}
        }},
        ...
    }}

    ERROR JSON
    {{"Error": "<Reason for error>"}}

""")


class PlanConv(PromptConv):
    
    def __init__(self, max_turns: int=None):

        # add introductory message
        super().__init__(
            max_turns
        )
        super().add_turn('assistant', 'Please enter the vocabulary you wish to learn.')
        self._plan = None
        self._error = None

    def add_turn(self, role: str, text: str) -> Conv:
        if role == 'assistant':
            try:
                result = json.loads(text)
                if 'Plan' in result:
                    self._plan = result['Plan']
                    self._error = None
                    super().add_turn(role, result['Plan'])
                elif 'Error' in result:
                    self._plan = None
                    self._error = result['Error']
                    super().add_turn(role, result['Error'])
                else:
                    self._error = None
                    self._plan = None
                    super().add_turn(role, text)
            except json.JSONDecodeError:
                self._error = 'Could not decode JSON.'
                super().add_turn(role, result)
        else:
            self._plan = None
            self._error = None
            super().add_turn(role, text)

    @property
    def user(self) -> str:

        return self.filter('user')[-1].text

    @property
    def error(self):
        return self._error

    @property
    def plan(self):
        return self._plan

    def reset(self):
        super().reset()
        self._plan = None
