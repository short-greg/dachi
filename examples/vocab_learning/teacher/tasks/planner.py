from dachi.struct import Prompt, Conv
import json


PLAN_PROMPT = Prompt(
    ['target_vocabulary'],
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

    TARGET VOCABULARY = {target_vocabulary}
""")


class PlanConv(Conv):
    
    def __init__(self, max_turns: int=None):

        # add introductory message
        super().__init__(
            ['system', 'assistant', 'user'], 
            max_turns, True
        )
        self.add_turn('system', None)
        self._plan = None
        self._error = None

    def set_system(self, **kwargs):

        self[0].text = PLAN_PROMPT.format(
            **kwargs
        )

    def add_turn(self, role: str, text: str) -> Conv:
        if role == 'assistant':
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
                self._error = "Unknown response from LLM"
                self._plan = None
                super().add_turn(role, text)
        else:
            self._plan = None
            self._error = None
            super().add_turn(role, text)

    @property
    def error(self):
        return self._error

    @property
    def plan(self):
        return self._plan

    def reset(self):
        super().reset()
        self.add_turn('system', None)
        self._plan = None
