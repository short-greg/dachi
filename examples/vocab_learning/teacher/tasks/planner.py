from dachi.struct import Prompt, Conv
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


class PlanConv(Conv):
    
    def __init__(self, max_turns: int=None):

        # add introductory message
        super().__init__(
            ['system', 'assistant', 'user'], 
            max_turns, True
        )
        self.add_turn('system', None)
        self._plan = None

    def set_system(self, heading: str=None, **kwargs):

        self[0].text = PLAN_PROMPT.format(
            **kwargs
        ).as_text(heading=heading)

    def add_turn(self, role: str, text: str) -> Conv:
        if role == 'assistant':
            result = json.loads(
                self.filter('assistant')[-1].text
            )
            if 'Plan' in result:
                self._plan = result['Plan']
                super().add_turn(role, result['Plan'])
            if 'Error' in result:
                self._plan = None
                super().add_turn(role, result['Error'])
        else:
            super().add_turn(role, text)

    @property
    def plan(self):
        return self._plan

    def reset(self):
        super().reset()
        self.add_turn('system', None)
        self._plan = None
