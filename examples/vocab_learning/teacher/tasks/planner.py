from dachi.behavior import Condition, Action, Task, Terminal, SangoStatus
from dachi.gengo import Prompt

def get_prompt():
    return Prompt(
        ['target_vocabulary'],
        """ 
        You are language teacher who teaches Japanese vocabulary to native English learners of Japanese.
        First, create a plan in JSON based on TEMPLATE JSON for the words to teach the learner based on the TARGET VOCABULARY received from the learner.

        If the learner response is not a list of Japanese vocabulary, then return an error message JSON as shown in ERROR JSON. 

        TEMPLATE JSON
        [
            {
            '<Japanese word>': {
                    'Translation': '<English translation>',
                    'Definition': '<Japanese definition', 
                }
            },
            ...
        ]

        TARGET VOCABULARY = {target_vocabulary}

        ERROR JSON
        {'Message': '<Reason for error>'}

        
        """
    )

class WaitingForInput(Condition):

    def __init__(self, input_name: str, input_state) -> None:
        super().__init__('Waiting For Input')
        self.input_name = input_name
        self.input_state = input_state

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.shared.get_or_set(self.plan_name, None)

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.plan_name] is not None

    def clone(self) -> 'WaitingForInput':
        return WaitingForInput(
            self.input_name, self.input_state
        )


class PlanPrompter(Action):

    def __init__(self, input_name: str, plan_name: str, prompt_name: str) -> None:
        super().__init__('Plan Learning')
        self.input_name = input_name
        self.plan_name = plan_name
        self.prompt_name = prompt_name
        self.tako = None

    def act(self, terminal: Terminal) -> SangoStatus:

        vocabulary = terminal.get(self.input_name)
        if vocabulary is None:
            return SangoStatus.FAILURE

        terminal.shared[self.input_name] = None
        prompt = get_prompt()
        prompt = prompt.format(vocabulary=vocabulary)
        terminal.shared[self.prompt_name] = prompt
        return SangoStatus.SUCCESS
        
    def clone(self) -> 'PlanPrompter':
        return PlanPrompter(
            self.input_name, self.plan_name, self.prompt_name
        )


class PlanGenerator(Action):

    def __init__(self, plan_name: str, prompt_name: str) -> None:
        super().__init__('Plan Learning')
        self.input_name = input_name
        self.plan_name = plan_name
        self.prompt_name = prompt_name
        self.tako = None

    def act(self, terminal: Terminal) -> SangoStatus:

        vocabulary = terminal.get(self.input_name)
        if vocabulary is None:
            return SangoStatus.FAILURE

        terminal.shared[self.input_name] = None
        prompt = get_prompt()
        prompt = prompt.format(vocabulary=vocabulary)
        terminal.shared[self.prompt_name] = prompt
        return SangoStatus.SUCCESS
        
    def clone(self) -> 'PlanPrompter':
        return PlanPrompter(
            self.input_name, self.plan_name, self.prompt_name
        )

