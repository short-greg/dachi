from dachi.behavior import Condition, Action, Task, Terminal, SangoStatus
from dachi.gengo import Prompt


class UpdateLesson(Action):

    def __init__(self, input_name: str, plan_name: str, prompt_name: str) -> None:
        super().__init__('Plan Learning')

    def act(self, terminal: Terminal) -> SangoStatus:
        pass
        
    def clone(self) -> 'UpdateLesson':
        pass


class Quiz(Action):

    def __init__(self, plan_name: str, prompt_name: str) -> None:
        super().__init__('Plan Learning')
        

    def act(self, terminal: Terminal) -> SangoStatus:
        pass
        
    def clone(self) -> 'Quiz':
        pass


class Explain(Action):

    def __init__(self, plan_name: str, prompt_name: str) -> None:
        super().__init__('Plan Learning')

    def act(self, terminal: Terminal) -> SangoStatus:
        pass
        
    def clone(self) -> 'Quiz':
        pass



