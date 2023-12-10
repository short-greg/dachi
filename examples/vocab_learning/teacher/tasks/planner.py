from dachi.behavior import Condition, Action, Task, Terminal, SangoStatus


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


class PlanLearning(Action):

    def __init__(self, input_name: str, plan_name: str, output_name: str) -> None:
        super().__init__('Plan Learning')
        self.input_name = input_name
        self.plan_name = plan_name
        self.output_name = output_name
        self.tako = None

    def act(self, terminal: Terminal) -> SangoStatus:

        if terminal.shared.get(self.input_name) is None:
            return SangoStatus.FAILURE

        terminal.shared[self.input_name] = None
        print(self.output_name)
        terminal.shared[self.output_name] = 'A simple message'
        return SangoStatus.SUCCESS
        
    def clone(self) -> 'PlanLearning':
        return PlanLearning(
            self.input_name, self.plan_name, self.output_name
        )
