from dachi.behavior import Condition, Action, Task, Terminal, SangoStatus


# Instead of defining this multiple times
# define it once
# Variable Condition
class WaitingForPlan(Condition):

    def __init__(self, plan_name: str) -> None:
        super().__init__('Waiting For Plan')
        self.plan_name = plan_name

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.shared[self.plan_name] = None

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.plan_name] is not None

    def clone(self) -> 'WaitingForPlan':
        return WaitingForPlan(
            self.plan_name
        )

class PlanLearning(Action):

    def __init__(self, plan_name: str, output_name: str) -> None:
        super().__init__('Plan Learning')
        self.plan_name = plan_name
        self.output_name = output_name
        self.tako = None

    def act(self, terminal: Terminal):

        # probe the tako to get the prompt
        # and the history
        terminal.shared[self.output_name] = 'A simple message'
        
    def clone(self) -> 'PlanLearning':
        return PlanLearning(
            self.plan_name, self.output_name
        )
