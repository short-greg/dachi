from dachi.behavior import Condition, Action, Terminal, SangoStatus


# Instead of defining this multiple times
# define it once
# Variable Condition
class WaitingForPlan(Condition):

    def __init__(self, signal_name: str) -> None:
        super().__init__('Waiting For Plan')
        self.signal_name = signal_name

    def __init_terminal__(self, terminal: Terminal):
        terminal.shared[self.signal_name] = None

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.signal_name] is not None


class PlanLearning(Action):

    def __init__(self, signal_name: str, message_handler) -> None:
        super().__init__('Plan Learning')
        self.signal_name = signal_name
        self.tako = None

    def act(self, terminal: Terminal):

        # probe the tako to get the prompt
        # and the history
        pass
        