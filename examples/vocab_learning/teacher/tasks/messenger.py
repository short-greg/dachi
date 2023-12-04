from dachi.behavior import Condition, Action, Terminal, SangoStatus


class MessageWaiting(Condition):

    def __init__(self, signal_name: str) -> None:
        super().__init__('Message Waiting')
        self.signal_name = signal_name

    def __init_terminal__(self, terminal: Terminal):
        terminal.shared[self.signal_name] = None

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.signal_name] is not None


class SendMessage(Action):

    def __init__(self, signal_name: str, message_handler) -> None:
        super().__init__('Send Message')
        self.signal_name = signal_name
        self.message_handler = message_handler

    def act(self, terminal: Terminal):

        message = terminal.shared.get(self.signal_name)
        if message is None:
            return SangoStatus.FAILURE
        if self.message_handler(message):
            terminal.shared[self.signal_name] = None
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE
