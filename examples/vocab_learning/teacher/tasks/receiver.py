from dachi.behavior import Condition, Action, Terminal, SangoStatus


class ResponseReceived(Condition):
    '''Receive response from the user
    '''

    def __init__(self, signal_name: str, response_register) -> None:
        super().__init__('Response Receiver')
        self.signal_name = signal_name
        response_register.register(self.signal_name)

    def __init_terminal__(self, terminal: Terminal):
        terminal.shared[self.signal_name] = terminal.shared.get(self.signal_name, None)

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.signal_name] is not None


class ProcessResponse(Action):

    def __init__(self, signal_name: str) -> None:
        super().__init__('Process Response')
        self.signal_name = signal_name

    def act(self, terminal: Terminal):

        message = terminal.shared.get(self.signal_name)
        if message is None:
            return SangoStatus.FAILURE
        if self.message_handler(message):
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE
