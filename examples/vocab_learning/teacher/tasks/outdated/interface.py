from dachi.behavior import Condition, Action, Task, Terminal, SangoStatus
from ...comm import IOHandler


class OutputWaiting(Condition):

    def __init__(self, output_name: str) -> None:
        super().__init__('Output Waiting')
        print('Output name: ', output_name)
        self.output_name = output_name

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.shared.get_or_set(self.output_name, None)

    def condition(self, terminal: Terminal) -> bool:
        
        # print(terminal.shared.get('message'))
        # print('Output Name', self.output_name)
        return terminal.shared[self.output_name] is not None

    def clone(self) -> Task:
        return OutputWaiting(self.output_name)

class OutputMessage(Action):

    def __init__(self, output_name: str, io_handler: IOHandler) -> None:
        super().__init__('Output Message')
        self.output_name = output_name
        self.io_handler = io_handler

    def act(self, terminal: Terminal):

        message = terminal.shared.get(self.output_name)
        if message is None:
            return SangoStatus.FAILURE
        if self.io_handler.post_bot_message(message):
            terminal.shared[self.output_name] = None
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE

    def clone(self) -> 'OutputMessage':
        return OutputMessage(
            self.output_name, self.io_handler
        )


# this is not actually necessary
class InputReceived(Condition):
    '''Receive response from the user
    '''
    def __init__(self, input_name: str, io_handler: IOHandler) -> None:
        """

        Args:
            input_name (str): 
            input_register (InputRegister): 
        """
        super().__init__('Input Receiver')
        self.input_name = input_name
        self.io_handler = io_handler
        io_handler.register(self.response_name)

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.shared.get_or_set(self.input_name, None)

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.input_name] is not None

    def clone(self) -> 'InputReceived':
        return OutputMessage(
            self.input_name, self.io_handler
        )


class ProcessInput(Action):

    def __init__(self, input_name: str, message_handler) -> None:
        super().__init__('Process Response')
        self.input_name = input_name
        self.message_handler = message_handler

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.shared.get_or_set(self.input_name, None)

    def act(self, terminal: Terminal):

        message = terminal.shared.get(self.input_name)
        if message is None:
            return SangoStatus.FAILURE
        
        if self.message_handler(message):
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE

    def clone(self) -> 'ProcessInput':
        return ProcessInput(
            self.input_name
        )
