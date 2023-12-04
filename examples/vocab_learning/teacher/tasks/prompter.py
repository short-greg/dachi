import threading
from dachi.behavior import Condition, Action, Terminal, SangoStatus
import openai


class PromptGenerated(Condition):
    '''Receive response from the user
    '''

    def __init__(self, prompt_signal: str) -> None:
        super().__init__('Response Receiver')
        self.prompt_signal = prompt_signal

    def __init_terminal__(self, terminal: Terminal):
        terminal.shared[self.prompt_signal] = terminal.shared.get(self.prompt_signal)

    def condition(self, terminal: Terminal) -> bool:
        
        return terminal.shared[self.prompt_signal] is not None


# Chat History
# Prompt

# https://stackoverflow.com/questions/76305207/openai-api-asynchronous-api-calls

class PromptLLM(Action):
    '''Receive response from the user
    '''

    def __init__(self, prompt_signal: str, llm_response_signal: str) -> None:
        super().__init__('Prompt LLM')
        self.prompt_signal = prompt_signal
        self.llm_response_signal = llm_response_signal

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage['sent_to_llm'] = False
        terminal.storage['llm_response'] = None

    def _prompt_llm(self, terminal: Terminal, prompt_messages):

        response = openai.ChatCompletion.create(
            model='gpt-4-turbo',
            messages=prompt_messages
        )
        terminal.storage['llm_response'] = response

    def act(self, terminal: Terminal) -> SangoStatus:
        
        llm_response = terminal.storage.get('llm_response')
        if llm_response is not None:
            
            terminal.shared[self.llm_response_signal] = llm_response
            terminal.storage['llm_response'] = None
            terminal.storage['sent_to_llm'] = False
            return SangoStatus.SUCCESS
        
        sent_to_llm = terminal.storage.get('send_to_llm')
        prompt = terminal.shared.get(self.prompt_signal)
        if sent_to_llm is None:

            prompt_messages = terminal.shared.get(self.prompt_signal)
            if prompt_messages is None:
                return SangoStatus.FAILURE
            thread = threading.Thread(target=prompt args=[terminal, prompt_messages])
            thread.start()
            terminal.shared[self.prompt_signal] = None

        # send message to the LLM
        return SangoStatus.RUNNING
