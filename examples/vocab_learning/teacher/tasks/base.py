from dachi.behavior import Action, Terminal, SangoStatus
from dachi.comm import Terminal
from .. import queries
from dachi.gengo import Conversation
from abc import abstractmethod


class PrepareConversation(Action):
    # create the quiz item
    # process the result

    def __init__(self, convo_name: str) -> None:
        super().__init__('Convo')
        self.convo_name = convo_name

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.shared.get_or_set(
            self.convo_name, Conversation(['System', 'AI', 'User'])
        )
        terminal.storage.get_or_set(self.posted, False)

    @abstractmethod
    def prepare_conversation(self, terminal: Terminal):
        pass

    def act(self, terminal: Terminal) -> SangoStatus:

        # Checkt his
        if terminal.storage.get(self.convo_name) is None:
            
            terminal.storage[self.convo_name] = self.prepare_conversation(terminal)

            return SangoStatus.SUCCESS
        
        return SangoStatus.SUCCESS
        
    def clone(self) -> 'PrepareConversation':
        return self.__class__(
            self.convo_name, self.query
        )


class ChatConversationAI(Action):
    # create the quiz item
    # process the result

    def __init__(self, ai_var: str, convo_var: str, query: queries.LLMQuery) -> None:
        super().__init__('Chat ConvO AI')
        self.query = query
        self.convo_var = convo_var
        self.ai_message = ai_var
        self.posted = 'posted'

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.get_or_set(self.posted, False)
        terminal.storage.get_or_set(self.ai_message, None)

    @abstractmethod
    def process_response(self, terminal: Terminal):
        pass

    def act(self, terminal: Terminal) -> SangoStatus:

        # I want to reduce this to a couple of lines
        # if terminal.shared.get('plan') is None:
        #     return SangoStatus.FAILURE

        converstation: Conversation = terminal.shared[self.convo_var]
        # if len(converstation) == 0:
        #     converstation.add_turn(
        #         'System', get_prompt(terminal.shared[self.plan_name])
        #     )

        if terminal.storage[self.posted] is False:
            self.query.post(converstation, on_response=self.ai_message)
            terminal.storage[self.posted] = True

        if terminal.shared[self.ai_message] is not None:
            terminal.storage[self.posted] = False

            completed, response = self.process_response(terminal)
            if completed is True:
                terminal.shared[self.convo_var].clear()
                return SangoStatus.FAILURE

            terminal.shared[self.convo_var].add_turn(
                'AI', response
            )

            return SangoStatus.SUCCESS
        
        return SangoStatus.RUNNING
        
    def clone(self) -> 'ChatConversationAI':
        return self.__class__(
            self.convo_var, self.plan_name, self.query
        )


class ChatUIResponse(Action):

    def __init__(self, conversation_var: str, ai_var: str, user_var: str, query: queries.UIQuery) -> None:
        super().__init__('Chat UI Response')
        self.ai_var = ai_var
        self.user_var = user_var
        self.query = query
        self.conversation_var = conversation_var

    def __init_terminal__(self, terminal: Terminal):
        
        terminal.storage.get_or_set('posted', False)
        terminal.shared.get_or_set(self.ai_var, None)

    def act(self, terminal: Terminal) -> SangoStatus:

        if terminal.shared.get(self.ai_var) is None:
            return SangoStatus.FAILURE

        # 
        if terminal.storage[self.posted] is False:
            self.query.post(
                terminal.shared[self.ai_var], 
                on_response=self.user_var
            )

        if terminal.shared[self.user_var] is not None:
            terminal.storage[self.posted] = False
            answer = terminal.shared[self.user_var]
            conversation: Conversation = terminal.shared[self.conversation_var]
            conversation.add_turn('User', answer)
            terminal.shared[self.user_var] = None
            return SangoStatus.SUCCESS

        return SangoStatus.RUNNING
        
    def clone(self) -> 'ChatUIResponse':
        return self.__class__(
            self.conversation_var, self.ai_var,
            self.user_var, self.query
        )
