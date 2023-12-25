from dachi.behavior import Action, SangoStatus
from dachi.comm import Terminal
from .. import queries
from dachi.gengo import Conversation, Prompt
from abc import abstractmethod


class PrepareConversation(Action):
    # create the quiz item
    # process the result

    def __init__(self, convo_name: str) -> None:
        super().__init__('Convo')
        self.convo_name = convo_name
        self.posted = 'posted'

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.cnetral.get_or_set(
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


class ConversationAI(Action):
    # create the quiz item
    # process the result

    def __init__(self, ai_message: str, convo_var: str, query: queries.LLMQuery) -> None:
        super().__init__('Chat Convo AI')
        self.query = query
        self.convo_var = convo_var
        self.ai_message = ai_message
        self.posted = 'posted'

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.get_or_set(self.posted, False)
        terminal.storage.get_or_set(self.ai_message, None)

    def process_response(self, terminal: Terminal) -> str:
        """Default behavior is just to return the message from the AI

        Args:
            terminal (Terminal): the terminal for the AI

        Returns:
            str: The response
        """
        return True, terminal.cnetral[self.ai_message]

    def act(self, terminal: Terminal) -> SangoStatus:
        """

        Args:
            terminal (Terminal): 

        Returns:
            SangoStatus: 
        """

        conversation: Conversation = terminal.cnetral[self.convo_var]

        if terminal.storage[self.posted] is False:
            self.query.post(
                conversation, on_response=self.ai_message
            )
            terminal.storage[self.posted] = True

        if terminal.cnetral[self.ai_message] is not None:
            terminal.storage[self.posted] = False

            success, response = self.process_response(terminal)
            
            terminal.cnetral[self.convo_var].add_turn(
                'AI', response
            )

            if success:

                return SangoStatus.SUCCESS
            return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING
        
    def clone(self) -> 'ConversationAI':
        return self.__class__(
            self.ai_message, self.convo_var,  self.query
        )



class CompletionAI(Action):
    # create the quiz item
    # process the result

    def __init__(self, ai_message: str, system_prompt: str, query: queries.LLMQuery) -> None:
        super().__init__('Chat Convo AI')
        self.query = query
        self.system_prompt = system_prompt
        self.ai_message = ai_message
        self.posted = 'posted'

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.get_or_set(self.posted, False)
        terminal.storage.get_or_set(self.ai_message, None)

    def process_response(self, terminal: Terminal) -> str:
        """Default behavior is just to return the message from the AI

        Args:
            terminal (Terminal): the terminal for the AI

        Returns:
            str: The response
        """
        return terminal.cnetral[self.ai_message]

    def act(self, terminal: Terminal) -> SangoStatus:
        """

        Args:
            terminal (Terminal): 

        Returns:
            SangoStatus: 
        """

        prompt: Prompt = terminal.cnetral[self.system_prompt]
        conversation = Conversation()
        conversation.add_turn('system', prompt.text)

        if terminal.storage[self.posted] is False:
            self.query.post(
                conversation, on_response=self.ai_message
            )
            terminal.storage[self.posted] = True

        if terminal.cnetral[self.ai_message] is not None:
            terminal.storage[self.posted] = False

            self.process_response(terminal)
            
            return SangoStatus.SUCCESS
        
        return SangoStatus.RUNNING
        
    def clone(self) -> 'CompletionAI':
        return self.__class__(
            self.ai_message, self.system_prompt,  self.query
        )


class UserConversationResponse(Action):

    def __init__(self, conversation_var: str, ai_var: str, user_var: str, query: queries.UIQuery) -> None:
        """Create 

        Args:
            conversation_var (str): _description_
            ai_var (str): _description_
            user_var (str): _description_
            query (queries.UIQuery): _description_
        """
        super().__init__('Chat UI Response')
        self.ai_var = ai_var
        self.user_var = user_var
        self.query = query
        self.conversation_var = conversation_var
        self.posted = 'posted'

    def __init_terminal__(self, terminal: Terminal):
        
        terminal.storage.get_or_set(self.posted, False)
        terminal.cnetral.get_or_set(self.ai_var, None)

    def act(self, terminal: Terminal) -> SangoStatus:

        if terminal.cnetral.get(self.ai_var) is None:
            return SangoStatus.FAILURE

        # 
        if terminal.storage[self.posted] is False:
            self.query.post(
                terminal.cnetral[self.ai_var], 
                on_response=self.user_var
            )

        if terminal.cnetral[self.user_var] is not None:
            terminal.storage[self.posted] = False
            answer = terminal.cnetral[self.user_var]
            conversation: Conversation = terminal.cnetral[self.conversation_var]
            conversation.add_turn('User', answer)
            terminal.cnetral[self.user_var] = None
            return SangoStatus.SUCCESS

        return SangoStatus.RUNNING
        
    def clone(self) -> 'UserConversationResponse':
        return self.__class__(
            self.conversation_var, self.ai_var,
            self.user_var, self.query
        )
