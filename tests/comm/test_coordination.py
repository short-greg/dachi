from dachi.behavior import _cooordination as coordination
from dachi.behavior._status import SangoStatus
from dachi.comm._coordination import Signal, Terminal
# from dachi.comm._cooordination import Signal, Terminal
from dachi.behavior._tasks import Action


class ATask(Action):

    def __init__(self) -> None:
        super().__init__('ATask')
        self.x = 1

    def receive(self, message: Signal):
        self.x = message.data['input']

    def act(self, terminal: Terminal):
        return SangoStatus.SUCCESS


class TestServer:

    def test_shared_returns_data_stored(self):

        server = coordination.Server()
        server.shared.add('x', 2)
        assert server.shared['x'] == 2

    # def test_send_triggers_a_message(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     server.send(coordination.Signal(coordination.SignalType.INPUT, 'x', {'input': 2}))
    #     assert task.x == 2

    # def test_send_not_triggered_if_canceled(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     server.cancel_receive(coordination.SignalType.INPUT, 'x', task)

    #     server.send(coordination.Signal(coordination.SignalType.INPUT, 'y', {'input': 2}))
    #     assert task.x == 1

    # def test_can_add_receiver_to_message_handler(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     assert server.receives_signal((coordination.SignalType.INPUT, 'x'))

    # def test_message_in_handler_with_message(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     message = coordination.Signal(coordination.SignalType.INPUT, 'x')
    #     server.receive(coordination.SignalType.INPUT, 'x', task)

    #     assert server.receives_signal(message)

    # def test_message_in_handler_with_type_and_name(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     message = coordination.Signal(coordination.SignalType.INPUT, 'x')
    #     server.receive(coordination.SignalType.INPUT, 'x', task)

    #     assert server.receives_signal(message)

    # def test_message_not_in_handler_with_type_and_name_after_removed_last(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     message = coordination.Signal(coordination.SignalType.INPUT, 'x')
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     server.cancel_receive(coordination.SignalType.INPUT, 'x', task)

    #     assert not server.receives_signal(message)

    # def test_receiver_not_in_handler_with_type_and_name_after_removed(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     message = coordination.Signal(coordination.SignalType.INPUT, 'x')
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     server.cancel_receive(coordination.SignalType.INPUT, 'x', task)
    #     assert not server.has_receiver(message.message_type, message.name, task)

    # def test_receiver_is_triggered_when_message_is_passed(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     server.send(coordination.Signal(coordination.SignalType.INPUT, 'x', {'input': 2}))
    #     assert task.x == 2

    # def test_receiver_is_not_triggered_when_message_is_passed(self):

    #     task = ATask()
    #     server = coordination.Server()
    #     server.register(task)
    #     server.receive(coordination.SignalType.INPUT, 'x', task)
    #     server.send(coordination.Signal(coordination.SignalType.INPUT, 'y', {'input': 2}))
    #     assert task.x == 1


class TestTerminal:

    def test_can_retrieve_value_stored(self):

        server = coordination.Server()
        terminal = coordination.Terminal(server)
        terminal.storage.add('x', 2)
        assert terminal.storage['x'] == 2

    def test_initialized_set_to_true_after_initialize(self):

        server = coordination.Server()
        terminal = coordination.Terminal(server)
        terminal.initialize()
        assert terminal.initialized is True

    def test_shared_gets_shared_value_for_terminal2(self):

        server = coordination.Server()
        terminal1 = coordination.Terminal(server)
        terminal2 = coordination.Terminal(server)
        terminal1.shared.add('x', 2)
        assert terminal2.shared['x'] == 2

    def test_server_returns_server(self):

        server = coordination.Server()
        terminal1 = coordination.Terminal(server)
        assert terminal1.server is server