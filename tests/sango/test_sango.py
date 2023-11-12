from kaijin.sango import _sango as sango
from kaijin.sango._sango import Data, Message, Terminal
import pytest


class SampleHook(sango.DataHook):

    def __init__(self):
        super().__init__('sample')
        self.called = False

    def __call__(self, data: Data, prev_value):
        self.called = True


class TestData:

    def test_set_data_value_is_1(self):

        data = sango.Data('x', 1)
        assert data.value == 1

    def test_set_data_name_is_correct(self):

        data = sango.Data('x', 1)
        assert data.name == 'x'

    def test_set_data_value_is_correct_after_updating(self):

        data = sango.Data('x', 1)
        data.update(2)
        assert data.value == 2

    def test_error_raised_if_check_fails(self):
        check = lambda x: (x >= 0, 'Value must be greater than or equal to 0')

        with pytest.raises(ValueError):
            sango.Data('x', -1, check)

    def test_data_hook_is_registered(self):

        data = sango.Data('x', 1)
        hook = SampleHook()
        data.register_hook(hook)
        assert data.has_hook(hook)

    def test_data_hook_is_removed(self):

        data = sango.Data('x', 1)
        hook = SampleHook()
        data.register_hook(hook)
        data.remove_hook(hook)
        assert not data.has_hook(hook)



class TestSynched:

    def test_set_synched_value_is_1(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        assert synched.value == 1

    def test_set_data_name_is_correct(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        assert synched.name == 'x'

    def test_set_data_value_is_correct_after_updating(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        data.update(2)
        assert synched.value == 2

    def test_error_raised_if_check_fails(self):
        check = lambda x: (x >= 0, 'Value must be greater than or equal to 0')

        data = sango.Data('x', 1, check)
        synched = sango.Synched('x', data)
        with pytest.raises(ValueError):
            synched.update(-1) 

    def test_data_hook_is_registered(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        hook = SampleHook()
        synched.register_hook(hook)
        assert synched.has_hook(hook)

    def test_data_hook_is_removed(self):

        data = sango.Data('x', 1)
        synched = sango.Synched('x', data)
        hook = SampleHook()
        synched.register_hook(hook)
        synched.remove_hook(hook)
        assert not synched.has_hook(hook)


class TestDataHook:
    
    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        data = sango.Data('x', 1)
        data.register_hook(hook)
        data.update(2)
        assert hook.called is True


class TestCompositeHook:
    
    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        hook2 = SampleHook()
        composite = sango.CompositeHook('x', [hook, hook2])
        data = sango.Data('x', 1)
        data.register_hook(composite)
        data.update(2)
        assert hook.called is True
        assert hook2.called is True


class TestStatus:
    
    def test_status_is_done_when_running(self):
        assert not sango.Status.RUNNING.is_done()
    
    def test_status_is_not_done_when_ready(self):
        assert not sango.Status.RUNNING.is_done()
    
    def test_status_is_done_when_failure(self):
        assert sango.Status.FAILURE.is_done()

    def test_status_is_done_when_success(self):
        assert sango.Status.SUCCESS.is_done()


class TestDataStore:

    def test_add_to_datastore_adds_value(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert data_store['x'] == 1

    def test_get_returns_none_if_not_in_data_store(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        value = data_store.get('y')
        assert value is None

    def test_add_to_datastore_adds_value(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert data_store.get('x') == 1

    def test_in_returns_true_if_x_is_there(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert 'x' in data_store

    def test_in_returns_false_if_x_is_not_there(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        assert 'y' not in data_store

    def test_update_changes_the_value(self):

        data_store = sango.DataStore()
        data_store.add('x', 1)
        data_store.update('x', 2)
        assert data_store['x'] == 2

    def test_data_hook_is_called_after_update(self):
        hook = SampleHook()
        data_store = sango.DataStore()
        data_store.add('x', 1)
        data_store.register_hook('x', hook)
        data_store.update('x', 2)
        assert hook.called is True


class ATask(sango.Action):

    def __init__(self) -> None:
        super().__init__('ATask')
        self.x = 1

    def receive(self, message: Message):
        print('Received message')
        self.x = message.data['input']

    def act(self, terminal: Terminal):
        return sango.Status.SUCCESS


class TestServer:

    def test_shared_returns_data_stored(self):

        server = sango.Server()
        server.shared.add('x', 2)
        assert server.shared['x'] == 2

    def test_send_triggers_a_message(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        server.receive(sango.MessageType.INPUT, 'x', task)
        server.send(sango.Message(sango.MessageType.INPUT, 'x', {'input': 2}))
        assert task.x == 2

    def test_send_not_triggered_if_canceled(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        server.receive(sango.MessageType.INPUT, 'x', task)
        server.cancel_receive(sango.MessageType.INPUT, 'x', task)

        server.send(sango.Message(sango.MessageType.INPUT, 'y', {'input': 2}))
        assert task.x == 1

    def test_can_add_receiver_to_message_handler(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        server.receive(sango.MessageType.INPUT, 'x', task)
        assert server.receives_message((sango.MessageType.INPUT, 'x'))

    def test_message_in_handler_with_message(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        message = sango.Message(sango.MessageType.INPUT, 'x')
        server.receive(sango.MessageType.INPUT, 'x', task)

        assert server.receives_message(message)

    def test_message_in_handler_with_type_and_name(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        message = sango.Message(sango.MessageType.INPUT, 'x')
        server.receive(sango.MessageType.INPUT, 'x', task)

        assert server.receives_message(message)

    def test_message_not_in_handler_with_type_and_name_after_removed_last(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        message = sango.Message(sango.MessageType.INPUT, 'x')
        server.receive(sango.MessageType.INPUT, 'x', task)
        server.cancel_receive(sango.MessageType.INPUT, 'x', task)

        assert not server.receives_message(message)

    def test_receiver_not_in_handler_with_type_and_name_after_removed(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        message = sango.Message(sango.MessageType.INPUT, 'x')
        server.receive(sango.MessageType.INPUT, 'x', task)
        server.cancel_receive(sango.MessageType.INPUT, 'x', task)
        assert not server.has_receiver(message.message_type, message.name, task)

    def test_receiver_is_triggered_when_message_is_passed(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        server.receive(sango.MessageType.INPUT, 'x', task)
        server.send(sango.Message(sango.MessageType.INPUT, 'x', {'input': 2}))
        assert task.x == 2

    def test_receiver_is_not_triggered_when_message_is_passed(self):

        task = ATask()
        server = sango.Server()
        server.register(task)
        server.receive(sango.MessageType.INPUT, 'x', task)
        server.send(sango.Message(sango.MessageType.INPUT, 'y', {'input': 2}))
        assert task.x == 1


class TestTerminal:

    def test_can_retrieve_value_stored(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        terminal.storage.add('x', 2)
        assert terminal.storage['x'] == 2

    def test_initialized_set_to_true_after_initialize(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        terminal.initialize()
        assert terminal.initialized is True

    def test_shared_gets_shared_value_for_terminal2(self):

        server = sango.Server()
        terminal1 = sango.Terminal(server)
        terminal2 = sango.Terminal(server)
        terminal1.shared.add('x', 2)
        assert terminal2.shared['x'] == 2

    def test_server_returns_server(self):

        server = sango.Server()
        terminal1 = sango.Terminal(server)
        assert terminal1.server is server


class SetStorageAction(sango.Action):

    def __init__(self, name: str, key: str='x', value: int=4) -> None:
        super().__init__(name)
        self.key = key
        self.value = value

    def act(self, terminal: Terminal) -> sango.Status:
        terminal.storage.update(self.key, self.value)

        if self.value < 0:
            return sango.Status.FAILURE

        return sango.Status.SUCCESS
    
    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.add(
            'count', 0, lambda x: (x >= 0, 'Value must be greater than or equal to 0')
        )

class SetStorageActionCounter(sango.Action):

    def __init__(self, name: str, key: str='x', value: int=4) -> None:
        super().__init__(name)
        self.key = key
        self.value = value

    def act(self, terminal: Terminal):

        print(terminal.shared.get('failure'))
        if terminal.shared.get('failure') is True:
            return sango.Status.FAILURE
        terminal.storage['count'] += 1
        if terminal.storage['count'] == 2:
            return sango.Status.SUCCESS
        if terminal.storage['count'] < 0:
            return sango.Status.FAILURE
        return sango.Status.RUNNING
    
    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.add(
            'count', 0
        )


class TestAction:

    def test_storage_action_returns_success(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        assert terminal.storage[action.key] == action.value

    def test_count_has_been_initialized_on_terminal(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        assert terminal.storage['count'] == 0

    def test_cannot_set_count_to_less_than_0(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        with pytest.raises(ValueError):
            terminal.storage['count'] = -1

    def test_storage_action_counter_returns_running_if_count_is_1(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action = SetStorageActionCounter('SetStorageAction')
        assert action.tick(terminal) == sango.Status.RUNNING

    def test_storage_action_counter_returns_success_if_count_is_2(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        assert action.tick(terminal) == sango.Status.SUCCESS


class TestSequence:

    def test_sequence_is_running_when_started(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageAction('SetStorageActionTask2')
        sequence = sango.Sequence(
            [action1, action2]
        )
        
        assert sequence.tick(terminal) == sango.Status.RUNNING
    
    def test_sequence_finished_after_three_ticks(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageActionCounter('SetStorageActionTask2')
        sequence = sango.Sequence(
            [action1, action2]
        )
        sequence.tick(terminal)
        sequence.tick(terminal)
        assert sequence.tick(terminal) == sango.Status.SUCCESS

    def test_sequence_fails_if_count_is_less_than_0(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageActionCounter('SetStorageActionTask2')
        sequence = sango.Sequence(
            [action1, action2]
        )
        terminal.shared['failure'] = True
        sequence.tick(terminal)
        
        assert sequence.tick(terminal) == sango.Status.FAILURE


class SampleCondition(sango.Condition):

    def __init__(self, x):

        super().__init__('Failure')
        self.x = x

    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage['x'] = self.x

    def condition(self, terminal: Terminal) -> bool:

        if self.x < 0:
            return False
        return True


class TestCondition:

    def test_storage_action_initializes_terminal(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        condition = SampleCondition(1)
        condition.tick(terminal)
        assert terminal.storage['x'] == 1

    def test_storage_action_returns_success(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        condition = SampleCondition(1)
        
        assert condition.tick(terminal) == sango.Status.SUCCESS

    def test_storage_action_returns_failure(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        condition = SampleCondition(-1)
        
        assert condition.tick(terminal) == sango.Status.FAILURE


class TestFallback:

    def test_fallback_is_running_when_started(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageAction('SetStorageActionTask2')
        fallback = sango.Fallback(
            [action1, action2]
        )
        
        assert fallback.tick(terminal) == sango.Status.SUCCESS
    
    def test_fallback_finished_after_three_ticks(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action1 = SetStorageAction('SetStorageAction', value=-1)
        action2 = SetStorageActionCounter('SetStorageActionTask2')
        fallback = sango.Fallback(
            [action1, action2]
        )
        assert fallback.tick(terminal) == sango.Status.RUNNING

    def test_fallback_fails_if_count_is_less_than_0_for_all(self):

        server = sango.Server()
        terminal = sango.Terminal(server)
        action1 = SetStorageAction('SetStorageAction', value=-1)
        action2 = SetStorageAction('SetStorageAction2', value=-1)
        fallback = sango.Fallback(
            [action1, action2]
        )
        terminal.shared['failure'] = True
        fallback.tick(terminal)
        assert fallback.tick(terminal) == sango.Status.FAILURE

