from dachi.behavior import _tasks as behavior
from dachi.behavior import _cooordination as coordination
from dachi.comm import Signal, Terminal
# from dachi.behavior._cooordination import Signal, Terminal
from dachi.behavior._status import SangoStatus
import pytest


class ATask(behavior.Action):

    def __init__(self) -> None:
        super().__init__('ATask')
        self.x = 1

    # def receive(self, message: Signal):
    #     self.x = message.data['input']

    def act(self, terminal: Terminal):
        return SangoStatus.SUCCESS


class SetStorageAction(behavior.Action):

    def __init__(self, name: str, key: str='x', value: int=4) -> None:
        super().__init__(name)
        self.key = key
        self.value = value

    def act(self, terminal: Terminal) -> SangoStatus:
        terminal.storage.update(self.key, self.value)

        if self.value < 0:
            return SangoStatus.FAILURE

        return SangoStatus.SUCCESS
    
    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.add(
            'count', 0, lambda x: (x >= 0, 'Value must be greater than or equal to 0')
        )

class SetStorageActionCounter(behavior.Action):

    def __init__(self, name: str, key: str='x', value: int=4) -> None:
        super().__init__(name)
        self.key = key
        self.value = value

    def act(self, terminal: Terminal):

        print(terminal.cnetral.get('failure'))
        if terminal.cnetral.get('failure') is True:
            return SangoStatus.FAILURE
        terminal.storage['count'] += 1
        if terminal.storage['count'] == 2:
            return SangoStatus.SUCCESS
        if terminal.storage['count'] < 0:
            return SangoStatus.FAILURE
        return SangoStatus.RUNNING
    
    def __init_terminal__(self, terminal: Terminal):
        super().__init_terminal__(terminal)
        terminal.storage.add(
            'count', 0
        )


class TestAction:

    def test_storage_action_returns_success(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        assert terminal.storage[action.key] == action.value

    def test_count_has_been_initialized_on_terminal(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        assert terminal.storage['count'] == 0

    def test_cannot_set_count_to_less_than_0(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        with pytest.raises(ValueError):
            terminal.storage['count'] = -1

    def test_storage_action_counter_returns_running_if_count_is_1(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action = SetStorageActionCounter('SetStorageAction')
        assert action.tick(terminal) == SangoStatus.RUNNING

    def test_storage_action_counter_returns_success_if_count_is_2(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action = SetStorageAction('SetStorageAction')
        action.tick(terminal)
        assert action.tick(terminal) == SangoStatus.SUCCESS


class TestSequence:

    def test_sequence_is_running_when_started(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageAction('SetStorageActionTask2')
        sequence = behavior.Sequence(
            [action1, action2]
        )
        
        assert sequence.tick(terminal) == SangoStatus.RUNNING
    
    def test_sequence_finished_after_three_ticks(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageActionCounter('SetStorageActionTask2')
        sequence = behavior.Sequence(
            [action1, action2]
        )
        sequence.tick(terminal)
        sequence.tick(terminal)
        assert sequence.tick(terminal) == SangoStatus.SUCCESS

    def test_sequence_fails_if_count_is_less_than_0(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageActionCounter('SetStorageActionTask2')
        sequence = behavior.Sequence(
            [action1, action2]
        )
        terminal.cnetral['failure'] = True
        sequence.tick(terminal)
        
        assert sequence.tick(terminal) == SangoStatus.FAILURE


class SampleCondition(behavior.Condition):

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

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        condition = SampleCondition(1)
        condition.tick(terminal)
        assert terminal.storage['x'] == 1

    def test_storage_action_returns_success(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        condition = SampleCondition(1)
        
        assert condition.tick(terminal) == SangoStatus.SUCCESS

    def test_storage_action_returns_failure(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        condition = SampleCondition(-1)
        
        assert condition.tick(terminal) == SangoStatus.FAILURE


class TestFallback:

    def test_fallback_is_running_when_started(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action1 = SetStorageAction('SetStorageAction')
        action2 = SetStorageAction('SetStorageActionTask2')
        fallback = behavior.Selector(
            [action1, action2]
        )
        
        assert fallback.tick(terminal) == SangoStatus.SUCCESS
    
    def test_fallback_finished_after_three_ticks(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action1 = SetStorageAction('SetStorageAction', value=-1)
        action2 = SetStorageActionCounter('SetStorageActionTask2')
        fallback = behavior.Selector(
            [action1, action2]
        )
        assert fallback.tick(terminal) == SangoStatus.RUNNING

    def test_fallback_fails_if_count_is_less_than_0_for_all(self):

        server = coordination.Server()
        terminal = behavior.Terminal(server)
        action1 = SetStorageAction('SetStorageAction', value=-1)
        action2 = SetStorageAction('SetStorageAction2', value=-1)
        fallback = behavior.Selector(
            [action1, action2]
        )
        terminal.cnetral['failure'] = True
        fallback.tick(terminal)
        assert fallback.tick(terminal) == SangoStatus.FAILURE
