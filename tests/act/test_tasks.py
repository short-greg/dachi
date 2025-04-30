from dachi.act import _tasks as behavior
from dachi.act._core import TaskStatus


class ATask(behavior.Action):

    def __init__(self, x: int=1):
        super().__init__()
        self.x = x

    def act(self, reset: bool=False):
        return TaskStatus.SUCCESS


class SetStorageAction(behavior.Action):

    def __init__(self, value: int=4):
        super().__init__()
        self.value = value

    def act(self, reset: bool=False) -> TaskStatus:

        if self.value < 0:
            return TaskStatus.FAILURE

        return TaskStatus.SUCCESS


class SampleCondition(behavior.Condition):

    def __init__(self, x: int=1):
        super().__init__()
        self.x = x

    def condition(self, reset: bool=False) -> bool:

        if self.x < 0:
            return False
        return True


class SetStorageActionCounter(behavior.Action):

    __store__ = ["value"]

    def __init__(self, value: int=4):
        super().__init__()
        self.value = value
        self._count = 0

    def act(self, reset: bool=False):

        if self.value == 0:
            return TaskStatus.FAILURE
        self._count += 1
        if self._count == 2:
            return TaskStatus.SUCCESS
        if self._count < 0:
            return TaskStatus.FAILURE
        return TaskStatus.RUNNING


class TestAction:

    def test_storage_action_count_is_1(self):
        action = SetStorageAction(value=1)
        assert action.tick() == TaskStatus.SUCCESS

    def test_store_action_returns_fail_if_fail(self):

        action = SetStorageAction(value=-1)
        assert action.tick() == TaskStatus.FAILURE

    def test_running_after_one_tick(self):

        action = SetStorageActionCounter(value=1)
        assert action.tick() == TaskStatus.RUNNING

    def test_success_after_two_tick(self):

        action = SetStorageActionCounter(value=2)
        action.tick()
        assert action.tick() == TaskStatus.SUCCESS

    def test_ready_after_reset(self):

        action = SetStorageActionCounter(value=2)
        action.tick()
        action.reset_status()
        assert action.status == TaskStatus.READY

    def test_load_state_dict_sets_state(self):

        action = SetStorageActionCounter(value=3)
        action2 = SetStorageActionCounter(value=2)
        action.tick()
        action2.load_state_dict(action.state_dict())
        assert action2.value == 3


class TestSequence:

    def test_sequence_is_running_when_started(self):

        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = behavior.Sequence(
            tasks=[action1, action2]
        )
        
        assert sequence.tick() == TaskStatus.RUNNING

    def test_sequence_is_success_when_finished(self):

        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=2)
        sequence = behavior.Sequence(
            tasks=[action1, action2]
        )
        sequence.tick()
        assert sequence.tick() == TaskStatus.SUCCESS

    def test_sequence_is_failure_less_than_zero(self):

        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = behavior.Sequence(
            tasks=[action1, action2]
        )
        sequence.tick()
        assert sequence.tick() == TaskStatus.FAILURE

    def test_sequence_is_ready_when_reset(self):

        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction(value=-1)
        sequence = behavior.Sequence(
            tasks=[action1, action2]
        )
        sequence.tick()
        sequence.tick()
        sequence.reset_status()
        assert sequence.status == TaskStatus.READY

    def test_sequence_finished_after_three_ticks(self):

        action1 = SetStorageAction(value=2)
        action2 = SetStorageActionCounter(value=3)
        sequence = behavior.Sequence(
            tasks=[action1, action2]
        )
        sequence.tick()
        sequence.tick()
        
        assert sequence.tick() == TaskStatus.SUCCESS


class TestCondition:

    def test_condition_returns_success(self):

        condition = SampleCondition(x=1)
        assert condition.tick() == TaskStatus.SUCCESS

    def test_condition_returns_failure(self):

        condition = SampleCondition(x=-1)
        assert condition.tick() == TaskStatus.FAILURE

    def test_condition_status_is_success(self):

        condition = SampleCondition(x=-1)
        condition.tick()
        assert condition.status == TaskStatus.FAILURE

    def test_condition_status_is_ready_after_reset(self):

        condition = SampleCondition(x=-1)
        condition.tick()
        condition.reset_status()
        assert condition.status == TaskStatus.READY


class TestFallback:

    def test_fallback_is_successful_after_one_tick(self):

        action1 = SetStorageAction(value=1)
        action2 = SetStorageAction()
        fallback = behavior.Selector(
            tasks=[action1, action2]
        )
        
        assert fallback.tick() == TaskStatus.SUCCESS
    
    def test_fallback_is_successful_after_two_ticks(self):

        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=1)
        fallback = behavior.Selector(
            tasks=[action1, action2]
        )
        fallback.tick()
        assert fallback.tick() == TaskStatus.SUCCESS
    
    def test_fallback_fails_after_two_ticks(self):

        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = behavior.Selector(
            tasks=[action1, action2]
        )
        fallback.tick()
        assert fallback.tick() == TaskStatus.FAILURE

    def test_fallback_running_after_one_tick(self):

        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = behavior.Selector(
            tasks=[action1, action2]
        )
        
        assert fallback.tick() == TaskStatus.RUNNING
    
    def test_fallback_ready_after_reset(self):

        action1 = SetStorageAction(value=-1)
        action2 = SetStorageAction(value=-1)
        fallback = behavior.Selector(
            tasks=[action1, action2]
        )
        fallback.tick()
        fallback.reset_status()
        assert fallback.status == TaskStatus.READY


class TestAsLongAs:

    def test_while_fails_if_failure(self):

        action1 = SetStorageActionCounter(
            value=0
        )
        action1._count = -1
        while_ = behavior.AsLongAs(
            task=action1, target_status=TaskStatus.FAILURE
        )

        while_.tick()
        action1.value = 1
        assert while_.status == TaskStatus.RUNNING

    def test_aslongas_fails_if_failure_after_two(self):

        action1 = SetStorageActionCounter(value=1)
        action1._count = 1
        action1.value = 4
        aslongas = behavior.AsLongAs(task=action1)

        aslongas.tick()
        action1.value = 0

        assert aslongas.tick() == TaskStatus.FAILURE


class TestUntil:

    def test_until_successful_if_success(self):

        action1 = SetStorageActionCounter(value=1)
        action1._count = 1
        until_ = behavior.Until(task=action1)

        assert until_.tick() == TaskStatus.SUCCESS

    def test_until_successful_if_success_after_two(self):

        action1 = SetStorageActionCounter(value=0)
        action1._count = 1
        until_ = behavior.Until(task=action1)
        until_.tick()
        action1.value = 1

        assert until_.tick() == TaskStatus.SUCCESS
