# import pytest
# from dachi.act import _core, _functional as F
# from dachi.act import TaskStatus
# from dachi.store import _data as store
# import typing
# from dachi import store
# from dachi.store import Context
# import time


# def sample_action(state: typing.Dict, x: int, reset: bool=False) -> _core.TaskStatus:

#     if reset:
#         state['val'] = 0

#     val = store.get_or_set(state, 'val', 0)
#     state['val'] = x + val

#     if state['val'] < 3:
#         return _core.TaskStatus.RUNNING
#     return _core.TaskStatus.SUCCESS


# def sample_cond(x: int, reset: bool=False) -> bool:

#     return x > 3


# class TestAction:

#     def test_action_returns_in_progress(self):
#         state = {}

#         status = sample_action(state, 2)
#         assert status.in_progress

#     def test_action_returns_success(self):
#         state = {}

#         status = sample_action(state, 2)
#         status2 = sample_action(state, 3)
#         assert status2.success

#     def test_action_with_negative_input(self):
#         state = {}

#         status = sample_action(state, -1)
#         assert status.in_progress

#     def test_action_with_large_input(self):
#         state = {}

#         status = sample_action(state, 1000)
#         assert status.success

#     def test_action_with_zero_input(self):
#         state = {}

#         status = sample_action(state, 0)
#         assert status.in_progress

#     def test_action_with_existing_state(self):
#         state = {'val': 1}

#         status = sample_action(state, 1)
#         assert status.in_progress
#         assert state['val'] == 2

#     def test_action_with_state_reaching_success(self):
#         state = {'val': 2}

#         status = sample_action(state, 1)
#         assert status.success
#         assert state['val'] == 3

#     def test_action_with_non_integer_input(self):
#         state = {}

#         with pytest.raises(TypeError):
#             sample_action(state, "string")

#     def test_action_with_missing_state_key(self):
#         state = {}

#         status = sample_action(state, 2)
#         assert 'val' in state
#         assert status.in_progress

#     def test_action_returns_in_progress(self):
#         state = {}

#         status = sample_action(state, 2)
#         assert status.in_progress

#     def test_action_returns_success(self):
#         state = {}

#         status = sample_action(state, 2)
#         status2 = sample_action(state, 3)
#         assert status2.success
    

# class TestCond:

#     def test_cond_returns_failure(self):

#         status = F.tick(F.condf(sample_cond, 2))
#         assert status.failure

#     def test_cond_returns_success(self):

#         status = F.tick(F.condf(sample_cond, 4))
#         assert status.success


# class TestSequence:

#     def test_sequence_empty_tasks(self):
#         state = store.ContextStorage()

#         status = F.sequence([], state.S)()
#         assert status.success

#     def test_sequence_single_task_success(self):
#         state = store.ContextStorage()

#         status = F.sequence([F.condf(sample_cond, 4)], state.S)()
#         assert status.success

#     def test_sequence_single_task_failure(self):
#         state = store.ContextStorage()

#         status = F.sequence([F.condf(sample_cond, 2)], state.S)()
#         assert status.failure

#     def test_sequence_multiple_tasks_all_success(self):
#         state = store.ContextStorage()

#         task = F.sequence([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)
#         status = task()
#         status = task()
#         assert status.success

#     def test_sequence_multiple_tasks_returns_running_after_reset(self):
#         state = store.ContextStorage()

#         task = F.sequence([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)
#         status = task()
#         status = task()
#         status = task(True)
#         assert status.running

#     def test_sequence_multiple_tasks_with_failure(self):
#         state = store.ContextStorage()

#         status = F.sequence([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()
#         assert status.failure

#     def test_sequence_intermediate_running(self):
#         state = store.ContextStorage()

#         status = F.tick(F.sequence([
#             F.action(sample_action, state.A, 2),
#             F.condf(sample_cond, 4)
#         ], state.S))
#         assert status.running

#     # TODO: Note: Must be careful not 
#     # to give the same state to multiple 
#     # tasks
#     def test_sequence_nested_sequence(self):
#         state = store.ContextStorage()
#         task = F.sequence([
#             F.condf(sample_cond, 4),
#             F.sequence([
#                 F.action(sample_action, state.A, 4),
#                 F.condf(sample_cond, 4)
#             ], state.S2)
#         ], state.S)
#         status = task()
#         status = task()
#         status = task()
#         assert status.success

#     def test_sequence_with_boolean_task(self):
#         state = store.ContextStorage()

#         task = F.sequence([
#             True,
#             F.condf(sample_cond, 4)
#         ], state.S)
#         status = task()
#         status = task()
#         assert status.success

#     def test_sequence_with_failure_boolean_task(self):
#         state = store.ContextStorage()

#         status = F.sequence([
#             False,
#             F.condf(sample_cond, 4)
#         ], state.S)()
#         assert status.failure

#     def test_sequence_with_mixed_task_types(self):
#         state = store.ContextStorage()

#         task = F.sequence([
#             F.condf(sample_cond, 4),
#             True,
#             F.action(sample_action, state.A, 4)
#         ], state.S)
#         status = task()
#         status = task()
#         status = task()
#         assert status.success

#     def test_sequence_with_no_context(self):
#         with pytest.raises(AttributeError):
#             F.sequence([
#                 F.condf(sample_cond, 4),
#                 F.action(sample_action, None, 4)
#             ], None)()


#     def test_sequence_executes_and_returns_running(self):
#         state = store.ContextStorage()

#         status = F.tick(F.sequence([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S))

#         # print(status)
#         assert status.running

#     def test_sequence_executes_and_returns_success(self):
#         state = store.ContextStorage()

#         status = F.sequence([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()

#         status = F.sequence([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()
#         assert status.success

#     def test_returns_success_if_no_tasks(self):
#         state = store.ContextStorage()

#         status = F.sequence([
#         ], state.S)()

#         assert status.success


#     def test_sequence_executes_and_returns_failure(self):
#         state = store.ContextStorage()

#         status = F.sequence([
#             F.condf(sample_cond, 0),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()

#         assert status.failure


# class TestSelector:

#     def test_selector_all_tasks_fail(self):
#         state = store.ContextStorage()

#         task = F.selector([
#             F.condf(sample_cond, 2),
#             F.condf(sample_cond, 1)
#         ], state.S)
#         status = task()
#         status = task()

#         assert status.failure

#     def test_selector_first_task_succeeds(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             F.condf(sample_cond, 4),
#             F.condf(sample_cond, 2)
#         ], state.S)()

#         assert status.success

#     def test_selector_second_task_succeeds(self):
#         state = store.ContextStorage()

#         task = F.selector([
#             F.condf(sample_cond, 2),
#             F.condf(sample_cond, 4)
#         ], state.S)
#         status = task()
#         status = task()

#         assert status.success

#     def test_selector_second_task_is_running_with_reset(self):
#         state = store.ContextStorage()

#         task = F.selector([
#             F.condf(sample_cond, 2),
#             F.condf(sample_cond, 4)
#         ], state.S)
#         status = task()
#         status = task()
#         status = task(reset=True)

#         assert status.running

#     def test_selector_mixed_task_types(self):
#         state = store.ContextStorage()

#         task = F.selector([
#             False,
#             F.condf(sample_cond, 4),
#             True
#         ], state.S)
#         status = task()
#         status = task()

#         assert status.success

#     def test_selector_with_no_tasks(self):
#         state = store.ContextStorage()

#         status = F.selector([], state.S)()

#         assert status.failure

#     def test_selector_with_boolean_task_success(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             True,
#             F.condf(sample_cond, 2)
#         ], state.S)()

#         assert status.success

#     def test_selector_with_boolean_task_failure(self):
#         state = store.ContextStorage()

#         task = F.selector([
#             False,
#             F.condf(sample_cond, 2)
#         ], state.S)
#         status = task()
#         status = task()

#         assert status.failure

#     def test_selector_nested_selector(self):
#         state = store.ContextStorage()

#         task = F.selector([
#             F.condf(sample_cond, 2),
#             F.selector([
#                 F.condf(sample_cond, 4),
#                 F.condf(sample_cond, 2)
#             ], state.S2)
#         ], state.S)
#         status = task()
#         status = task()

#         assert status.success

#     def test_selector_executes_and_returns_running(self):
#         state = store.ContextStorage()

#         status = F.tick(F.selector([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], state.S))

#         assert status.running

#     def test_selector_executes_and_returns_success(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()

#         assert status.success

#     def test_selector_executes_and_returns_running(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 2)
#         ], state.S)()

#         assert status.running

#     def test_selector_executes_and_returns_running(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()

#         assert status.running

#     def test_selector_executes_and_returns_success(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()

#         status = F.selector([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()
#         assert status.success

#     def test_returns_failure_if_no_tasks(self):
#         state = store.ContextStorage()

#         status = F.selector([
#         ], state.S)()

#         assert status.failure

#     def test_selector_executes_and_returns_success(self):
#         state = store.ContextStorage()

#         status = F.selector([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ], state.S)()

#         assert status.success


# class TestParallel:

#     def test_parallel_returns_failure_if_one_fails(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ])()

#         assert status.failure

#     def test_parallel_returns_failure_if_one_fails(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ])()

#         assert status.failure

#     def test_parallel_returns_success_if_all_succeed(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 4)
#         ])()

#         assert status.success

#     def test_parallel_returns_running_if_one_is_running(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 4),
#             F.action(sample_action, state.A, 2)
#         ])()

#         assert status.running

#     def test_parallel_with_custom_succeeds_on(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 4),
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], succeeds_on=2)()

#         assert status.success

#     def test_parallel_with_custom_fails_on(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 2),
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], fails_on=2, succeeds_on=2)()

#         assert status.failure

#     def test_parallel_with_no_tasks(self):
#         status = F.parallel([])()

#         assert status.success

#     def test_parallel_with_nested_parallel(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 4),
#             F.parallel([
#                 F.condf(sample_cond, 2),
#                 F.action(sample_action, state.A, 4)
#             ])
#         ])()

#         assert status.failure

#     def test_parallel_with_success_priority(self):

#         status = F.parallel([
#             F.condf(sample_cond, 4),
#             F.condf(sample_cond, 2),
#             F.condf(sample_cond, 4),
#             F.condf(sample_cond, 2),
#         ], success_priority=True, succeeds_on=2, fails_on=2)()

#         assert status.success

#     def test_parallel_without_success_priority(self):
#         state = store.ContextStorage()

#         status = F.parallel([
#             F.condf(sample_cond, 4),
#             F.condf(sample_cond, 2),
#             F.action(sample_action, state.A, 4)
#         ], success_priority=False)()

#         assert status.failure


# class TestAsLongAs:

#     def test_unless_returns_failure_if_failed(self):
#         state = store.ContextStorage()

#         status = F.aslongas(
#             F.sequence([
#                 F.condf(sample_cond, 2),
#                 F.action(sample_action, state.A, 4)
#             ], state.S)
#         , state)()

#         assert status.failure

#     def test_unless_returns_running_if_succeeded(self):
#         state = store.ContextStorage()

#         status = F.aslongas(
#             F.sequence([
#                 F.condf(sample_cond, 4),
#                 F.action(sample_action, state.A, 4)
#             ], state.S)
#         )()

#         status = F.aslongas(
#             F.sequence([
#                 F.condf(sample_cond, 4),
#                 F.action(sample_action, state.S, 4)
#             ], state.A)
#         )()

#         assert status.running


# class TestParallelf:

#     def test_parallelf_all_tasks_succeed(self):
#         state = store.ContextStorage()

#         def task_generator(x):
#             yield F.condf(sample_cond, x)
#             yield F.action(sample_action, state.A, x)

#         status = F.parallelf(task_generator, 4)()
#         assert status.success

#     def test_parallelf_all_tasks_fail(self):
#         state = store.ContextStorage()

#         def task_generator(x):
#             yield F.condf(sample_cond, x)
#             yield F.action(sample_action, state.A, x)

#         status = F.parallelf(task_generator, 2)()
#         status = F.parallelf(task_generator, 2)()
#         assert status.failure

#     def test_parallelf_mixed_results_with_success_priority(self):
#         state = store.ContextStorage()

#         def task_generator(x):
#             yield F.condf(sample_cond, x)
#             yield F.action(sample_action, state.A, x)

#         status = F.parallelf(task_generator, 4, succeeds_on=1, fails_on=2, success_priority=True)()
#         assert status.success

#     def test_parallelf_mixed_results_without_success_priority(self):
#         state = store.ContextStorage()

#         def task_generator(x):
#             yield F.condf(sample_cond, x)
#             yield F.condf(sample_cond, x)

#         status = F.parallelf(task_generator, 2, succeeds_on=1, fails_on=2, success_priority=False)()
#         assert status.failure

#     def test_parallelf_no_tasks(self):
#         def task_generator():
#             if False:  # No tasks generated
#                 yield

#         status = F.parallelf(task_generator)()
#         assert status.success

#     def test_parallelf_nested_parallelf(self):

#         def task_generator(x):
#             yield F.condf(sample_cond, x)
#             yield F.parallelf(
#                 lambda y: (F.condf(sample_cond, y) for _ in range(2)), x
#             )

#         status = F.parallelf(task_generator, 2)()
#         assert status.failure

#     def test_parallelf_custom_succeeds_on(self):
#         state = store.ContextStorage()

#         def task_generator(x):
#             yield F.condf(sample_cond, x)
#             yield F.condf(sample_cond, x - 2)
#             yield F.action(sample_action, state.A, x)

#         status = F.parallelf(task_generator, 4, succeeds_on=2)()
#         assert status.success

#     def test_parallelf_custom_fails_on(self):
#         state = store.ContextStorage()

#         def task_generator(x):
#             yield F.condf(sample_cond, x - 2)
#             yield F.condf(sample_cond, x - 2)
#             yield F.action(sample_action, state.A, x)

#         status = F.parallelf(task_generator, 4, fails_on=2, succeeds_on=2)()
#         assert status.failure


# class TestUntil:

#     def test_until_returns_running_if_failed(self):
#         state = store.ContextStorage()

#         status = F.until(
#             F.sequence([
#                 F.condf(sample_cond, 2),
#                 F.action(sample_action, 4)
#             ], state.S)
#         )()

#         assert status.running

#     def test_unless_returns_success_if_succeeded(self):
#         state = store.ContextStorage()

#         status = F.until(
#             F.sequence([
#                 F.condf(sample_cond, 4),
#                 F.action(sample_action, state.A, 4)
#             ],state.S)
#         )()

#         status = F.until(
#             F.sequence([
#                 F.condf(sample_cond, 4),
#                 F.action(sample_action, state.A, 4)
#             ], state.S)
#         )()

#         assert status.success


# class TestNot:

#     def test_not_returns_success_if_failed(self):

#         status = F.not_(
#             F.condf(sample_cond, 2),
#         )()

#         assert status.success

#     def test_not_returns_failed_if_success(self):

#         status = F.tick(F.not_(
#             F.condf(sample_cond, 4),
#         ))

#         assert status.failure


# class TestTick:

#     def test_tick_with_callable_task(self):
#         def mock_task(reset: bool=False):
#             return TaskStatus.RUNNING

#         status = F.tick(mock_task)
#         assert status == TaskStatus.RUNNING

#     def test_tick_with_invalid_task_type(self):
#         with pytest.raises(TypeError):
#             F.tick("invalid_task")

#     def test_tick_with_task_returning_failure(self):
#         def mock_task(reset: bool=False):
#             return TaskStatus.FAILURE

#         status = F.tick(mock_task)
#         assert status == TaskStatus.FAILURE

#     def test_tick_with_task_returning_running(self):
#         def mock_task(reset: bool=False):
#             return TaskStatus.RUNNING

#         status = F.tick(mock_task)
#         assert status == TaskStatus.RUNNING

#     def test_tick_with_task_returning_success(self):
#         def mock_task(reset: bool=False):
#             return TaskStatus.SUCCESS

#         status = F.tick(mock_task)
#         assert status == TaskStatus.SUCCESS

#     def test_tick_with_task_raising_exception(self):
#         def mock_task(reset: bool=False):
#             raise ValueError("Task failed")

#         with pytest.raises(ValueError):
#             F.tick(mock_task)

#     def test_tick_with_task_returning_none(self):
#         def mock_task(reset: bool=False):
#             return None

#         status = F.tick(mock_task)
#         assert status is None

#     def test_tick_with_task_returning_custom_status(self):
#         class CustomStatus:
#             def __init__(self, state):
#                 self.state = state

#         def mock_task(reset: bool=False):
#             return CustomStatus("custom_state")

#         status = F.tick(mock_task)
#         assert isinstance(status, CustomStatus)
#         assert status.state == "custom_state"


# class TestThreaded:

#     def test_threadedf2_initial_status(self):
#         ctx = Context()
#         def mock_task():
#             return TaskStatus.SUCCESS

#         task = F.threaded_task(mock_task, ctx)
#         status = task()
#         assert status == TaskStatus.RUNNING

#     def test_threadedf2_success_status(self):
#         ctx = Context()
#         def mock_task():
#             return TaskStatus.SUCCESS

#         task = F.threaded_task(mock_task, ctx)
#         task()
#         task()
#         assert ctx['thread_status'] == TaskStatus.SUCCESS

#     def test_threadedf2_failure_status(self):
#         ctx = Context()
#         def mock_task():
#             raise ValueError("Task failed")

#         task = F.threaded_task(mock_task, ctx)
#         with pytest.raises(ValueError):
#             task()
#             task()
#             # assert ctx['thread_status'] == TaskStatus.FAILURE

#     def test_threadedf2_with_reset(self):
#         ctx = Context()
#         def mock_task():
#             return TaskStatus.SUCCESS

#         task = F.threaded_task(mock_task, ctx)
#         task(reset=True)
#         assert ctx['tick_id'] is not None

#     def test_threadedf2_with_callback(self):
#         ctx = Context()
#         callback_called = False

#         def mock_task():
#             return TaskStatus.SUCCESS

#         def callback(ctx):
#             nonlocal callback_called
#             callback_called = True

#         task = F.threaded_task(mock_task, ctx, callback=callback)
#         task()
#         task()
#         assert callback_called

#     def test_threadedf2_with_streamed_task(self):
#         ctx = Context()
#         def mock_task():
#             yield TaskStatus.RUNNING
#             yield TaskStatus.SUCCESS

#         task = F.streamed_task(mock_task, ctx)
#         status = task()
#         assert status == TaskStatus.RUNNING

#     def test_threadedf2_with_invalid_context(self):
#         with pytest.raises(RuntimeError):
#             ctx = Context()
#             def mock_task():
#                 return TaskStatus.SUCCESS

#             ctx['task_id'] = id(mock_task)
#             task = F.threaded_task(lambda: None, ctx)
#             task()

#     # TODO: Think how to handle this case where the user accidentally
#     #  sends the same context.
#     #  I should have a general approach.. That prevents a context
#     #  being reused
#     # def test_threadedf2_with_multiple_threads(self):
#     #     ctx = Context()
#     #     def mock_task():
#     #         return TaskStatus.SUCCESS

#     #     task1 = F.threadedf2(mock_task, ctx)
#     #     task2 = F.threadedf2(mock_task, ctx)

#     #     status1 = task1()
#     #     status2 = task2()

#     #     assert status1 == TaskStatus.RUNNING
#     #     assert status2 == TaskStatus.RUNNING

#     def test_threadedf2_with_long_running_task(self):
#         ctx = Context()
#         def mock_task():
#             time.sleep(0.1)
#             return TaskStatus.SUCCESS

#         task = F.threaded_task(mock_task, ctx)
#         status = task()
#         assert status == TaskStatus.RUNNING

#     def test_threadedf2_with_to_status_conversion(self):
#         ctx = Context()
#         def mock_task():
#             return "custom_result"

#         def to_status(result):
#             return TaskStatus.SUCCESS if result == "custom_result" else TaskStatus.FAILURE

#         task = F.threaded_task(mock_task, ctx, to_status=to_status)
#         task()
#         task()
#         assert ctx['thread_status'] == TaskStatus.SUCCESS
