# from dachi.act import _core


# class TestTaskStatus(object):

#     def test_task_status_is_done_if_failed(self):

#         assert _core.TaskStatus.FAILURE.is_done

#     def test_task_status_is_done_if_success(self):

#         assert _core.TaskStatus.SUCCESS.is_done

#     def test_task_status_is_not_done_if_running(self):

#         assert not _core.TaskStatus.RUNNING.is_done

#     def test_task_status_in_progress_if_running(self):

#         assert _core.TaskStatus.RUNNING.in_progress

#     def test_task_status_success_if_SUCCESS(self):

#         assert _core.TaskStatus.SUCCESS.success

#     def test_task_status_not_success_if_FAILURE(self):

#         assert not _core.TaskStatus.FAILURE.success

#     def test_or_returns_success_if_one_success(self):

#         assert (_core.TaskStatus.SUCCESS | _core.TaskStatus.FAILURE).success

#     def test_or_returns_success_if_one_success_and_running(self):

#         assert (_core.TaskStatus.SUCCESS | _core.TaskStatus.RUNNING).success

#     def test_or_returns_running_if_failure_and_running(self):

#         assert (_core.TaskStatus.FAILURE | _core.TaskStatus.RUNNING).running

#     def test_and_returns_success_if_one_failure(self):

#         assert (_core.TaskStatus.SUCCESS & _core.TaskStatus.FAILURE).failure

#     def test_or_returns_success_if_one_success_and_running(self):

#         assert (_core.TaskStatus.FAILURE & _core.TaskStatus.RUNNING).failure

#     def test_or_returns_running_if_failure_and_running(self):

#         assert (_core.TaskStatus.SUCCESS & _core.TaskStatus.RUNNING).running

#     def test_invert_converts_failure_to_success(self):

#         assert (_core.TaskStatus.FAILURE.invert()).success

#     def test_invert_converts_success_to_failure(self):

#         assert (_core.TaskStatus.SUCCESS.invert()).failure


# class TestFromBool(object):

#     def test_from_bool_returns_success_for_true(self):
#         assert _core.from_bool(True) == _core.TaskStatus.SUCCESS

#     def test_from_bool_returns_failure_for_false(self):
#         assert _core.from_bool(False) == _core.TaskStatus.FAILURE



