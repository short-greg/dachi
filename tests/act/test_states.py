
# @pytest.mark.asyncio
# class TestStateMachine:
    
#     async def test_success_path(self):

#         def state_b():
#             return TaskStatus.SUCCESS
#         def state_a():
#             return state_b
#         sm = behavior.StateMachine(init_state=state_a)
#         assert await sm.tick() is TaskStatus.RUNNING
#         assert await sm.tick() is TaskStatus.SUCCESS

#     async def test_immediate_failure(self):
#         sm = behavior.StateMachine(init_state=lambda: TaskStatus.FAILURE)
#         assert await sm.tick() is TaskStatus.FAILURE

#     async def test_reset(self):
#         sm = behavior.StateMachine(init_state=lambda: TaskStatus.SUCCESS)
#         await sm.tick(); sm.reset()
#         assert sm.status is TaskStatus.READY
