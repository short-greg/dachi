import asyncio
import typing
from typing import Any
from dachi.proc import _process as core
from dachi.proc import Param, Module
from dachi.proc import _process
from dachi.inst import Cue
import numpy as np


# class TestParam(object):

#     def test_get_x_from_param(self):

#         cue = Param(
#             name='X', cue='x'
#         )
#         assert cue.render() == 'x'

#     # def test_param_with_instruction_passed_in(self):

#     #     instruction = _core.Cue(
#     #         text='x', out=_core.StructRead(
#     #             name='F1',
#     #             out_cls=SimpleStruct
#     #         )
#     #     )

#     #     param = _core.Param(
#     #         name='X', instruction=instruction
#     #     )
#     #     assert param.render() == 'x'

# #     def test_read_reads_the_object(self):

# #         instruction = _core.Cue(
# #             text='x', out=_core.StructRead(
# #                 name='F1',
# #                 out_cls=SimpleStruct
# #             )
# #         )
# #         param = _core.Param(
# #             name='X', instruction=instruction
# #         )
# #         simple = SimpleStruct(x='2')
# #         assert param.reads(simple.to_text()).x == '2'

# class TestParam:

#     def test_param_renders_the_instruction(self):

#         param = Param(
#             name='p',
#             cue=Cue(
#                 text='simple instruction'
#             )
#         )
#         target = param.cue.render()
#         assert param.render() == target

#     def test_param_updates_the_instruction(self):

#         param = Param(
#             name='p',
#             cue=Cue(
#                 text='simple instruction'
#             ),
#             training=True
#         )
#         target = 'basic instruction'
#         param.update(target)
#         assert param.render() == target