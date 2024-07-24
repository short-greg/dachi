from dachi._core import _instruct as core
from dachi._core._core import Struct, str_formatter
from .test_core import SimpleStruct


# class TestOp:

#     def test_op_outputs_an_instruction(self):

#         role = Role(name='Assistant', duty='You are a helpful assistant')

#         text = 'Evaluate the user'
#         instruction = _core.op(
#             [role], text, 'Evaluate'
#         )
#         assert 'Assistant' in instruction.text



# class TestBullet(object):

#     def test_bullet_outputs_x(self):
#         out = core.bullet(
#             ['x', 'y']
#         )
#         assert '-x' in out.render()

#     def test_bullet_outputs_y(self):
#         out = core.bullet(
#             ['x', 'y']
#         )
#         assert '-y' in out.render()

# class TestNumbered(object):

#     def test_numbered_outputs_x(self):
#         out = core.numbered(
#             ['x', 'y']
#         )
#         assert '2. y' in out.render()

#     def test_numbered_outputs_x(self):
#         out = core.numbered(
#             ['x', 'y'], numbering='roman'
#         )
#         assert 'i. y' in out.render()


# class TestFill(object):

#     def test_fill_updates_text(self):

#         out = core.fill(
#             '{x}', x=2
#         )
#         assert out.text == '2'

#     def test_fill_updates_first_item(self):

#         out = core.fill(
#             '{x} {x2}', x=2
#         )
#         assert out.text == '2 {x2}'

#     def test_fill_updates_second_item(self):

#         out = core.fill(
#             '{x} {x2}', x2=2
#         )
#         assert out.text == '{x} 2'

#     def test_fill_updates_output_of_struct(self):

#         struct = SimpleStruct(
#             x='{x}'
#         )

#         out = core.fill(
#             struct, x2=2
#         )
#         assert out.text == '2'
