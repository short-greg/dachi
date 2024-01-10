from typing import Dict
from dachi.struct import _prompting as prompting


class DummyComponent(prompting.Component):

    def __init__(self, x=1):
        super().__init__()
        self.x = x

    @property
    def y(self) -> int:
        return 2

    def as_text(self) -> str:
        return f"{self.x}"
    
    def as_dict(self) -> Dict:
        return {'x': 1}


class TestF:

    def test_f_returns_3_if_2_passed(self):

        f = prompting.F(lambda x: x + 1)
        assert f(2) == 3

    def test_f_returns_3_if_2_passed_in_kwargs(self):

        f = prompting.F(lambda x: x + 1, x=2)
        assert f() == 3

    def test_f_returns_3_if_2_passed_in_args(self):

        f = prompting.F(lambda x: x + 1, 2)
        assert f() == 3


class TestR:

    def test_r_returns_data(self):

        dummy = DummyComponent()
        f = prompting.R(dummy, "x")
        assert f() == 1

    def test_r_returns_2_if_y_called(self):

        dummy = DummyComponent()
        f = prompting.R(dummy, 'y')
        assert f() == 2


class TestComponent:

    def test_get_returns_value(self):
        dummy = DummyComponent()
        assert dummy.get('x') == 1

    def test_set_sets_value(self):
        dummy = DummyComponent()
        dummy.set('x', 2)
        assert dummy.x == 2

    def test_r_returns_R(self):
        dummy = DummyComponent(3)
        r = dummy.r('x')
        assert r() == 3


class TestComponent:

    def test_get_returns_value(self):
        dummy = DummyComponent()
        assert dummy.get('x') == 1


class TestPromptComponent:

    def test_prompt_format_removes_args(self):

        prompt = prompting.Prompt(
            ['x'], """Answer is {x}"""
        )
        answer = prompt.format(x=1)
        assert answer.text == """Answer is 1"""

    def test_prompt_leaves_args_if_not_provided(self):

        prompt = prompting.Prompt(
            ['x'], """Answer is {x}"""
        )
        answer = prompt.format()
        assert answer.text == """Answer is {x}"""

    def test_prompt_as_text_adds_header(self):

        prompt = prompting.Prompt(
            ['x'], """Answer is {x}"""
        )
        answer = prompt.format(x=2).as_text('==Prompt==')
        print(answer)
        assert answer == (
            f'==Prompt==\n'
            f'Answer is 2'
        )

    def test_prompt_spawn_spawns_same_prompt(self):

        prompt = prompting.Prompt(
            ['x'], """Answer is {x}"""
        )
        prompt2 = prompt.spawn()
        answer = prompt2.format(x=2)
        assert answer.text == """Answer is 2"""


class TestCompletion:

    def test_completion_response_is_true(self):

        prompt = prompting.Completion(
            prompting.Prompt([], """Answer is 1"""), 'True',
        )
        assert prompt.as_dict()['response'] == 'True'

    def test_completion_formats_the_prompt(self):

        prompt = prompting.Completion(
            prompting.Prompt(['x'], """Answer is {x}"""),
        )
        formatted = prompt.format_prompt(x=2)
        
        assert formatted.prompt.text == "Answer is 2"


class TestText:

    def test_text_returns_text_in_dict(self):

        text = prompting.Text(
            "Answer is 2"
        )
        assert text.as_text() == "Answer is 2"


class TestTurn:

    def test_spawn_creates_new_turn(self):

        text = prompting.Turn(
            prompting.Role('user'), 'x'
        )
        assert text.text == 'x'

    def test_as_text_returns_turn(self):

        text = prompting.Turn(
            prompting.Role('user'), 'x'
        )
        assert text.as_text() == "user: x"


class TestConv:

    def test_add_turn_adds_turn(self):

        conv = prompting.Conv()
        conv.add_turn('system', 'name three people')

        assert conv[0].text == 'name three people'

    def test_reset_removes_all_turns(self):

        conv = prompting.Conv()
        conv.add_turn('system', 'name three people')
        conv.reset()

        assert len(conv) == 0

    def test_as_text_returns_conv_as_text(self):

        conv = prompting.Conv()
        conv.add_turn('system', 'name three people')
        conv.add_turn('user', 'y j z')

        assert conv.as_text() == (
            'system: name three people\n'
            'user: y j z'
        )

    def test_spawn_creates_new_conv(self):

        conv = prompting.Conv()
        conv.add_turn('system', 'name three people')
        conv.add_turn('user', 'y j z')
        conv = conv.spawn()

        assert conv.as_text() == (
            ''
        )


class TestStoreList:

    def test_state_dict_returns_dict(self):

        conv = prompting.StoreList()
        conv.append(1)

        assert conv.state_dict()[0] == 1

    def test_as_dict_returns_dict(self):

        conv = prompting.StoreList()
        conv.append(1)

        assert conv.as_dict()[0] == 1

    def test_spawn_creates_new_store_list(self):

        conv = prompting.StoreList()
        conv.append(1)
        conv2 = conv.spawn()
        print(conv2)
        assert conv2.as_dict()[0] == 1

    def test_spawn_creates_new_store_list_with_turn(self):

        conv = prompting.StoreList()
        conv.append(prompting.Text('text'))
        conv2 = conv.spawn()
        assert conv2.as_dict()[0].text == 'text'
