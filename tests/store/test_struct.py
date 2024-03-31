from dachi.store import _struct


class TestRole:

    def test_role_sets_name(self):

        assistant = _struct.Role(name='assistant')
        assert assistant.name == 'assistant'

    def test_role_has_default_descr(self):

        assistant = _struct.Role(name='assistant')
        assert assistant.descr == ''


class TestText:

    def test_text_has_text_field(self):

        text = _struct.Text(text='assistant')
        assert text.text == 'assistant'

