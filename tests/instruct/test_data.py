from dachi.op import _data as _structs
from .._structs import Role


class TestRef:

    def test_ref_does_not_output_text(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _structs.Ref(desc=role)
        assert 'Helpful Assistant' in ref.desc.render()

    def test_name_returns_name_of_reference(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _structs.Ref(desc=role)
        # ref = ref.update(role='Helpful Assistant')
        assert ref.name == 'Assistant'

    def test_text_is_empty_string(self):

        role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
        ref = _structs.Ref(desc=role)
        # ref = ref.update(role='Helpful Assistant')
        assert ref.render() == role.name


class TestRendering:

    def test_text_for_description_is_correct(self):
        
        role = Role(name='Assistant', duty='You are a helpful assistant')
        text = role.render()

        assert text == f"""
        # Role Assistant

        {role.duty}
        """

    def test_text_for_description_is_correct_after_updating(self):
        
        role = Role(name='Assistant', duty='You are a helpful {role}')
        
        role = role.update(role='Sales Assistant')
        text = role.render()
        assert 'Sales Assistant' in text
