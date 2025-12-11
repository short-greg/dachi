import pytest
from .._structs import Role

"""Unit tests for new criterion system (_criterion.py)

This suite covers the EvalField-based criterion architecture including:
- BaseCriterion auto-generation
- Concrete criterion types (PassFail, Likert, NumericalRating)
- Critic evaluation executor
"""

class TestTerm:
    def test_init(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        assert term.name == "AI"
        assert term.definition == "Artificial Intelligence"
        assert term.meta == {"source": "Wikipedia"}

    def test_getitem_single_key(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        assert term["source"] == "Wikipedia"

    def test_getitem_multiple_keys(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia", usage="Technology")
        assert term[["source", "usage"]] == ["Wikipedia", "Technology"]

    def test_setitem_single_key(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        term["source"] = "Encyclopedia"
        assert term.meta["source"] == "Encyclopedia"

    def test_setitem_multiple_keys(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia", usage="Technology")
        term[["source", "usage"]] = ["Encyclopedia", "Science"]
        assert term.meta["source"] == "Encyclopedia"
        assert term.meta["usage"] == "Science"


class TestGlossary:
    def test_add(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        assert glossary.terms["AI"] == term

    def test_get_term(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        retrieved_term = glossary["AI"]
        assert retrieved_term == term

    def test_remove_term(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        glossary.remove("AI")
        assert "AI" not in glossary.terms

    def test_update_term(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        updated_term = Term(name="AI", definition="Advanced Intelligence", source="Encyclopedia")
        glossary.update(updated_term)
        assert glossary.terms["AI"].definition == "Advanced Intelligence"
        assert glossary.terms["AI"].meta["source"] == "Encyclopedia"

    def test_get_nonexistent_term(self):
        glossary = Glossary()
        with pytest.raises(KeyError):
            glossary["Nonexistent"]

    def test_remove_nonexistent_term(self):
        glossary = Glossary()
        with pytest.raises(KeyError):
            glossary.remove("Nonexistent")


import pytest
from dachi.inst import Term, Glossary


class TestTerm:
    def test_init(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        assert term.name == "AI"
        assert term.definition == "Artificial Intelligence"
        assert term.meta == {"source": "Wikipedia"}

    def test_getitem_single_key(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        assert term["source"] == "Wikipedia"

    def test_getitem_multiple_keys(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia", usage="Technology")
        assert term[["source", "usage"]] == ["Wikipedia", "Technology"]

    def test_setitem_single_key(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        term["source"] = "Encyclopedia"
        assert term.meta["source"] == "Encyclopedia"

    def test_setitem_multiple_keys(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia", usage="Technology")
        term[["source", "usage"]] = ["Encyclopedia", "Science"]
        assert term.meta["source"] == "Encyclopedia"
        assert term.meta["usage"] == "Science"


class TestGlossary:
    def test_add(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        assert glossary.terms["AI"] == term

    def test_get_term(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        retrieved_term = glossary["AI"]
        assert retrieved_term == term

    def test_remove_term(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        glossary.remove("AI")
        assert "AI" not in glossary.terms

    def test_update_term(self):
        glossary = Glossary()
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        glossary.add(term)
        updated_term = Term(name="AI", definition="Advanced Intelligence", source="Encyclopedia")
        glossary.update(updated_term)
        assert glossary.terms["AI"].definition == "Advanced Intelligence"
        assert glossary.terms["AI"].meta["source"] == "Encyclopedia"

    def test_get_nonexistent_term(self):
        glossary = Glossary()
        with pytest.raises(KeyError):
            glossary["Nonexistent"]

    def test_remove_nonexistent_term(self):
        glossary = Glossary()
        with pytest.raises(KeyError):
            glossary.remove("Nonexistent")





# class TestRef:

#     def test_ref_does_not_output_text(self):

#         role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
#         ref = _structs.Ref(desc=role)
#         assert 'Helpful Assistant' in ref.desc.render()

#     def test_name_returns_name_of_reference(self):

#         role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
#         ref = _structs.Ref(desc=role)
#         # ref = ref.update(role='Helpful Assistant')
#         assert ref.name == 'Assistant'

#     def test_text_is_empty_string(self):

#         role = Role(name='Assistant', duty='You are a helpful Helpful Assistant')
#         ref = _structs.Ref(desc=role)
#         # ref = ref.update(role='Helpful Assistant')
#         assert ref.render() == role.name
