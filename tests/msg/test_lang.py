import pytest
from dachi.msg import Term, Glossary


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

    def test_render_no_meta(self):
        term = Term(name="AI", definition="Artificial Intelligence")
        assert term.render() == "AI: Artificial Intelligence"

    def test_render_with_meta(self):
        term = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia", usage="Technology")
        rendered = term.render()
        assert "AI: Artificial Intelligence" in rendered
        assert "source: Wikipedia" in rendered
        assert "usage: Technology" in rendered


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

    def test_render_glossary(self):
        glossary = Glossary()
        term1 = Term(name="AI", definition="Artificial Intelligence", source="Wikipedia")
        term2 = Term(name="ML", definition="Machine Learning", source="Encyclopedia")
        glossary.add(term1)
        glossary.add(term2)
        rendered = glossary.render()
        assert "AI: Artificial Intelligence" in rendered
        assert "source: Wikipedia" in rendered
        assert "ML: Machine Learning" in rendered
        assert "source: Encyclopedia" in rendered

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


