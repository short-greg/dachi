from dachi.db import _core as core
import datetime
import pandas as pd

manager = core.DFConceptManager()

print(manager)

class SimpleConcept(manager.Concept):

    name: str
    time: int


class TestDFConceptManager:

    def test_add_concept_creates_new_concept(self):
        
        SimpleConcept.create()
        assert isinstance(manager._concepts['SimpleConcept'], pd.DataFrame)

    def test_add_row_adds_a_new_row(self):
        
        concept = SimpleConcept(name='x', time=1)
        concept.save()
        print(manager._concepts['SimpleConcept'])