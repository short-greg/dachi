import typing
from dachi.store import _concept
from dachi.store._concept import (
    Col, RepIdx, Concept, ConceptManager
)
import faiss
import numpy as np
import pandas as pd
import pydantic

from dataclasses import field
from dataclasses import dataclass


class Person(_concept.Concept):

    __manager__: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager

    name: str
    age: int


# class PersonRep(_concept.Concept):

#     manager: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager

#     name: str
#     age: int


class TestConcept:

    def test_concept_adds_columns(self):

        person = Person(name='X', age=10)
        assert person.age == 10

    def test_can_get_data_after_saving(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        data = _concept.concept_manager.get_data(Person)
        assert data.loc[0, 'age'] == 10

    def test_can_get_data_after_saving_two_rows(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        person2 = Person(name='Y', age=20)
        person2.save()
        data = _concept.concept_manager.get_data(Person)
        assert data.loc[1, 'name'] == 'Y'

    def test_only_one_row_if_saved_twice(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        person.name = 'Z'
        person.save()
        data = _concept.concept_manager.get_data(Person)
        assert data.loc[0, 'name'] == 'Z'
        assert len(data.index) == 1

    def test_filter_gets_correct_person(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        person2 = Person(name='Y', age=20)
        person2.save()
        query = Person.filter(Col('age') == 20)
        
        persons = []
        for c in query:
            persons.append(c)
        assert persons[0].age == 20

    def test_exclude_gets_correct_person(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        person2 = Person(name='Y', age=20)
        person2.save()
        query = Person.exclude(Col('age') == 20)
        
        persons = []
        for c in query:
            persons.append(c)
        assert persons[0].age == 10



class TestCol:

    def test_eq_retrieves_less_than(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') == 2
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 4

    def test_lt_retrieves_less_than(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') < 2
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 3

    def test_gt_retrieves_greater_than(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') > 1
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 4

    def test_ge_retrieves_greater_than_or_equal(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') >= 2
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 4

    def test_ge_retrieves_greater_than_or_equal(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') >= 2
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 4


class TestComp:

    def test_and_retrieves_correct_row(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = (Col('x') == 2) & (Col('y') == 4)
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 4

    def test_or_retrieves_correct_row(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = (Col('x') == 2) | (Col('y') == 9)
        df2 = comp(df)
        assert df2.iloc[0]['y'] == 4


class TestRepIdx:

    def test_rep_idx_creates_rep(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepIdx('x', faiss.IndexFlatL2(4), emb)
        
        idx.add(0, 'hi')
        assert len(idx) == 1

    def test_rep_idx_removes_from_rep(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepIdx('x', faiss.IndexFlatL2(4), emb)
        
        idx.add(0, 'hi')
        idx.remove(0)
        assert len(idx) == 0
    
    def test_rep_idx_adds_multiple(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepIdx('x', faiss.IndexFlatL2(4), emb)
        
        idx.add(0, 'hi')
        idx.add(1, 'bye')
        assert len(idx) == 2

    def test_rep_idx_gets_only_one_similar(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepIdx('x', faiss.IndexFlatL2(4), emb)
        
        idx.add(0, 'hi')
        idx.add(1, 'bye')
        similar = idx.like(['x'], 1)

        assert len(similar) == 1


class TestRepFactory:

    def test_rep_factory_creates_rep_idx(self):

        emb = lambda x: np.random.randn(len(x), 4)
        factory = RepIdx.F(
            'x', faiss.IndexFlatL2, emb, 4
        )
        idx = factory()
        
        idx.add(0, 'hi')
        assert len(idx) == 1

    def test_rep_idx_removes_from_rep(self):

        emb = lambda x: np.random.randn(len(x), 4)
        factory = RepIdx.F(
            'x', faiss.IndexFlatL2, emb, 4
        )
        idx = factory()
        
        idx.add(0, 'hi')
        idx.remove(0)
        assert len(idx) == 0


class PersonWRep(_concept.Concept):

    manager: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager
    name: str
    age: int

    @dataclass
    class __rep__(_concept.RepMap):

        name: RepIdx = _concept.RepIdx.field(
            'name', faiss.IndexFlatL2, 
            lambda x: np.random.randn(len(x), 4), 4
        )


class TestConceptWithRepField(object):
    
    def test_build_produces_new_person(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        data = _concept.concept_manager.get_data(Person)
        assert data.loc[0, 'age'] == 10


# How do I incorportae similarity

# Sim, Comp => both of these return 
# How to 

# for each index => set boolean to True
# 
