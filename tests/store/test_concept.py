import typing
from dachi.store1 import _concept
from dachi.store1._concept import (
    Col, 
    Like, ColSelector, R,
    StrSelector, FSelector, Selection,
    Join, Derived
)
from dachi.store1._rep import (
    Rep, RepFactory, RepLookup
)
from dachi.store1 import _rep
import faiss
import numpy as np
import pandas as pd
import pytest

from dataclasses import field
from dataclasses import dataclass


class Person(_concept.Concept):

    __manager__: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager

    name: str
    age: int


class DummyIdxMap(Rep):

    def __init__(self, n=2):

        emb = lambda x: np.array(x)
        idx = RepLookup('x', faiss.IndexFlatL2(n), emb, True)
        self.r = idx


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
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 4

    def test_lt_retrieves_less_than(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') < 2
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 3

    def test_gt_retrieves_greater_than(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') > 1
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 4

    def test_ge_retrieves_greater_than_or_equal(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') >= 2
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 4

    def test_ge_retrieves_greater_than_or_equal(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = Col('x') >= 2
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 4


class TestComp:

    def test_and_retrieves_correct_row(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = (Col('x') == 2) & (Col('y') == 4)
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 4

    def test_or_retrieves_correct_row(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        comp = (Col('x') == 2) | (Col('y') == 9)
        df2 = comp(df, None)
        assert df2.iloc[0]['y'] == 4


class TestIndex:

    def test_rep_idx_creates_rep(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepLookup('x', faiss.IndexFlatL2(4), emb, True)
        
        idx.add(0, 'hi')
        assert len(idx) == 1

    def test_rep_idx_removes_from_rep(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepLookup('x', faiss.IndexFlatL2(4), emb, True)
        
        idx.add(0, 'hi')
        idx.remove(0)
        assert len(idx) == 0
    
    def test_rep_idx_adds_multiple(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepLookup('x', faiss.IndexFlatL2(4), emb, True)
        
        idx.add(0, 'hi')
        idx.add(1, 'bye')
        assert len(idx) == 2

    def test_rep_idx_gets_only_one_similar(self):

        emb = lambda x: np.random.randn(len(x), 4)
        idx = RepLookup('x', faiss.IndexFlatL2(4), emb, True)
        
        idx.add(0, 'hi')
        idx.add(1, 'bye')
        similar = idx.like(['x'], 1)

        assert len(similar) == 1


class TestIdxFactory:

    def test_rep_factory_creates_rep_idx(self):

        emb = lambda x: np.random.randn(len(x), 4)
        factory = RepLookup.F(
            'x', faiss.IndexFlatL2, emb, 4
        )
        idx = factory()
        
        idx.add(0, 'hi')
        assert len(idx) == 1

    def test_rep_idx_removes_from_rep(self):

        emb = lambda x: np.random.randn(len(x), 4)
        factory = RepLookup.F(
            'x', faiss.IndexFlatL2, emb, 4
        )
        idx = factory()
        
        idx.add(0, 'hi')
        idx.remove(0)
        assert len(idx) == 0


class TestConceptWithRepField(object):
    
    def test_build_produces_new_person(self):

        Person.build()
        person = Person(name='X', age=10)
        person.save()
        data = _concept.concept_manager.get_data(Person)
        assert data.loc[0, 'age'] == 10


class TestSim:

    def test_sim_returns_similarity_for_first(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        sim = R('r', np.array([3, 4]), 1)
        similarity = sim(df, idx_map)
        assert 1 in similarity.indices
        assert 0 not in similarity.indices

    def test_sim_returns_both_values(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        sim = R('r', np.array([3, 4]), 2)
        similarity = sim(df, idx_map)
        assert 1 in similarity.indices
        assert 0 in similarity.indices

    def test_add_creates_agg_sim(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        sim = R('r', np.array([3, 4]), 2)
        sim2 = R('r', np.array([1, 3]), 2)
        sim3 = sim + sim2
        similarity = sim3(df, idx_map)
        assert 1 in similarity.indices
        assert 0 in similarity.indices

    def test_add_results_in_best_getting_chosen(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        sim = R('r', np.array([3, 4]), 1)
        sim2 = R('r', np.array([3, 3.5]), 1)
        sim3 = sim + sim2
        similarity = sim3(df, idx_map)
        assert 1 in similarity.indices
        assert 0 not in similarity.indices

    def test_add_results_in_both_getting_chosen(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        sim = R('r', np.array([3, 4]), 1)
        sim2 = R('r', np.array([1, 3]), 1)
        sim3 = 0.1 * sim + 2.0 * sim2
        similarity = sim3(df, idx_map)
        assert 1 in similarity.indices
        assert 0 in similarity.indices


class TestColSelector:

    def test_annotate_adds_columns_to_a_df(self):

        selector = ColSelector(
            'z', Col('x')
        )
        
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        annotated = selector.annotate(df, None)
        assert (
            (annotated['x'] == annotated['z']).all()
        )

    def test_annotate_adds_columns_to_a_df(self):

        selector = ColSelector(
            'z', Col('b')
        )
        
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        with pytest.raises(KeyError):
            selector.annotate(df, None)

    def test_select_has_added_z_and_removed_x(self):

        selector = ColSelector(
            'z', Col('x')
        )
        
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        annotated = selector.annotate(df, None)
        annotated = selector.select(None, df)
        assert (annotated['z'] == df['x']).all()
        assert 'x' not in annotated.columns.values


class TestSelection:

    def test_annotate_adds_columns(self):

        selection = Selection(
            [StrSelector('z', 'x'),
             StrSelector('a', 'y')]
        )
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        annotated = selection.annotate(
            df, None
        )
        assert (
            annotated['a'] == annotated['y']
        ).all()
        assert (
            annotated['x'] == annotated['z']
        ).all()

    def test_select_creates_fresh_df(self):

        selection = Selection(
            [StrSelector('z', 'x'),
             StrSelector('a', 'y')]
        )
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        annotated = selection.annotate(
            df, None
        )
        annotated = selection.select(
            df
        )
        assert (
            annotated['a'] == df['y']
        ).all()
        assert (
            annotated['z'] == df['x']
        ).all()
        assert 'x' not in annotated.columns.values
        assert 'y' not in annotated.columns.values


class TestStrSelector:

    def test_annotate_adds_columns_to_a_df(self):

        selector = StrSelector(
            'z', 'x'
        )
        
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        annotated = selector.annotate(df, None)
        assert (
            (annotated['x'] == annotated['z']).all()
        )

    def test_annotate_adds_columns_to_a_df(self):

        selector = StrSelector(
            'z', 'b'
        )
        
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        with pytest.raises(KeyError):
            selector.annotate(df, None)

    def test_select_has_added_z_and_removed_x(self):

        selector = StrSelector(
            'z', 'x'
        )
        
        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        annotated = selector.annotate(df, None)
        annotated = selector.select(None, df)
        assert (annotated['z'] == df['x']).all()
        assert 'x' not in annotated.columns.values


class TestLike:

    def test_like_returns_similarity_for_first(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        like = Like(R('r', np.array([3, 4]), 1), 1)
        result = like(df, idx_map)
        assert 1 in result.index
        assert 0 not in result.index

    def test_like_returns_similarity_for_first_when_topk(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))
        like = Like(R('r', np.array([3, 4]), 2), 1)
        result = like(df, idx_map)
        print(result)
        assert 1 in result.index
        assert 0 not in result.index

    def test_like_and_compare_can_both_be_used(self):

        df = pd.DataFrame(
            {'x': [1, 2], 'y': [3, 4]}
        )
        
        idx_map = DummyIdxMap()
        idx_map.r.add(0, np.array([1, 3]))
        idx_map.r.add(1, np.array([3, 4]))

        comp = Like(R('r', np.array([3, 4]), 2)) & (
            Col('x') < 2
        ) 
        result = comp(df, idx_map)
        assert 0 in result.index
        assert 1 not in result.index


class PersonWRep(_concept.Concept):

    __manager__: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager
    name: str
    age: int

    @dataclass
    class __rep__(_rep.Rep):

        age: RepLookup = _rep.RepLookup.field(
            'age', faiss.IndexFlatL2, 
            lambda x: np.array([x], dtype=np.float32), 1
        )


class Purchaser(_concept.Concept):

    __manager__: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager
    name: str
    amount: float

    @dataclass
    class __rep__(_rep.Rep):
        pass


def build_person():

    PersonWRep.build()
    person = PersonWRep(name='X', age=10)
    person2 = PersonWRep(name='X2', age=15)
    person.save()
    person2.save()


def build_purchaser():

    Purchaser.build()
    person = Purchaser(name='X', amount=15)
    person2 = Purchaser(name='X2', amount=30)
    person3 = Purchaser(name='Y', amount=30)
    person.save()
    person2.save()
    person3.save()


class TestJoin:

    def test_join_adds_two_dataframes_together(self):

        df = pd.DataFrame(
            {'name': ['X', 'Y'], 'y': [3, 4]}
        )
        query = PersonWRep.all()
        join = Join(
            'person', query, 'name', 'name'
        )
        joined = join.join(df)
        print(joined)
        assert joined['name'].isin(['X']).any()
        assert not joined['name'].isin(['X2']).any()
        assert joined['person.age'].isin([10]).any()
        

class TestConceptWithRepField:

    def test_that_embeddings_are_added(self):

        PersonWRep.build()
        person = PersonWRep(name='X', age=10)
        person2 = PersonWRep(name='X2', age=15)
        person.save()
        person2.save()
        # map_ = PersonWRep.__manager__._field_reps[PersonWRep.model_name()]
        
        result = PersonWRep.filter(
            Like(R('age', 10, 2))
        ).data()
        assert len(result.index) == 2

    def test_that_embeddings_are_added(self):

        PersonWRep.build()
        person = PersonWRep(name='X', age=10)
        person2 = PersonWRep(name='X2', age=15)
        person.save()
        person2.save()
        
        result = PersonWRep.filter(
            Like(R('age', 15, 1))
        ).data()
        assert len(result.index) == 1
        assert 1 in result.index
        assert 0 not in result.index

    def test_that_embeddings_are_not_added_if_not_saved(self):

        PersonWRep.build()
        PersonWRep(name='X', age=10)
        PersonWRep(name='X2', age=15)
        PersonWRep.__manager__._field_reps[PersonWRep.concept_name()]
        
        result = PersonWRep.filter(
            Like(R('age', 2, 1))
        ).data()
        assert len(result.index) == 0

    def test_that_like_works_in_place_of_filter(self):

        PersonWRep.build()
        person = PersonWRep(name='X', age=10)
        person2 = PersonWRep(name='X2', age=15)
        person.save()
        person2.save()
        
        result = PersonWRep.like(
            R('age', 15, 1)
        ).data()
        assert len(result.index) == 1
        assert 1 in result.index
        assert 0 not in result.index

    def test_join_two_concepts(self):

        build_person()
        build_purchaser()

        query = PersonWRep.all()
        query2 = Purchaser.all()
        query2 = query2.join(
            query, 'person', on_left='name', on_right='name'
        )
        df = query2.data()
        assert not df['name'].isin(['Y']).any()
        assert df['name'].isin(['X']).any()

    def test_join_two_concepts_with_filter(self):

        build_person()
        build_purchaser()

        query = PersonWRep.all()
        query2 = Purchaser.all()
        query2 = query2.join(
            query, 'person', on_='name', on_right='name'
        )
        query2 = query2.filter(Col('person.age') > 10)
        df = query2.df()
        assert not df['name'].isin(['Y']).any()
        assert not df['name'].isin(['X']).any()
        assert df['name'].isin(['X2']).any()

class TestDerivedConcept:

    def test_that_select_gets_df(self):

        build_person()

        query = PersonWRep.filter(
            Like(R('age', 10, 2))
        )
        query = query.select(person_age='age')
        df = query.data()
        assert 'name' not in df.columns.values
        assert 'person_age' in df.columns.values

    def test_that_select_produces_derived_concepts(self):

        build_person()

        query = PersonWRep.filter(
            Like(R('age', 10, 2))
        )
        query = query.select(person_age='age')
        for c in query:
            assert isinstance(c, Derived)


class BuyerWithRep(_concept.ConceptRepMixin, PersonWRep):

    # __manager__: typing.ClassVar[_concept.ConceptManager] = _concept.concept_manager
    purchaser: bool

    @dataclass
    class __rep__(_rep.Rep):

        id: RepLookup = _rep.RepLookup.field(
            'id', faiss.IndexFlatL2, 
            lambda x: np.array([x], dtype=np.float32), 1
        )

        age: RepLookup = _rep.RepLookup.field(
            'age', faiss.IndexFlatL2, 
            lambda x: np.array([[x[0] * 0.5]], dtype=np.float32), 1
        )


class TestBuyerRep:

    def test_representation_is_added(self):

        PersonWRep.__manager__.reset()
        PersonWRep.build()
        BuyerWithRep.build()
        assert BuyerWithRep.rep_name() in PersonWRep.__manager__._field_reps

    def test_cannot_build_rep_if_no_concept(self):

        PersonWRep.__manager__.reset()
        with pytest.raises(RuntimeError):
            BuyerWithRep.build()

    def test_that_embeddings_are_added_if_saved(self):

        PersonWRep.__manager__.reset()

        PersonWRep.build()
        BuyerWithRep.build()
        buyer1 = BuyerWithRep(name='X', age=10, purchaser=True)
        buyer2 = BuyerWithRep(name='X2', age=15, purchaser=False)
        buyer1.save()
        buyer2.save()
        
        result = BuyerWithRep.filter(
            Like(R('age', 2, 1))
        ).data()
        assert len(result.index) == 1

    def test_that_person_does_not_return_purchaser(self):

        PersonWRep.__manager__.reset()

        PersonWRep.build()
        BuyerWithRep.build()
        buyer1 = BuyerWithRep(name='X', age=10, purchaser=True)
        buyer2 = BuyerWithRep(name='X2', age=15, purchaser=False)
        buyer1.save()
        buyer2.save()
        
        result = PersonWRep.filter(
            Like(R('age', 2, 1))
        ).data()
        print(result.columns.values)
        assert 'purchaser' not in result.columns.values

    def test_that_buyer_does_return_purchaser(self):

        PersonWRep.__manager__.reset()

        PersonWRep.build()
        BuyerWithRep.build()
        buyer1 = BuyerWithRep(name='X', age=10, purchaser=True)
        buyer2 = BuyerWithRep(name='X2', age=15, purchaser=False)
        buyer1.save()
        buyer2.save()
        
        result = BuyerWithRep.filter(
            Like(R('age', 2, 1))
        ).data()
        print(result.columns.values)
        assert 'purchaser' in result.columns.values

    def test_that_id_rep_returns_one_row(self):

        PersonWRep.__manager__.reset()

        PersonWRep.build()
        BuyerWithRep.build()
        buyer1 = BuyerWithRep(name='X', age=10, purchaser=True)
        buyer2 = BuyerWithRep(name='X2', age=15, purchaser=False)
        buyer1.save()
        buyer2.save()
        
        result = BuyerWithRep.filter(
            Like(R('id', 1, 1))
        ).data()
        assert len(result.index) == 1
