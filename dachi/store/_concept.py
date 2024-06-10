# 1st party
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields, MISSING, field
import typing
from typing_extensions import Self
from functools import wraps
import inspect
import math
from dataclasses import InitVar

# 3rd party
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from pydantic import Field
from ._base import conceptmethod

from ._rep import Rep, Sim


class BaseConcept(BaseModel):
    pass


class Concept(BaseConcept):

    __manager__: typing.ClassVar['ConceptManager'] = None

    id: int = None

    # y_idx: typing.Optional[np.array] = None

    @dataclass
    class __rep__(Rep):
        pass

    @classmethod
    def columns(cls, dtypes: bool=False) -> typing.List[str]:
        schema = cls.model_json_schema()

        columns = list(schema['properties'].keys())
        if dtypes:
            defaults, default_factories = [], []
            for name, field in cls.model_fields.items():
                defaults.append(field.default)
                default_factories.append(field.default_factory)
            types = [cls.__annotations__[c] if c in cls.__annotations__ else None for c in columns]

            return columns, types, defaults, default_factories
        
        return columns

    @conceptmethod
    def get_data(cls) -> pd.DataFrame:
        return cls.__manager__.get_data(cls)

    @conceptmethod
    def get_rep(cls) -> Rep:
        return cls.__manager__.get_rep(cls)

    @classmethod
    def manager(cls) -> 'ConceptManager':
        return cls.__manager__

    @conceptmethod
    def reps(cls) -> typing.List[str]:
        
        return [key for key, _ in cls.__rep__]

    @conceptmethod
    def build(cls):
        cls.__manager__.add_concept(cls)

    @conceptmethod
    def get(cls, id) -> 'Concept':

        df = cls.__manager__.get(
            cls
        )
        return cls(
            df.loc[id][cls.columns()]
        )

    def save(self):
        self.__manager__.add_row(self)

    @conceptmethod
    def to_df(cls) -> pd.DataFrame:
        return cls.__manager__.get_data(
            cls, cls.columns()
        )

    def to_dict(self) -> typing.Dict:
        return self.model_dump()

    def to_series(self) -> pd.DataFrame:
        return pd.Series(self.to_dict())

    @conceptmethod
    def filter(cls, comp: 'Comp'):
        return ConceptQuery(
            cls, comp
        )
    
    @conceptmethod
    def all(cls):
        return ConceptQuery(
            cls
        )

    @conceptmethod
    def like(cls, sim: 'BaseR', k: int=None):
        return ConceptQuery(
            cls, Like(sim, k)
        )

    @conceptmethod
    def exclude(cls, comp: 'Comp'):
        return ConceptQuery(
            cls, ~comp
        )

    @classmethod
    def concept_name(cls):
        
        return cls.__name__
    
    @classmethod
    def model_name(cls):
        
        return cls.__name__

    @classmethod
    def ctype(cls) -> typing.Type['Concept']:
        return cls


class Derived(BaseConcept):

    data: typing.Dict[str, typing.Any]
    
    def __getattr__(self, key: str) -> typing.Any:

        if key not in self.data:
            raise AttributeError(f'{key} is not a member of the derived concept.')
        return self.data[key]



class ConceptRepMixin(object):

    @conceptmethod
    def like(cls, comp: 'Comp'):
        return ConceptQuery(
            cls, comp
        )

    @conceptmethod
    def build(cls):
        cls.manager().add_rep(cls)

    @classmethod
    def model_name(cls) -> str:
        concept_cls = None
        for base in cls.__bases__:
            if inspect.isclass(base) and issubclass(base, Concept):
                concept_cls = base
                return concept_cls.concept_name()
        return None

    @classmethod
    def rep_name(cls):
        
        return cls.__name__

    @classmethod
    def concept_name(cls):
        
        return cls.concept().concept_name()
    
    @classmethod
    def model_name(cls):
        
        return cls.__name__

    @classmethod
    def concept(cls) -> typing.Type[Concept]:

        for base in cls.__bases__:
            if issubclass(base, Concept):
                return base
        return None


class Val(object):

    def __init__(self, val) -> None:
        self.val = val

    def query(self, df: pd.DataFrame, rep_map: Rep):
        return self.val


class Filter(object):

    @abstractmethod
    def query(self, df: pd.DataFrame, rep_map: Rep) -> pd.Series:
        pass

    def __call__(self, df: pd.DataFrame, rep_map: Rep) -> pd.DataFrame:
        """

        Args:
            df (pd.DataFrame): The dataframe to retrieve from
            rep_map (RepMap): The repmap to retrieve from

        Returns:
            pd.DataFrame: 
        """
        return df[self.query(df, rep_map)]

    def __xor__(self, other) -> 'Comp':

        return Comp(self, other, lambda lhs, rhs: lhs ^ rhs)

    def __and__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs & rhs)

    def __or__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs | rhs)
    
    def __invert__(self):

        return Comp(None, self, lambda lhs, rhs: ~rhs)


class Comp(Filter):

    def __init__(
        self, lhs: typing.Union['Comp', 'Col', typing.Any], 
        rhs: typing.Union['Comp', 'Col', typing.Any], 
        f: typing.Callable[[typing.Any, typing.Any], bool]
    ) -> None:
        """_summary_

        Args:
            lhs (typing.Union[BinComp;, Col;, typing.Any]): _description_
            rhs (typing.Union[BinComp;, &#39;Col&#39;, typing.Any]): _description_
            f (typing.Callable[[typing.Any, typing.Any], bool]): _description_
        """
        
        if not isinstance(lhs, Filter) and not isinstance(lhs, Col):
            lhs = Val(lhs)

        if not isinstance(rhs, Filter) and not isinstance(rhs, Col):
            rhs = Val(rhs)
        
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def query(self, df: pd.DataFrame, rep_map: Rep) -> pd.Series:
        """

        Args:
            df (pd.DataFrame): 

        Returns:
            The filter by comparison: 
        """
        print(type(self.lhs))
        lhs = self.lhs.query(df, rep_map)
        print('Result: ', type(lhs))
        rhs = self.rhs.query(df, rep_map)
        print('executing ', type(lhs), type(rhs))
        return self.f(lhs, rhs)


class BaseR(ABC):

    @abstractmethod
    def query(self, df: pd.DataFrame, rep_map: Rep) -> 'Sim':
        pass
    
    def __call__(self, df: pd.DataFrame, rep_map: Rep) -> 'Sim':

        return self.query(df, rep_map)

    def __mul__(self, other) -> Self:

        return AggR(
            self, other, lambda x, y: x * y
        )

    def __rmul__(self, other) -> Self:

        return AggR(
            self, other, lambda x, y: x * y
        )        

    def __add__(self, other) -> Self:

        return AggR(
            self, other, lambda x, y: x + y
        )

    def __radd__(self, other) -> Self:

        return AggR(
            self, other, lambda x, y: x + y
        )
    
    def __sub__(self, other) -> Self:

        return AggR(
            self, other, lambda x, y: x - y
        )
    
    def __rsub__(self, other) -> Self:

        return AggR(
            self, other, lambda x, y: y - x
        )
    # How to limit the "similarity"


class R(BaseR):

    def __init__(self, name: str, val, k: int):
        """

        Args:
            name (str): 
            val (): 
            k (int): 
        """
        # val could also be a column
        # or a compare
        self.name = name
        self.val = val
        self.k = k

    def query(self, df: pd.DataFrame, rep_map: Rep) -> 'Sim':
        """
        Args:
            rep_map (RepMap): 
            df (pd.DataFrame): 

        Returns:
            Similarity: 
        """
        indices = df.index.tolist()

        if isinstance(self.val, Col):
            val = self.val.query(df).index
        elif isinstance(self.val, typing.List):
            val = self.val
        else:
            val = [self.val]

        rep_idx = rep_map[self.name]
        
        return rep_idx.like(val, self.k, subset=indices)


class AggR(BaseR):

    def __init__(self, lhs, rhs, f: typing.Callable[[typing.Any, typing.Any], 'Sim']):
        """Aggregate the similarity

        Args:
            lhs: The left hand side of the aggregation
            rhs: The right hand side of the aggregation
            f (typing.Callable[[typing.Any, typing.Any], Similarity;]): 
        """
        # val could also be a column
        # or a compare
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def query(self, df: pd.DataFrame, idx_map: Rep) -> 'Sim':
        """
        Args:
            idx_map (IdxMap): 
            df (pd.DataFrame): 

        Returns:
            Similarity: 
        """
        if isinstance(self.lhs, BaseR):
            lhs = self.lhs.query(df, idx_map)
        else:
            lhs = self.lhs
        
        if isinstance(self.rhs, BaseR):
            rhs = self.rhs.query(df, idx_map)
        else:
            rhs = self.rhs
        s = self.f(lhs, rhs)
        return s


class Like(Filter):

    def __init__(self, sim: BaseR, k: int=None):
        """

        Args:
            sim (BaseSim): The similarities to use
        """
        self.sim = sim
        self.k = k

    def query(self, df: pd.DataFrame, rep_map: Rep) -> pd.Series:
        """

        Args:
            rep_map (RepMap): The RepMap for the concept
            df (pd.DataFrame): The DataFrame for the concept

        Returns:
            pd.Series: The 
        """
        similarity = self.sim(df, rep_map)

        # TODO: retrieve the top k

        similarity = similarity.topk(self.k)

        series = pd.Series(
            np.full((len(df.index)), False, np.bool_),
            df.index
        )
        print(similarity.indices)
        series[similarity.indices] = True
        return series

    def __call__(self, df: pd.DataFrame, rep_map: Rep) -> pd.DataFrame:
        
        return df[self.query(df, rep_map)]


class Col(object):
    """A column in the model
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): Name of the column
        """
        self.name = name
    
    def __eq__(self, other) -> 'Comp':
        """Check the eqquality of two columns

        Args:
            other : The other Col com compare with

        Returns:
            Comp: The comparison for equality
        """
        return Comp(self, other, lambda lhs, rhs: lhs == rhs)
    
    def __lt__(self, other) -> 'Comp':
        """Check whether column is less than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than
        """
        return Comp(self, other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other) -> 'Comp':
        """Check whether column is less than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than or equal
        """

        return Comp(self, other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other) -> 'Comp':
        """Check whether column is greater than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than
        """
        return Comp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other) -> 'Comp':
        """Check whether column is greater than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than or equal to
        """
        return Comp(self, other, lambda lhs, rhs: lhs >= rhs)

    def query(self, df: pd.DataFrame, rep_map: Rep) -> typing.Any:
        """Retrieve the column

        Args:
            df (pd.DataFrame): The DataFrame to retrieve from

        Returns:
            typing.Any: The 
        """
        return df[self.name]


class F(ABC):
    
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class Selector(ABC):
    
    def __init__(self, alias: str, to_select: typing.Union[Col, str, F]):

        self.alias = alias
        self.to_select = to_select

    @abstractmethod
    def annotate(self, df: pd.DataFrame, rep_map) -> pd.DataFrame:
        pass

    def select(
        self, cur_df: typing.Union[pd.DataFrame, None], 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        
        if cur_df is None:
            cur_df = pd.DataFrame()
        cur_df[self.alias] = df[self.alias]
        return cur_df


class ColSelector(Selector):
    
    def __init__(self, alias: str, to_select: Col):

        self.alias = alias
        self.to_select = to_select

    def annotate(self, df: pd.DataFrame, rep_map) -> pd.DataFrame:
        
        if self.alias in df.columns.values:
            raise ValueError('')

        try:        
            df[self.alias] = self.to_select.query(df, rep_map)
        except KeyError:
            raise KeyError(f'The key {self.to_select} is not present in the dataframe being annotated.')

        return df


class StrSelector(Selector):
    
    def __init__(self, alias: str, to_select: str):

        self.alias = alias
        self.to_select = to_select

    def annotate(self, df: pd.DataFrame, rep_map) -> pd.DataFrame:

        if self.alias in df.columns.values:
            raise ValueError('')

        try:
            df[self.alias] = df[self.to_select]
        except KeyError:
            raise KeyError(f'The key {self.to_select} is not present in the dataframe being annotated.')
        return df


def create_selector(alias: str, select):

    if isinstance(select, Col):
        return ColSelector(alias, select)
    if isinstance(select, str):
        return StrSelector(alias, select)
    if isinstance(select, F):
        return FSelector(alias, select)
    raise ValueError(
        f'Cannot create selector for an object of type {type(select)}'
    )


class FSelector(Selector):
    
    def __init__(self, alias: str, to_select: F):

        self.alias = alias
        self.to_select = to_select

    def annotate(self, df: pd.DataFrame, rep_map) -> pd.DataFrame:
        if self.alias in df.columns.values:
            raise ValueError('')

        df[self.alias] = F(df)
        return df


class Selection(object):

    def __init__(self, selectors: typing.List[Selector]=None):

        self.selectors = selectors or []

    def annotate(self, df: pd.DataFrame, rep_map) -> pd.DataFrame:
        
        cur_df = None
        for selector in self.selectors:
            cur_df = selector.annotate(df, rep_map)
        return cur_df

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        
        cur_df = None
        for selector in self.selectors:
            cur_df = selector.select(cur_df, df)

        return cur_df


C = typing.TypeVar('C', bound=BaseConcept)


class BaseQuery(ABC):

    @abstractmethod
    def filter(self, comp: Comp) -> Self:
        pass

    @abstractmethod
    def select(self, **kwargs) -> 'DerivedQuery':
        pass
    
    @abstractmethod
    def __iter__(self) -> typing.Iterator['BaseConcept']:
        pass
        
    @abstractmethod
    def df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def join(self, query: 'BaseQuery', alias: str, on_: str, how: str='inner') -> 'DerivedQuery':
        pass

    def inner(self, query: 'BaseQuery', alias: str, on_: str) -> Self:
        return self.join(query, alias, on_, 'inner')

    def left(self, query: 'BaseQuery', alias: str, on_: str) -> Self:
        return self.join(query, alias, on_, 'left')

    def right(self, query: 'BaseQuery', alias: str, on_: str) -> Self:
        return self.join(query, alias, on_, 'right')


class Join(object):

    def __init__(self, alias: str, query: BaseQuery, on_: str, how: str='inner'):

        self.query = query
        self.on_ = on_
        self.alias = alias
        self.how = how

    def join(self, df: pd.DataFrame) -> pd.DataFrame:
        
        join_df = self.query.df()
        join_df = join_df.copy()

        columns = [f'{self.alias}.{column}' for column in join_df.columns.values]

        join_df = join_df.set_axis(columns, axis=1)
        return df.merge(
            join_df, left_on=self.on_, right_on=f'{self.alias}.{self.on_}', how=self.how
        )


class ConceptQuery(BaseQuery, typing.Generic[C]):

    def __init__(
        self, concept_cls: typing.Type[C], 
        comp: Filter=None, 
        joined: typing.Dict[str, Join]=None
    ):
        """
        Args:
            concept_cls (typing.Type[C]): 
            comp (BaseComp): 
        """
        self.concept = concept_cls
        self.comp = comp
        self.joined: typing.Dict[str, Join] = joined or {}

    def filter(self, comp: Comp) -> Self:

        if self.comp is None:
            comp = comp
        else:
            comp = comp & comp
        return ConceptQuery[C](
            self.concept, comp,
            self.joined
        )
    
    def join(self, query: BaseQuery, alias: str, on_: str, how: str='inner') -> 'ConceptQuery[C]':
        
        if alias in self.joined:
            raise KeyError(f'Already  {alias} already defined.')
        joined = {
            alias: Join(alias, query, on_, how=how),
            **self.joined
        }
        return ConceptQuery[C](
            self.concept, self.comp, joined
        )

    def select(self, **kwargs) -> 'DerivedQuery':
        # 1) uses an alias.. a='x', y=Col(['x', 'y']), x=F('x') + F('y')
        # 1) if it is a 
        # selection = [*self.selection.selectors]
        selection = []
        for k, v in kwargs.items():
            if not isinstance(v, Selector):
                selector = create_selector(k, v)

            selection.append(selector)
        return DerivedQuery(
            self.concept, Selection(selection), 
            self.comp, self.joined
        )
    
    def __iter__(self) -> typing.Iterator[C]:

        concept = self.concept.get_data()
        idx = self.concept.get_rep()

        for k, to_join in self.joined.items():
            concept = to_join.join(
                concept
            )
        
        if self.comp is not None:
            sub_df = self.comp(concept, idx)
        else:
            sub_df = concept

        for _, row in sub_df.iterrows():
            yield self.concept(**row.to_dict())
        
    def df(self) -> pd.DataFrame:

        concept = self.concept.get_data()
        idx = self.concept.get_rep()

        for k, to_join in self.joined.items():
            concept = to_join.join(
                concept
            )

        if self.comp is None:
            return concept

        return self.comp(concept, idx)


class DerivedQuery(BaseQuery):

    def __init__(
        self, base: typing.Type[BaseConcept], 
        selection: Selection, comp: Comp=None,
        joined: Join=None
    ):
        """
        Args:
            concept_cls (typing.Type[C]): 
            comp (BaseComp): 
        """
        self.base = base
        self.comp = comp
        self.selection = selection
        self.joined: typing.Dict[str, Join] = joined or {}

    def filter(self, comp: Comp) -> Self:

        if self.comp is None:
            comp = comp
        else:
            comp = comp & comp
        return DerivedQuery(
            self.base, 
            self.selection, 
            comp, self.joined
        )
    
    def join(self, query: BaseQuery, alias: str, on_: str, how: str='inner') -> 'DerivedQuery':
        
        if alias in joined:
            raise KeyError(f'Already  {alias} already defined.')
        joined = {
            alias: Join(alias, query, on_, how=how)
            **self.joined
        }
        return DerivedQuery(
            self.concept, self.selection, 
            self.comp, joined
        )

    def select(self, **kwargs) -> Self:
        # 1) uses an alias.. 
        # a='x', y=Col(['x', 'y']), x=F('x') + F('y')
        # 1) if it is a 
        selection = [*self.selection.selection]
        for k, v in kwargs.items():
            selection.append(Selector(k, v))
        return DerivedQuery(
            self.base, selection, self.comp, self.joined
        )
    
    def __iter__(self) -> typing.Iterator[C]:
        # else create the concept
        # How to handle selections?
        concept = self.base.get_data()
        idx = self.base.get_rep()
        concept = self.selection.annotate(concept, idx)

        sub_df = self.comp(concept, idx)
        sub_df = self.selection.select(sub_df)

        for _, row in sub_df.iterrows():
            yield Derived(data=row.to_dict())
        
    def df(self) -> pd.DataFrame:

        concept = self.base.get_data()
        idx = self.base.get_rep()
        concept = self.selection.annotate(concept, idx)
        # How to handle it if the user 
        # i think the easiest way to is to prevent
        # overwriting a field name in annotate
        concept = self.comp(concept, idx)
        return self.selection.select(concept)


class ConceptManager(object):

    def __init__(self):

        self._concepts = {}
        self._field_reps: typing.Dict[str, Rep] = {}
        self._concept_reps: typing.Dict[str, typing.List] = {}
        self._ids = {}

    def reset(self):
        self._concepts = {}
        self._field_reps: typing.Dict[str, Rep] = {}
        self._concept_reps: typing.Dict[str, typing.List] = {}
        self._ids = {}

    def add_concept(self, concept: typing.Type[Concept]):

        columns, dtypes, _, _ = concept.columns(dtypes=True)

        df = pd.DataFrame(
            columns=columns
        )
        df = df.astype(dict(zip(columns, dtypes)))

        self._concepts[concept.concept_name()] = df
        self._field_reps[concept.concept_name()] = concept.__rep__()
        self._ids[concept.concept_name()] = 0

    def add_rep(self, rep: typing.Type[ConceptRepMixin]):

        if not issubclass(rep, Concept):
            raise ValueError('Cannot build Rep unless mixed with a concept.')
        columns, dtypes, defaults, default_factories = rep.columns(True)
        
        if rep.concept_name() not in self._concepts:
            raise RuntimeError(f'There is no concept named {rep.concept_name()} so the Rep {rep.model_name()} cannot be built')
        df = self.get_data(rep.concept())

        for c, dtype, default, default_factory, in zip(columns, dtypes, defaults, default_factories):
            if default_factory is not None and default is None:
                df[c] = default_factory()
            elif default is not None:
                df[c] = default
            else:
                df[c] = None
        # add columns for the representation
        df = df.astype(dict(zip(columns, dtypes)))
        self._concepts[rep.concept_name()] = df
        base_rep = self._field_reps[rep.concept_name()]
        self._field_reps[rep.model_name()] = rep.__rep__(base_rep=base_rep)

    def get_rep(self, concept: typing.Type[Concept]) -> Rep:

        return self._field_reps[concept.model_name()]

    def get_data(self, concept: typing.Type[Concept]) -> pd.DataFrame:

        return self._concepts[concept.concept_name()][concept.columns()]
    
    def add_row(self, concept: Concept):

        try:
            df = self._concepts[concept.concept_name()]
        except KeyError:
            raise KeyError(
                f'No concept named {concept.concept_name()}. '
                'Has it been built with Concept.build()?')
        if concept.id is None:
            concept.id = self._ids[concept.concept_name()]

            # Find a better way to handle this
            self._ids[concept.concept_name()] += 1
        
        rep = self._field_reps[concept.model_name()]
        df.loc[concept.id] = concept.to_dict()
        rep.add(concept.to_dict())
    
    def delete_row(self, concept: Concept):

        try:
            df: pd.DataFrame = self._concepts[concept.concept_name()]
        except KeyError:
            raise KeyError(
                f'No concept named {concept.concept_name()}. '
                'Has it been built with Concept.build()?')

        if concept.id is None:
            # TODO: decide how to handle this
            return
        
        rep = self._field_reps[concept.model_name()]
        df.drop(index=concept.id)
        rep.drop(concept.id)
        concept.id = None



concept_manager = ConceptManager()



""" 
# query.join(query, on=.., name=...)
# # # how about select?? Selet determines some aliases
# # # query.select(...) # create
# sub_query (.joined)
# annotations (.annotations)
# selection (.select)
#    .select twice will add multiple selections
# if there are selections or annotations it will not return the
# base concept class

"""


# like( )  <= 
# Sim() <= I want this to return numerical values

# This makes it a comparison
# Sim() returns a "Similarity"
# like(Sim() + Sim(), N=10))

# # I want to make it something like this
# like(0.5 * Sim('R', Comp) + 0.5 Sim()

# Sim() + 


# Think about how to handle this
# Have ConceptQuery
# And DerivedQuery
# 


# TODO: Change this... Have the similarity
# contain all indices + a "chosen"
# if not "chosen" will be 0


# TODO: Add in Maximize/Minimize into similarity
#    These values must depend on the index used
#    This will affect adding and subtracting the similarities



# TODO: ad in a base class

