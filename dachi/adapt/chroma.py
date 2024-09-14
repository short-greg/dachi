import typing

from ..store import (
    VectorizedStore, Comp, Join, Like, 
    QF, get_uuid, coalesce, UNCHANGED,
    VectorizedQuery
)
from ..store import df_sort, df_join, df_select

import pandas as pd
import chromadb
import uuid

# 1) Collection
# 2) 


# @classmethod
# def create(cls, collection: str, distance: str):

#     client = chromadb.Client()
#     collection = client.create_collection(
#         name=collection
#     )


def chroma_filter(comp: Comp):
    
    pass


def to_df(query_results: typing.Dict[typing.List]) -> pd.DataFrame:

    ids = query_results.get("ids")
    embeddings = query_results.get("embeddings", None)
    metadatas = query_results.get("metadatas", None)
    documents = query_results.get("documents", None)

    d = {
        'id': ids
    }
    if embeddings is not None:
        d['embeddings'] = embeddings
    if metadatas is not None:
        d.update(
            pd.DataFrame(metadatas).to_dict(orient="records")
        )
    if documents is not None:
        d['documents'] = documents

    return pd.DataFrame(d)


class ChromaDBStore(VectorizedStore):

    def __init__(
        self, collection: str, path: str,
        select: typing.Dict[str, str | QF] = None, 
        like: 'Like'=None, where: Comp = None, 
        order_by: typing.List[str] = None, limit: int = None,
        embedder: typing.Callable[[typing.Any], typing.List[typing.List]] = None   

    ):
        """
        Args:
            df (pd.DataFrame): 
        """
        super().__init__(
            select, like, where, order_by, limit
        )
        self._path = path
        self._collection_name = collection
        self._client = chromadb.Client(persistent=path)
        self._collection = self._client.get_or_create_collection(
            name=collection, path=path
        )
        self._ids = self._collection.get()['ids']
        self._embedder = embedder

    @property
    def collection(self):
        return self._collection

    def store(
        self, document, index=None, embedding: typing.Optional[typing.List]=None, **meta
    ):
        
        meta = {
            k: [meta] for k, meta in meta.items()
        }

        return self.bulk_store(
            [document], [index], [embedding], **meta
        )

    def bulk_store(
        self, documents: typing.List, 
        index: typing.Optional[typing.List]=None, 
        embeddings: typing.List[typing.List]=None, 
        **kwargs
    ):
        
        if index is None:
            index = [get_uuid() for i in range(len(documents))]

        if embeddings is not None:
            kwargs['embeddings'] = embeddings
        elif self._embedder is not None:
            kwargs['embeddings'] = self._embedder(documents)

        kwargs = {
            'documents': documents,
            'ids': [index],
            **kwargs
        }

        self._collection.add(
            **kwargs
        )

    @property
    def query(self) -> 'ChromaDBQuery':
        return ChromaDBQuery(self)


class ChromaDBQuery(VectorizedQuery):

    def __init__(self, store: 'ChromaDBStore', select: typing.Dict[str, str | QF] = None, like: Like = None, where: Comp = None, order_by: typing.List[str] = None, limit: int = None) -> None:
        super().__init__(
            select, like, where, order_by, limit)

    def values(
        self, 
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        where: typing.Optional[Comp]=None,
        order_by: typing.Optional[typing.List[str]]=None,
        limit: typing.Optional[int]=None, 
        like: typing.Optional[Like]=None
    ) -> pd.DataFrame:
        
        collection = self._store.collection

        kwargs = chroma_filter(where, like)

        if like is None:
            results = collection.get(
                **kwargs,
                n_results=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
        else:
            results = collection.query(
                **kwargs,
                n_results=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
        results = to_df(results)

        if order_by is not None:
            df = df_sort(df, order_by)

        if select is not None:
            df = df_select(df, select)

        return df


# Bring back "query" but have one for each
# ChromaDBStore
# ChromaDBQuery
# => this seems better to me thinking about it more
# .query()
# 
