import typing

from ..store import VectorStore, Comp, Join, Like, QF
from ..store import df_sort, df_join, df_select

import pandas as pd
import chromadb

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


class ChromaDBStore(VectorStore):

    def __init__(self, collection: str, path: str):
        """

        Args:
            df (pd.DataFrame): 
        """
        super().__init__()
        self.client = chromadb.Client(persistent=path)
        self._collection = self.client.get_or_create_collection(
            name=collection, path=path
        )
        # client.get_or_create_collection("local_collection")

    def store(
        self, index=None, **kwargs
    ):
        pass
        # if index is None:
        #     self._df.iloc[len(self._df.index)] = kwargs
        # else:
        #     self._df.loc[max(self._df.index)] = kwargs

    def bulk_store(
        self, index=None, **kwargs
    ):
        pass
        # df = pd.DataFrame(kwargs, index)
        # if index is None:
        #     self._df = pd.concat(
        #         [self._df, df]
        #     )
        # else:
        #     self._df = self._df.reindex(df.index.union(df.index))
        #     self._df.update(df)

    def retrieve(
        self,
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        joins: typing.Optional[typing.List[Join]]=None,
        where: typing.Optional[Comp]=None,
        order_by: typing.Optional[typing.List[str]]=None,
        limit: typing.Optional[int]=None, 
        like: typing.Optional[Like]=None
    ) -> pd.DataFrame:

        kwargs = chroma_filter(where, like)

        if like is None:
            results = self._collection.get(
                **kwargs,
                n_results=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
        else:
            results = self._collection.query(
                **kwargs,
                n_results=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
        results = to_df(results)

        # Technically JOIN should be done prior to getting
        # the similarity.. 
        # One way to handle this is to 
        joins = joins or []
        for join in joins:
            store = df_join(store, join)

        if order_by is not None:
            df = df_sort(df, order_by)

        if select is not None:
            df = df_select(df, select)

        return df
