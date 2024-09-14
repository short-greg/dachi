from dachi.store import _dataframe as st_df, _core as st_core
import pandas as pd
import numpy as np


class TestDFStore(object):

    def test_df_returns_the_dataframe(self):

        df = pd.DataFrame(
            np.random.random((2, 4)), 
            columns=['w', 'x', 'y', 'z']
        )
        store = st_df.DFStore(df)
        assert store.df is df

    def test_retrieve_returns_the_dataframe(self):

        df = pd.DataFrame(
            np.random.random((2, 4)), 
            columns=['w', 'x', 'y', 'z']
        )
        store = st_df.DFStore(df)
        assert store.query.values() is df

    def test_retrieve_returns_sorted_dataframe(self):

        df = pd.DataFrame(
            np.random.random((2, 4)), 
            columns=['w', 'x', 'y', 'z']
        )
        store = st_df.DFStore(df)
        retrieved = store.query.order_by('x').values()
        print(retrieved)
        print(df.sort_values('x'))

        assert (
            retrieved.values == df.sort_values('x').values
        ).all()

    def test_retrieve_returns_filtered_dataframe(self):

        df = pd.DataFrame(
            np.random.random((6, 4)), 
            columns=['w', 'x', 'y', 'z']
        )
        store = st_df.DFStore(df)
        comp = st_core.Key('x') >= 0.5
        retrieved = store.query.where(comp).values()

        assert (
            retrieved['x'] >= 0.5
        ).all()

    def test_retrieve_returns_selected_dataframe(self):

        df = pd.DataFrame(
            np.random.random((6, 4)), 
            columns=['w', 'x', 'y', 'z']
        )
        store = st_df.DFStore(df)
        retrieved = store.query.select(t='x', w='w').values()
        assert 't' in retrieved.columns.values
        assert 'y' not in retrieved.columns.values
        assert 'x' not in retrieved.columns.values
        assert 'w' in retrieved.columns.values
