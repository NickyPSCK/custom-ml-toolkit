import pandas as pd
from typing import List, Optional, Any


class ClusterProfiler:
    '''Class to perform cluster profiling on a dataset, providing summaries for numerical and categorical features
    within each cluster.
    '''
    def __init__(
        self,
        cluster_col: str,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> None:
        '''Initializes the ClusterProfiler with the cluster column and lists of numerical and categorical columns.

        Args:
            cluster_col (str): The column name representing the cluster.
            numerical_cols (List[str]): List of numerical column names.
            categorical_cols (List[str]): List of categorical column names.
        '''

        self._cluster_col = cluster_col

        if numerical_cols is None:
            self._numerical_cols = list()
        else:
            self._numerical_cols = numerical_cols

        if categorical_cols is None:
            self._categorical_cols = list()
        else:
            self._categorical_cols = categorical_cols

    @staticmethod
    def _profile_cluster(
        df: pd.DataFrame,
        cluster_col: str
    ) -> pd.DataFrame:
        '''Profiles the distribution of the cluster column by counting the occurrences and calculating percentages.

        Args:
            df (pd.DataFrame): The dataframe to profile.
            cluster_col (str): The column name representing the cluster.

        Returns:
            pd.DataFrame: A dataframe containing cluster distribution statistics.
        '''

        result_df = df.groupby(cluster_col)[[cluster_col]]\
            .count()\
            .rename(columns={cluster_col: 'n'})
        result_df['percent'] = 100 * result_df['n'] / result_df['n'].sum()
        result_df = result_df.T
        result_df['feature_name'] = '_cluster_stat_'
        result_df.columns.name = 'cluster'
        result_df.index.name = 'feature_values'
        result_df = result_df.reset_index(drop=False)
        result_df = result_df.set_index(['feature_name', 'feature_values'])

        return result_df

    @staticmethod
    def _profile_cat_feature(
        df: pd.DataFrame,
        cluster_col: str,
        target_col: str
    ) -> pd.DataFrame:
        '''Profiles a categorical feature by counting occurrences within each cluster and calculating percentages.

        Args:
            df (pd.DataFrame): The dataframe to profile.
            cluster_col (str): The column name representing the cluster.
            target_col (str): The column name representing the categorical feature.

        Returns:
            pd.DataFrame: A dataframe containing the counts and percentages of the categorical feature for each cluster.
        '''
        result_df = df\
            .groupby([cluster_col, target_col])[[target_col]]\
            .count()\
            .rename(columns={target_col: '_count_'})\
            .reset_index(drop=False)\
            .pivot(
                index=[target_col],
                columns=[cluster_col],
                values=['_count_']
            )\
            .reset_index()\
            .rename(columns={target_col: 'feature_values'})

        result_df.columns = [col_name[1] if col_name[1] != '' else col_name[0] for col_name in result_df.columns]
        result_df['feature_name'] = target_col
        result_df = result_df.set_index(['feature_name', 'feature_values'])
        result_df = 100 * result_df / result_df.sum(axis=0)
        result_df.columns.name = 'cluster'

        return result_df

    @staticmethod
    def _profile_num_features(
        df: pd.DataFrame,
        cluster_col: str,
        target_cols: List[Any],
        agg: Optional[List[str]] = None
    ) -> pd.DataFrame:
        '''Profiles numerical features by calculating statistics (e.g., max, min, mean, median, std) within each cluster.

        Args:
            df (pd.DataFrame): The dataframe to profile.
            cluster_col (str): The column name representing the cluster.
            target_cols (List[Any]): The list of numerical column names to profile.
            agg (Optional[List[str]]): The aggregation functions to apply (default is ['max', 'min', 'mean', 'median', 'std']).

        Returns:
            pd.DataFrame: A dataframe containing aggregated statistics for numerical features within each cluster.
        '''
        if agg is None:
            agg = ['max', 'min', 'mean', 'median', 'std']

        result_df = df\
            .groupby([cluster_col])[target_cols]\
            .agg(agg)\
            .T
        result_df.index.names = ['feature_name', 'feature_values']
        result_df.columns.name = 'cluster'

        return result_df

    def summary(
        self,
        df: pd.DataFrame,
        num_agg: Optional[List[str]] = None
    ) -> pd.DataFrame:
        '''Summarizes the profiling of clusters, including numerical and categorical features.

        Args:
            df (pd.DataFrame): The dataframe to summarize.
            num_agg (Optional[List[str]]): The aggregation functions for numerical features (default is None, which uses 'max', 'min', 'mean', 'median', 'std').

        Returns:
            pd.DataFrame: A dataframe containing the profiling summary for clusters, numerical features, and categorical features.
        '''
        profiles = list()

        profile = self._profile_cluster(
            df=df,
            cluster_col=self._cluster_col
        )
        profiles.append(profile)

        profile = self._profile_num_features(
            df=df,
            cluster_col=self._cluster_col,
            target_cols=self._numerical_cols,
            agg=num_agg
        )
        profiles.append(profile)

        for cat_col in self._categorical_cols:
            profile = self._profile_cat_feature(
                df=df,
                cluster_col=self._cluster_col,
                target_col=cat_col
            )
            profiles.append(profile)

        return pd.concat(profiles, axis=0)
