import pandas as pd


class ClusterProfiler:
    def __init__(
        self,
        cluster_col: str,
        numerical_cols: list,
        categorical_cols: list
    ):

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
    ):

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
    ):
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
        target_cols: list,
        agg: list = None
    ):
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
        num_agg: list = None
    ):
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
