import pickle
from typing import Literal, Union, Optional, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


class SupportMissingOneHotEncoder(BaseEstimator, TransformerMixin):

    ''' Supported missing values one hot encoder

    Example: the difference between OneHotEncoder and SupportMissingOneHotEncoder

    Code:
    ```
        train_data = pd.DataFrame(data={'a': ['cat', 'bird', 'dog', np.nan],
                                        'b': ['red', 'green', 'blue', 'red']})
        test_data = pd.DataFrame(data={'a': ['dog', 'cat', np.nan, 'bird'],
                                       'b': ['red', 'green', 'blue', 'black']})
        print(f'---------------{"ORIGINAL TRAIN":^25}---------------')
        print(train_data)
        print(f'---------------{"ORIGINAL TEST":^25}---------------')
        print(test_data)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.set_output(transform='pandas')
        print(f'---------------{"ENCODED TRAIN (OHE)":^25}---------------')
        print(ohe.fit_transform(train_data))
        print(f'---------------{"ENCODED TEST (OHE)":^25}---------------')
        print(ohe.transform(test_data))

        smohe = SupportMissingOneHotEncoder()
        print(f'---------------{"ENCODED TRAIN (SMOHE)":^25}---------------')
        print(smohe.fit_transform(train_data))
        print(f'---------------{"ENCODED TEST (SMOHE)":^25}---------------')
        print(smohe.transform(test_data))
    ```
    Result:
    ```
        ---------------     ORIGINAL TRAIN      ---------------
            a      b
        0   cat    red
        1  bird  green
        2   dog   blue
        3   NaN    red
        ---------------      ORIGINAL TEST      ---------------
            a      b
        0   dog    red
        1   cat  green
        2   NaN   blue
        3  bird  black
        ---------------   ENCODED TRAIN (OHE)   ---------------
        a_bird  a_cat  a_dog  a_nan  b_blue  b_green  b_red
        0     0.0    1.0    0.0    0.0     0.0      0.0    1.0
        1     1.0    0.0    0.0    0.0     0.0      1.0    0.0
        2     0.0    0.0    1.0    0.0     1.0      0.0    0.0
        3     0.0    0.0    0.0    1.0     0.0      0.0    1.0
        ---------------   ENCODED TEST (OHE)    ---------------
        a_bird  a_cat  a_dog  a_nan  b_blue  b_green  b_red
        0     0.0    0.0    1.0    0.0     0.0      0.0    1.0
        1     0.0    1.0    0.0    0.0     0.0      1.0    0.0
        2     0.0    0.0    0.0    1.0     1.0      0.0    0.0
        3     1.0    0.0    0.0    0.0     0.0      0.0    0.0
        ---------------  ENCODED TRAIN (SMOHE)  ---------------
        a_bird  a_cat  a_dog  b_blue  b_green  b_red
        0     0.0    1.0    0.0     0.0      0.0    1.0
        1     1.0    0.0    0.0     0.0      1.0    0.0
        2     0.0    0.0    1.0     1.0      0.0    0.0
        3     NaN    NaN    NaN     0.0      0.0    1.0
        ---------------  ENCODED TEST (SMOHE)   ---------------
        a_bird  a_cat  a_dog  b_blue  b_green  b_red
        0     0.0    0.0    1.0     0.0      0.0    1.0
        1     0.0    1.0    0.0     0.0      1.0    0.0
        2     NaN    NaN    NaN     1.0      0.0    0.0
        3     1.0    0.0    0.0     NaN      NaN    NaN
    ```
    '''
    def __init__(
        self,
        drop_binary: bool = True
    ):
        '''
        Args:
            drop_binary: (bool) = True for removing the first category in each feature with two categories,
            False otherwise. Default to True.
        '''
        self.drop_binary = drop_binary
        self._unknown_value = np.nan

        self._ohe = OneHotEncoder(
            handle_unknown='ignore',
            drop=None,
            sparse_output=False
        )
        self._ohe.set_output(transform='pandas')

    @property
    def BaseOneHotEncoder(
        self
    ) -> OneHotEncoder:
        '''OneHotEncoder: Fitted OneHotEncoder'''
        return self._ohe

    @staticmethod
    def _create_representative_df(
        data_df: pd.DataFrame
    ) -> pd.DataFrame:
        '''Creates a representative DataFrame from the given data for fitting the OneHotEncoder.

        Args:
            data_df (pd.DataFrame): Raw data for creating representative catagories in train set.
        Returns:
            pd.DataFrame: Representative data frame

        '''
        cat_representative_dict = OrderedDict()
        for col in data_df.columns:
            unique_values = data_df[col].dropna().unique()
            cat_representative_dict[col] = pd.Series(unique_values)
        cat_representative_df = pd.DataFrame(cat_representative_dict)
        cat_representative_df = cat_representative_df.ffill()
        cat_representative_df = cat_representative_df.astype('object')
        return cat_representative_df

    def _replace_unknow_value(
        self,
        encoded_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        '''Replaces unknown values (NaNs) in the encoded DataFrame with a placeholder value.

        Args:
            encoded_data_df (pd.DataFrame): Encoded data with potential unknown values to replace.
        
        Returns:
            pd.DataFrame: Processed encoded data frame with unknown values replaced.
        '''
        processed_encoded_data_list = list()
        for i, feature in enumerate(self.feature_names_in_):
            encoded_feature_cols = [str(feature) + '_' + str(cat) for cat in self._ohe.categories_[i]]
            encoded_each_feature = encoded_data_df[encoded_feature_cols].copy()
            unknown_index = (encoded_each_feature.sum(axis=1) == 0)
            encoded_each_feature[unknown_index] = self._unknown_value

            if self.drop_binary and (encoded_each_feature.shape[1] == 2):
                encoded_each_feature = encoded_each_feature.drop(encoded_feature_cols[-1], axis=1)
                processed_encoded_data_list.append(encoded_each_feature)
            else:
                processed_encoded_data_list.append(encoded_each_feature)

        processed_encoded_data_df = pd.concat(processed_encoded_data_list, axis=1)
        return processed_encoded_data_df

    def fit(
        self,
        X: pd.DataFrame,
        y=None
    ):
        '''Fits the SupportMissingOneHotEncoder to the data.

        Args:
            X (pd.DataFrame): Training data of categorical features.
            y: Ignored, needed for compatibility with sklearn pipeline.
        
        Returns:
            self: The fitted transformer.
        '''
        representative_x_df = self._create_representative_df(X)
        self._ohe.fit(
            X=representative_x_df
        )
        self.feature_names_in_ = self._ohe.feature_names_in_
        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        '''Transforms the input data by replacing unknown values and encoding categorical features.

        Args:
            X (pd.DataFrame): Data to be transformed, assumed to be of categorical type with missing values represented as NaN.
        
        Returns:
            pd.DataFrame: Transformed data with unknown values replaced and categorical features one-hot encoded.
        '''
        check_is_fitted(self)
        X = X.astype('object')
        encoded_x_df = self._ohe.transform(X)
        return self._replace_unknow_value(encoded_data_df=encoded_x_df)

    def inverse_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        '''Inverse transforms the encoded data back to the original categories, replacing placeholders with NaNs.

        Args:
            X (pd.DataFrame): Encoded data to be inverted, with placeholder values for unknown categories.
        
        Returns:
            pd.DataFrame: Dataframe with placeholder values replaced by NaNs, and original categories restored.
        '''
        check_is_fitted(self)
        return pd.DataFrame(
            self._ohe.inverse_transform(X.fillna(0)),
            columns=self.feature_names_in_
        )

    def get_feature_names_out(self):
        '''Retrieves feature names after transformation, accounting for whether to drop binary columns.

        Returns:
            List[str]: List of feature names after transformation.
        '''
        check_is_fitted(self)

        feature_names_out = list()
        for i, feature in enumerate(self.feature_names_in_):
            encoded_feature_cols = [str(feature) + '_' + str(cat) for cat in self._ohe.categories_[i]]
            if self.drop_binary and (len(encoded_feature_cols) == 2):
                feature_names_out += [encoded_feature_cols[0]]
            else:
                feature_names_out += encoded_feature_cols

        return feature_names_out


class SupportMissingCategoricalEncoder(BaseEstimator, TransformerMixin):
    '''Dataset categorical encoder for features and target.
    '''
    def __init__(
        self,
        numerical_cols: Optional[list] = None,
        norminal_cols: Optional[list] = None,
        ordinal_cols: Optional[list] = None,
        drop_binary: bool = True,
        oe_unknown_value: Any = np.nan,
        oe_missing_value: Any = np.nan,
    ):
        '''
        Args:
            numerical_cols (:obj:`list`): Numerical column names in the data set.
            norminal_cols (:obj:`list`): Norminal column names in the data set.
            ordinal_cols (:obj:`list`): Ordinal column names in the data set.
            drop_binary: (bool) = True for removing the first category in each norminal feature with two categories,
                False otherwise. Default to True.
            oe_unknown_value: (any): unknown_value parmeter in OrdinalEncoder.
            oe_missing_value: (any) encoded_missing_value parmeter in OrdinalEncoder.
        '''

        if numerical_cols is not None:
            self._numerical_cols = numerical_cols
        else:
            self._numerical_cols = list()

        if norminal_cols is not None:
            self._norminal_cols = norminal_cols
        else:
            self._norminal_cols = list()

        if ordinal_cols is not None:
            self._ordinal_cols = ordinal_cols
        else:
            self._ordinal_cols = list()

        self._drop_binary = drop_binary
        self._oe_unknown_value = oe_unknown_value
        self._oe_missing_value = oe_missing_value

        self._smohe = None
        self._oe = None
        self.all_cols = (
            self._numerical_cols
            + self._norminal_cols
            + self._ordinal_cols
        )

    @classmethod
    def load(
        cls,
        path
    ):
        with open(path, 'rb') as f:
            cls = pickle.load(f)
        return cls

    @property
    def smoe(
        self
    ) -> SupportMissingOneHotEncoder:
        '''SupportMissingOneHotEncoder: SupportMissingOneHotEncoder'''
        return self._smohe

    @property
    def oe(
        self
    ) -> OrdinalEncoder:
        '''OrdinalEncoder: OrdinalEncoder'''
        return self._oe

    def save(
        self,
        path: str
    ):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def check_input(
            self,
            X: pd.DataFrame
    ):
        missing_keys = list()
        for col_name in self.all_cols:
            if col_name not in X.columns:
                missing_keys.append(col_name)

        if len(missing_keys) > 0:
            raise KeyError(f'''None of {missing_keys} are in the [columns]''')

        return X.copy()

    def fit(
        self,
        X: pd.DataFrame,
        y=None
    ):
        X = self.check_input(X)

        self.feature_names_in_ = list()
        for col_name in X.columns:
            if col_name in self.all_cols:
                self.feature_names_in_.append(col_name)

        X = X[self.feature_names_in_]

        if len(self._norminal_cols) > 0:
            self._smohe = SupportMissingOneHotEncoder(
                drop_binary=self._drop_binary
            )
            self._smohe.fit(X[self._norminal_cols])

        if len(self._ordinal_cols) > 0:
            self._oe = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=self._oe_unknown_value,
                encoded_missing_value=self._oe_missing_value
            )
            self._oe.set_output(transform='pandas')
            self._oe.fit(X[self._ordinal_cols])

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        check_is_fitted(self)
        X = self.check_input(X)

        if len(self._norminal_cols) > 0:
            transformed_norminal_cols_df = self._smohe.transform(
                X[self._norminal_cols]
            )
        else:
            transformed_norminal_cols_df = pd.DataFrame()

        if len(self._ordinal_cols) > 0:
            transformed_ordinal_cols_df = self._oe.transform(
                X[self._ordinal_cols]
            )
        else:
            transformed_ordinal_cols_df = pd.DataFrame()

        X_tr = pd.concat(
            [
                X[self._numerical_cols],
                transformed_norminal_cols_df,
                transformed_ordinal_cols_df
            ],
            axis=1
        )

        return X_tr

    def inverse_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        check_is_fitted(self)

        if len(self._norminal_cols) > 0:
            norminal_cols_df = self._smohe.inverse_transform(
                X[self._smohe.get_feature_names_out()]
            )
        else:
            norminal_cols_df = pd.DataFrame()

        if len(self._ordinal_cols) > 0:
            ordinal_cols_df = pd.DataFrame(
                self._oe.inverse_transform(
                    X[self._oe.get_feature_names_out()]
                ),
                columns=self._oe.feature_names_in_
            )
        else:
            ordinal_cols_df = pd.DataFrame()

        X = pd.concat(
            [
                X[self._numerical_cols],
                norminal_cols_df,
                ordinal_cols_df
            ],
            axis=1
        )[self.feature_names_in_]

        return X

    def get_norminal_feature_name_out(self):
        if len(self._norminal_cols) > 0:
            return list(self._smohe.get_feature_names_out())
        else:
            return list()

    def get_feature_names_out(self):
        check_is_fitted(self)
        feature_names_out = list()
        feature_names_out += self._numerical_cols
        if len(self._norminal_cols) > 0:
            feature_names_out += list(self._smohe.get_feature_names_out())
        if len(self._ordinal_cols) > 0:
            feature_names_out += list(self._oe.get_feature_names_out())
        return feature_names_out


class SupportMissingDatasetEncoder:
    '''Dataset categorical encoder for features and target.
    '''
    def __init__(
        self,
        numerical_cols: Optional[list] = None,
        norminal_cols: Optional[list] = None,
        ordinal_cols: Optional[list] = None,
        target_col: Optional[str] = None,
        drop_binary: bool = True,
        oe_unknown_value: Any = np.nan,
        oe_missing_value: Any = np.nan,
        encode_target: bool = True,
    ):
        '''

        Args:
            numerical_cols (:obj:`list`): Numerical column names in the data set.
            norminal_cols (:obj:`list`): Norminal column names in the data set.
            ordinal_cols (:obj:`list`): Ordinal column names in the data set.
            target_col (:obj:`str`): Target column name in the data set.
            drop_binary: (bool) = True for removing the first category in each norminal feature with two categories,
                False otherwise. Default to True.
            oe_unknown_value: (any): unknown_value parmeter in OrdinalEncoder.
            oe_missing_value: (any) encoded_missing_value parmeter in OrdinalEncoder.
            encode_target: (bool): True for encoding the target, False otherwise. Default to True.

        '''

        self._target_col = target_col
        self._encode_target = encode_target

        self._fe = SupportMissingCategoricalEncoder(
            numerical_cols=numerical_cols,
            norminal_cols=norminal_cols,
            ordinal_cols=ordinal_cols,
            drop_binary=drop_binary,
            oe_unknown_value=oe_unknown_value,
            oe_missing_value=oe_missing_value,
        )

        if self._encode_target:
            self._le = LabelEncoder()
        else:
            self._le = None

    @classmethod
    def load(
        cls,
        path
    ):
        with open(path, 'rb') as f:
            cls = pickle.load(f)
        return cls

    @property
    def features_encoder(
        self
    ) -> SupportMissingCategoricalEncoder:
        '''SupportMissingCategoricalEncoder: SupportMissingCategoricalEncoder'''
        return self._fe

    @property
    def target_encoder(
        self
    ) -> LabelEncoder:
        '''LabelEncoder: Fitted feature encoder'''
        return self._le

    @property
    def classes_(
        self
    ) -> list:
        if self._le is None:
            return self._le
        else:
            return self._le.classes_

    def save(
        self,
        path: str
    ):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def fit(
        self,
        data_df
    ):
        data_df = data_df.copy()
        self._fe.fit(data_df)
        if self._encode_target:
            self._le.fit(data_df[self._target_col])

    def transform(
        self,
        data_df: pd.DataFrame,

    ) -> pd.DataFrame:
        encoded_data_df = self._fe.transform(data_df)
        if self._target_col in data_df.columns:
            if self._encode_target:
                target_series = self._le.transform(data_df[self._target_col])
                encoded_data_df[self._target_col] = target_series
            else:
                encoded_data_df[self._target_col] = data_df[self._target_col].copy()

        return encoded_data_df

    def fit_transform(
        self,
        data_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.fit(data_df=data_df)
        return self.transform(
            data_df=data_df
        )

    def inverse_transform_target(
        self,
        y: pd.Series
    ) -> pd.Series:
        return self._le.inverse_transform(y=y)

    def transform_target(
        self,
        y: pd.Series
    ) -> pd.Series:
        return self._le.transform(y)


# class DatasetEncoder:
#     '''Dataset categorical encoder for features and target.
#     '''
#     def __init__(
#         self,
#         cat_cols: Optional[list] = None,
#         num_cols: Optional[list] = None,
#         target_col: Optional[str] = None,
#         encode_target: bool = True,
#         encode_mode: Literal['onehot', 'ordinal'] = 'onehot',
#         ordinal_unknown_value=np.nan,
#         ordinal_encoded_missing_value=np.nan,
#         drop_binary: bool = True
#     ):
#         '''
#         Args:
#             cat_cols (:obj:`list`): Categorical column names in the data set.
#             num_cols (:obj:`list`): Numerical column names in the data set.
#             target_col (:obj:`str`): Target column name in the data set.
#             encode_target: (bool): True for encoding the target, False otherwise. Default to True.
#             encode_mode (:obj:`Literal['onehot', 'ordinal']`): Mode for encoding categorical features.
#                 Default to 'onehot'.
#             ordinal_unknown_value: (any): unknown_value parmeter in OrdinalEncoder.
#             ordinal_encoded_missing_value: (any) encoded_missing_value parmeter in OrdinalEncoder.
#             drop_binary: (bool) = True for removing the first category in each feature with two categories, False otherwise. Default to True.
#         '''
#         if cat_cols is not None:
#             self._cat_cols = cat_cols
#         else:
#             self._cat_cols = list()

#         self._num_cols = num_cols
#         self._target_col = target_col
#         self._encode_mode = encode_mode
#         self._encode_target = encode_target

#         if self._encode_mode == 'onehot':
#             self._ce = SupportMissingOneHotEncoder(
#                 drop_binary=drop_binary
#             )
#         elif self._encode_mode == 'ordinal':
#             self._ce = OrdinalEncoder(
#                 handle_unknown='use_encoded_value',
#                 unknown_value=ordinal_unknown_value,
#                 encoded_missing_value=ordinal_encoded_missing_value
#             )
#             self._ce.set_output(transform='pandas')
#         else:
#             raise ValueError('encode_mode must be either onehot or ordinal')

#         if self._encode_target:
#             self._le = LabelEncoder()

#     @classmethod
#     def load(
#         cls,
#         path
#     ):
#         with open(path, 'rb') as f:
#             cls = pickle.load(f)
#         return cls

#     @property
#     def features_encoder(
#         self
#     ) -> Union[SupportMissingOneHotEncoder, OrdinalEncoder]:
#         '''Union[SupportMissingOneHotEncoder, OrdinalEncoder]: Fitted feature encoder'''
#         return self._ce

#     @property
#     def target_encoder(
#         self
#     ) -> LabelEncoder:
#         '''LabelEncoder: Fitted feature encoder'''
#         return self._le

#     def save(
#         self,
#         path: str
#     ):
#         with open(path, 'wb') as f:
#             pickle.dump(self, f)

#     def fit(
#         self,
#         data_df
#     ):
#         data_df = data_df.copy()
#         self._data_columns = list(data_df.columns)
#         if self._num_cols is None:
#             self._num_cols = list()
#             for col in self._data_columns:
#                 if col not in (self._cat_cols + [self._target_col]):
#                     self._num_cols.append(col)

#         if len(self._cat_cols) != 0:
#             cat_features_df = data_df[self._cat_cols]
#             self._ce.fit(cat_features_df)

#         if self._encode_target:
#             target_data = data_df[self._target_col]
#             self._le.fit(target_data)

#     def transform(
#         self,
#         data_df: pd.DataFrame,
#         is_inference: bool = False
#     ) -> pd.DataFrame:

#         if len(self._cat_cols) != 0:
#             cat_features_df = data_df[self._cat_cols]
#             transformed_cat_features_df = self._ce.transform(cat_features_df)
#         else:
#             transformed_cat_features_df = pd.DataFrame()

#         num_features_df = data_df[self._num_cols]
#         transformed_data_df = pd.concat(
#             [num_features_df, transformed_cat_features_df],
#             axis=1
#         )

#         if self._encode_target and (not is_inference):
#             target_series = self._le.transform(data_df[self._target_col])
#             transformed_data_df[self._target_col] = target_series

#         return transformed_data_df

#     def fit_transform(
#         self,
#         data_df: pd.DataFrame
#     ) -> pd.DataFrame:
#         self.fit(data_df=data_df)
#         return self.transform(
#             data_df=data_df,
#             is_inference=False
#         )

#     def inverse_transform_target(
#         self,
#         target: pd.Series
#     ) -> pd.Series:
#         return self._le.inverse_transform(y=target)

#     def transform_target(
#         self,
#         target: pd.Series
#     ) -> pd.Series:
#         return self._le.transform(target)
