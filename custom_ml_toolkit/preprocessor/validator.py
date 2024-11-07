from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd


class DataFrameValidator:
    def __init__(
        self,
        required_cols: Optional[list] = None,
        bool_cols: Optional[list] = None,  # Non-nullable
        int_cols: Optional[list] = None,  # Non-nullable
        float_cols: Optional[list] = None,
        str_cols: Optional[list] = None,
        dt_col_formats: Optional[Dict[str, str]] = None,
        literal_col_values: Optional[Dict[str, List[str]]] = None,
        non_nullable_cols: Optional[list] = None,
        # col_min_max_constrains: Optional[dict] = None,
        # col_func_constrain: Optional[dict] = None,

    ):
        self._required_cols = list(set(self._defalut(required_cols, list)))
        self._dtype_col_dict = {
            'bool': list(set(self._defalut(bool_cols, list))),
            'int': list(set(self._defalut(int_cols, list))),
            'float': list(set(self._defalut(float_cols, list))),
            'str': list(set(self._defalut(str_cols, list))),
            'dt': self._defalut(dt_col_formats, dict),
            'literal': self._defalut(literal_col_values, dict)
        }
        self._check_duplicate_dtypes()
        self._non_nullable_cols = self._create_non_nullable_cols(
            non_nullable_cols=non_nullable_cols
        )

    def _create_non_nullable_cols(
        self,
        non_nullable_cols: list
    ) -> list:
        non_nullable_cols = list(set(non_nullable_cols))
        non_nullable_by_dtype_cols = (
            self._dtype_col_dict['bool']
            + self._dtype_col_dict['int']
        )
        missing_non_nullable_cols = list()
        for col in non_nullable_by_dtype_cols:
            if col not in non_nullable_cols:
                missing_non_nullable_cols.append(col)

        non_nullable_cols += missing_non_nullable_cols

        print(f'Warning: boolean and int columns: {missing_non_nullable_cols} have been added to non nullable columns.')
        return non_nullable_cols

    def _check_duplicate_dtypes(self) -> None:
        cols = list()
        for dtype in self._dtype_col_dict:
            cols += list(self._dtype_col_dict[dtype])
        count_cols = Counter(cols)
        duplicate_dtype_cols = [col for col in count_cols if count_cols[col] > 1]
        if len(duplicate_dtype_cols) > 0:
            raise ValueError(
                f'the column(s): {duplicate_dtype_cols} should belong to only one data type.'
            )

    @staticmethod
    def _defalut(v: object, o: object):
        if v is None:
            return o()
        else:
            return v

    @staticmethod
    def check_bool(
        series: pd.Series,
    ):
        # Untill this become non-experiment
        # https://pandas.pydata.org/docs/user_guide/boolean.html
        formatted_series = series.copy()
        error_descs = list()
        try:
            formatted_series = formatted_series.astype(
                dtype=bool, copy=True, errors='raise'
            )
            error = False
            series = formatted_series
        except ValueError:
            error = True
            error_descs.append('Error: Invalid boolean value(s).')
        return series, error, error_descs

    @staticmethod
    def check_int(
        series: pd.Series,
    ):
        # Untill this become non-experiment
        # https://pandas.pydata.org/docs/user_guide/integer_na.html
        formatted_series = series.copy()
        error_descs = list()
        try:
            series_float = formatted_series.astype(
                dtype=float, copy=True, errors='raise'
            )
            formatted_series = series_float.astype(
                dtype=int, copy=True, errors='raise'
            )
            if series_float.sum() > formatted_series.sum():
                error = True
                error_descs.append('Warning: Loss of precision.')
            else:
                error = False
            series = formatted_series
        except ValueError:
            error = True
            error_descs.append('Error: Invalid integer value(s).')
        return series, error, error_descs

    @staticmethod
    def check_float(
        series: pd.Series
    ):
        formatted_series = series.copy()
        error_descs = list()
        try:
            formatted_series[formatted_series == ''] = None
            formatted_series = formatted_series.astype(
                dtype=float, copy=True, errors='raise'
            )
            error = False
            series = formatted_series
        except ValueError:
            error = True
            error_descs.append('Error: Invalid float value(s).')
        return series, error, error_descs

    @staticmethod
    def check_str(
        series: pd.Series,
    ):
        formatted_series = series.copy()
        error_descs = list()
        try:
            null_index = formatted_series.isna()
            formatted_series = formatted_series.astype(
                dtype=str, copy=True, errors='raise'
            )
            formatted_series[null_index] = ''
            error = False
            series = formatted_series
        except ValueError:
            error = True
            error_descs.append('Error: Invalid string value(s).')
        return series, error, error_descs

    @staticmethod
    def check_dt(
        series: pd.Series,
        format: str
    ):
        formatted_series = series.copy()
        error_descs = list()
        try:
            formatted_series[formatted_series == ''] = None
            formatted_series = pd.to_datetime(
                formatted_series, errors='raise', format=format
            )
            error = False
            series = formatted_series
        except ValueError:
            error = True
            error_descs.append(f'Error: Invalid datetime format ({format}).')
        return series, error, error_descs

    @staticmethod
    def check_literal(
        series: pd.Series,
        literals: List[str]
    ):
        formatted_series = series.copy()
        error_descs = list()
        try:
            null_index = formatted_series.isna()
            formatted_series = formatted_series.astype(
                dtype=str, copy=True, errors='raise'
            )
            formatted_series[null_index] = None
            formatted_series[formatted_series == ''] = None

            assert ((~formatted_series.isna()) & (~formatted_series.isin(literals))).sum() == 0

            error = False
            series = formatted_series
        except ValueError:
            error = True
            error_descs.append(f'Error: Invalid literal value(s) {literals}.')
        except AssertionError:
            error = True
            error_descs.append(f'Error: Invalid literal value(s) {literals}.')
        return series, error, error_descs

    def _check_missing_required_cols(
        self,
        columns: list
    ):
        target_cols = set(columns)
        required_cols = set(self._required_cols)
        missing_required_cols = list(required_cols.difference(target_cols))
        existing_required_cols = list(required_cols.intersection(target_cols))
        not_required_cols = list(target_cols.difference(required_cols))

        return missing_required_cols, existing_required_cols, not_required_cols

    def _check_null_values(
        self,
        df: pd.DataFrame
    ) -> list:

        count_null_values = df[self._non_nullable_cols]\
            .isna()\
            .sum(axis=0)\
            .to_dict()

        null_in_non_nullable_cols = [
            col for col in count_null_values if count_null_values[col] > 0
        ]
        return null_in_non_nullable_cols

    def _check_types(
        self,
        df: pd.DataFrame,
        formatted: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        df = df.copy()
        type_errors = dict()
        for type_name in self._dtype_col_dict:

            for col in self._dtype_col_dict[type_name]:
                if col in df.columns:

                    if isinstance(self._dtype_col_dict[type_name], dict):
                        series, error, error_descs = getattr(
                            self,
                            f'check_{type_name}'
                        )(df[col], self._dtype_col_dict[type_name][col])
                    else:
                        series, error, error_descs = getattr(
                            self,
                            f'check_{type_name}'
                        )(df[col])

                    if formatted:
                        df[col] = series

                    if error:
                        for error in error_descs:
                            if error in type_errors:
                                type_errors[error].append(col)
                            else:
                                type_errors[error] = [col]
        return df, type_errors

    def validate(
        self,
        df: pd.DataFrame,
        formatted: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        errors = dict()

        missing_req_cols, _, not_req_cols = self._check_missing_required_cols(
            columns=df.columns
        )
        null_in_non_nullable_cols = self._check_null_values(df=df)

        df, type_errors = self._check_types(
            df=df,
            formatted=formatted
        )

        if len(missing_req_cols) > 0:
            errors['Error: Missing required column(s).'] = missing_req_cols

        if len(not_req_cols) > 0:
            errors['Warning: Contain no required columns(s).'] = not_req_cols

        if len(null_in_non_nullable_cols) > 0:
            errors['Error: Have null value(s) in non nullable columns(s).'] = null_in_non_nullable_cols

        errors.update(type_errors)

        return df, errors
