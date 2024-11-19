from collections import Counter
from typing import Any, List, Dict, Tuple, Optional
import pandas as pd

pd.DataFrame()

class DataFrameFormatter:
    '''Format the Pandas DataFrame as per specified format.
    
    '''
    def __init__(
        self,
        required_cols: Optional[list] = None,
        bool_cols: Optional[list] = None,  # Non-nullable
        int_cols: Optional[list] = None,  # Non-nullable
        float_cols: Optional[list] = None,
        str_cols: Optional[list] = None,
        dt_col_formats: Optional[Dict[Any, str]] = None,
        literal_col_values: Optional[Dict[Any, List[str]]] = None,
        non_nullable_cols: Optional[list] = None,
        distinct_keys: Optional[list] = None
    ):
        '''Pandas DataFrame Formatter
        Args:
            required_cols (:obj:`list`, optional): Description of `param1`.
            bool_cols (:obj:`list`, optional): A list of column name to be formatted as boolean type. 
            int_cols (:obj:`list`, optional): A list of column name to be formatted as integer type.
            float_cols (:obj:`list`, optional): A list of column name to be formatted as floating point type.
            str_cols (:obj:`list`, optional): A list of column name to be formatted as string type
            dt_col_formats (:obj:`dict`, optional): A dictionary where the keys are 
                column names to be formatted as datetime type
                and the values are the format to parse datetime.
                See strftime documentation for more information on choices:
                https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

        '''
        # Add distinct keys
        self._required_cols = list(set(self._defalut(required_cols, list)))
        self._dtype_col_dict = self._create_dtype_col_dict(
            bool_cols=bool_cols,
            int_cols=int_cols,
            float_cols=float_cols,
            str_cols=str_cols,
            dt_col_formats=dt_col_formats,
            literal_col_values=literal_col_values
        )
        self._check_duplicate_dtypes()
        self._distinct_keys = list(set(self._defalut(distinct_keys, list)))
        self._non_nullable_cols = self._create_non_nullable_cols(
            non_nullable_cols=non_nullable_cols
        )

    def _create_dtype_col_dict(
            self,
            bool_cols: Optional[list] = None,  # Non-nullable
            int_cols: Optional[list] = None,  # Non-nullable
            float_cols: Optional[list] = None,
            str_cols: Optional[list] = None,
            dt_col_formats: Optional[Dict[str, str]] = None,
            literal_col_values: Optional[Dict[str, List[str]]] = None
    ) -> None:
        return {
            'bool': list(set(self._defalut(bool_cols, list))),
            'int': list(set(self._defalut(int_cols, list))),
            'float': list(set(self._defalut(float_cols, list))),
            'str': list(set(self._defalut(str_cols, list))),
            'dt': self._defalut(dt_col_formats, dict),
            'literal': self._defalut(literal_col_values, dict)
        }

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

    def _create_non_nullable_cols(
        self,
        non_nullable_cols: list
    ) -> list:
        non_nullable_cols = list(set(non_nullable_cols))
        non_nullable_by_dtype_cols = set(
            self._dtype_col_dict['bool']
            + self._dtype_col_dict['int']
            + self._distinct_keys
        )
        missing_non_nullable_cols = list()
        for col in non_nullable_by_dtype_cols:
            if col not in non_nullable_cols:
                missing_non_nullable_cols.append(col)

        non_nullable_cols += missing_non_nullable_cols

        print(f'Warning: boolean, int and distinct key columns: {missing_non_nullable_cols} have been added to non nullable columns.')
        return non_nullable_cols

    @staticmethod
    def _defalut(
        input: object,
        default_object: object
    ) -> object:
        if input is None:
            return default_object()
        else:
            if not isinstance(input, default_object):
                raise TypeError('Invalid input type')
            return input

    @staticmethod
    def check_bool(
        series: pd.Series,
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
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
            is_formatted = True
        except ValueError:
            error = True
            error_descs.append('Error: Invalid boolean value(s).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_int(
        series: pd.Series,
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
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
            is_formatted = True
        except ValueError:
            error = True
            error_descs.append('Error: Invalid integer value(s).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_float(
        series: pd.Series
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        formatted_series = series.copy()
        error_descs = list()
        try:
            formatted_series[formatted_series == ''] = None
            formatted_series = formatted_series.astype(
                dtype=float, copy=True, errors='raise'
            )
            error = False
            series = formatted_series
            is_formatted = True
        except ValueError:
            error = True
            error_descs.append('Error: Invalid float value(s).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_str(
        series: pd.Series,
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        formatted_series = series.copy()
        error_descs = list()
        try:
            null_index = formatted_series.isna()
            formatted_series = formatted_series.astype(
                dtype=str, copy=True, errors='raise'
            )
            formatted_series[null_index] = None
            error = False
            series = formatted_series
            is_formatted = True
        except ValueError:
            error = True
            error_descs.append('Error: Invalid string value(s).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_dt(
        series: pd.Series,
        format: str
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        formatted_series = series.copy()
        error_descs = list()
        try:
            formatted_series[formatted_series == ''] = None
            formatted_series = pd.to_datetime(
                formatted_series, errors='raise', format=format
            )
            error = False
            series = formatted_series
            is_formatted = True
        # except ParserError:
        #     error = True
        #     error_descs.append(f'Error: Invalid datetime format ({format}).')
        #     is_formatted = False
        except ValueError:
            error = True
            error_descs.append(f'Error: Invalid datetime format ({format}).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_literal(
        series: pd.Series,
        literals: List[str]
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
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
            is_formatted = True
        except ValueError:
            error = True
            error_descs.append(f'Error: Invalid literal value(s) {literals}.')
            is_formatted = False
        except AssertionError:
            error = True
            error_descs.append(f'Error: Invalid literal value(s) {literals}.')
            is_formatted = False
        return is_formatted, series, error, error_descs

    def _check_missing_required_cols(
        self,
        columns: list
    ) -> Tuple[List[str], List[str], List[str]]:
        target_cols = set(columns)
        required_cols = set(self._required_cols)
        missing_required_cols = list(required_cols.difference(target_cols))
        existing_required_cols = list(required_cols.intersection(target_cols))
        not_required_cols = list(target_cols.difference(required_cols))

        return missing_required_cols, existing_required_cols, not_required_cols

    def _check_types(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, dict, list, list]:
        df = df.copy()
        formatted_cols = list()
        unformatted_cols = list()
        type_errors = dict()
        for type_name in self._dtype_col_dict:
            for col in self._dtype_col_dict[type_name]:
                if col in df.columns:

                    if isinstance(self._dtype_col_dict[type_name], dict):
                        # 'dt', 'literal'
                        is_formatted, df[col], error, error_descs = getattr(
                            self,
                            f'check_{type_name}'
                        )(df[col], self._dtype_col_dict[type_name][col])
                    elif isinstance(self._dtype_col_dict[type_name], list):
                        # 'bool', 'int', 'float', 'str'
                        is_formatted, df[col], error, error_descs = getattr(
                            self,
                            f'check_{type_name}'
                        )(df[col])
                    else:
                        # Add Proper Error
                        raise ValueError('Input error')

                    if is_formatted:
                        formatted_cols.append(col)
                    else:
                        unformatted_cols.append(col)

                    if error:
                        for error in error_descs:
                            if error in type_errors:
                                type_errors[error].append(col)
                            else:
                                type_errors[error] = [col]
        return df, type_errors, formatted_cols, unformatted_cols

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

    def _check_distinct_keys(
        self,
        df: pd.DataFrame,
        null_in_non_nullable_cols: list,
    ) -> bool:
        df = df.copy()
        null_disticnt_keys = list()
        for col in self._distinct_keys:
            if col in null_in_non_nullable_cols:
                null_disticnt_keys.append(col)
        try:
            assert len(null_disticnt_keys) == 0
            assert df[self._distinct_keys].drop_duplicates().shape[0] == df.shape[0]
            valid_distinct_keys = False
        except AssertionError:
            valid_distinct_keys = True
        return valid_distinct_keys

    def format(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, dict, list, list]:
        errors = dict()

        missing_req_cols, _, not_req_cols = self._check_missing_required_cols(
            columns=df.columns
        )

        df, type_errors, type_formatted_cols, type_unformatted_cols = self._check_types(df=df)

        null_in_non_nullable_cols = self._check_null_values(df=df)

        valid_distinct_keys = self._check_distinct_keys(
            df=df,
            null_in_non_nullable_cols=null_in_non_nullable_cols
        )

        if len(missing_req_cols) > 0:
            errors['Error: Missing required column(s).'] = missing_req_cols

        if len(not_req_cols) > 0:
            errors['Warning: Contain no required columns(s).'] = not_req_cols

        if len(null_in_non_nullable_cols) > 0:
            errors['Error: Have null value(s) in non nullable columns(s).'] = null_in_non_nullable_cols

        if valid_distinct_keys:
            errors['Error: Distinct keys are not valid.'] = self._distinct_keys

        errors.update(type_errors)

        return df, errors, type_formatted_cols, type_unformatted_cols
