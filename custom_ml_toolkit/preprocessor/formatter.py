import warnings
from collections import Counter
from typing import Any, List, Dict, Tuple, Optional
import pandas as pd
from pandas.errors import ParserError


class DataFrameFormatter:
    '''Format the Pandas DataFrame as per specified format.
    Provides methods to enforce data types, check for missing required columns, validate
    non-null constraints, and ensure uniqueness of composite keys in a DataFrame.
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
        '''Initializes the DataFrameFormatter with specified formatting rules.
        Args:
            required_cols (list, optional): Description of param1.
            bool_cols (list, optional): A list of column name to be formatted as boolean type.
            int_cols (list, optional): A list of column name to be formatted as integer type.
            float_cols (list, optional): A list of column name to be formatted as floating point type.
            str_cols (list, optional): A list of column name to be formatted as string type
            dt_col_formats (dict, optional): A dictionary where keys are column names to be
                formatted as datetime type, and values are the format strings used to parse these columns.
                Refer to Python's strftime documentation for supported formats:
                https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
            literal_col_values (dict, optional): A dictionary where keys are column names and
                values are lists of valid literal strings for those columns. Used to validate that
                values in the specified columns match the provided list.
            non_nullable_cols (list, optional): A list of column names that must not contain
                null values. Validation will fail if nulls are detected in these columns.
            distinct_keys (list, optional): A list of column names that together form a
                composite key. Validation ensures the combination of these columns has unique
                values across the DataFrame, enforcing the uniqueness of the composite key.
        '''

        self._required_cols = list(set(self._default(required_cols, list)))
        self._dtype_col_dict = self._create_dtype_col_dict(
            bool_cols=bool_cols,
            int_cols=int_cols,
            float_cols=float_cols,
            str_cols=str_cols,
            dt_col_formats=dt_col_formats,
            literal_col_values=literal_col_values
        )
        self._check_duplicate_dtypes()
        self._distinct_keys = list(set(self._default(distinct_keys, list)))
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
    ) -> dict:
        '''Creates a dictionary mapping data types to their associated column lists.

        Args:
            bool_cols (list, optional): Columns to format as boolean.
            int_cols (list, optional): Columns to format as integer.
            float_cols (list, optional): Columns to format as float.
            str_cols (list, optional): Columns to format as string.
            dt_col_formats (dict, optional): Datetime columns with their formats.
            literal_col_values (dict, optional): Literal columns with valid values.

        Returns:
            dict: A dictionary mapping data types to their respective columns or formats.
        '''
        return {
            'bool': list(set(self._default(bool_cols, list))),
            'int': list(set(self._default(int_cols, list))),
            'float': list(set(self._default(float_cols, list))),
            'str': list(set(self._default(str_cols, list))),
            'dt': self._default(dt_col_formats, dict),
            'literal': self._default(literal_col_values, dict)
        }

    def _check_duplicate_dtypes(self) -> None:
        '''Validates that no column is assigned to more than one data type.

        Raises:
            ValueError: If duplicate columns exist in multiple data types.
        '''
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
        '''Creates the final list of non-nullable columns by merging specified and inferred columns.

        Args:
            non_nullable_cols (list): Initial list of non-nullable columns.

        Returns:
            list: Updated list of non-nullable columns, including inferred columns.

        Prints:
            Warning: Columns inferred from bool, int, and distinct_keys are added.
        '''
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

        warnings.warn(
            f"Warning: boolean, int, and distinct key columns: {missing_non_nullable_cols} "
            "have been added to non-nullable columns.",
            UserWarning,
        )
        return non_nullable_cols

    @staticmethod
    def _default(
        input: object,
        default_object: object
    ) -> object:
        '''Ensures a default object is returned if input is None, and validates its type.

        Args:
            input (object): Input value.
            default_object (object): Expected type or callable to create default.

        Returns:
            object: The input or a default object.

        Raises:
            TypeError: If input type is invalid.
        '''
        if input is None:
            return default_object()
        else:
            if not isinstance(input, default_object):
                raise TypeError(
                    f'Invalid input type. Expected {type(default_object).__name__}, '
                    f'but got {type(input).__name__}.'
                )
            return input

    @staticmethod
    def check_bool(
        series: pd.Series,
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        '''Validates and formats a series as boolean.

        Args:
            series (pd.Series): Input series.

        Returns:
            tuple: (is_formatted, formatted_series, has_error, error_messages)
        '''
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
            error_descs.append('Invalid data type (boolean).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_int(
        series: pd.Series,
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        '''Validates and formats a series as integer.

        Args:
            series (pd.Series): Input series.

        Returns:
            tuple: (is_formatted, formatted_series, has_error, error_messages)
        '''
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
                error_descs.append('Loss of precision (integer).')
            else:
                error = False
            series = formatted_series
            is_formatted = True
        except ValueError:
            error = True
            error_descs.append('Invalid data type (integer).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_float(
        series: pd.Series
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        '''Validates and formats a series as float.

        Args:
            series (pd.Series): Input series.

        Returns:
            tuple: (is_formatted, formatted_series, has_error, error_messages)
        '''
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
            error_descs.append('Invalid data type (float).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_str(
        series: pd.Series,
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        '''Validates and formats a series as string.

        Args:
            series (pd.Series): Input series.

        Returns:
            tuple: (is_formatted, formatted_series, has_error, error_messages)
        '''
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
            error_descs.append('Invalid data type (string).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_dt(
        series: pd.Series,
        format: str
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        '''Validates and formats a series as datetime using a specified format.

        Args:
            series (pd.Series): Input series.
            format (str): Datetime format string.
                Refer to Python's strftime documentation for supported formats:
                https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        Returns:
            tuple: (is_formatted, formatted_series, has_error, error_messages)
        '''
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
        except (ValueError, ParserError):
            error = True
            error_descs.append(f'Datetime formatting issues ({format}).')
            is_formatted = False
        return is_formatted, series, error, error_descs

    @staticmethod
    def check_literal(
        series: pd.Series,
        literals: List[str]
    ) -> Tuple[bool, pd.Series, bool, List[str]]:
        '''Validates and formats a series against specified literal values.

        Args:
            series (pd.Series): Input series.
            literals (list of str): Valid literal values.

        Returns:
            tuple: (is_formatted, formatted_series, has_error, error_messages)
        '''
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
        except (ValueError, AssertionError):
            error = True
            error_descs.append(f'Invalid literal value {literals}.')
            is_formatted = False
        return is_formatted, series, error, error_descs

    def _check_missing_required_cols(
        self,
        columns: list
    ) -> Tuple[List[str], List[str], List[str]]:
        '''Checks for missing required columns in the DataFrame.

        Args:
            columns (list): Columns present in the DataFrame.

        Returns:
            tuple: (missing_required_cols, existing_required_cols, not_required_cols)
        '''
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
        '''Validates and formats columns based on their specified data types.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: (formatted DataFrame, type_errors, formatted_cols, unformatted_cols)
        '''
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
        '''Checks for null values in non-nullable columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            list: Columns with null values in non-nullable columns.
        '''
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
        '''Validates the uniqueness of composite keys in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            null_in_non_nullable_cols (list): Columns with null values.

        Returns:
            bool: Whether the distinct keys are valid (True if invalid).
        '''
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
        '''Applies all validation and formatting rules to a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: (formatted DataFrame, errors, formatted_cols, unformatted_cols)

            Posible "errors" keys:
                - Missing required columns.
                - Not required columns.
                - Invalid data type (Boolean).
                - Loss of precision (integer).
                - Invalid data type (integer).
                - Invalid data type (float).
                - Invalid data type (string).
                - Datetime formatting issues ({format}).
                - Invalid literal value {literals}.
                - Null values in non-nullable columns.
                - Duplicate composite key violations.
        '''
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
            errors['Missing required columns.'] = missing_req_cols

        if len(not_req_cols) > 0:
            errors['Not required columns.'] = not_req_cols

        if len(null_in_non_nullable_cols) > 0:
            errors['Null values in non-nullable columns.'] = null_in_non_nullable_cols

        if valid_distinct_keys:
            errors['Duplicate composite key violations.'] = self._distinct_keys

        errors.update(type_errors)

        return df, errors, type_formatted_cols, type_unformatted_cols
