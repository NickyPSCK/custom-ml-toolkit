from typing import Tuple
import pandas as pd


def melt_list_of_list(list_of_list):
    result_list = list()
    for member in list_of_list:
        if isinstance(member, list):
            result_list += melt_list_of_list(member)
        else:
            result_list.append(member)
    return result_list


def split_cal_and_num(
    col_names: list,
    all_cat_cols: list,
    all_num_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cat_columns = list()
    num_columns = list()
    for col_name in col_names:
        if col_name in all_cat_cols:
            cat_columns.append(col_name)
        elif col_name in all_num_cols:
            num_columns.append(col_name)
    return num_columns, cat_columns