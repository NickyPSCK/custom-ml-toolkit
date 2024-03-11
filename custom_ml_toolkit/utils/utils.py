import pandas as pd


def merge_data(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    df_keys: list,
    right_df_keys: list,
    key_dtype: list
):

    left_df = left_df.copy()
    right_df = right_df.copy()

    left_noise_suffix = '_ORG_MNS'
    right_noise_suffix = '_EXT_NMS'

    new_left_df_keys = list()
    for col_name, dtype in zip(df_keys, key_dtype):
        new_col_name = col_name + left_noise_suffix
        left_df[new_col_name] = left_df[col_name].astype(dtype)
        new_left_df_keys.append(new_col_name)

    new_right_df_keys = list()
    for col_name, dtype in zip(right_df_keys, key_dtype):
        new_col_name = col_name + right_noise_suffix
        right_df[new_col_name] = right_df[col_name].astype(dtype)
        new_right_df_keys.append(new_col_name)

    left_df = left_df.merge(right_df,
                            how='left',
                            left_on=new_left_df_keys,
                            right_on=new_right_df_keys)

    drop_cols = new_left_df_keys + new_right_df_keys
    left_df = left_df.drop(drop_cols, axis=1)

    return left_df
