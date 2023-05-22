import numpy as np
import pandas as pd


def filter_out_negatives(df, cols):
    for c in cols:
        df = df[df[c] >= 0]
    return df


def filter_by_zeros(df, columns1):
    """Function divides dataframe into 2 new dataframes. In first, data where all specified columns are near-zero are
    replaced by Nan. Second is complimentary dataframe, containing only data with near-zero values in specified
    columns.

            Parameters:
                df (DataFrame): Input dataframe
                columns (list): Columns that ALL must be zeros to remove row in first stage

            Returns:
                df1 (DataFrame): Dataframe without rows with near-zero values in specified columns
                df2 (DataFrame): Dataframe with only rows with near-zero values in specified columns
    """
    df_z = df.copy(deep=True)
    df_nz = df.copy(deep=True)
    df_res_nz = df.copy(deep=True)
    df_res_z_c1 = df.copy(deep=True)
    for col in columns1:
        mean = df[col].mean()
        margin = 0.01 * mean
        min_v = min(df[col])
        df_z = df_z.loc[(df[col] < min_v+margin) & (df[col] > min_v-margin)]
    df_z.loc[:] = np.inf
    df_res_nz.update(df_z)
    df_res_nz.replace(np.inf, np.nan, inplace=True)

    df_nz = df_res_z_c1.drop(df_z.index.values)
    df_nz.loc[:] = np.inf
    df_res_z_c1.update(df_nz)
    df_res_z_c1.replace(np.inf, np.nan, inplace=True)

    # df_res_nz = df_res_nz.dropna().reset_index(drop=True)
    # df_res_z_c1 = df_res_z_c1.dropna().reset_index(drop=True)
    df_res_nz = df_res_nz.dropna()
    df_res_z_c1 = df_res_z_c1.dropna()

    return df_res_nz, df_res_z_c1

