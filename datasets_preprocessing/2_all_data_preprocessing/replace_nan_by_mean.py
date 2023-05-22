import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    path = (os.path.realpath(os.path.join(os.path.dirname(__file__),
                                               '..', '..', '..', 'processed_data', 'all_data', "all_data.csv")))

    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'all_data'))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    df = pd.read_csv(path, index_col=False)

    # df = df.astype(float, errors='ignore')
    df = df.replace(' ', '', regex=True)
    df = df.apply(pd.to_numeric)
    print(df.head(15))
    print(df.mean())
    df = df.fillna(df.mean())

    print(df.describe(include='all'))

    print(df.head(15))
    df.to_csv(os.path.join(path_target_data, "all_data_without_nan.csv"), index=False)
