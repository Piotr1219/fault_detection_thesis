import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    paths = []
    folders = ['archive', 'carOBD-master', 'Dataset', 'Kia_soul', 'OBD-II-Dataset']
    for i in range(len(folders)):
        paths.append(os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..', '..', 'processed_data', 'datasets', folders[i])))

    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'all_data'))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    dfs = []

    for i, path in enumerate(paths):
        for filename in os.listdir(path):
            print(filename)
            f = os.path.join(path, filename)
            if os.path.isfile(f) and filename[-4:] == '.csv':
                df = pd.read_csv(f, index_col=False)
                dfs.append(df)

    df_res = pd.concat(dfs)
    df_res = df_res[df_res['Engine Coolant Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
    df_res = df_res[df_res['Intake Air Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
    df_res = df_res[df_res['MAF [g/s]'].astype(str).str.contains(":").fillna(False) == False]
    df_res = df_res[df_res['Intake Manifold Pressure [kPa]'].astype(str).str.contains(":").fillna(False) == False]
    df_res = df_res[df_res['Throttle Position [%]'].astype(str).str.contains(":").fillna(False) == False]

    print(df_res.describe(include='all'))

    print(df_res.head(15))
    df_res.to_csv(os.path.join(path_target_data, ("all_data.csv")), index=False)
