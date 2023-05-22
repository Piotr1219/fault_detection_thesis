import json
import pandas as pd
import numpy as np
import os
import time


def fill(df, means):
    # df=df.mask(df == '')
    # df = df.astype(float, errors='ignore')
    df = df.replace(' ', '', regex=True)
    df = df.apply(pd.to_numeric)
    cols = ['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                    'Engine RPM [RPM]', 'Vehicle Speed [km/h]', 'Intake Air Temperature [C]', 'MAF [g/s]',
                    'Throttle Position [%]', 'Ambient Air Temperature [C]', 'Accelerator Pedal Position [%]',
                    'Steering Wheel Angle [deg]']
    for c in cols:
        if c in df:
            df[c] = df[c].fillna(df.mean()[c])
        else:
            df[c] = means[c]
    return df


if __name__ == '__main__':
    paths = []
    folders = ['archive', 'carOBD-master', 'Dataset', 'Kia_soul', 'OBD-II-Dataset']
    for i in range(len(folders)):
        paths.append(os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..', '..', 'processed_data', 'datasets', folders[i])))

    path_all = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                             '..', '..', '..', 'processed_data', 'all_data',
                                             'all_data_without_nan.csv'))

    paths_target = []
    for i in range(len(folders)):
        paths_target.append(os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                          '..', '..', '..', 'processed_data', 'datasets_with_inserted',
                                                          folders[i])))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    df_all = pd.read_csv(path_all, index_col=False)
    all_mean = df_all.mean()

    dfs = []

    for i, path in enumerate(paths):
        for filename in os.listdir(path):
            print(filename)
            f = os.path.join(path, filename)
            if os.path.isfile(f) and filename[-4:] == '.csv':
                df = pd.read_csv(f, index_col=False)
                df = fill(df, all_mean)

                print(df.describe(include='all'))
                print(df.head(15))

                df.to_csv(os.path.join(paths_target[i], filename), index=False)


