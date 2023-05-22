import json
import pandas as pd
import numpy as np
import os

units = {}


def unit(df1, unit1):
    df1[c] = df1[c].str.replace(unit1, '')
    df1 = df1.rename(columns={c: c + ' [' + unit1 + ']'})
    if c not in units.keys():
        units[c] = unit1
    return df1


if __name__ == '__main__':
    path = '../../datasets/archive/'
    path_target_data = '../../datasets/archive/processed_data/'
    for filename in os.listdir(path):
        print(filename)
        f = os.path.join(path, filename)
        if os.path.isfile(f) and filename[-4:] == '.csv':
            df = pd.read_csv(f)
            cols = df.columns
            for c in cols:
                if c in ['EQUIV_RATIO', 'TROUBLE_CODES', 'DTC_NUMBER']:
                    df = df.drop(columns=[c])
                elif '%' in str(df[c].iat[0]):
                    df = unit(df, '%')
                elif 'kPa' in str(df[c].iat[0]):
                    df = unit(df, 'kPa')
                elif 'RPM' in str(df[c].iat[0]):
                    df = unit(df, 'RPM')
                elif 'C' in str(df[c].iat[0]):
                    df = unit(df, 'C')
                elif 'g/s' in str(df[c].iat[0]):
                    df = unit(df, 'g/s')
                elif 'km/h' in str(df[c].iat[0]):
                    df = unit(df, 'km/h')

            pd.set_option('display.max_colwidth', 20)
            pd.set_option('display.max_columns', 40)
            pd.set_option('display.max_rows', 40)
            pd.set_option('display.expand_frame_repr', False)
            print(df.head(20))
            # df.to_csv(os.path.join(path_target_data, filename))

    units = list(units.items())
    units = np.array(units)
    units = {'parameter': units[:, 0],
             'unit': units[:, 1]}
    units = pd.DataFrame(units)
    print(units)

    units.to_csv(os.path.join(path_target_data, 'Units.csv'))
