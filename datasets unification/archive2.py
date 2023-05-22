import json
import pandas as pd
import numpy as np
import os


def unit1(df1, unit1, c):
    print(c)
    df1[c] = df1[c].astype(str)
    # print(df1.dtypes)
    df1[c] = df1[c].str.replace(unit1, '')
    df1 = df1.rename(columns={c: c + ' [' + unit1 + ']'})
    return df1


if __name__ == '__main__':
    path = '../../datasets/archive/'
    path_target_data = '../../datasets/archive/processed_data/'
    units = pd.read_csv(path_target_data + 'Units.csv')
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
                    df = unit1(df, '%', c)
                elif 'kPa' in str(df[c].iat[0]):
                    df = unit1(df, 'kPa', c)
                elif 'RPM' in str(df[c].iat[0]):
                    df = unit1(df, 'RPM', c)
                elif 'C' in str(df[c].iat[0]):
                    df = unit1(df, 'C', c)
                elif 'g/s' in str(df[c].iat[0]):
                    df = unit1(df, 'g/s', c)
                elif 'km/h' in str(df[c].iat[0]):
                    df = unit1(df, 'km/h', c)
                # print(list(units['parameter']))
                elif c in list(units['parameter']):
                    # print('c', c)
                    # print(units[units['parameter'] == 'BAROMETRIC_PRESSURE'])
                    # print(units[units['parameter'] == c]['unit'].values[0])
                    # print("unit")
                    # print(units[units['parameter'] == c]['unit'][0])
                    df = unit1(df, units[units['parameter'] == c]['unit'].values[0], c)

            pd.set_option('display.max_colwidth', 20)
            pd.set_option('display.max_columns', 40)
            pd.set_option('display.max_rows', 40)
            pd.set_option('display.expand_frame_repr', False)
            print(df.head(20))
            df.to_csv(os.path.join(path_target_data, filename))


