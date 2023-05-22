import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         '..', '..', '..', 'datasets', 'archive', 'processed_data'))
    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'datasets', 'archive'))
    pd.set_option('display.max_colwidth', 30)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    for filename in os.listdir(path):
        print(filename)
        f = os.path.join(path, filename)
        if os.path.isfile(f) and filename[-4:] == '.csv':
            df = pd.read_csv(f)
            dfs = []
            names = []
            if filename == 'exp1_14drivers_14cars_dailyRoutes.csv':
                for i in range(1, 15):
                    dfs.append(df.loc[df['VEHICLE_ID'] == ('car'+str(i))].reset_index().copy(deep=True))
                    names.append('car' + str(i) + '_' + filename)
                for df1 in dfs:
                    df1['Time'] = pd.to_datetime(df1['TIMESTAMP'], unit='ms')
                    df1['Time'] = df1['Time'] - df1['Time'][0]
                    df1['Time'] = df1['Time'].dt.total_seconds()
            else:
                df['Time'] = df.index
                dfs.append(df)
                names.append(filename)

            for i, df in enumerate(dfs):
                df = df.rename(columns={'ENGINE_COOLANT_TEMP [C]': 'Engine Coolant Temperature [C]',
                                        'INTAKE_MANIFOLD_PRESSURE [kPa]': 'Intake Manifold Pressure [kPa]',
                                        'ENGINE_RPM [RPM]': 'Engine RPM [RPM]',
                                        'SPEED [km/h]': 'Vehicle Speed [km/h]',
                                        'AIR_INTAKE_TEMP [C]': 'Intake Air Temperature [C]',
                                        'MAF [g/s]': 'MAF [g/s]',
                                        'THROTTLE_POS [%]': 'Throttle Position [%]',
                                        'AMBIENT_AIR_TEMP [C]': 'Ambient Air Temperature [C]',
                                        'Accelerator Pedal Position D [%]': 'Accelerator Pedal Position [%]'})

                # print(df.head(2))
                df = df[['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                        'Engine RPM [RPM]', 'Vehicle Speed [km/h]', 'Intake Air Temperature [C]', 'MAF [g/s]',
                        'Throttle Position [%]', 'Ambient Air Temperature [C]']]

                df = df.replace(',', '.', regex=True)
                df = df[df.isnull().sum(axis=1) < 3]

                df = df[df['Engine Coolant Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
                df = df[df['Intake Air Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
                df = df[df['MAF [g/s]'].astype(str).str.contains(":").fillna(False) == False]
                df = df[df['Intake Manifold Pressure [kPa]'].astype(str).str.contains(":").fillna(False) == False]
                df = df[df['Throttle Position [%]'].astype(str).str.contains(":").fillna(False) == False]

                df = df.reset_index()
                df['Time'] = df['Time'] - df['Time'][0]
                # df = df.dropna()
                df = df[['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                        'Engine RPM [RPM]', 'Vehicle Speed [km/h]', 'Intake Air Temperature [C]', 'MAF [g/s]',
                        'Throttle Position [%]', 'Ambient Air Temperature [C]']]
                print(df.describe(include='all'))
                print(df.head(15))
                df.to_csv(os.path.join(path_target_data, names[i]), index=False)
