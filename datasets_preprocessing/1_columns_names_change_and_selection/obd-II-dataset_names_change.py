import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         '..', '..', '..', 'datasets', 'OBD-II-Dataset', 'processed_data'))
    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'datasets', 'OBD-II-Dataset'))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    for filename in os.listdir(path):
        print(filename)
        f = os.path.join(path, filename)
        if os.path.isfile(f) and filename[-4:] == '.csv':
            df = pd.read_csv(f)
            df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S.%f")
            df['Time'] = df['Time'] - df['Time'][0]
            df['Time'] = df['Time'].dt.total_seconds()

            df = df.rename(columns={'Engine Coolant Temperature [°C]': 'Engine Coolant Temperature [C]',
                                    'Engine Coolant Temperature [Â°C]': 'Engine Coolant Temperature [C]',
                                    'Intake Manifold Absolute Pressure [kPa]': 'Intake Manifold Pressure [kPa]',
                                    'Vehicle Speed Sensor [km/h]': 'Vehicle Speed [km/h]',
                                    'Intake Air Temperature [°C]': 'Intake Air Temperature [C]',
                                    'Intake Air Temperature [Â°C]': 'Intake Air Temperature [C]',
                                    'Air Flow Rate from Mass Flow Sensor [g/s]': 'MAF [g/s]',
                                    'Absolute Throttle Position [%]': 'Throttle Position [%]',
                                    'Ambient Air Temperature [°C]': 'Ambient Air Temperature [C]',
                                    'Ambient Air Temperature [Â°C]': 'Ambient Air Temperature [C]',
                                    'Accelerator Pedal Position D [%]': 'Accelerator Pedal Position [%]'})

            df = df[['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                    'Engine RPM [RPM]', 'Vehicle Speed [km/h]', 'Intake Air Temperature [C]', 'MAF [g/s]',
                    'Throttle Position [%]', 'Ambient Air Temperature [C]', 'Accelerator Pedal Position [%]']]

            df = df.replace(',', '.', regex=True)

            df = df[df['Engine Coolant Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['Intake Air Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['MAF [g/s]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['Intake Manifold Pressure [kPa]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['Throttle Position [%]'].astype(str).str.contains(":").fillna(False) == False]

            print(df.head(15))
            df.to_csv(os.path.join(path_target_data, filename), index=False)
