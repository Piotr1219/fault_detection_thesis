import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    paths = []
    folders = ['A', 'B', 'C', 'D']
    for i in range(len(folders)):
        paths.append(os.path.realpath(os.path.join(os.path.dirname(__file__),
                                             '..', '..', '..', 'datasets', 'Dataset', folders[i])))

    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'datasets', 'Dataset'))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    for i, path in enumerate(paths):
        for filename in os.listdir(path):
            print(filename)
            f = os.path.join(path, filename)
            if os.path.isfile(f) and filename[-4:] == '.csv':
                df = pd.read_csv(f, index_col=False)
                df['Time'] = df.index
                print(df.head(2))

                df = df.rename(columns={'cooling_temperature': 'Engine Coolant Temperature [C]',
                                        'inhale_pressure': 'Intake Manifold Pressure [kPa]',
                                        'engine_speed': 'Engine RPM [RPM]',
                                        'car_speed': 'Vehicle Speed [km/h]',
                                        'INTAKE_AIR_TEMP ()': 'Intake Air Temperature [C]',
                                        'Air Flow Rate from Mass Flow Sensor [g/s]': 'MAF [g/s]',
                                        'throttle_position': 'Throttle Position [%]',
                                        'Ambient Air Temperature [°C]': 'Ambient Air Temperature [C]',
                                        'accelerator_position': 'Accelerator Pedal Position [%]',
                                        'steering_wheel_angle': 'Steering Wheel Angle [deg]'})

                df = df[['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                        'Engine RPM [RPM]', 'Vehicle Speed [km/h]',
                        'Throttle Position [%]', 'Accelerator Pedal Position [%]', 'Steering Wheel Angle [deg]']]

                df = df.replace(',', '.', regex=True)

                df = df[df['Engine Coolant Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
                df = df[df['Intake Manifold Pressure [kPa]'].astype(str).str.contains(":").fillna(False) == False]
                df = df[df['Throttle Position [%]'].astype(str).str.contains(":").fillna(False) == False]

                print(df.head(15))
                df.to_csv(os.path.join(path_target_data, (folders[i] + '_' + filename)), index=False)
