import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         '..', '..', '..', 'datasets', 'carOBD-master', 'obdiidata'))
    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'datasets', 'carOBD-master'))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    for filename in os.listdir(path):
        print(filename)
        f = os.path.join(path, filename)
        if os.path.isfile(f) and filename[-4:] == '.csv':
            df = pd.read_csv(f, index_col=False)
            df['Time'] = df['ENGINE_RUN_TINE ()']
            print(df.head(2))

            df = df.rename(columns={'COOLANT_TEMPERATURE ()': 'Engine Coolant Temperature [C]',
                                    'INTAKE_MANIFOLD_PRESSURE ()': 'Intake Manifold Pressure [kPa]',
                                    'ENGINE_RPM ()': 'Engine RPM [RPM]',
                                    'VEHICLE_SPEED ()': 'Vehicle Speed [km/h]',
                                    'INTAKE_AIR_TEMP ()': 'Intake Air Temperature [C]',
                                    'Air Flow Rate from Mass Flow Sensor [g/s]': 'MAF [g/s]',
                                    'THROTTLE ()': 'Throttle Position [%]',
                                    'Ambient Air Temperature [°C]': 'Ambient Air Temperature [C]',
                                    'PEDAL_D ()': 'Accelerator Pedal Position [%]'})

            df = df.drop_duplicates(subset=['Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                    'Engine RPM [RPM]', 'Vehicle Speed [km/h]', 'Intake Air Temperature [C]',
                    'Throttle Position [%]', 'Accelerator Pedal Position [%]'])

            df = df[df['Engine Coolant Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['Intake Air Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['Intake Manifold Pressure [kPa]'].astype(str).str.contains(":").fillna(False) == False]
            df = df[df['Throttle Position [%]'].astype(str).str.contains(":").fillna(False) == False]

            df = df.reset_index()
            df['Time'] = df.index

            df = df[['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                    'Engine RPM [RPM]', 'Vehicle Speed [km/h]', 'Intake Air Temperature [C]',
                    'Throttle Position [%]', 'Accelerator Pedal Position [%]']]

            df = df.replace(',', '.', regex=True)
            print(df.head(15))
            df.to_csv(os.path.join(path_target_data, filename), index=False)
