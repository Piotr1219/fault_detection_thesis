import json
import pandas as pd
import numpy as np
import os
import time

if __name__ == '__main__':
    path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         '..', '..', '..', 'datasets'))
    path_target_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                     '..', '..', '..', 'processed_data', 'datasets'))
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.expand_frame_repr', False)

    filename = 'Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv'
    print(filename)
    f = os.path.join(path, filename)
    if os.path.isfile(f) and filename[-4:] == '.csv':
        df = pd.read_csv(f)
        df['Time'] = df['Time(s)']

        df = df.rename(columns={'Engine_coolant_temperature': 'Engine Coolant Temperature [C]',
                                'Intake_air_pressure': 'Intake Manifold Pressure [kPa]',
                                'Engine_speed': 'Engine RPM [RPM]',
                                'Vehicle_speed': 'Vehicle Speed [km/h]',
                                'Intake Air Temperature [°C]': 'Intake Air Temperature [C]',
                                'Air Flow Rate from Mass Flow Sensor [g/s]': 'MAF [g/s]',
                                'Throttle_position_signal': 'Throttle Position [%]',
                                'Ambient Air Temperature [°C]': 'Ambient Air Temperature [C]',
                                'Accelerator_Pedal_value': 'Accelerator Pedal Position [%]',
                                'Steering_wheel_angle': 'Steering Wheel Angle [deg]'})

        df = df[['Time', 'Engine Coolant Temperature [C]', 'Intake Manifold Pressure [kPa]',
                'Engine RPM [RPM]', 'Vehicle Speed [km/h]',
                'Throttle Position [%]', 'Accelerator Pedal Position [%]', 'Steering Wheel Angle [deg]']]

        df = df[df['Engine Coolant Temperature [C]'].astype(str).str.contains(":").fillna(False) == False]
        df = df[df['Intake Manifold Pressure [kPa]'].astype(str).str.contains(":").fillna(False) == False]
        df = df[df['Throttle Position [%]'].astype(str).str.contains(":").fillna(False) == False]

        print(df.head(15))
        df.to_csv(os.path.join(path_target_data, filename), index=False)
