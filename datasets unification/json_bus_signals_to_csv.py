import json
import pandas as pd
import numpy as np


if __name__ == '__main__':
    path = '../../datasets/camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json'
    path_target_data = '../../datasets/camera_lidar/20180810_150607/bus/data_processed_v2.csv'
    path_target_units = '../../datasets/camera_lidar/20180810_150607/bus/units.csv'
    f = open(path)
    file = json.load(f)
    keys = file.keys()
    # units = pd.DataFrame(columns=['parameter', 'unit'])
    units = []
    data = pd.DataFrame(columns=['timestamp'])
    for k in keys:
        units.append([k, file[k]["unit"]])
        values = np.array(file[k]["values"])
        temp = pd.DataFrame(values, columns=['timestamp', k])
        data = data.merge(temp.set_index('timestamp'), on='timestamp', how='outer')

    units = np.array(units)
    units = {'parameter': units[:, 0],
             'unit': units[:, 1]}
    units = pd.DataFrame(units)

    data = data.sort_values(by=['timestamp'])

    data['tst_diff'] = data.diff()['timestamp']
    data['tst_diff'] = (data['tst_diff'] > 3000).astype(int)
    data['measurement'] = data['tst_diff'].cumsum()
    data = data.groupby(data['measurement'], as_index=False).mean()
    data = data.drop(columns=['tst_diff', 'measurement'])
    data = data.set_index('timestamp')
    data = data.fillna(method="ffill")
    data = data.dropna()

    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.expand_frame_repr', False)
    print(units)
    print(data)

    data.to_csv(path_target_data)
    units.to_csv(path_target_units)

