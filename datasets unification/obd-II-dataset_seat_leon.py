import json
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    path = '../../datasets/OBD-II-Dataset/'
    path_target_data = '../../datasets/OBD-II-Dataset/processed_data/'
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and filename[-4:] == '.csv':
            df = pd.read_csv(f)
            df['index'] = df.index
            df['index'] = (((df.index+1) % 10) == 0).astype(int)
            df = df[df['index'] == 1]
            df = df.set_index('Time')

            pd.set_option('display.max_colwidth', 20)
            pd.set_option('display.max_columns', 40)
            pd.set_option('display.max_rows', 40)
            pd.set_option('display.expand_frame_repr', False)
            print(df)
            df.to_csv(os.path.join(path_target_data, filename))
