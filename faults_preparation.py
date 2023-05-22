import pandas as pd
import random
import numpy as np
from itertools import chain


def health_score(df_start, df, faulty_params):
    for f in faulty_params:
        df[f + '_health'] = 1 - abs((df[f] - df_start[f]) / df_start[f])
        df.loc[df[f + '_health'] < 0, f + '_health'] = 0
        df.loc[(df[f] == 0) & (np.isnan(df[f + '_health'])), f + '_health'] = 1
        df.loc[(np.isnan(df[f + '_health'])) & (df[f] != 0), f + '_health'] = 1 - abs((df[f]) / df_start[f].mean())
    return df


def get_indexes(df, faulty_params, time, period_length, period_diff, same_periods):
    indexes = np.array(list(df.index.values))
    indexes_errors = []
    if same_periods:
        ind_errors = np.array([])
        while ind_errors.size / indexes.size < time:
            start_index = random.randint(0, indexes.size)
            period = int(period_length + period_length * random.uniform(-period_diff, period_diff))
            if not np.in1d(ind_errors, indexes[start_index:start_index + period]).any():
                ind_errors = np.append(ind_errors, indexes[start_index:start_index + period]).astype(int)
        ind_errors.sort()
        for i in range(len(faulty_params)):
            indexes_errors.append(ind_errors)
    else:
        for i in range(len(faulty_params)):
            ind_errors = np.array([])
            while ind_errors.size / indexes.size < time:
                start_index = random.randint(0, indexes.size)
                period = int(period_length + period_length * random.uniform(-period_diff, period_diff))
                if not np.in1d(ind_errors, indexes[start_index:start_index + period]).any():
                    ind_errors = np.append(ind_errors, indexes[start_index:start_index + period].astype(int))
            ind_errors.sort()
            indexes_errors.append(ind_errors)
    return indexes_errors


def get_indexes_separate(df, faulty_params, time, period_length, period_diff, same_periods):
    indexes = list(df.index.values)
    indexes_errors = []
    if same_periods:
        ind_errors = []
        while len(list(chain.from_iterable(ind_errors))) / len(indexes) < time:
            start_index = random.randint(0, len(indexes))
            period = int(period_length + period_length * random.uniform(-period_diff, period_diff))
            if not bool(
                    list(set(list(chain.from_iterable(ind_errors))) & set(indexes[start_index:start_index + period]))):
                ind_errors.append(indexes[start_index:start_index + period])
        ind_errors.sort()
        for i in range(len(faulty_params)):
            indexes_errors.append(ind_errors)
    else:
        for i in range(len(faulty_params)):
            ind_errors = []
            while len(list(chain.from_iterable(ind_errors))) / len(indexes) < time:
                start_index = random.randint(0, len(indexes))
                period = int(period_length + period_length * random.uniform(-period_diff, period_diff))
                if not bool(list(set(list(chain.from_iterable(ind_errors))) & set(
                        indexes[start_index:start_index + period]))):
                    ind_errors.append(indexes[start_index:start_index + period])
            ind_errors.sort()
            indexes_errors.append(ind_errors)
    return indexes_errors


def get_indexes_2(df, faulty_params, time, period_length, period_diff, same_periods, separate_sequences=False):
    indexes = np.array(list(df.index.values))
    indexes_errors = []
    if same_periods:
        if separate_sequences:
            ind_errors = []
        else:
            ind_errors = np.array([])
        while ind_errors.size / indexes.size < time:
            start_index = random.randint(0, indexes.size)
            period = int(period_length + period_length * random.uniform(-period_diff, period_diff))
            if separate_sequences and not np.in1d(ind_errors, indexes[start_index:start_index + period]).any():
                ind_errors.append(indexes[start_index:start_index + period])
            elif not separate_sequences and not np.in1d(ind_errors, indexes[start_index:start_index + period]).any():
                ind_errors = np.append(ind_errors, indexes[start_index:start_index + period]).astype(int)
        ind_errors.sort()
        for i in range(len(faulty_params)):
            indexes_errors.append(ind_errors)
    else:
        for i in range(len(faulty_params)):
            if separate_sequences:
                ind_errors = []
            else:
                ind_errors = np.array([])
            while ind_errors.size / indexes.size < time:
                start_index = random.randint(0, indexes.size)
                period = int(period_length + period_length * random.uniform(-period_diff, period_diff))
                if separate_sequences and not np.in1d(ind_errors, indexes[start_index:start_index + period]).any():
                    ind_errors.append(indexes[start_index:start_index + period])
                elif not separate_sequences and not np.in1d(ind_errors,
                                                            indexes[start_index:start_index + period]).any():
                    ind_errors = np.append(ind_errors, indexes[start_index:start_index + period].astype(int))
            ind_errors.sort()
            indexes_errors.append(ind_errors)
    return indexes_errors


def insert_fault_erratic(df, faulty_params, intensity, time, period_length, period_diff=0.5, same_periods=False, keep_range=False):
    """Function inserts erratic error in data

            Parameters:
                df (DataFrame): Input dataframe
                faulty_params (list): Columns in df where error will be inserted
                intensity (float): Maximum error value in comparison to standard deviation of column values
                time (float): 0-1 Fraction of elements where errors will be inserted
                period_length (int): Number of elements in continuous period to insert errors
                period_diff (float): Maximum variance of period_length as a fraction of period_length
                same_periods (bool): Will errors for all measurements be inserted in the same time periods
                keep_range (bool): If this value is True, min and max values of data column are kept the same after
                    adding error. It prevents errors, which are big spikes in data.

            Returns:
                df (DataFrame): Dataframe with inserted errors
    """
    indexes_errors = get_indexes(df, faulty_params, time, period_length, period_diff, same_periods)
    df_start = df.copy(deep=True)
    errors_count = 0

    for i in range(len(faulty_params)):
        # mean = df[faulty_params[i]].mean()
        stdev = df[faulty_params[i]].std()
        dat = np.array(df.loc[np.array(indexes_errors[i]), [faulty_params[i]]]).swapaxes(0, 1)
        range_min = np.amin(dat)
        range_max = np.amax(dat)
        errors = np.random.uniform(-intensity, intensity, size=(len(dat[0]),)) * stdev
        errors = np.array([errors])
        dat = dat + errors
        if keep_range:
            dat[dat > range_max] = range_max
            dat[dat < range_min] = range_min
        df.loc[np.array(indexes_errors[i]), [faulty_params[i]]] = dat.swapaxes(0, 1)
        # set new target column for errors
        df['error_' + faulty_params[i]] = np.zeros(len(df)).astype(int)
        df.loc[np.array(indexes_errors[i]), ['error_' + faulty_params[i]]] = \
            np.array([np.ones(len(indexes_errors[i])).astype(int)]).swapaxes(0, 1)
        errors_count += df['error_' + faulty_params[i]].value_counts()
    df = health_score(df_start, df, faulty_params)

    print("Errors inserted: ", errors_count)
    return df


def insert_fault_hardover(df, faulty_params, top_values, time, period_length, period_diff=0.5, same_periods=False):
    """Function inserts hardover error in data. It means temporary maximal signal value for sensor.

            Parameters:
                df (DataFrame): Input dataframe
                faulty_params (list): Columns in df where error will be inserted
                top_values (list): Maximum possible values of signals for selected faulty_parameters
                time (float): 0-1 Fraction of elements where errors will be inserted
                period_length (int): Number of elements in continuous period to insert errors
                period_diff (float): Maximal difference of period_length in comparison to period_length
                same_periods (bool): Will errors for all measurements be inserted in the same time periods

            Returns:
                df (DataFrame): Dataframe with inserted errors
    """
    indexes_errors = get_indexes(df, faulty_params, time, period_length, period_diff, same_periods)
    df_start = df.copy(deep=True)
    errors_count = 0

    for i in range(len(faulty_params)):
        dat = np.array(df.loc[indexes_errors[i], [faulty_params[i]]]).swapaxes(0, 1)
        errors = np.repeat(top_values[i], len(dat[0]))
        errors = np.array([errors])
        dat = errors
        df.loc[indexes_errors[i], [faulty_params[i]]] = dat.swapaxes(0, 1)
        # set new target column for errors
        df['error_' + faulty_params[i]] = np.zeros(len(df))
        df.loc[indexes_errors[i], ['error_' + faulty_params[i]]] = \
            np.array([np.ones(indexes_errors[i].size)]).swapaxes(0, 1)
        errors_count += df['error_' + faulty_params[i]].value_counts()
    df = health_score(df_start, df, faulty_params)

    print("Errors inserted: ", errors_count)
    return df


def insert_fault_spike(df, faulty_params, intensity, time, same_periods=False):
    """Function inserts spike errors in data. It means big random single values increase.

            Parameters:
                df (DataFrame): Input dataframe
                faulty_params (list): Columns in df where error will be inserted
                intensity (float): Multiplication factor for signal value to create spike error
                time (float): 0-1 Fraction of elements where errors will be inserted
                same_periods (bool): Will errors for all measurements be inserted in the same time periods

            Returns:
                df (DataFrame): Dataframe with inserted errors
    """
    indexes_errors = get_indexes(df, faulty_params, time, 1, 0, same_periods)
    df_start = df.copy(deep=True)
    errors_count = 0

    for i in range(len(faulty_params)):
        dat = np.array(df.loc[np.array(indexes_errors[i]), [faulty_params[i]]]).swapaxes(0, 1)
        errors = np.repeat(intensity, len(dat[0]))
        errors = np.array([errors])
        dat = dat * errors
        df.loc[np.array(indexes_errors[i]), [faulty_params[i]]] = dat.swapaxes(0, 1)
        # set new target column for errors
        df['error_' + faulty_params[i]] = np.zeros(len(df))
        df.loc[np.array(indexes_errors[i]), ['error_' + faulty_params[i]]] = \
            np.array([np.ones(len(indexes_errors[i]))]).swapaxes(0, 1)
    df = health_score(df_start, df, faulty_params)

    print("Errors inserted: ", errors_count)
    return df


def insert_fault_drift(df, faulty_params, intensity, time, period_length, period_diff=0.5, same_periods=False):
    """Function inserts drift error in data.
    For half of period_length it rises and for the other half stays flat on max value.

            Parameters:
                df (DataFrame): Input dataframe
                faulty_params (list): Columns in df where error will be inserted
                intensity (float): Maximum error value in comparison to mean column value
                time (float): 0-1 Fraction of elements where errors will be inserted
                period_length (int): Number of elements in continuous period to insert errors
                period_diff (float): Maximal difference of period_length in comparison to period_length
                same_periods (bool): Will errors for all measurements be inserted in the same time periods

            Returns:
                df (DataFrame): Dataframe with inserted errors
    """
    indexes_errors = get_indexes_separate(df, faulty_params, time, period_length, period_diff, same_periods)
    df_start = df.copy(deep=True)
    errors_count = 0

    for i in range(len(faulty_params)):
        df['error_' + faulty_params[i]] = np.zeros(len(df))
        errors = np.array([])
        indexes_all = list(chain.from_iterable(indexes_errors[i]))
        dat = np.array(df.loc[np.array(indexes_all), [faulty_params[i]]]).swapaxes(0, 1)
        for indexes in indexes_errors[i]:
            errors1 = np.linspace(1, intensity, num=int(len(indexes) / 2))
            errors2 = np.ones(len(indexes) - int(len(indexes) / 2)) * intensity
            errors = np.concatenate((errors, np.concatenate((errors1, errors2))))
        dat = dat * errors
        df.loc[np.array(indexes_all), [faulty_params[i]]] = dat.swapaxes(0, 1)
        # add binary error target
        df.loc[np.array(indexes_all), ['error_' + faulty_params[i]]] = \
            np.array([np.ones(len(indexes_all))]).swapaxes(0, 1)
    df = health_score(df_start, df, faulty_params)

    print("Errors inserted: ", errors_count)
    return df


def insert_empty_error_columns(df):
    for i in range(len(df.columns)):
        df['error_' + df.columns[i]] = np.zeros(len(df))
        df[df.columns[i] + '_health'] = np.ones(len(df))
    return df
