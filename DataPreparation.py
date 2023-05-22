import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data, round_half_away_from_zero
from faults_preparation import insert_fault_erratic, insert_fault_hardover, insert_fault_spike, insert_fault_drift, \
    insert_empty_error_columns
from preprocessing import filter_by_zeros, filter_out_negatives


def visualize_data_time(data, params, start_index=0, period_length=10000):
    """
                visualize_data_time(data, params, start_index=0, period_length=10000):
                    Plot of selected parameters of data DataFrame, in time, for selected time range
    """
    for i in range(len(params)):
        plt.subplot(len(params), 1, i + 1)
        plt.plot(data[start_index:start_index + period_length].index,
                 data[start_index:start_index + period_length][params[i]], color='red')
        plt.ylabel(params[i])

    plt.xlabel('index')
    # plt.savefig('vis2.svg')
    plt.show()


class DataPreparation:
    """Class containing multiple dataframes with differently processed data

            Attributes
            ----------
            data : DataFrame
                data to work with
            data_without_errors : DataFrame
                data straight form the car -  without inserting errors
            data_drive : DataFrame
                subset of data with velocity greater than 0
            data_stationary : DataFrame
                subset of data, where velocity is close to 0
            scaler : sklearn.preprocessing.StandardScaler
                used for normalizing data DataFrame
            data_scaled : DataFrame
                data normalized to 0-1 range
            data_drive_scaled : DataFrame
                data_drive normalized to 0-1 range
            data_stationary_scaled : DataFrame
                data_stationary normalized to 0-1 range


            Methods
            -------
            visualize_data():
                Shows scatter plot of column1 vs column2, where columns can be selected from data
            visualize_data_error_comparison(params, start_index=0, period_length=10000):
                Plot of selected parameters in time, comparing plain data to data with inserted errors
            visualize_data_time(data, params, start_index=0, period_length=10000):
                Plot of selected parameters of data DataFrame, in time, for selected time range
            data_correlation():
                Correlation between each 2 data columns
            insert_error(error_col, err_type=['erratic'], keep_range=False)
                Insert error in data, by calling functions appropriate to error type.

    """

    def __init__(self, file):
        # data reading and scaling
        self.filename = file
        self.data = read_data(file)
        self.error_col = None
        self.errors_types = None
        self.data_err = None
        self.data_drive_err = None
        self.data_drive_acc0_err = None
        self.data_drive = None
        self.data_stationary = None
        self.data_drive_acc0 = None
        self.data_stationary_err = None
        self.scaler = None
        self.data_scaled = None
        self.data_drive_scaled = None
        self.data_stationary_scaled = None
        self.data_scaled_err = None
        self.data_drive_scaled_err = None
        self.data_stationary_scaled_err = None
        self.data_scaled_chunks = None
        self.data_drive_scaled_chunks = None
        self.data_stationary_scaled_chunks = None
        self.data_scaled_err_chunks = None
        self.data_drive_scaled_err_chunks = None
        self.data_stationary_scaled_err_chunks = None
        self.cols = list(self.data.columns)
        self.cols.remove('Time')
        # self.data = self.data.drop(['Time', 'index'], axis=1, errors='ignore')
        self.data = self.data.dropna().reset_index(drop=True)
        if 'Accelerator Pedal Position [%]' in self.cols:
            self.data = filter_out_negatives(self.data, ['Vehicle Speed [km/h]', 'Accelerator Pedal Position [%]'])
        else:
            self.data = filter_out_negatives(self.data, ['Vehicle Speed [km/h]'])
        self.data_without_errors = self.data.copy(deep=True)
        self.data = insert_empty_error_columns(self.data)


    @staticmethod
    def time_chunks(data, min_length=5):
        time_delta = data['Time'][1] - data['Time'][0]
        # dividing by 3 to consider gaps shorter than 3 records still as continuous data
        data['time_diff'] = (data['Time'].diff() / time_delta - 1) / 3
        data['time_diff'] = data['time_diff'].round(0)
        data['time_chunk'] = data['time_diff'].cumsum()
        grouped = data.groupby(['time_chunk'])
        # drop unnecessary columns and remove data chunks shorter than min_length records
        dataframes = [group.drop(['time_diff', 'time_chunk', 'Time', 'index'], axis=1, errors='ignore').reset_index(drop=True) for _, group in
                      grouped if len(group) >= min_length]
        return dataframes

    def scaled_data_all_to_chunks(self):
        self.data_scaled_chunks = self.time_chunks(self.data_scaled)
        # num = []
        # for i in range(2, 50):
        #     self.data_drive_scaled_chunks = self.time_chunks(self.data_drive_scaled, min_length=i)
        #     num.append(len(self.data_drive_scaled_chunks))
        # fig, ax = plt.subplots(figsize=(12, 8))
        # ax.plot(range(2, 50), num)
        # plt.xlabel('Długość pojedynczej sekwencji (wyrażona w liczbie próbek)')
        # plt.ylabel('Liczba sekwencji')
        # plt.savefig('dlugosc_sekwencji_a_liczba.eps')
        # plt.show()
        self.data_drive_scaled_chunks = self.time_chunks(self.data_drive_scaled)
        self.data_stationary_scaled_chunks = self.time_chunks(self.data_stationary_scaled)
        self.data_scaled_err_chunks = self.time_chunks(self.data_scaled_err)
        self.data_drive_scaled_err_chunks = self.time_chunks(self.data_drive_scaled_err)
        self.data_stationary_scaled_err_chunks = self.time_chunks(self.data_stationary_scaled_err)

    def insert_error(self, error_col, time=0.05, err_type=['erratic'], keep_range=False, same_periods=False, top_values=None):
        # inserting different fault types
        self.error_col = error_col
        self.errors_types = err_type
        self.data_err = self.data.copy(deep=True)
        if 'erratic' in err_type:
            self.data_err = insert_fault_erratic(self.data_err.copy(deep=True), error_col,
                                                 intensity=4, time=time, period_length=15, same_periods=same_periods, keep_range=keep_range)
        elif 'hardover' in err_type:
            self.data_err = insert_fault_hardover(self.data_err.copy(deep=True), error_col,
                                                  top_values=top_values, time=time, period_length=15,
                                                  same_periods=same_periods)
        elif 'spike' in err_type:
            self.data_err = insert_fault_spike(self.data_err.copy(deep=True), error_col,
                                               intensity=3, time=time, same_periods=same_periods)
        elif 'drift' in err_type:
            self.data_err = insert_fault_drift(self.data_err.copy(deep=True), error_col,
                                               intensity=3, time=time, period_length=20, same_periods=same_periods)

    def filter_out_statinary_and_drive_data(self):
        self.data_drive, self.data_stationary = \
            filter_by_zeros(self.data, ['Vehicle Speed [km/h]'])
        self.data_drive_err = self.data_err.iloc[self.data_drive.index]
        self.data_stationary_err = self.data_err.iloc[self.data_stationary.index]
        self.data_drive = self.data_drive.reset_index(drop=True)
        self.data_stationary = self.data_stationary.reset_index(drop=True)
        self.data_drive_err = self.data_drive_err.reset_index(drop=True)
        self.data_stationary_err = self.data_stationary_err.reset_index(drop=True)
        if 'Accelerator Pedal Position [%]' in self.cols:
            self.data_drive, self.data_drive_acc0 = \
                filter_by_zeros(self.data_drive, ['Accelerator Pedal Position [%]'])
            self.data_drive_acc0_err = self.data_drive_err.iloc[self.data_drive_acc0.index]
            self.data_drive_err = self.data_drive_err.iloc[self.data_drive.index]
            self.data_drive = self.data_drive.reset_index(drop=True)
            self.data_drive_acc0 = self.data_drive_acc0.reset_index(drop=True)
            self.data_drive_err = self.data_drive_err.reset_index(drop=True)
            self.data_drive_acc0_err = self.data_drive_acc0_err.reset_index(drop=True)

    def scale_all_data(self):
        # data scaling
        self.scaler = MinMaxScaler().fit(self.data[self.cols])
        self.data_scaled = self.data.copy(deep=True)
        self.data_scaled[self.cols] = pd.DataFrame(self.scaler.transform(self.data[self.cols]), columns=self.cols)
        self.data_drive_scaled = self.data_drive.copy(deep=True)
        self.data_drive_scaled[self.cols] = pd.DataFrame(self.scaler.transform(self.data_drive[self.cols]),
                                                         columns=self.cols)
        self.data_stationary_scaled = self.data_stationary.copy(deep=True)
        self.data_stationary_scaled[self.cols] = pd.DataFrame(self.scaler.transform(self.data_stationary[self.cols]),
                                                              columns=self.cols)
        self.data_scaled_err = self.data_err.copy(deep=True)
        self.data_scaled_err[self.cols] = pd.DataFrame(self.scaler.transform(self.data_err[self.cols]),
                                                       columns=self.cols)
        self.data_drive_scaled_err = self.data_drive_err.copy(deep=True)
        self.data_drive_scaled_err[self.cols] = pd.DataFrame(self.scaler.transform(self.data_drive_err[self.cols]),
                                                             columns=self.cols)
        self.data_stationary_scaled_err = self.data_stationary_err.copy(deep=True)
        self.data_stationary_scaled_err[self.cols] = pd.DataFrame(
            self.scaler.transform(self.data_stationary_err[self.cols]),
            columns=self.cols)

    def visualize_data(self):
        # Data visualization
        plt.figure(figsize=(8, 6))
        # plt.scatter(self.data.Fuel_consumption, self.data.Accelerator_Pedal_value, s=0.2)
        # plt.scatter(self.data_scaled.Fuel_consumption, self.data_scaled.Intake_air_pressure, s=0.2)
        plt.scatter(self.data['Accelerator Pedal Position [%]'], self.data['Intake Manifold Pressure [kPa]'], s=0.2)
        plt.xlabel('Accelerator Pedal Position [%]')
        plt.ylabel('Intake Manifold Pressure [kPa]')
        plt.savefig('comp2.svg')
        plt.show()

    def visualize_distribution(self):
        # Data visualization
        plt.figure(figsize=(8, 6))
        for c in self.cols:
            self.data_scaled[c].plot(kind='kde')
        plt.xlim(0, 1)
        plt.legend(self.cols)
        plt.savefig('distr.svg')
        plt.show()

    def visualize_data_error_comparison(self, params, start_index=0, period_length=10000):
        # Data visualization
        fig, ax = plt.subplots(len(params), 1, figsize=(10, 6))
        fig.tight_layout(h_pad=1.7)
        plt.subplots_adjust(top=0.93)
        for i in range(len(params)):

            ax[i].plot(self.data_err[start_index:start_index + period_length].index,
                     self.data_err[start_index:start_index + period_length][params[i]], color='red')
            ax[i].plot(self.data_without_errors[start_index:start_index + period_length].index,
                     self.data_without_errors[start_index:start_index + period_length][params[i]], color='blue')
            ax[i].set_title(params[i])

        # plt.xlabel('próbka')
        plt.savefig('comp1.svg')
        plt.show()

    def data_correlation(self):
        corr = self.data[self.cols].corr(method='pearson')
        print(corr)
        names = list(corr.columns)
        names = [n[0: n.find('[') - 1] for n in names]
        figure = plt.figure(figsize=(18, 16))
        ax = figure.add_subplot(111)
        ax.matshow(corr, cmap='coolwarm')
        ax.set_xticks(np.arange(0, 9, 1))
        ax.set_yticks(np.arange(0, 9, 1))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.xticks(rotation=30, ha='left')
        for (i, j), z in np.ndenumerate(corr):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=12)
        plt.savefig('corr.svg')
        plt.show()
