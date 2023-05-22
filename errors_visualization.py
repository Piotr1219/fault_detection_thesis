import pandas as pd

from ClusteringMethods import ClusteringMethods, plot_2d, plot_3d
from LSTM_model import LSTM_model
from LSTM_autoencoder import LSTM_autoencoder
from LSTM_autoencoder_sequence import LSTM_autoencoder_sequence
from XGBoost_detection import XGBoost_detection
from XGBoost_sequence import XGBoost_sequence
from PCA_detection import PCA_detection
from DBSCAN_detection import DBSCAN_detection
from DataPreparation import DataPreparation, visualize_data_time
from DimensionReduction import DimensionReduction
from faults_preparation import insert_fault_erratic
from utils import metrics, to_0_1_range
from clusters_run import run_all_clustering_methods

import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.width', 2000)

    file = 'kia_soul'
    # file = 'seat_leon'
    # file = 'seat_leon_s'
    # file = 'Dataset_a_all_1'
    # file = 'carobd_drive1'
    # file = 'car1_exp1'
    # error_col = ['Intake Manifold Pressure [kPa]', 'Throttle Position [%]']
    # error_col = ['Intake Manifold Pressure [kPa]', 'Engine Coolant Temperature [C]', 'Engine RPM [RPM]',
    #              'Vehicle Speed [km/h]', 'Throttle Position [%]']

    # tutaj same periods false
    error_col = ['Intake Manifold Pressure [kPa]', 'Engine Coolant Temperature [C]', 'Engine RPM [RPM]']
    # tutaj same periods true (może później sprawdzić też z false)
    # error_col = ['Vehicle Speed [km/h]', 'Throttle Position [%]']

    fig, ax = plt.subplots(4, 1, figsize=(10, 6))
    fig.suptitle('Ciśnienie w kolektorze dolotowym [kPa]')
    fig.tight_layout(h_pad=1.0)
    # plt.subplots_adjust(top=2.0)

    for i in range(4):

        data = DataPreparation(file)
        top_val = []
        for c in error_col:
            top_val.append(max(data.data[c]))
        # error_col parameter below is a list
        # data.insert_error(error_col=error_col, err_type=['erratic'], keep_range=False)
        if i==0:
            data.insert_error(error_col=error_col, time=0.05, err_type=['erratic'], keep_range=True, same_periods=False,  top_values=top_val)
        elif i==1:
            data.insert_error(error_col=error_col, time=0.05, err_type=['hardover'], keep_range=True, same_periods=False,  top_values=top_val)
        elif i==2:
            data.insert_error(error_col=error_col, time=0.05, err_type=['spike'], keep_range=True, same_periods=False, top_values=top_val)
        elif i==3:
            data.insert_error(error_col=error_col, time=0.05, err_type=['drift'], keep_range=True, same_periods=False, top_values=top_val)
        # data.insert_error(error_col=error_col, time=0.02, err_type=['erratic', 'hardover', 'spike', 'drift'], keep_range=True, same_periods=False, top_values=top_val)

        data.filter_out_statinary_and_drive_data()

        ax[i].plot(data.data_drive_err['Intake Manifold Pressure [kPa]'][0:150].index, data.data_drive_err['Intake Manifold Pressure [kPa]'][0:150], c='red', label='Wartość błędna')
        ax[i].plot(data.data_drive['Intake Manifold Pressure [kPa]'][0:150].index, data.data_drive['Intake Manifold Pressure [kPa]'][0:150], c='blue', label='Wartość poprawna')
        ax[i].set_ylim(20, 250)
        if i==0:
            ax[i].set_title('Erratic')
        if i==1:
            ax[i].set_title('Hardover')
        if i==2:
            ax[i].set_title('Spike')
        if i==3:
            ax[i].set_title('Drift')

    plt.legend()
    plt.savefig('errors1.svg')
    plt.show()