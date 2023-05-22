import pandas as pd

from ClusteringMethods import ClusteringMethods, plot_2d, plot_3d
from LSTM_model import LSTM_model
from LSTM_autoencoder import LSTM_autoencoder
from LSTM_autoencoder_sequence import LSTM_autoencoder_sequence
from XGBoost_detection import XGBoost_detection
from XGBoost_sequence import XGBoost_sequence
from PCA_detection import PCA_detection
from DBSCAN_detection import DBSCAN_detection
from DataPreparation import DataPreparation
from DimensionReduction import DimensionReduction
from faults_preparation import insert_fault_erratic
from utils import metrics, to_0_1_range
from clusters_run import run_all_clustering_methods
from VAR_model import VAR_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.width', 2000)

    # file = 'kia_soul'
    # file = 'seat_leon'
    # file = 'Dataset_a_all_1'
    # file = 'carobd_drive1'
    # file = 'car1_exp1'
    files = ['kia_soul', 'seat_leon', 'Dataset_a_all_1', 'carobd_drive1', 'car1_exp1']
    # files = ['car1_exp1']
    for file in files:
        # error_col = ['Intake Manifold Pressure [kPa]', 'Throttle Position [%]']
        # error_col = ['Intake Manifold Pressure [kPa]', 'Engine Coolant Temperature [C]', 'Engine RPM [RPM]',
        #              'Vehicle Speed [km/h]', 'Throttle Position [%]']

        # tutaj same periods false
        # error_col = ['Intake Manifold Pressure [kPa]', 'Engine Coolant Temperature [C]', 'Engine RPM [RPM]']
        # tutaj same periods true (może później sprawdzić też z false)
        error_col = ['Vehicle Speed [km/h]', 'Throttle Position [%]']



        data = DataPreparation(file)
        top_val = []
        for c in error_col:
            top_val.append(max(data.data[c]))
        # error_col parameter below is a list
        # data.insert_error(error_col=error_col, err_type=['erratic'], keep_range=False)
        # data.insert_error(error_col=error_col, time=0.05, err_type=['erratic'], keep_range=True, same_periods=False,  top_values=top_val)
        # data.insert_error(error_col=error_col, time=0.05, err_type=['hardover'], keep_range=True, same_periods=False,  top_values=top_val)
        # data.insert_error(error_col=error_col, time=0.05, err_type=['spike'], keep_range=True, same_periods=False, top_values=top_val)
        # data.insert_error(error_col=error_col, time=0.05, err_type=['drift'], keep_range=True, same_periods=False, top_values=top_val)
        data.insert_error(error_col=error_col, time=0.02, err_type=['erratic', 'hardover', 'spike', 'drift'], keep_range=True, same_periods=False, top_values=top_val)

        data.filter_out_statinary_and_drive_data()
        data.scale_all_data()
        data.scaled_data_all_to_chunks()

        methods = ClusteringMethods(data)
        reduction = DimensionReduction()

        # methods.visualize_data()
        # methods.visualize_data_error_comparison(params=['Accelerator_Pedal_value', 'Fuel_consumption'])
        # methods.visualize_data_error_comparison(params=['Accelerator Pedal Position [%]', 'MAF [g/s]'], period_length=100)
        # visualize_data_time(data=methods.data_drive, params=['Accelerator_Pedal_value', 'Fuel_consumption'])
        # visualize_data_time(data=methods.data_drive, params=['Accelerator Pedal Position [%]', 'MAF [g/s]'])
        # methods.data_correlation()

        for r in ['PCA', 'UMAP', 'TSNE']:
            run_all_clustering_methods(methods, reduction=r, visualization=False)

        # reduction.dimension_reduction_visualization(data=methods.data.data_drive_scaled_err[methods.data.cols], method='UMAP')
        # reduction.dimension_reduction_visualization(data=methods.data.data_drive_scaled_err[methods.data.cols], method='PCA')
        # reduction.dimension_reduction_visualization(data=methods.data.data_drive_scaled_err[methods.data.cols], method='TSNE')

        lstm_model = LSTM_model(data)
        lstm_model.train_model(visualization=False)
        lstm_model.time_prediction(visualization=False)
        lstm_model.fault_detection(visualization=False, threshold_arr=[0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98, 0.96, 0.93, 0.9, 0.8])

        lstm_autoencoder = LSTM_autoencoder(data)
        lstm_autoencoder.train_model(visualization=False)
        lstm_autoencoder.fault_detection(threshold_arr=[0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.96, 0.93, 0.9, 0.8], visualization=False)

        lstm_autoencoder_sequence = LSTM_autoencoder_sequence(data)
        lstm_autoencoder_sequence.train_model(visualization=False)
        lstm_autoencoder_sequence.fault_detection(threshold_arr=[0.999, 0.998, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.96, 0.93, 0.9, 0.8], visualization=False)

        xgboost_detection = XGBoost_detection(data)
        xgboost_detection.train_model(visualization=False)
        xgboost_detection.fault_detection(visualization=False)

        xgboost_sequence = XGBoost_sequence(data)
        xgboost_sequence.train_model(visualization=False)
        xgboost_sequence.fault_detection(visualization=False)

        pca_detection = PCA_detection(data)
        pca_detection.train_model(visualization=False)
        pca_detection.fault_detection(threshold_arr=[0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98], visualization=False)

        # dbscan_detection = DBSCAN_detection(data)
        # dbscan_detection.fault_detection(eps_arr=[0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0], visualization=False)

        var_model = VAR_model(data)
        var_model.train_model(visualization=False)
        var_model.fault_detection(visualization=False,
                                  threshold_arr=[0.999, 0.998, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.96, 0.93, 0.9,
                                                 0.8])

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
