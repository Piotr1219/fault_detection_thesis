import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import train_test_split
from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class DBSCAN_detection:
    """Class for detection method based on clustering using DBSCAN method.
        We treat data sequences as vectors of given number of dimensions. Next we compute correlation matrix between all
        vectors. Correlation matrix is provided to this model. It is expected that all correlations
        should be included in one cluster. All values corresponding to points out of main cluster are considered faulty.

            Attributes
            ----------
            data : DataPreparation
                Class object containing differently prepared data.
            sequence_length : int
                Length of data sequence provided to model.
            train_x_err : list
                List of Dataframes with training data, with included columns containing information about errors.
            train_y_err : list
                List containing Series objects, which are last rows of sequences from train_x_err
            train_y_no_err : list
                The same data as train_y_err, but not containing errors.
            train_x : list
                List containing sequences of training data, without additional information about errors.

            Methods
            -------
            fault_detection(threshold_arr, visualization=False)
                Based on DBSCAN fit_predict method, distances between vectors (sequences) corresponding to all
                parameters in current time sequence, are assigned to clusters. Values that are not included in main
                jego sygnacluster are considered faulty.

    """

    def __init__(self, data):
        self.data = data
        self.sequence_length = 6
        _, _, self.train_x_err, self.train_y_err = to_sequences(self.data.data_drive_scaled_err_chunks, self.data.cols,
                                                                                      self.sequence_length, errors=True,
                                                                                      current_in=True)
        _, _, _, self.train_y_no_err = to_sequences(self.data.data_drive_scaled_chunks, self.data.cols,
                                                                                      self.sequence_length, errors=True,
                                                                                      current_in=True)
        self.train_x = []
        for seq in self.train_x_err:
            self.train_x.append(seq[self.data.cols])

    def fault_detection(self, eps_arr, visualization=False):
        for eps in eps_arr:
            network_ano = []
            dbscan = DBSCAN(eps=eps, min_samples=3, metric="precomputed")

            for i, sequence in enumerate(self.train_x):

                preds = dbscan.fit_predict(
                    # pairwise_distances correlation computes correlation between all parameters vectors for current
                    # time sequence
                    np.nan_to_num(pairwise_distances(sequence.T, metric='correlation'))
                )
                if (preds > 0).any():
                    ano_features = list(sequence.columns[np.where(preds > 0)[0]])
                    network_ano.append(ano_features)
                else:
                    network_ano.append(None)

            errors_detected = []
            for detected in network_ano:
                if detected is not None:
                    detected_set = set(detected)
                    errors_detected.append([1 if val in detected_set else 0 for i, val in enumerate(self.data.cols)])
                else:
                    errors_detected.append([0 for i in range(len(self.data.cols))])

            errors_detected = np.array(errors_detected)
            self.train_y_err = pd.DataFrame(self.train_y_err)
            self.train_y_no_err = pd.DataFrame(self.train_y_no_err)

            testErrY = self.data.scaler.inverse_transform(self.train_y_err[self.data.cols])
            testY = self.data.scaler.inverse_transform(self.train_y_no_err[self.data.cols])

            for err in self.data.error_col:
                column_number = self.data.cols.index(err)
                test_err_errors = self.train_y_err['error_' + err].values.astype(int)
                testPredict = errors_detected[:, column_number]
                accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(test_err_errors,
                                                                                            testPredict,
                                                                                            visualization=False)
                print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                      specificity, " F1_score: ", F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
                save_data("DBSCAN_model", '-', 'Eps=' + str(eps), self.data.filename, self.data.error_col,
                          err, self.data.errors_types, accuracy, sensitivity, specificity,
                          F1_score, AUC, detection_rate, sheet_name="DBSCAN_sequence")

                if visualization:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    l=100
                    ax.plot(range(len(testErrY[:, column_number]))[0:l], testErrY[:, column_number][0:l], c='red', zorder=1, label='Wartość błędna')
                    ax.plot(range(len(testY[:, column_number]))[0:l], testY[:, column_number][0:l], c='blue', zorder=2, label='Wartość poprwana')
                    testPredict = testPredict.astype(float)
                    testPredict[testPredict == 0] = np.NaN
                    ax.scatter(range(len(testPredict))[0:l], (testPredict * testErrY[:, column_number])[0:l], c='magenta', zorder=3, label='Wykryty błąd')
                    plt.title(err)
                    plt.xlabel('próbka')
                    plt.legend()
                    plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                    plt.show()





