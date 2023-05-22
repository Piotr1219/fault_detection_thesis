import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

import scipy.cluster.hierarchy as sch

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import train_test_split
from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class PCA_detection:
    """Class for detection method based on reconstruction using PCA method.
        Parameter is considered faulty if difference of input value and value after reconstruction is above threshold
        percentage of mean value.

            Attributes
            ----------
            data : DataPreparation
                Class object containing differently prepared data.
            train : DataFrame
                DataFrame with training data without errors.
            test : DataFrame
                DataFrame with test data without errors.
            test_err : DataFrame
                DataFrame with test data including errors.
            X_train : np.array
                Data contained in train DataFrame, reshaped.
            X_test : np.array
                Data contained in test DataFrame, reshaped.
            X_test_err : np.array
                Data contained in test_err DataFrame, reshaped.
            max_diffs : np.array
                Maximum value of difference between input and reconstructed value, considered to be not faulty.

            Methods
            -------
            train_model(visualization):
                Method creates and trains model. Optionally it can show plot of data distribution and correlations.
            fault_detection(threshold_arr, visualization=False)
                Based on max_diff array, the threshold value for each parameter is set. In next step, data including
                faults is being reconstructed. Elements are considered as errors when difference between reconstruction
                and actual data is above threshold value. Optionally data can be plotted with marked detected and
                true errors.

    """

    def __init__(self, data):
        self.data = data
        self.train, self.test = train_test_split(self.data.data_drive_scaled[self.data.cols], test_size=0.3)
        self.test_err = self.data.data_drive_scaled_err.iloc[self.test.index]
        self.X_train = self.train.values.reshape(self.train.shape[0], 1, self.train.shape[1])
        self.X_test = self.test.values.reshape(self.test.shape[0], 1, self.test.shape[1])
        self.X_test_err = self.test_err[self.data.cols].values.reshape(self.test_err[self.data.cols].shape[0], 1,
                                                                       self.test_err[self.data.cols].shape[1])
        self.model = None
        self.max_diffs = None
        self.corr_mat = pairwise_distances(self.train.T, metric='correlation')

    def train_model(self, visualization=False):
        print(self.train.shape)
        print(self.train.columns)
        self.model = PCA(0.7, random_state=33)
        self.model.fit(self.train)

        if visualization:
            ### PLOT SERIES DISTRIBUTIONS ###
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 2, 1)
            self.train.plot(legend=False, xlabel='timesteps', ax=plt.gca())
            plt.subplot(1, 2, 2)
            self.train.plot(kind='density', legend=False, ax=plt.gca())
            plt.show()

            ### PLOT SERIES CORRELATIONS ###
            plt.figure(figsize=(12, 10))
            sns.heatmap(self.train.corr(), annot=False, cmap='bwr')
            plt.title('correlation matrix')
            plt.ylabel('series')
            plt.xlabel('series')
            plt.show()

    def fault_detection(self, threshold_arr, visualization=False):
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2])
        self.X_test_err = self.X_test_err.reshape(self.X_test_err.shape[0], self.X_test_err.shape[2])

        testX = self.data.scaler.inverse_transform(self.X_test)
        testErrX = self.data.scaler.inverse_transform(self.X_test_err)

        for threshold in threshold_arr:
            # determine threshold values for each column
            errors = []
            for index, row in pd.concat([self.train, self.test]).iterrows():
                Xt = self.model.inverse_transform(
                    self.model.transform(
                        np.array(row).reshape(1, -1)
                    )
                )
                errors.append((np.array(row).reshape(1, -1) - Xt)[0])
            errors = np.array(errors)
            self.max_diffs = []
            for i in range(len(errors[0])):
                self.max_diffs.append(np.percentile(errors[:, i], threshold * 100))
            # check results for test data with errors
            for err in self.data.error_col:
                time.sleep(1)
                errors = []
                column_number = self.data.cols.index(err)
                for index, row in self.test_err[self.data.cols].iterrows():
                    Xt = self.model.inverse_transform(
                        self.model.transform(
                            np.array(row).reshape(1, -1)
                        )
                    )
                    error = (np.array(row).reshape(1, -1) - Xt)[0]
                    error = error[column_number]
                    if error > self.max_diffs[column_number]:
                        errors.append(1)
                    else:
                        errors.append(0)
                errors_detected = np.array(errors)
                test_err_errors = self.test_err['error_' + err].values.astype(int)
                accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(test_err_errors,
                                                                                            errors_detected,
                                                                                            visualization=False)
                print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                      specificity, " F1_score: ", F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
                save_data("PCA_model", '-', 'Threshold=' + str(threshold), self.data.filename, self.data.error_col,
                          err, self.data.errors_types, accuracy, sensitivity, specificity,
                          F1_score, AUC, detection_rate, sheet_name="PCA_detection")

                if visualization:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    l=100
                    ax.plot(range(len(testErrX[:, column_number]))[0:l], testErrX[:, column_number][0:l], c='red', zorder=1, label='Wartość błędna')
                    ax.plot(range(len(testX[:, column_number]))[0:l], testX[:, column_number][0:l], c='blue', zorder=2, label='Wartość poprwana')
                    errors_detected = errors_detected.astype(float)
                    errors_detected[errors_detected == 0] = np.NaN
                    ax.scatter(range(len(errors_detected))[0:l], (errors_detected * testErrX[:, column_number])[0:l], c='magenta',
                               zorder=3, label='Wykryty błąd')
                    plt.title(err)
                    plt.xlabel('próbka')
                    plt.legend()
                    plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                    plt.show()


                    d = sch.distance.pdist(self.corr_mat)
                    L = sch.linkage(d, method='ward')
                    ind = sch.fcluster(L, d.max(), 'distance')
                    dendrogram = sch.dendrogram(L, no_plot=True)

                    labels = dendrogram['leaves']
                    corr_mat_cluster = pairwise_distances(
                        pd.concat([self.train.iloc[:, [i]] for i in labels], axis=1).T,
                        metric='correlation'
                    )

                    plt.figure(figsize=(18, 5))
                    ax1 = plt.axes()
                    plt.subplot(1, 2, 1)
                    dendrogram = sch.dendrogram(L, no_plot=False)
                    plt.title('dendrogram')
                    plt.ylabel('distance')
                    plt.xlabel('series')
                    print(plt.xticks())
                    print(type(plt.xticks()))
                    print(type(plt.xticks()[1][1]))
                    labels = [item.get_text() for item in plt.xticks()[1]]
                    labels = [self.train.columns[int(i)] for i in labels]
                    plt.xticks(plt.xticks()[0], labels, rotation='vertical')

                    plt.subplot(1, 2, 2)
                    plt.imshow(corr_mat_cluster, cmap='bwr')
                    plt.title('correlation matrix')
                    plt.ylabel('series')
                    plt.xlabel('series')
                    plt.xticks(range(len(self.train.columns)), labels, rotation=90)
                    plt.yticks(range(len(self.train.columns)), labels)
                    plt.show()
