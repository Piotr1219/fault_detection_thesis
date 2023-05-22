import numpy as np
import math
from statsmodels.tsa.api import VAR
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import train_test_split

from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class VAR_model:
    """Class for detection method based on time series prediction using LSTM.

            Attributes
            ----------
            data : DataPreparation
                Class object containing differently prepared data.
            sequence_length : int
                Length (in time samples) of data sequence, which is input to model.
            train_chunks : list of DataFrame
                Chunks of continuous data, used as training data
            test_chunks : list of DataFrame
                Chunks of continuous data, used as test data
            train_x : np.array
                Array of training data sequences of length sequence_length, made of data from train_chunks
            train_y : np.array
                Array of elements following respective train_x sequences.
            train_x_err : list of DataFrame
                List of training data sequences train_x, containing additional columns with information about errors.
            train_y_err : list of DataFrame
                Array of elements in train_y, containing additional columns with information about errors.
            test_x : np.array
                Array of test data sequences of length sequence_length, made of data from train_chunks
            test_y : np.array
                Array of elements following respective test_x sequences.
            test_err_chunks : list of DataFrame
                The same chunks as test_chunks, but with inserted errors.
            test_err_x : np.array
                The same data as test_x but with inserted errors.
            test_err_y : np.array
                The same data as test_y but with inserted errors.
            test_x_errors_col : list of Dataframe
                List of training data sequences test_err_x, containing additional columns with information about errors.
            test_y_errors_col : list of Dataframe
                List of training data sequences test_err_y, containing additional columns with information about errors.


            Methods
            -------
            predicted_data_comparison(data, data_predict, c, chunks_test_seq_len, name):
                Plots original data and data predicted by model to visually inspect model quality.
            train_model(visualization):
                Method creates and trains model. Optionally it can show plot od losses and train data errors distribution
            time_prediction(visualization):
                This method predicts data using previously trained model. Optionally it can plot original and predicted
                data for visual comparison
            fault_detection(threshold_arr, visualization=False)
                This method first computes prediction error for training and test data without faults. Then, based on
                threshold_err value, the threshold value is set. In next step results are predicted for faulty data. Elements
                are considered as errors when difference between predicted and actual data is above threshold value.
                Optionally data can be plotted with marked detected and true errors.

    """
    def __init__(self, data):
        self.data = data
        self.sequence_length = 6
        self.train, self.test = train_test_split(self.data.data_drive_scaled, test_size=0.3)
        self.test_err = self.data.data_drive_scaled_err.iloc[self.test.index]
        # , self.test_err = train_test_split(self.data.data_drive_scaled_err, test_size=0.3)
        self.train.columns = self.train.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        self.test.columns = self.test.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        self.test_err.columns = self.test_err.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        self.cols = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.cols]
        self.error_col = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.error_col]

        self.train_x = self.train[self.cols].reset_index(drop=True)
        self.test_x = self.test[self.cols].reset_index(drop=True)
        self.test_err_x = self.test_err[self.cols].reset_index(drop=True)
        self.errors_info = ['error_' + s for s in self.cols]
        self.train_y = self.train[self.errors_info]
        self.test_y = self.test[self.errors_info]
        self.test_err_y = self.test_err[self.errors_info]


    def train_model(self, visualization=False):
        self.model = VAR(self.train_x)
        self.model_fitted = self.model.fit(self.sequence_length)
        print(self.model_fitted.summary())


    def fault_detection(self, threshold_arr, visualization=False):
        trainPredict = []
        testPredict = []
        testErrPredict = []
        trainY = self.data.data_drive.iloc[self.train.index][self.sequence_length:][self.data.cols].values
        testY = self.data.data_drive.iloc[self.test.index][self.sequence_length:][self.data.cols].values
        testErrY = self.data.data_drive_err.iloc[self.test_err.index][self.sequence_length:][self.data.cols].values

        for i in range(len(self.train_x)-self.sequence_length):
            d = self.train_x.values[i:i+self.sequence_length]
            fc = self.model_fitted.forecast(d, 1)
            trainPredict.append(fc)
        for i in range(len(self.test_x)-self.sequence_length):
            d = self.test_x.values[i:i + self.sequence_length]
            fc = self.model_fitted.forecast(d, 1)
            testPredict.append(fc)
        for i in range(len(self.test_err_x)-self.sequence_length):
            d = self.test_err_x.values[i:i + self.sequence_length]
            fc = self.model_fitted.forecast(d, 1)
            testErrPredict.append(fc)

        self.train_y = self.train_y[self.sequence_length:]
        self.test_y = self.test_y[self.sequence_length:]
        self.test_err_y = self.test_err_y[self.sequence_length:]

        trainPredict = np.squeeze(np.stack(trainPredict, axis=0))
        trainPredict = self.data.scaler.inverse_transform(trainPredict)
        # trainY = self.data.scaler.inverse_transform(self.train_y)
        testPredict = np.squeeze(np.stack(testPredict, axis=0))
        testPredict = self.data.scaler.inverse_transform(np.stack(testPredict, axis=0))
        # testY = self.data.scaler.inverse_transform(self.test_y)
        testErrPredict = np.squeeze(np.stack(testErrPredict, axis=0))
        testErrPredict = self.data.scaler.inverse_transform(np.stack(testErrPredict, axis=0))
        # testErrY = self.data.scaler.inverse_transform(self.test_err_y)
        # minimal threshold to consider element faulty, as a percentage of maximum prediction difference for training
        # data

        for threshold in threshold_arr:
            for err, error in zip(self.error_col, self.data.error_col):
                time.sleep(1)
                column_number = self.data.cols.index(error)

                train_diff = trainPredict[:, column_number] - trainY[:, column_number]
                test_diff = testPredict[:, column_number] - testY[:, column_number]
                train_test_diff = np.concatenate((train_diff, test_diff), axis=0)
                max_diff = np.percentile(train_test_diff, threshold * 100)

                # ones in test_err_diffs mean detected errors
                test_err_diffs = testErrPredict[:, column_number] - testErrY[:, column_number]
                test_err_diffs = np.where(test_err_diffs < max_diff, 0, 1)
                test_err_errors = np.array(self.test_err_y['error_' + err]).astype(int)
                accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(test_err_errors,
                                                                                            test_err_diffs,
                                                                                            visualization=False)
                print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                      specificity, " F1_score: ",
                      F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
                save_data("VAR_model", '-', 'Threshold=' + str(threshold), self.data.filename, self.data.error_col,
                          err,
                          self.data.errors_types, accuracy, sensitivity, specificity,
                          F1_score, AUC, detection_rate, sheet_name="VAR_test")

                if visualization:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(len(testErrY[:, column_number])), testErrY[:, column_number], c='red', zorder=1,
                            label='Wartość błędna')
                    ax.plot(range(len(testY[:, column_number])), testY[:, column_number], c='blue', zorder=2,
                            label='Wartość poprwana')
                    test_err_diffs = test_err_diffs.astype(float)
                    test_err_diffs[test_err_diffs == 0] = np.NaN
                    ax.scatter(range(len(test_err_diffs)), test_err_diffs * testErrY[:, column_number], c='magenta',
                               zorder=3, label='Wykryty błąd')
                    plt.title(err)
                    plt.xlabel('próbka')
                    plt.legend()
                    # plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                    plt.show()
