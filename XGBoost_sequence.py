from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import seed
import tensorflow as tf
import time

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from xgboost import XGBClassifier

from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class XGBoost_sequence:
    """Class for detection method based on XGBoost. The model is trained using data with marked errors (0-1 values).

            Attributes
            ----------
            data : DataPreparation
                Class object containing differently prepared data.
            train : DataFrame
                DataFrame with training data containing errors.
            validation : DataFrame
                DataFrame with validation data containing errors.
            test : DataFrame
                DataFrame with test data containing errors.
            train_x : DataFrame
                DataFrame with train data containing only columns with measurements.
            validation_x : DataFrame
                DataFrame with validation data containing only columns with measurements.
            test_x : DataFrame
                DataFrame with test data containing only columns with measurements.
            test_no_err : DataFrame
                Test data without errors.
            errors_info : list
                List of names of columns containing information about error occurrence.
            train_y : DataFrame
                Data containing information about error occurrence in train dataset.
            validation_y : DataFrame
                Data containing information about error occurrence in validation dataset.
            test_y : DataFrame
                Data containing information about error occurrence in test dataset.

            Methods
            -------
            train_model(visualization):
                Method creates and trains model. Optionally it can show plot od losses during training.
            fault_detection(threshold_arr, visualization=False)
                This method makes predictions for test data, using previously trained model. Then it evaluates results
                using different metrics. Optionally data can be plotted with marked detected and true errors.

    """

    def __init__(self, data):
        self.data = data

        for i, d in enumerate(self.data.data_drive_scaled_err_chunks):
            self.data.data_drive_scaled_err_chunks[i].columns = d.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        for i, d in enumerate(self.data.data_drive_scaled_chunks):
            self.data.data_drive_scaled_chunks[i].columns = d.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        self.sequence_length = 6
        self.train_chunks, self.test_chunks = chunks_train_test_split(self.data.data_drive_scaled_err_chunks,
                                                                      split_train=0.7)
        self.train_chunks, self.validation_chunks = chunks_train_test_split(self.train_chunks,
                                                                      split_train=0.9)
        _, self.test_chunks_no_err = chunks_train_test_split(self.data.data_drive_scaled_chunks,
                                                                      split_train=0.7)
        self.cols = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.cols]
        self.errors_info = ['error_' + s for s in self.cols]
        self.error_col = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.error_col]
        self.train_x, _, _, self.train_y_err = to_sequences(self.train_chunks, self.cols, self.sequence_length,
                                                            errors=True, current_in=True)
        self.test_x, self.test_y, _, self.test_y_err = to_sequences(self.test_chunks, self.cols, self.sequence_length,
                                                          errors=True, current_in=True)
        self.validation_x, _, _, self.validation_y_err = to_sequences(self.validation_chunks, self.cols,
                                                                      self.sequence_length, errors=True,
                                                                      current_in=True)
        self.test_x_no_err, self.test_y_no_err, _, self.test_y_no_err_err = to_sequences(self.test_chunks_no_err, self.cols, self.sequence_length,
                                                          errors=True, current_in=True)
        self.train_x = self.train_x.reshape((self.train_x.shape[0], self.train_x.shape[1]*self.train_x.shape[2]))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], self.test_x.shape[1]*self.test_x.shape[2]))
        self.validation_x = self.validation_x.reshape((self.validation_x.shape[0],
                                                       self.validation_x.shape[1]*self.validation_x.shape[2]))
        errors_info_col_numbers = [self.train_y_err[0].index.get_loc(c) for c in self.errors_info]
        self.test_y_plot = self.test_y.copy()
        self.train_y = np.array(self.train_y_err)[:, errors_info_col_numbers]
        self.test_y = np.array(self.test_y_err)[:, errors_info_col_numbers]
        self.validation_y = np.array(self.validation_y_err)[:, errors_info_col_numbers]


        # self.train, self.test = train_test_split(self.data.data_drive_scaled_err, test_size=0.3)
        # self.train.columns = self.train.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        # self.test.columns = self.test.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        # self.cols = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.cols]
        # self.train, self.validation = train_test_split(self.train, test_size=0.1)
        # self.test_no_err = self.data.data_drive_scaled.iloc[self.test.index]
        # self.train_x = self.train[self.cols]
        # self.test_x = self.test[self.cols]
        # self.validation_x = self.validation[self.cols]
        # self.errors_info = ['error_' + s for s in self.cols]
        # self.train_y = self.train[self.errors_info]
        # self.test_y = self.test[self.errors_info]
        # self.validation_y = self.validation[self.errors_info]

        # self.X_train = self.train.values.reshape(self.train.shape[0], 1, self.train.shape[1])
        # self.X_test = self.test.values.reshape(self.test.shape[0], 1, self.test.shape[1])
        self.models = None

    def train_model(self, visualization=False):
        self.models = {}
        for err in self.error_col:
            column_number = self.cols.index(err)
            self.models[err] = XGBClassifier(base_score=0.5, learning_rate=0.5, early_stopping_rounds=10)
            # self.model.summary()
            nb_epochs = 50
            batch_size = 10
            # history = self.model.fit(self.train_x, self.train_y,
            #           eval_set=[(X_val.drop(['target'], axis=1), X_val['target'])], early_stopping_rounds=10)
            # train_x = np.array(self.train_x)
            # train_y = np.array(self.train_y)[:, i]
            train_y = self.train_y[:, column_number]
            validation_y = self.validation_y[:, column_number]
            history = self.models[err].fit(self.train_x, train_y,
                                           eval_set=[(self.validation_x, validation_y)])
            if visualization:
                # plot the training losses
                fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
                ax.plot(history['loss'], 'b', label='Train', linewidth=2)
                ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
                ax.set_title('Model loss', fontsize=16)
                ax.set_ylabel('Loss (mae)')
                ax.set_xlabel('Epoch')
                ax.legend(loc='upper right')
                plt.show()

    def fault_detection(self, visualization=False):
        testErrY = self.data.scaler.inverse_transform(self.test_y_plot)
        testY = self.data.scaler.inverse_transform(self.test_y_no_err)
        for err, error in zip(self.error_col, self.data.error_col):
            time.sleep(1)
            column_number = self.cols.index(err)
            # make predictions
            test_x = np.array(self.test_x)
            test_y = np.array(self.test_y)[:, column_number]

            testPredict = self.models[err].predict(test_x)

            # testX = self.data.scaler.inverse_transform(self.test_no_err)
            # testErrX = self.data.scaler.inverse_transform(self.test_x)
            # testX = self.data.data_drive.iloc[self.test.index]
            # testErrX = self.data.data_drive_err.iloc[self.test.index]

            accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(test_y.astype(int),
                                                                                        testPredict,
                                                                                        visualization=False)
            print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                  specificity, " F1_score: ", F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
            save_data("XGBoost_model", '-', '-', self.data.filename, self.data.error_col,
                      err, self.data.errors_types, accuracy, sensitivity, specificity,
                      F1_score, AUC, detection_rate, sheet_name="XGBoost_sequence_test")

            if visualization:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(len(testErrY[:, column_number])), testErrY[:, column_number], c='red', zorder=1, label='Wartość błędna')
                ax.plot(range(len(testY[:, column_number])), testY[:, column_number], c='blue', zorder=2, label='Wartość poprwana')
                testPredict = testPredict.astype(float)
                testPredict[testPredict == 0] = np.NaN
                ax.scatter(range(len(testPredict)), testPredict * testErrY[:, column_number], c='magenta', zorder=3, label='Wykryty błąd')
                plt.title(err)
                plt.xlabel('próbka')
                plt.legend()
                plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                plt.show()
