from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import seed
import tensorflow as tf

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from xgboost import XGBClassifier

from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class XGBoost_detection:
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
        self.train, self.test = train_test_split(self.data.data_drive_scaled_err, test_size=0.3)
        self.train.columns = self.train.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        self.test.columns = self.test.columns.str.translate("".maketrans({"[": "{", "]": "}", "<": "^"}))
        self.cols = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.cols]
        self.error_col = [c.translate("".maketrans({"[": "{", "]": "}", "<": "^"})) for c in self.data.error_col]
        self.train, self.validation = train_test_split(self.train, test_size=0.1)
        # self.test_no_err = self.data.data_drive_scaled.iloc[self.test.index]
        self.train_x = self.train[self.cols]
        self.test_x = self.test[self.cols]
        self.validation_x = self.validation[self.cols]
        self.errors_info = ['error_' + s for s in self.cols]
        self.train_y = self.train[self.errors_info]
        self.test_y = self.test[self.errors_info]
        self.validation_y = self.validation[self.errors_info]

        # self.X_train = self.train.values.reshape(self.train.shape[0], 1, self.train.shape[1])
        # self.X_test = self.test.values.reshape(self.test.shape[0], 1, self.test.shape[1])
        self.models = None

    def train_model(self, visualization=False):
        self.models = {}
        for err in self.error_col:
            self.models[err] = XGBClassifier(base_score=0.5, learning_rate=0.5)
            # self.model.summary()
            nb_epochs = 50
            batch_size = 10
            history = self.models[err].fit(self.train_x, self.train_y['error_' + err],
                                           eval_set=[(self.validation_x, self.validation_y['error_' + err])], early_stopping_rounds=10)
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
        for err, error in zip(self.error_col, self.data.error_col):
            # make predictions
            testPredict = self.models[err].predict(self.test_x)

            # testX = self.data.scaler.inverse_transform(self.test_no_err)
            # testErrX = self.data.scaler.inverse_transform(self.test_x)
            testX = self.data.data_drive.iloc[self.test.index]
            testErrX = self.data.data_drive_err.iloc[self.test.index]

            accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(np.array(self.test_y['error_' + err]).astype(int),
                                                                                        testPredict,
                                                                                        visualization=False)
            print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                  specificity, " F1_score: ", F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
            save_data("XGBoost_model", '-', '-', self.data.filename, self.data.error_col,
                      err, self.data.errors_types, accuracy, sensitivity, specificity,
                      F1_score, AUC, detection_rate, sheet_name="XGBoost_test")

            if visualization:
                fig, ax = plt.subplots(figsize=(10, 6))
                l = 100
                ax.plot(range(len(testErrX[error].iloc[0:l])), testErrX[error].iloc[0:l], c='red', zorder=1, label='Wartość błędna')
                ax.plot(range(len(testX[error].iloc[0:l])), testX[error].iloc[0:l], c='blue', zorder=2, label='Wartość poprwana')
                testPredict = testPredict.astype(float)
                testPredict[testPredict == 0] = np.NaN
                ax.scatter(range(len(testPredict))[0:l], (testPredict * testErrX[error]).iloc[0:l], c='magenta', zorder=3, label='Wykryty błąd')
                plt.title(err)
                plt.xlabel('próbka')
                plt.legend()
                plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                plt.show()
