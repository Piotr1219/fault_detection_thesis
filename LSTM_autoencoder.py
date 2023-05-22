from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import time

from numpy.random import seed
import tensorflow as tf

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class LSTM_autoencoder:
    """Class for detection method based on single element reconstruction using LSTM based autoencoder.
        Data element is considered faulty if reconstruction error of element is above threshold.

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

            Methods
            -------
            train_model(visualization):
                Method creates and trains model. Optionally it can show plot od losses and train data errors distribution
            fault_detection(threshold_arr, visualization=False)
                This method first computes reconstruction error for training and test data without faults. Then, based on
                threshold_err value, the threshold value is set. In next step, data including faults is being
                reconstructed. Elements are considered as errors when difference between reconstruction and actual data
                is above threshold value. Optionally data can be plotted with marked detected and true errors.

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

    def train_model(self, visualization=False):
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True,
                  kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(self.X_train.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(self.X_train.shape[2]))(L5)
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
        nb_epochs = 50
        batch_size = 10
        history = self.model.fit(self.X_train, self.X_train, epochs=nb_epochs, batch_size=batch_size,
                            validation_split=0.05).history
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

            # plot the loss distribution of the training set
            X_pred = self.model.predict(self.X_train)
            X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
            X_pred = pd.DataFrame(X_pred, columns=self.train.columns)
            X_pred.index = self.train.index

            scored = pd.DataFrame(index=self.train.index)
            Xtrain = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[2])
            scored['MAPE'] = np.mean(np.abs(X_pred - Xtrain)/np.mean(Xtrain, axis=0), axis=1)
            plt.figure(figsize=(16, 9), dpi=80)
            plt.title('Mean Absolute Percentage Error Distribution', fontsize=16)
            sns.distplot(scored['MAPE'], bins=20, kde=True, color='blue')
            plt.show()


    def fault_detection(self, threshold_arr, visualization=False):
        # make predictions
        trainPredict = self.model.predict(self.X_train)
        testPredict = self.model.predict(self.X_test)
        testErrPredict = self.model.predict(self.X_test_err)

        # invert predictions back to prescaled values
        # This is to compare with original input values
        # Since we used minmaxscaler we can now use scaler.inverse_transform
        # to invert the transformation.
        trainPredict = trainPredict.reshape(trainPredict.shape[0], trainPredict.shape[2])
        testPredict = testPredict.reshape(testPredict.shape[0], testPredict.shape[2])
        testErrPredict = testErrPredict.reshape(testErrPredict.shape[0], testErrPredict.shape[2])
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[2])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2])
        self.X_test_err = self.X_test_err.reshape(self.X_test_err.shape[0], self.X_test_err.shape[2])

        trainPredict = self.data.scaler.inverse_transform(trainPredict)
        trainX = self.data.scaler.inverse_transform(self.X_train)
        testPredict = self.data.scaler.inverse_transform(testPredict)
        testX = self.data.scaler.inverse_transform(self.X_test)
        testErrPredict = self.data.scaler.inverse_transform(testErrPredict)
        testErrX = self.data.scaler.inverse_transform(self.X_test_err)
        # minimal threshold to consider element faulty, as a percentage of maximum prediction difference for training
        # data

        for threshold in threshold_arr:
            for err in self.data.error_col:
                time.sleep(1)
                column_number = self.data.cols.index(err)
                train_diff = trainPredict[:, column_number] - trainX[:, column_number]
                test_diff = testPredict[:, column_number] - testX[:, column_number]
                train_test_diff = np.concatenate((train_diff, test_diff), axis=0)
                max_diff = np.percentile(train_test_diff, threshold * 100)
                # max_diff = threshold * 0.5 * (train_max_diff + test_max_diff)

                # ones in test_err_diffs mean detected errors
                test_err_diffs = testErrPredict[:, column_number] - testErrX[:, column_number]
                test_err_diffs = np.where(test_err_diffs < max_diff, 0, 1)
                test_err_errors = self.test_err['error_' + err].values.astype(int)
                accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(test_err_errors,
                                                                                            test_err_diffs,
                                                                                            visualization=False)
                print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                      specificity, " F1_score: ", F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
                save_data("LSTM_model", '-', 'Threshold=' + str(threshold), self.data.filename, self.data.error_col,
                          err, self.data.errors_types, accuracy, sensitivity, specificity,
                          F1_score, AUC, detection_rate, sheet_name="LSTM_autoencoder_test")

                if visualization:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(len(testErrX[:, column_number])), testErrX[:, column_number], c='red', zorder=1, label='Wartość błędna')
                    ax.plot(range(len(testX[:, column_number])), testX[:, column_number], c='blue', zorder=2, label='Wartość poprwana')
                    test_err_diffs = test_err_diffs.astype(float)
                    test_err_diffs[test_err_diffs == 0] = np.NaN
                    ax.scatter(range(len(test_err_diffs)), test_err_diffs * testErrX[:, column_number], c='magenta', zorder=3, label='Wykryty błąd')
                    plt.title(err)
                    plt.xlabel('próbka')
                    plt.legend()
                    plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                    plt.show()



