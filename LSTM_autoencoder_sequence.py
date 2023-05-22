from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import time

from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class LSTM_autoencoder_sequence:
    """Class for detection method based on time sequence reconstruction using LSTM based autoencoder.
        Data element is considered faulty if reconstruction error of sequence ending with this element is above threshold.

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
            train_model(visualization):
                Method creates and trains model. Optionally it can show plot od losses and train data errors distribution
            fault_detection(threshold_arr, visualization=False)
                This method first computes reconstruction error for training and test data without faults. Then, based on
                threshold_err value, the threshold value is set. In next step, faulty data sequences are being
                reconstructed. Elements are considered as errors when difference between reconstruction and actual data
                is above threshold value.

    """
    def __init__(self, data):
        self.data = data
        self.sequence_length = 6
        self.current_in = False
        self.train_chunks, self.test_chunks = chunks_train_test_split(self.data.data_drive_scaled_chunks,
                                                                      split_train=0.7)
        self.train_x, self.train_y, self.train_x_err, self.train_y_err = to_sequences(self.train_chunks, self.data.cols,
                                                                                      self.sequence_length, errors=True,
                                                                                      current_in=self.current_in)
        self.train_y = self.train_y.reshape(self.train_y.shape[0], 1, self.train_y.shape[1])
        self.test_x, self.test_y = to_sequences(self.test_chunks, self.data.cols, self.sequence_length,
                                                current_in=self.current_in)
        self.test_y = self.test_y.reshape(self.test_y.shape[0], 1, self.test_y.shape[1])
        _, self.test_err_chunks = chunks_train_test_split(self.data.data_drive_scaled_err_chunks, split_train=0.7)
        self.test_err_x, self.test_err_y, self.test_x_errors_col, self.test_y_errors_col = \
            to_sequences(self.test_err_chunks, self.data.cols, self.sequence_length, errors=True,
                         current_in=self.current_in)
        self.test_err_y = self.test_err_y.reshape(self.test_err_y.shape[0], 1, self.test_err_y.shape[1])


    def train_model(self, visualization=False):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(self.train_x.shape[1]))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.train_x.shape[2])))
        self.model.compile(optimizer='adam', loss='mae')
        self.model.summary()

        history = self.model.fit(self.train_x, self.train_y, epochs=50, batch_size=16, validation_split=0.1,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                            shuffle=False)

        if visualization:
            # plot the training losses
            fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
            ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
            ax.plot(history.history['val_loss'], 'r', label='Validation', linewidth=2)
            ax.set_title('Model loss', fontsize=16)
            ax.set_ylabel('Loss (mae)')
            ax.set_xlabel('Epoch')
            ax.legend(loc='upper right')
            plt.show()

            # plot the loss distribution of the training set
            train_pred = self.model.predict(self.train_x)

            r1 = np.abs(train_pred - self.train_x)
            r2 = r1/(np.mean(np.mean(self.train_x, axis=1), axis=0))
            train_MAPE = np.mean(np.mean(r2, axis=1), axis=1)
            plt.hist(train_MAPE)
            plt.show()

    def fault_detection(self, threshold_arr, visualization=False):
        # make predictions
        trainPredict = self.model.predict(self.train_x)
        testPredict = self.model.predict(self.test_x)
        testErrPredict = self.model.predict(self.test_err_x)

        # invert predictions back to prescaled values
        # This is to compare with original input values
        # Since we used minmaxscaler we can now use scaler.inverse_transform
        # to invert the transformation.

        # trainPredict = self.data.scaler.inverse_transform(trainPredict)
        # trainY = self.data.scaler.inverse_transform(self.train_y)
        # testPredict = self.data.scaler.inverse_transform(testPredict)
        # testY = self.data.scaler.inverse_transform(self.test_y)
        # testErrPredict = self.data.scaler.inverse_transform(testErrPredict)
        # testErrY = self.data.scaler.inverse_transform(self.test_err_y)

        # minimal threshold to consider element faulty, as a percentage of maximum prediction difference for training
        # data

        for threshold in threshold_arr:
            for err in self.data.error_col:
                time.sleep(1)
                column_number = self.data.cols.index(err)

                train_diff = trainPredict[:, :, column_number] - self.train_x[:, :, column_number]
                test_diff = testPredict[:, :, column_number] - self.test_x[:, :, column_number]
                train_diff = np.mean(train_diff, axis=1)
                test_diff = np.mean(test_diff, axis=1)

                train_test_diff = np.concatenate((train_diff, test_diff), axis=0)
                max_diff = np.percentile(train_test_diff, threshold * 100)

                # ones in test_err_diffs mean detected errors
                test_err_diffs = testErrPredict[:, :, column_number] - self.test_err_x[:, :, column_number]
                test_err_diffs = np.mean(test_err_diffs, axis=1)
                test_err_diffs = np.where(test_err_diffs < max_diff, 0, 1)
                test_err_errors = pd.DataFrame(self.test_y_errors_col)['error_' + err].values.astype(int)
                accuracy, sensitivity, specificity, F1_score, AUC, detection_rate = metrics(test_err_errors,
                                                                                            test_err_diffs,
                                                                                            visualization=False)
                print("Error", err, " Accuracy: ", accuracy, " Sensitivity: ", sensitivity, " Specificity: ",
                      specificity, " F1_score: ",
                      F1_score, " AUC: ", AUC, " Detection rate: ", detection_rate)
                save_data("LSTM_model", '-', 'Threshold=' + str(threshold), self.data.filename, self.data.error_col, err,
                          self.data.errors_types, accuracy, sensitivity, specificity,
                          F1_score, AUC, detection_rate, sheet_name="LSTM_autoencoder_sequence_test")

                if visualization:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    p = self.test_y[:, :, column_number][:, 0]
                    test_err_y_unscaled = self.data.scaler.inverse_transform(self.test_err_y[:, 0])
                    test_y_unscaled = self.data.scaler.inverse_transform(self.test_y[:, 0])
                    ax.plot(range(len(test_err_y_unscaled[:, column_number])), test_err_y_unscaled[:, column_number], c='red', zorder=1, label='Wartość błędna')
                    ax.plot(range(len(test_y_unscaled[:, column_number])), test_y_unscaled[:, column_number], c='blue', zorder=2, label='Wartość poprwana')
                    test_err_diffs = test_err_diffs.astype(float)
                    test_err_diffs[test_err_diffs == 0] = np.NaN
                    ax.scatter(range(len(test_err_diffs)), test_err_diffs * test_err_y_unscaled[:, column_number], c='magenta', zorder=3, label='Wykryty błąd')
                    plt.title(err)
                    plt.xlabel('próbka')
                    plt.legend()
                    plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                    plt.show()

