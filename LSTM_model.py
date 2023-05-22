import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
import time

from utils import chunks_train_test_split, to_sequences, metrics, save_data
from datetime import datetime


class LSTM_model:
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
        self.train_chunks, self.test_chunks = chunks_train_test_split(self.data.data_drive_scaled_chunks,
                                                                      split_train=0.7)
        self.train_x, self.train_y, self.train_x_err, self.train_y_err = to_sequences(self.train_chunks, self.data.cols,
                                                                                      self.sequence_length, errors=True)
        self.test_x, self.test_y = to_sequences(self.test_chunks, self.data.cols, self.sequence_length)
        _, self.test_err_chunks = chunks_train_test_split(self.data.data_drive_scaled_err_chunks, split_train=0.7)
        self.test_err_x, self.test_err_y, self.test_x_errors_col, self.test_y_errors_col = \
            to_sequences(self.test_err_chunks, self.data.cols, self.sequence_length, errors=True)

    def predicted_data_comparison(self, data, data_predict, c, chunks_test_seq_len, name):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(data[:, c])), data[:, c], label='Pomierzone wartości')
        ax.scatter(range(len(data_predict[:, c])), data_predict[:, c], s=0.5, c='red', label='Przewidywane wartości')
        chunk_ends = np.zeros(len(data[:, c]))
        np.put(chunk_ends, np.cumsum(chunks_test_seq_len)-1, np.ones(len(chunks_test_seq_len)))
        chunk_ends[chunk_ends == 0] = np.nan
        # ax.scatter(range(len(chunk_ends)), chunk_ends * data[:, c], c='magenta')
        for xc in np.where(chunk_ends == 1)[0]:
            plt.axvline(x=xc, c='lightgrey')
        plt.ylim(min(data[:, c]) * 0.9, max(data[:, c]) * 1.1)
        # plt.title(name + ' ' + self.data.cols[c])
        plt.title(self.data.cols[c])
        plt.legend()
        plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
        plt.show()

    def train_model(self, visualization=False):
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', input_shape=(self.train_x.shape[1], self.train_x.shape[2]),
                            return_sequences=True))
        self.model.add(LSTM(64, activation='relu', return_sequences=True))
        self.model.add(LSTM(32, activation='relu', return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.train_y.shape[1]))

        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        # fit the model
        history = self.model.fit(self.train_x, self.train_y, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

        if visualization:
            # plot the training losses
            fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
            ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
            ax.plot(history.history['val_loss'], 'r', label='Validation', linewidth=2)
            ax.set_title('Model loss', fontsize=16)
            ax.set_ylabel('Loss (mae)')
            ax.set_xlabel('Epoch')
            ax.legend(loc='upper right')
            plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
            plt.show()

            # plot the loss distribution of the training set
            Y_pred = self.model.predict(self.train_x)

            Y_pred = pd.DataFrame(Y_pred, columns=self.data.cols)
            Y_train = pd.DataFrame(self.train_y, columns=self.data.cols)

            scored = pd.DataFrame(index=pd.DataFrame(self.train_y_err).index)
            scored['MAPE'] = np.mean(np.abs(Y_pred - Y_train)/np.mean(Y_train, axis=0), axis=1)
            plt.figure(figsize=(16, 9), dpi=80)
            plt.title('Mean Absolute Percentage Error Distribution', fontsize=16)
            sns.distplot(scored['MAPE'], bins=20, kde=True, color='blue')
            plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
            plt.show()

    def time_prediction(self, visualization=False):
        # make predictions
        trainPredict = self.model.predict(self.train_x)
        testPredict = self.model.predict(self.test_x)
        testErrPredict = self.model.predict(self.test_err_x)

        # invert predictions back to prescaled values
        # This is to compare with original input values
        # SInce we used minmaxscaler we can now use scaler.inverse_transform
        # to invert the transformation.
        trainPredict = self.data.scaler.inverse_transform(trainPredict)
        trainY = self.data.scaler.inverse_transform(self.train_y)
        testPredict = self.data.scaler.inverse_transform(testPredict)
        testY = self.data.scaler.inverse_transform(self.test_y)
        testErrPredict = self.data.scaler.inverse_transform(testErrPredict)
        testErrY = self.data.scaler.inverse_transform(self.test_err_y)

        chunks_train_seq_len = []
        for c in self.train_chunks:
            chunks_train_seq_len.append(len(c) - self.sequence_length)
        chunks_test_seq_len = []
        for c in self.test_chunks:
            chunks_test_seq_len.append(len(c) - self.sequence_length)
        chunks_test_err_seq_len = []
        for c in self.test_err_chunks:
            chunks_test_err_seq_len.append(len(c) - self.sequence_length)

        for c in range(len(self.data.cols)):
            time.sleep(1)
            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY[:, c], trainPredict[:, c]))
            trainScoreRel = trainScore / np.mean(trainY[:, c])
            print(self.data.cols[c] + 'Train Score: %.2f RMSE' % (trainScore), ', Relative (error to mean value): ' + str(trainScoreRel))
            if visualization:
                self.predicted_data_comparison(trainY, trainPredict, c, chunks_train_seq_len, 'Train')

            testScore = math.sqrt(mean_squared_error(testY[:, c], testPredict[:, c]))
            testScoreRel = testScore / np.mean(testY[:, c])
            print(self.data.cols[c] + 'Test Score: %.2f RMSE' % (testScore), ', Relative (error to mean value): ' + str(testScoreRel))
            if visualization:
                self.predicted_data_comparison(testY, testPredict, c, chunks_test_seq_len, 'Test')

            testErrScore = math.sqrt(mean_squared_error(testErrY[:, c], testErrPredict[:, c]))
            testErrScoreRel = testErrScore / np.mean(testErrY[:, c])
            print(self.data.cols[c] + 'Test Score: %.2f RMSE' % (testErrScore), ', Relative (error to mean value): ' + str(testErrScoreRel))
            if visualization:
                self.predicted_data_comparison(testErrY, testErrPredict, c, chunks_test_err_seq_len, 'TestErr')

    def fault_detection(self, threshold_arr, visualization=False):
        # make predictions
        trainPredict = self.model.predict(self.train_x)
        testPredict = self.model.predict(self.test_x)
        testErrPredict = self.model.predict(self.test_err_x)

        # invert predictions back to prescaled values
        # This is to compare with original input values
        # Since we used minmaxscaler we can now use scaler.inverse_transform
        # to invert the transformation.
        trainPredict = self.data.scaler.inverse_transform(trainPredict)
        trainY = self.data.scaler.inverse_transform(self.train_y)
        testPredict = self.data.scaler.inverse_transform(testPredict)
        testY = self.data.scaler.inverse_transform(self.test_y)
        testErrPredict = self.data.scaler.inverse_transform(testErrPredict)
        testErrY = self.data.scaler.inverse_transform(self.test_err_y)
        # minimal threshold to consider element faulty, as a percentage of maximum prediction difference for training
        # data

        for threshold in threshold_arr:
            for err in self.data.error_col:
                time.sleep(1)
                column_number = self.data.cols.index(err)

                train_diff = trainPredict[:, column_number] - trainY[:, column_number]
                test_diff = testPredict[:, column_number] - testY[:, column_number]
                train_test_diff = np.concatenate((train_diff, test_diff), axis=0)
                max_diff = np.percentile(train_test_diff, threshold * 100)

                # ones in test_err_diffs mean detected errors
                test_err_diffs = testErrPredict[:, column_number] - testErrY[:, column_number]
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
                          F1_score, AUC, detection_rate, sheet_name="LSTM_test")

                if visualization:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(len(testErrY[:, column_number])), testErrY[:, column_number], c='red', zorder=1, label='Wartość błędna')
                    ax.plot(range(len(testY[:, column_number])), testY[:, column_number], c='blue', zorder=2, label='Wartość poprwana')
                    test_err_diffs = test_err_diffs.astype(float)
                    test_err_diffs[test_err_diffs == 0] = np.NaN
                    ax.scatter(range(len(test_err_diffs)), test_err_diffs * testErrY[:, column_number], c='magenta', zorder=3, label='Wykryty błąd')
                    plt.title(err)
                    plt.xlabel('próbka')
                    plt.legend()
                    plt.savefig(datetime.now().strftime("%H:%M:%S") + '.svg')
                    plt.show()
