import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Conv1D
from keras.layers.core import Dense, Activation, Dropout, Lambda, RepeatVector
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from utils import *

def load_data(data, time_step=20, after_day=1, validate_percent=0.8):
    seq_length = time_step + after_day
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])

    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    return [x_train, y_train, x_validate, y_validate]

def base_model(feature_len=1, after_day=1, input_shape=(20, 1)):
    model = Sequential()

    model.add(Conv1D(10, kernel_size=5, input_shape=input_shape,  activation='relu', padding='valid', strides=1))
    model.add(LSTM(100, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.25))

    # one to many
    model.add(RepeatVector(after_day))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(100, activation='relu', kernel_initializer='uniform')))
    model.add(TimeDistributed(Dense(feature_len, activation='linear', kernel_initializer='uniform')))

    return model

if __name__ == '__main__':
    class_list = ['50', '51', '52', '53', '54', '55', '56', '57', '58',
                  '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']

    scaler = MinMaxScaler(feature_range=(0, 1))

    validate_percent = 0.9
    time_step = 20
    after_day = 5
    batch_size = 64
    epochs = 100
    output = []

    model_name = sys.argv[0].replace(".py", "")

    for index in range(len(class_list)):
        _class = class_list[index]
        print('******************************************* class 00{} *******************************************'.format(_class))

        # read data from csv, return data: (Samples, feature)
        data = file_processing(
            'data/20180427_process/20180427_{}.csv'.format(_class))
        feature_len = data.shape[1]

        # normalize data
        data = normalize_data(data, scaler, feature_len)

        # test data
        x_test = data[-time_step:]
        x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

        # get train and validate data
        x_train, y_train, x_validate, y_validate = load_data(
            data, time_step=time_step, after_day=after_day, validate_percent=validate_percent)

        print('train data: ', x_train.shape, y_train.shape)
        print('validate data: ', x_validate.shape, y_validate.shape)


        # model complie
        input_shape = (time_step, feature_len)
        model = base_model(feature_len, after_day, input_shape)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        plot_model_architecture(model, model_name=model_name)

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_validate, y_validate))
        model_class_name = model_name + '_00{}'.format(_class)
        #save_model(model, model_name=model_class_name)

        print('-' * 100)
        train_score = model.evaluate(x_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (train_score[0], math.sqrt(train_score[0])))

        validate_score = model.evaluate(x_validate, y_validate, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (validate_score[0], math.sqrt(validate_score[0])))

        train_predict = model.predict(x_train)
        validate_predict = model.predict(x_validate)
        test_predict = model.predict(x_test)

        # ?????????????????????????????????????????????
        train_predict = inverse_normalize_data(train_predict, scaler)
        y_train = inverse_normalize_data(y_train, scaler)
        validate_predict = inverse_normalize_data(validate_predict, scaler)
        y_validate = inverse_normalize_data(y_validate, scaler)
        test_predict = inverse_normalize_data(test_predict, scaler)

        # 3 or 0: close ?????????, 0:5?????????
        ans = np.append(y_validate[-1, -1, 3], test_predict[-1, 0:5, 3])
        output.append(ans)
        #print("output: \n", output)

        # plot predict situation (save in images/result)
        file_name = 'result_' + model_name + '_00{}'.format(_class)
        plot_predict(y_validate, validate_predict, file_name=file_name)

        # plot loss (save in images/loss)
        file_name = 'loss_' + model_name + '_00{}'.format(_class)
        plot_loss(history, file_name)

    output = np.array(output)
    print(output)
    generate_output(output, model_name=model_name, class_list=class_list)
