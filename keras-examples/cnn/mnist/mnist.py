import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from keras.utils.visualize_util import plot

from sklearn.preprocessing import MinMaxScaler

# MNISTの数字分類
# 参考: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py


def build_cnn(input_shape, nb_filters, filter_size, pool_size):
    model = Sequential()

    model.add(Convolution2D(nb_filters,
                            filter_size[0], filter_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1]))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def plot_history(history, outdir):
    # 精度の履歴をプロット
    plt.figure()
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, 'acc.png'))

    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, 'loss.png'))


def visualize_filter(model):
    # 最初の畳み込み層の重みを取得
    # tf => (nb_row, nb_col, nb_channel, nb_filter)
    # th => (nb_filter, nb_channel, nb_row, nb_col)
    W = model.layers[0].get_weights()[0]

    # 次元を並べ替え
    if K.image_dim_ordering() == 'tf':
        # (nb_filter, nb_channel, nb_row, nb_col)
        W = W.transpose(3, 2, 0, 1)

    nb_filter, nb_channel, nb_row, nb_col = W.shape

    # 32個（手抜きで固定）のフィルタの重みを描画
    plt.figure()
    for i in range(nb_filters):
        # フィルタの画像
        im = W[i, 0]

        # 重みを0-255のスケールに変換
        scaler = MinMaxScaler(feature_range=(0, 255))
        im = scaler.fit_transform(im)

        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap="gray")
    plt.show()


if __name__ == "__main__":
    batch_size = 128
    nb_classes = 10
    nb_epoch = 100

    img_rows, img_cols = 28, 28
    nb_filters = 32
    filter_size = (5, 5)
    pool_size = (2, 2)

    # MNISTデータのロード
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 画像集合を表す4次元テンソルに変形
    # keras.jsonのimage_dim_orderingがthのときはチャネルが2次元目、tfのときはチャネルが4次元目にくる
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 画素を0.0-1.0の範囲に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # one-hot-encoding
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # 畳み込みニューラルネットワークを構築
    model = build_cnn(input_shape, nb_filters, filter_size, pool_size)

    # モデルのサマリを表示
    model.summary()
    plot(model, show_shapes=True, to_file='result_mnist/model.png')

    # モデルをコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 学習前の1層目のフィルタを可視化
    visualize_filter(model)

    # Early-stopping
    early_stopping = EarlyStopping()

    # モデルの訓練
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    # 学習後の1層目のフィルタを可視化
    visualize_filter(model)

    # 学習履歴をプロット
    plot_history(history, 'result_mnist')

    # モデルの評価
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test loss:', loss)
    print('Test acc:', acc)
