# 텐서보드 시험용

import datetime

import tensorflow as tf
import numpy as np
x_train = [np.ones((1, 2))]
y_train = [np.ones(1)]

model = tf.keras.models.Sequential([tf.keras.layers.Dense(2, input_shape=(2, ))])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(x_train,
          y_train,
          batch_size=1,
          epochs=1,
          callbacks=[tensorboard_callback])