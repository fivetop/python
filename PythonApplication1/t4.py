# 텐서보드 시험용

import datetime

import tensorflow as tf
import numpy as np

logdir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

@tf.function
def forward_and_backward(x, y, w, b, lr=tf.constant(0.01)):

    with tf.name_scope('logits'):
        logits = tf.matmul(x, w) + b

    with tf.name_scope('loss'):
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        reduced = tf.reduce_sum(loss_fn)


    return reduced

# inputs
x = tf.convert_to_tensor(np.ones([1, 2]), dtype=tf.float32)
y = tf.convert_to_tensor(np.array([1]))
# params
w = tf.Variable(tf.random.normal([2, 2]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1, 2]), dtype=tf.float32)

loss_val = forward_and_backward(x, y, w, b)

with writer.as_default():
    tf.summary.trace_export(
        name='NN',
        step=0,
        profiler_outdir=logdir)