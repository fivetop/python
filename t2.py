# 텐서보드 시험용

import tensorflow as tf
import datetime

log_dir = "logs/m11" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tf.summary.trace_on(graph=True, profiler=True)
writer = tf.summary.create_file_writer(log_dir)

with tf.name_scope('scope_alpha'):
    a = tf.constant(5, dtype=tf.int32, name='const_a')
    b = tf.constant(10, dtype=tf.int32, name='const_b')
    c = tf.add(a, b, name='add_c')

with tf.name_scope('scope_beta'):
    d = tf.constant(500, dtype=tf.int32, name='const_a')
    e = tf.constant(1000, dtype=tf.int32, name='const_b')
    f = tf.add(d, e, name='add_c')

output = tf.add(c, f)

tf.print(output)
with writer.as_default():
    tf.summary.trace_export(name="ss", step=0, profiler_outdir=log_dir)
