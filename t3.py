# 텐서보드 시험용

import datetime

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
# Some function to convert into a graph
@tf.function
def my_fun(x):
    return 2 * x
# Test
a = tf.constant(10, tf.float32)
b = my_fun(a)
tf.print(b)


with tf.name_scope('scope_alpha'):
    a = tf.constant(5, dtype=tf.int32, name='const_a')
    b = tf.constant(10, dtype=tf.int32, name='const_b')
    c = tf.add(a, b, name='add_c')

with tf.name_scope('scope_beta'):
    d = tf.constant(500, dtype=tf.int32, name='const_a')
    e = tf.constant(1000, dtype=tf.int32, name='const_b')
    f = tf.add(d, e, name='add_c')

output = tf.add(c, f)

# 20
# Log the function graph
log_dir = "logs\m1" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    # Get concrete graph for some given inputs
    func_graph = my_fun.get_concrete_function(a).graph
    # Write the graph
    summary_ops_v2.graph(func_graph.as_graph_def(), step=0)
writer.close()