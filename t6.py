# 간단한 계산
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.constant([5],dtype=tf.float32)
b = tf.constant([10],dtype=tf.float32)
c = tf.constant([2],dtype=tf.float32)

d = a*b+c

sess = tf.compat.v1.Session()
result = sess.run(d)
print(result)
