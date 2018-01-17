import numpy as np
import tensorflow as tf


def PairwiseEuclidian(X,Y):
    return tf.reduce_sum((tf.expand_dims(X, 1) - tf.expand_dims(Z, 0))**2, 2)

################################################################################


init = tf.global_variables_initializer()


sess = tf.InteractiveSession()
sess.run(init)

X = tf.constant([[1,2],[3,4]])
Z = tf.constant([[1,2],[3,4],[5,6]])

print(sess.run(X))
print(sess.run(Z))
print(sess.run(PairwiseEuclidian(X,Z)))

