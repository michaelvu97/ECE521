import tensorflow as tf
import numpy as np

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Basic Logistic Regresion Model

X = tf.placeholder(tf.float64); # [Nxd]
Y = tf.placeholder(tf.float64); # [N]

one = tf.constant([1], dtype=tf.float64)

# w = tf.expand_dims(tf.Variable(tf.zeros([2], dtype=tf.float64)),1) # [d]
# b = tf.Variable(tf.zeros([1], dtype=tf.float64)) # [1]
w = tf.expand_dims(tf.constant([1,2], dtype=tf.float64),1) # [d]
b = tf.constant([0], dtype=tf.float64) # [1]

lambda_weight_penalty = tf.constant(0.01, dtype=tf.float64)

# yhat = sigmoid(w^TX + b) in R^[N]
yHat = tf.sigmoid(tf.add(tf.matmul(X, w), b))

Loss_Data = \
        tf.reduce_sum( \
            tf.negative(tf.multiply(Y, tf.log(yHat))) - \
            tf.multiply(one - Y, tf.log(one - yHat)) \
        ) / tf.cast(tf.shape(Y)[0], tf.float64);

Loss_Weights = (lambda_weight_penalty / tf.constant(2.0, dtype=tf.float64)) * tf.reduce_sum(w**2)

Loss = Loss_Data + Loss_Weights

print(sess.run(Loss, feed_dict={X:np.array([[1,2],[3,4]]), Y:np.array([1,-1])}))