import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

# Basic Logistic Regresion Model

X = tf.placeholder(tf.float64); # [Nxd]
Y = tf.placeholder(tf.float64); # [N]

one = tf.constant([1], dtype=tf.float64)

w = tf.expand_dims(tf.get_variable("weights", shape=[2], dtype=tf.float64, \
        initializer=tf.constant_initializer([-1,-1])), 1) # [d]

b = tf.get_variable(shape=[1], dtype=tf.float64, name="bias", \
        initializer=tf.constant_initializer(5)) # [1]

lambda_weight_penalty = tf.constant(0.01, dtype=tf.float64)

# Xw + b, no sigmoid since that is handled in the cross entropy function
yhat = tf.add(tf.matmul(X, w), b)

Loss_Data = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))

Loss_Weights = (lambda_weight_penalty / tf.constant(2.0, dtype=tf.float64)) * tf.reduce_sum(w**2)

Loss = Loss_Data + Loss_Weights

learning_rate = 0.0001

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(Loss)

init = tf.global_variables_initializer()
sess.run(init)

epochLoss = []

for iteration in range(20000):
    sess.run(optimizer, feed_dict={X:np.array([[1,2],[3,4]]), Y:np.array([1,0])})
    epochLoss.append(sess.run(Loss, feed_dict={X:np.array([[1,2],[3,4]]), Y:np.array([1,0])}))

plt.plot(np.arange(20000), epochLoss)
plt.show()
