import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

################################################################################

# The given code to get the data

with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

################################################################################

n_iterations = 5000
batch_size = 500
learning_rate = 0.001
epoch_size = 7

################################################################################

trainData = np.reshape(trainData, [3500, -1])
validData = np.reshape(validData, [100, -1])
testData = np.reshape(testData, [testData.shape[0], -1])

dimension = trainData.shape[1]

X = tf.placeholder(tf.float64); # [Nxd]
Y = tf.placeholder(tf.float64); # [N]

training_set = {
    X: trainData,
    Y: trainTarget
}

validation_set = {
    X: validData,
    Y: validTarget
}

test_set = {
    X: testData,
    Y: testTarget
}

################################################################################
'''
LOGISTIC REGRESSION
'''

w = tf.Variable(tf.zeros([dimension,1], dtype=tf.float64), name="weights", dtype=tf.float64)

b = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64, name="bias")

# Xw + b, no sigmoid since that is handled in the cross entropy function
yhat = tf.add(tf.matmul(X, w), b)

Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))

Accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, tf.round(tf.sigmoid(yhat))), tf.float64))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss)


# Wipe the previous optimized weights and bias
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

logistic_training_losses = []
logistic_training_accuracy = []

start_point = 0

logistic_training_losses.append(sess.run(Loss, feed_dict=training_set))
logistic_training_accuracy.append(sess.run(Accuracy, feed_dict=training_set))

for iteration in range(n_iterations):

    # Once step of optimization

    batch = {
        X: trainData[start_point : start_point + batch_size],
        Y: trainTarget[start_point : start_point + batch_size]
    }

    sess.run(optimizer, feed_dict=batch)

    start_point = (start_point + batch_size) % len(trainData)

    # Check the result of this epoch
    if ((iteration + 1) % epoch_size == 0):
        logistic_training_losses \
                .append(sess.run(Loss, feed_dict=training_set))
        logistic_training_accuracy \
                .append(sess.run(Accuracy, feed_dict=training_set))

'''
LINEAR REGRESSION
'''

# Redefine Loss and Accuracy function
MSE = tf.reduce_mean(((yhat - Y)**2)/2)
LinearOptimizer = tf.train.AdamOptimizer(learning_rate).minimize(MSE)
LinearAccuracy = tf.reduce_mean(tf.cast(tf.equal(Y, tf.round(yhat)), tf.float64))

linear_training_losses = []
linear_training_accuracy = []


init = tf.global_variables_initializer()
sess.run(init)

start_point = 0
linear_training_losses.append(sess.run(MSE, feed_dict=training_set))
linear_training_accuracy.append(sess.run(LinearAccuracy, feed_dict=training_set))


for iteration in range(n_iterations):

    batch = {
        X: trainData[start_point : start_point + batch_size],
        Y: trainTarget[start_point : start_point + batch_size]
    }

    sess.run(LinearOptimizer, feed_dict=batch)

    start_point = (start_point + batch_size) % len(trainData)

    # Check the result of this epoch
    if ((iteration + 1) % epoch_size == 0):
        linear_training_losses \
                .append(sess.run(MSE, feed_dict=training_set))
        linear_training_accuracy \
                .append(sess.run(LinearAccuracy, feed_dict=training_set))

plt.figure(1)

plt.plot(logistic_training_losses, label="Logistic Loss")
plt.plot(linear_training_losses, label="Linear Loss")

plt.legend()
plt.title("Training Loss: Linear vs Logistic, Learning Rate = " + str(learning_rate))
plt.show()

plt.plot(logistic_training_accuracy, label="Logistic")
plt.plot(linear_training_accuracy, label="Linear")
plt.legend()
plt.title("Training Accuracy: Linear vs Logistic, Learning Rate = " + str(learning_rate))
plt.show()
