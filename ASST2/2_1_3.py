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
learningRates = [.005, .001, .0001]
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

w = tf.Variable(tf.zeros([dimension,1], dtype=tf.float64), name="weights", dtype=tf.float64)

b = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64, name="bias")

# Xw + b, no sigmoid since that is handled in the cross entropy function
yhat = tf.add(tf.matmul(X, w), b)

Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))

Accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, tf.round(tf.sigmoid(yhat))), tf.float64))

class AccuracyGraph:
    best_training_accuracy = []
    best_validation_accuracy = []
    best_test_accuracy = []
    best_epoch_validation_accuracy = 0.0
    best_learning_rate = 0.0

linear_graph = AccuracyGraph()
logistic_graph = AccuracyGraph()

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

for learning_rate in learningRates:

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss)

    epoch_training_accuracy = []
    epoch_validation_accuracy = []
    epoch_test_accuracy = []

    '''
    LOGISTIC REGRESSION
    '''

    # Wipe the previous optimized weights and bias
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    start_point = 0

    # Pre-training accuracy datapoint
    epoch_training_accuracy.append(sess.run(Accuracy, feed_dict=training_set))
    epoch_validation_accuracy.append(sess.run(Accuracy, feed_dict=validation_set))
    epoch_test_accuracy.append(sess.run(Accuracy, feed_dict=test_set))

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
            epoch_training_accuracy.append(sess.run(Accuracy, feed_dict=training_set))
            epoch_validation_accuracy.append(sess.run(Accuracy, feed_dict=validation_set))
            epoch_test_accuracy.append(sess.run(Accuracy, feed_dict=test_set))

    if epoch_validation_accuracy[-1] > logistic_graph.best_epoch_validation_accuracy:
        logistic_graph.best_epoch_validation_accuracy = epoch_validation_accuracy[-1]
        logistic_graph.best_learning_rate = learning_rate
        logistic_graph.best_training_accuracy = epoch_training_accuracy
        logistic_graph.best_validation_accuracy = epoch_validation_accuracy
        logistic_graph.best_test_accuracy = epoch_test_accuracy

'''
NORMAL LINEAR REGRESSION
'''

trans = tf.matrix_inverse(tf.matmul(tf.transpose(X), X))
xy = tf.matmul(tf.transpose(X), Y)
wls = tf.matmul(trans, xy)

# Using least squares weight, check the validation, training, and test accuracy
LinearRegressionAccuracy = tf.reduce_mean( \
        tf.cast(tf.equal(tf.round(tf.matmul(X, wls)),Y), tf.float64))

W_lin = tf.Variable(tf.zeros([dimension + 1, 1], dtype=tf.float64), name="linear_weights", dtype=tf.float64)

LinearRegressionAccuracyFixedWeights = tf.reduce_mean(\
        tf.cast(tf.equal(tf.round(tf.matmul(X, W_lin)), Y), tf.float64))

graph_length = len(logistic_graph.best_training_accuracy)

# Modify the data sets to add b value for normal linear regression
training_set[X] = np.concatenate([np.full([training_set[X].shape[0], 1], 1), training_set[X]], 1)
validation_set[X] = np.concatenate([np.full([validation_set[X].shape[0], 1], 1), validation_set[X]], 1)
test_set[X] = np.concatenate([np.full([test_set[X].shape[0], 1], 1), test_set[X]], 1)

computed_W = sess.run(wls, feed_dict=training_set)

# Python in gross
training_set = {
    X: training_set[X],
    Y: training_set[Y],
    W_lin: computed_W
}
validation_set = {
    X: validation_set[X],
    Y: validation_set[Y],
    W_lin: computed_W
}
test_set = {
    X: test_set[X],
    Y: test_set[Y],
    W_lin: computed_W
}

linear_graph.best_training_accuracy = np.full((graph_length), sess.run(LinearRegressionAccuracyFixedWeights, feed_dict=training_set))
linear_graph.best_validation_accuracy = np.full((graph_length), sess.run(LinearRegressionAccuracyFixedWeights, feed_dict=validation_set))
linear_graph.best_test_accuracy = np.full((graph_length), sess.run(LinearRegressionAccuracyFixedWeights, feed_dict=test_set))

plt.figure(1)

plt.plot(logistic_graph.best_training_accuracy, label=("Logistic"))
plt.plot(linear_graph.best_training_accuracy, label=("Linear"))
plt.legend()
plt.title("Training Accuracy, Learning Rate = " + str(logistic_graph.best_learning_rate))
plt.show()

plt.plot(logistic_graph.best_validation_accuracy, label=("Logistic"))
plt.plot(linear_graph.best_validation_accuracy, label=("Linear"))
plt.legend()
plt.title("Validation Accuracy, Learning Rate = " + str(logistic_graph.best_learning_rate))
plt.show()

plt.plot(logistic_graph.best_test_accuracy, label=("Logistic"))
plt.plot(linear_graph.best_test_accuracy, label=("Linear"))
plt.legend()
plt.title("Test Accuracy, Learning Rate = " + str(logistic_graph.best_learning_rate))
plt.show()

