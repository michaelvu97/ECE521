import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

################################################################################
"""
Minified test/train/valid datasets for quicker testing of neural network models
DISABLE THIS ONCE READY TO TRAIN THE NN ON THE FULL DATASET
"""
testing = False

if testing:
    # Clip all of the datasets
    trainData   = trainData[:150]
    trainTarget = trainTarget[:150]
    validData   = validData[:10]
    validTarget = validTarget[:10]
    testData    = testData[:20]
    testTarget  = testTarget[:20]

################################################################################

trainData = np.reshape(trainData, [trainData.shape[0], -1])
validData = np.reshape(validData, [validData.shape[0], -1])
testData = np.reshape(testData, [testData.shape[0], -1])

# def WeightedSumLayer(inputTensor, numHiddenUnits): 
#     """
#     Takes activations from a previous layer and returns the weighted sum of the
#     inputs for the current hidden layer (described by numHiddenUnits).
#     """
#     X = tf.placeholder(tf.float64)

#     # input shape[1] is the dimension of the input images
#     W = tf.get_variable("W", shape=[inputTensor.shape[1], numHiddenUnits],
#             dtype=tf.float64, 
#             initializer=tf.contrib.layers.xavier_initializer())

#     sess = tf.Session()
#     init = tf.global_variables_initializer()

#     sess.run(init)

#     Sum = tf.matmul(X, W)

#     # Result shape: N x numHiddenUnits
#     return sess.run(Sum, feed_dict={X: inputTensor})

numHiddenUnits = 1000
numClasses = 10
inputDimension = trainData.shape[1]
# learningRates = [0.01, 0.001, 0.0001]
# Through testing, a learning rate of 0.001 yielded the fastest convergence
bestLearningRate = 0.001

"""
Theses will change once we use the full dataset
"""
epochSize = 5
n_iterations = 500
batch_size = 3000

if testing:
    epochSize = 5
    n_iterations = 500
    batch_size = 30

#To ease computation, indices with right label are matched to 1, rest are 0.

def one_hot_encoding(data):
    encoding = np.zeros((len(data), numClasses))
    encoding[np.arange(len(data)), data] = 1
    return encoding

trainTarget = one_hot_encoding(trainTarget)
validTarget = one_hot_encoding(validTarget)
testTarget  = one_hot_encoding(testTarget)

Bias1 = tf.get_variable("b1_{0}".format(bestLearningRate), shape=[1, numHiddenUnits], dtype=tf.float64)
Bias2 = tf.get_variable("b2_{0}".format(bestLearningRate), shape=[1, numClasses], dtype=tf.float64)

Y = tf.placeholder(tf.float64)
X0 = tf.placeholder(tf.float64)

W1 = tf.get_variable("W1_{0}".format(bestLearningRate), shape=[inputDimension, numHiddenUnits],
        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
W2 = tf.get_variable("W2_{0}".format(bestLearningRate), shape=[numHiddenUnits, numClasses],
        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

S1 = tf.matmul(X0, W1) + Bias1
X1 = tf.nn.relu(S1)

S2 = tf.matmul(X1, W2) + Bias2

# Now determine the output classification
y_hat = S2

Loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = Y)
)

ClassificationError = tf.reduce_mean(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1)), tf.float32)
)

Optimizer = tf.train.AdamOptimizer(learning_rate = bestLearningRate).minimize(Loss);

training_set = {
    X0: trainData,
    Y: trainTarget
}

validation_set = {
    X0: validData,
    Y: validTarget
}

testing_set = {
    X0: testData,
    Y: testTarget
}


start_point = 0

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

epoch_training_loss = []
epoch_validation_loss = []
epoch_testing_loss = []
epoch_training_error = []
epoch_validation_error = []
epoch_testing_error = []

epoch_training_loss.append(sess.run(Loss, feed_dict=training_set))
epoch_validation_loss.append(sess.run(Loss, feed_dict=validation_set))
epoch_testing_loss.append(sess.run(Loss, feed_dict=testing_set))

epoch_training_error.append(sess.run(ClassificationError, feed_dict=training_set))
epoch_validation_error.append(sess.run(ClassificationError, feed_dict=validation_set))
epoch_testing_error.append(sess.run(ClassificationError, feed_dict=testing_set))

for i in range(n_iterations):

    batch = {
        X0: trainData[start_point : start_point + batch_size],
        Y: trainTarget[start_point : start_point + batch_size]
    }

    sess.run(Optimizer, feed_dict=batch)

    start_point = (start_point + batch_size) % len(trainData)

    if (i + 1) % epochSize == 0:
        epoch_training_loss.append(sess.run(Loss, feed_dict=training_set))
        epoch_validation_loss.append(sess.run(Loss, feed_dict=validation_set))
        epoch_testing_loss.append(sess.run(Loss, feed_dict=testing_set))

        epoch_training_error.append(sess.run(ClassificationError, feed_dict=training_set))
        epoch_validation_error.append(sess.run(ClassificationError, feed_dict=validation_set))
        epoch_testing_error.append(sess.run(ClassificationError, feed_dict=testing_set))
        print("{0}%".format(i * 100.0 / (1.0 *n_iterations)))

plt.plot(epoch_training_loss, label="Training")
plt.plot(epoch_validation_loss, label="Validation")
plt.plot(epoch_testing_loss, label="Testing")
plt.legend()
plt.title("Cross Entropy Loss, Learning Rate = " + str(bestLearningRate))
plt.show()

plt.plot(epoch_training_error, label="Training")
plt.plot(epoch_validation_error, label="Validation")
plt.plot(epoch_testing_error, label="Testing")
plt.title("Classification Error, Learning Rate = " + str(bestLearningRate))
plt.show()