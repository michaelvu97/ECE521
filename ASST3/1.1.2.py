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

trainData = np.reshape(trainData, [trainData.shape[0], -1])
validData = np.reshape(validData, [validData.shape[0], -1])
testData = np.reshape(testData, [testData.shape[0], -1])

inputImageDimension = trainData[0].shape[0]
layer_variable_suffix = 1

def WeightedSumLayer(inputTensor, numHiddenUnits): 
    """
    Takes activations from a previous layer and returns the weighted sum of the
    inputs for the current hidden layer (described by numHiddenUnits).
    """

    global layer_variable_suffix
    global inputImageDimension

    W = tf.get_variable("W_{0}".format(layer_variable_suffix), 
            shape=[inputTensor.shape[1], numHiddenUnits],
            dtype=tf.float64, 
            initializer=tf.contrib.layers.xavier_initializer(uniform = False))

    b = tf.get_variable("b_{0}".format(layer_variable_suffix), 
            shape=[1, numHiddenUnits],
            dtype=tf.float64,
            initializer=tf.zeros_initializer())

    layer_variable_suffix = layer_variable_suffix + 1

    Y = tf.matmul(inputTensor, W) + b

    return Y, W, b

numHiddenUnits = 1000
numClasses = 10
inputDimension = trainData.shape[1]
# learningRates = [0.01, 0.001, 0.0001]
# Through testing, a learning rate of 0.001 yielded the fastest convergence
bestLearningRate = 0.001

lambda_weight_penalty = tf.constant(0.0003, dtype=tf.float64)

epochSize = 5
n_iterations = 5000
batch_size = 3000

#To ease computation, indices with right label are matched to 1, rest are 0.

def one_hot_encoding(data):
    encoding = np.zeros((len(data), numClasses))
    encoding[np.arange(len(data)), data] = 1
    return encoding

trainTarget = one_hot_encoding(trainTarget)
validTarget = one_hot_encoding(validTarget)
testTarget  = one_hot_encoding(testTarget)

Y = tf.placeholder(tf.float64, [None, numClasses])
X0 = tf.placeholder(tf.float64, [None, inputImageDimension])

S1, W1, b1 = WeightedSumLayer(X0, numHiddenUnits)

X1 = tf.nn.relu(S1)

S2, W2, b2 = WeightedSumLayer(X1, numClasses)

# Now determine the output classification
y_hat = S2

WeightDecay = 0.5 * lambda_weight_penalty * (
        tf.reduce_sum(W1 ** 2) + 
        tf.reduce_sum(W2 ** 2)
)

Loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = Y)
) + WeightDecay

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

minValidationError = [];
minValidationError.append(epoch_validation_error[-1]);

minValidationLoss = [];
minValidationLoss.append(epoch_validation_loss[-1]);

earlyStoppingIndex_error = 0;
earlyStoppingIndex_loss = 0;

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
        if (epoch_validation_error[-1] < minValidationError[-1]):
            minValidationError.append(epoch_validation_error[-1])
            earlyStoppingIndex_error = len(minValidationError) - 1
        else:
            minValidationError.append(minValidationError[-1])

        if (epoch_validation_loss[-1] < minValidationLoss[-1]):
            minValidationLoss.append(epoch_validation_loss[-1])
            earlyStoppingIndex_loss = len(minValidationLoss) - 1
        else:
            minValidationLoss.append(minValidationLoss[-1])

        epoch_testing_error.append(sess.run(ClassificationError, feed_dict=testing_set))
        print("{0}%".format(i * 100.0 / (1.0 *n_iterations)))

print ("At early stopping point (error): ")
print ("\tTrainingError = " + str(epoch_training_error[earlyStoppingIndex_error]))
print ("\tValidationError = " + str(epoch_validation_error[earlyStoppingIndex_error]))
print ("\tTestingError = " + str(epoch_testing_error[earlyStoppingIndex_error]))

print ("At early stopping point (loss) : ")
print ("\tTrainingLoss = " + str(epoch_training_loss[earlyStoppingIndex_loss]))
print ("\tValidationLoss = " + str(epoch_validation_loss[earlyStoppingIndex_loss]))
print ("\tTestingLoss = " + str(epoch_testing_loss[earlyStoppingIndex_loss]))

plt.plot(epoch_training_loss, label="Training")
plt.plot(epoch_validation_loss, label="Validation")
plt.plot(epoch_testing_loss, label="Testing")
plt.plot(minValidationLoss, label="Min Validation Loss", linestyle='--')
plt.legend()
plt.title("Cross Entropy Loss, Learning Rate = " + str(bestLearningRate))
plt.show()

plt.plot(epoch_training_error, label="Training")
plt.plot(epoch_validation_error, label="Validation")
plt.plot(epoch_testing_error, label="Testing")
plt.plot(minValidationError, label="Min Validation Error", linestyle='--')
plt.legend()
plt.title("Classification Error, Learning Rate = " + str(bestLearningRate))
plt.show()

plt.plot(epoch_testing_error)
plt.title("Testing Error with One Hidden Layer (1000 units)")
plt.show()
