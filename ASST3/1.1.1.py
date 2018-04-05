import tensorflow as tf
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
testing = True

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

layer_variable_suffix = 1

inputImageDimension = trainData[0].shape[0]
print(inputImageDimension)

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


X = tf.placeholder(tf.float64, [None, inputImageDimension])

result, W1, b1 = WeightedSumLayer(X, 5)

res2, W2, b2 = WeightedSumLayer(result, 5)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

print(result)
print(result.shape)

j = sess.run(res2, feed_dict={X: trainData})

print(j)
print(j.shape)
