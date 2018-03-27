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

def WeightedSumLayer(inputTensor, numHiddenUnits): 
    """
    Takes activations from a previous layer and returns the weighted sum of the
    inputs for the current hidden layer (described by numHiddenUnits).
    """

    """
    TODO DO BIAS
    """

    X = tf.placeholder(tf.float64)

    # input shape[1] is the dimension of the input images
    W = tf.get_variable("W", shape=[inputTensor.shape[1], numHiddenUnits],
            dtype=tf.float64, 
            initializer=tf.contrib.layers.xavier_initializer())

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    Sum = tf.matmul(X, W)

    # Result shape: N x numHiddenUnits
    return sess.run(Sum, feed_dict={X: inputTensor})

result = WeightedSumLayer(trainData, 5)

print(result)
print(result.shape)