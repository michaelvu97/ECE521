import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

import pprint

pp = pprint.PrettyPrinter(indent=4)
pp.pprint("test")

with np.load("notMNIST.npz") as data:
    np.random.seed(521)
    Data, Target = data ["images"], data["labels"]
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

def WeightedSumLayer(inputTensor, numHiddenUnits, useDropOut): 
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

    #Decide whether or not to use dropout
    if useDropOut:
        dropout_W = tf.nn.dropout(W, 0.5)
        Y = tf.matmul(inputTensor, dropout_W) + b
    else:
        Y = tf.matmul(inputTensor, W) + b

    return Y, W, b

numClasses = 10
inputDimension = trainData.shape[1]
epochSize = 5
n_iterations = 5000
batch_size = 3000

def one_hot_encoding(data):
    encoding = np.zeros((len(data), numClasses))
    encoding[np.arange(len(data)), data] = 1
    return encoding

trainTarget = one_hot_encoding(trainTarget)
validTarget = one_hot_encoding(validTarget)
testTarget  = one_hot_encoding(testTarget)

    
learningRateOptions = np.arange(-7.5, -4.5, 0.1) # Log this
dropOutOptions = [True, False]
numLayerOptions = range(1, 6) 
numHiddenUnitsOptions = range(100, 600, 100)
weightDecayOptions = np.arange(-9, -6, 0.1) # Log this


def Simulation(index):

    # These were found from someone on piazza, which yielded 0.0362 valid error
    # and 0.072 test error
    learningRate = 0.00221
    dropout = True
    numLayers = 2
    numHiddenUnits = 430
    weightDecay = 0.00043

    lambda_weight_penalty = tf.constant(weightDecay, dtype=tf.float64)

    Y = tf.placeholder(tf.float64, [None, numClasses])
    X0 = tf.placeholder(tf.float64, [None, inputImageDimension])

    # Build the Neural Net

    Weights = [] # starts at 1.
    Biases = []
    Signals = []
    Signals_out =[]
    X = [X0]

    S1, W1, b1 = WeightedSumLayer(X[0], numHiddenUnits, dropout)
    S1_out = tf.matmul(X[0], W1)

    Signals.append(S1)
    Weights.append(W1)
    Biases.append(b1)

    Signals_out.append(S1_out)

    X.append(tf.nn.relu(S1))

    # Intermediate hidden layer connections
    for i in range (1, numLayers):
        Si, Wi, bi = WeightedSumLayer(X[i], numHiddenUnits, dropout)


        Signals.append(Si)
        Weights.append(Wi)
        Biases.append(bi)

        X.append(tf.nn.relu(Si))

        Si_out = tf.matmul(tf.nn.relu(Signals_out[i-1]), Wi) + bi
        Signals_out.append(Si_out)

    SLast, WLast, bLast = WeightedSumLayer(X[-1], numClasses, dropout)

    SLast_out = tf.matmul(tf.nn.relu(Signals_out[-1]), WLast) + bLast

    Signals.append(SLast)
    Weights.append(WLast)
    Biases.append(bLast)
    Signals_out.append(SLast_out)


    y_hat = SLast
    y_hat_out = SLast_out

    print(Weights)

    WeightDecay = lambda_weight_penalty * tf.add_n([tf.nn.l2_loss(w) for w in Weights]);

    Loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = Y)
    ) + WeightDecay

    Loss_out = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = y_hat_out, labels = Y)
    ) + WeightDecay

    ClassificationError = tf.reduce_mean(tf.cast(
            tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1)), tf.float32)
    )
    
    ClassificationError_out = tf.reduce_mean(tf.cast(
        tf.not_equal(tf.argmax(y_hat_out, 1), tf.argmax(Y, 1)), tf.float32)
    )

    Optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(Loss);

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

    epoch_training_loss.append(sess.run(Loss_out, feed_dict=training_set))
    epoch_validation_loss.append(sess.run(Loss_out, feed_dict=validation_set))
    epoch_testing_loss.append(sess.run(Loss_out, feed_dict=testing_set))

    epoch_training_error.append(sess.run(ClassificationError_out, feed_dict=training_set))
    epoch_validation_error.append(sess.run(ClassificationError_out, feed_dict=validation_set))
    epoch_testing_error.append(sess.run(ClassificationError_out, feed_dict=testing_set))

    minValidationError = [];
    minValidationError.append(epoch_validation_error[-1]);

    for i in range(n_iterations):

        batch = {
            X0: trainData[start_point : start_point + batch_size],
            Y: trainTarget[start_point : start_point + batch_size]
        }

        sess.run(Optimizer, feed_dict=batch)

        start_point = (start_point + batch_size) % len(trainData)

        if (i + 1) % epochSize == 0:
            epoch_training_loss.append(sess.run(Loss_out, feed_dict=training_set))
            epoch_validation_loss.append(sess.run(Loss_out, feed_dict=validation_set))
            epoch_testing_loss.append(sess.run(Loss_out, feed_dict=testing_set))

            epoch_training_error.append(sess.run(ClassificationError_out, feed_dict=training_set))
            epoch_validation_error.append(sess.run(ClassificationError_out, feed_dict=validation_set))
            if (epoch_validation_error[-1] < minValidationError[-1]):
                minValidationError.append(epoch_validation_error[-1])
            else:
                minValidationError.append(minValidationError[-1])

            epoch_testing_error.append(sess.run(ClassificationError_out, feed_dict=testing_set))
            print("{0}%".format(i * 100.0 / (1.0 *n_iterations)))

    print("PARAMETERS")
    print("Hidden units : " + str(numHiddenUnits))
    print("Layers : " + str(numLayers))
    print("Learning Rate : " + str(learningRate))
    print("Decay : " + str(weightDecay))
    print("Dropout : " + str(dropout))
    print("END PARAMETERS")


    plt.plot(epoch_training_loss, label="Training")
    plt.plot(epoch_validation_loss, label="Validation")
    plt.plot(epoch_testing_loss, label="Testing")
    plt.legend()
    plt.title("Cross Entropy Loss, Learning Rate = " + str(learningRate))
    # plt.show()

    plt.plot(epoch_training_error, label="Training")
    plt.plot(epoch_validation_error, label="Validation")
    plt.plot(epoch_testing_error, label="Testing")
    plt.plot(minValidationError, label="Min Validation Error", linestyle='--')
    plt.legend()
    plt.title("Classification Error, Learning Rate = " + str(learningRate))
    plt.show()

    plt.plot(epoch_testing_error)
    plt.title("Testing Error with One Hidden Layer (1000 units)")
    plt.show()

    return {
        "parameters": {
            "numHiddenUnits": numHiddenUnits,
            "numLayers": numLayers,
            "learningRate": learningRate,
            "decay": weightDecay,
            "dropout" : dropout
        },
        "results": {
            "minValidationError" : minValidationError[-1],
            "minTestingError": np.amin(epoch_testing_error)
        }
    }

paramsAndResults = []

for i in range(1):
    paramsAndResults.append(Simulation(i))

pp.pprint(paramsAndResults)