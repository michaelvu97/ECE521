import tensorflow as tf
import numpy as np
import time

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


# Given parameters
# One iteration is to run through an entire mini batch (size 500 here)
# Note that one epoch is one runthrough of the entire training data set
# This means that we will be training for about 2857 epochs
# We should be plotting training loss function per # epochs, meaning one entry every
# 7 iterations
trainDataSize = 3500
batchSizes = [500, 1500, 3500]
numIterations = 20000

# This was the best learning rate chosen from 1.1
learningRate = .005

# Input and output data
X = tf.placeholder(tf.float64)
Y = tf.placeholder(tf.float64)


for batchSize in batchSizes:

    # Weights
    w = tf.expand_dims(tf.Variable(tf.zeros([784], dtype=tf.float64), name="weights"), 1)
    b = tf.Variable(tf.zeros([1], dtype = tf.float64), name = "bias")

    # Reshape the input so that each picture is a vector rather than a matrix
    # Mention in the report that we used this numpy operation to speed things up
    trainData = np.reshape(trainData, [3500, -1])

    # Create a prediction function
    yhat = tf.add(tf.matmul(X, w), b)

    # The function to minimize
    MSE = tf.reduce_mean(((yhat - Y)**2)/2)

    # I needed to re create the variables. The easiest way to do this seemed to be to
    # start a new session for each time I recreated the variables
    ################################################################################
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)
    ################################################################################
    
    # Where each batch begins in the training data
    startPoint = 0

    # Create the optimizer. Every time a session is run, 
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(MSE)

    # Record the time before training starts
    startTime = time.time()

    for iteration in range(numIterations):

        # Run the optimizer
        sess.run(optimizer, feed_dict={X: trainData[startPoint : startPoint + batchSize], Y: trainTarget[startPoint : startPoint + batchSize]})
        
        # Update the start of the batch as needed
        startPoint = (startPoint + batchSize) % trainDataSize
    
    # Get the time it took to train
    trainTime = time.time() - startTime

    # Calculate the MSE after each batch and print it
    tempMSE = sess.run(MSE, feed_dict={X: trainData, Y: trainTarget})

    print("Batch size:", batchSize, " MSE:", tempMSE, " Time taken:", trainTime)
