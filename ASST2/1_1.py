import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
batchSize = 500
numIterations = 20000
learningRates = [.005, .001, .0001]

# Input and output data
X = tf.placeholder(tf.float64)
Y = tf.placeholder(tf.float64)


for rate in learningRates:

    # Reshape the input so that each picture is a vector rather than a matrix
    # Mention in the report that we used this numpy operation to speed things up
    trainData = np.reshape(trainData, [3500, -1])

    # Weights
    w = tf.expand_dims(tf.Variable(tf.zeros([784], dtype=tf.float64), name="weights"), 1)
    # w = tf.Variable(tf.truncated_normal(shape = [trainData.shape[1], 1], stddev = 0.1, dtype = tf.float64))
    b = tf.Variable(tf.zeros([1], dtype = tf.float64), name = "bias")
    # b = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64))

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
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(MSE)

    # The losses at each epoch and the epoch itself
    epochLoss = []
    epoch = np.arange(2857)

    for iteration in range(numIterations):

        # Run the optimizer
        sess.run(optimizer, feed_dict={X: trainData[startPoint : startPoint + batchSize], Y: trainTarget[startPoint : startPoint + batchSize]})
        
        # Update the start of the batch as needed
        startPoint = (startPoint + batchSize) % trainDataSize
    
        # If an epoch has completed, then update the epoch loss
        if((iteration + 1)%7 == 0):
            tempMSE = sess.run(MSE, feed_dict={X: trainData, Y: trainTarget})
            #print("MSE: ", tempMSE)
            epochLoss.append(tempMSE)
    
    # After optimizing, plot the epoch and the loss
    epochLoss = np.array(epochLoss)
    label = "MSE with learning rate " + str(rate)
    plt.plot(epoch, epochLoss, label=label)
    print(sess.run(MSE, feed_dict={X: trainData, Y: trainTarget}))

# Create the plot
plt.legend()
plt.title("MSE Value at Each Epoch for Various Learning Rates")
plt.show()
