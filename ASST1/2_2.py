import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def PairwiseEuclidian(X,Y):
    return tf.reduce_sum((tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))**2, 2)

def PickKNearest(distMatrix, k):
	dists, indices_k = tf.nn.top_k(-distMatrix, k)

	#This is the number of training tensors
	#Use this to determine how long to make the responsibility tensor
	trainingNums = tf.shape(distMatrix)[1]

	#This creates a tensor of shape [trainingNums], then expands it for the
	#element wise comparison in tf.equal
	index = tf.range(trainingNums)
	index = tf.expand_dims(index, 1)

	#This prepares the tensor for the element wise comparison
	indices_k = tf.expand_dims(indices_k, 1)

	#This creates a matrix of 0s and 1s that represent whether or not a training point is used
	#then divides it by k to obtain the correct tensor for each new test input
	#The responsibility vectors are row vectors
	responsibilites = tf.reduce_sum(tf.to_float(tf.equal(index, indices_k)), 2)/tf.to_float(k)

	return responsibilites

def MSE(Ytest, Ynew):
	return tf.reduce_mean(tf.reduce_sum(((Ynew - Ytest)**2)/2,1))

def predict(R, Y):
	return(tf.matmul(R, tf.cast(Y, tf.float32)))

################################################################################

init = tf.global_variables_initializer()


sess = tf.InteractiveSession()
sess.run(init)

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
         + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


# The list of K values to test
ks = [1,3,5,50]

# Variables to keep track of the min error and the min K
minValidError = 1000
minK = 0


# Create the X axis points for the graph
X = np.linspace(0.0,11.0,num=1000)[:,np.newaxis]

i = 1

for k in ks:

	# Make tensors to represent the inputs and outputs for the data sets
	trainX = tf.constant(trainData)
	trainY = tf.constant(trainTarget)

	validX = tf.constant(validData)
	validY = tf.constant(validTarget)

	testX = tf.constant(testData)
	testY = tf.constant(testTarget)


	# Compute the R matrices, predictions, and MSEs for each training set
	validR = PickKNearest(PairwiseEuclidian(validX, trainX), k)
	#print(sess.run(validR))
	validPredict = predict(validR, trainY)
	validMSE = MSE(tf.cast(validY, tf.float32), validPredict)

	trainR = PickKNearest(PairwiseEuclidian(trainX, trainX), k)
	trainPredict = predict(trainR, trainY)
	trainMSE = MSE(tf.cast(trainY, tf.float32), trainPredict)

	testR = PickKNearest(PairwiseEuclidian(testX, trainX), k)
	testPredict = predict(testR, trainY)
	testMSE = MSE(tf.cast(testY, tf.float64), tf.cast(testPredict, tf.float64))


	# Print each error and keep track of the min error so far
	print("")
	print("Using k = %d" %k)
	print("training error: %f" %sess.run(trainMSE))
	print("valid error: %f" %sess.run(validMSE))
	print("test error: %f" %sess.run(testMSE))
	currError = sess.run(validMSE)
	if currError < minValidError:
		minValidError = currError
		minK = k

	# Plot the inputs and outputs
	Y = sess.run(predict(PickKNearest(PairwiseEuclidian(X, trainData), k), trainTarget))
	plt.figure(i)
	i = i + 1
	plt.plot(trainData,trainTarget,'.')
	plt.plot(X,Y,'-')
	plt.show()

# Print the final minimum K value
print("")
print("Min error is with k = %d" %(minK))

