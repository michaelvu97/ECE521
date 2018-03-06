import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Given code to load data
with np.load("notMNIST.npz") as data:

	Data, Target = data ["images"], data["labels"]
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data = Data[randIndx]/255.
	Target = Target[randIndx]
	
	#Data has 18720 instances of 28-by-28 images, divided
	#into training, validation and testing data.
	
	#Reshape converts image data from matrix to vector.
	
	trainData, trainTarget = Data[:15000], Target[:15000]
	trainData = np.reshape(trainData, (trainData.shape[0], 784)) #28 * 28 = 784
	
	validData, validTarget = Data[15000:16000], Target[15000:16000]
	validData = np.reshape(validData, (validData.shape[0], 784))
	
	testData, testTarget = Data[16000:], Target[16000:]
	testData = np.reshape(testData, (testData.shape[0], 784))

#To ease computation, indices with right label are matched to 1, rest are 0.
def one_hot_encoding(data):
	encoding = np.zeros((len(data), 10))
	encoding[np.arange(len(data)), data] = 1
	return encoding

#Encode the labels using one hot encoding
trainTarget = one_hot_encoding(trainTarget)
validTarget = one_hot_encoding(validTarget)
testTarget = one_hot_encoding(testTarget)

#Values we will be using for training model -
valLambda = 0.01
numIterations = 100
batchSize = 500
learningRate = 0.001
numClasses = 10
numBatches = int(trainData.shape[0]/batchSize) #30

#Graph Input Variables
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#Set model weights
W = tf.Variable(tf.truncated_normal(shape = [784,10], dtype = tf.float32, stddev = 0.1))
b = tf.Variable(tf.zeros([10]))

#Construct Model
yhat = tf.matmul(X,W) + b
predictedY = tf.nn.softmax(tf.matmul(X,W) + b)

#Calculate Cross Entropy Loss
dataLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = yhat, labels = Y))

weightLoss = (valLambda/2) * tf.reduce_sum(tf.square(W))

totalLoss = dataLoss + weightLoss

#Prediction Step - Choose class with highest probability as label
prediction = tf.equal(tf.argmax(predictedY,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32)) 

#The gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
train = optimizer.minimize(totalLoss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

trainLoss = np.empty(numBatches * numIterations)
validLoss = np.empty(numBatches * numIterations)
trainAccuracy = np.empty(numBatches * numIterations)
validAccuracy = np.empty(numBatches * numIterations)

startPoint = 0

#Training cycle
for i in range(numIterations):
	#Loop over all batches
	for batch in range(numBatches):
		#Pick the data points
		batchX = trainData[startPoint: startPoint + batchSize]
		batchY = trainTarget[startPoint: startPoint + batchSize]
		
		_, error, weight, bias, y = sess.run([train, totalLoss, W, b, predictedY], feed_dict = {X: batchX, Y: batchY})
		
		startPoint = (startPoint + batchSize) % len(trainData)
		
		index =  numBatches*i+batch
		#Calculate losses and accuracy for training and validation sets
		trainLoss[index] = totalLoss.eval(feed_dict = {X: trainData, Y: trainTarget})
		validLoss[index] = totalLoss.eval(feed_dict = {X: validData, Y: validTarget})
		trainAccuracy[index] = accuracy.eval(feed_dict = {X: trainData, Y: trainTarget})
		validAccuracy[index] = accuracy.eval(feed_dict = {X: validData, Y: validTarget})
		
	print('Iteration: %2d, Train Loss: %0.3f, Valid Loss: %0.3f' % (i, trainLoss[index], validLoss[index]))
	print('Iteration: %2d, Train Accuracy: %0.3f, Valid Accuracy: %0.3f' % (i, trainAccuracy[index], validAccuracy[index]))
	
testAccuracy = accuracy.eval(feed_dict = {X: testData, Y: testTarget})
print("Test Accuracy: " + str(testAccuracy))

	