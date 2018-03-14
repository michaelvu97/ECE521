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
numIterations = 20000
batchSize = 500
learningRate = 0.001
numClasses = 10
epochSize = 30
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

epochTrainLoss = []
epochTrainAccuracy = []
epochValidLoss = []
epochValidAccuracy = []

startPoint = 0

bestTestAccuracy = 0.0

#Training cycle
for i in range(numIterations):
	
	#Pick the data points
	batchX = trainData[startPoint: startPoint + batchSize]
	batchY = trainTarget[startPoint: startPoint + batchSize]
	_, error, weight, bias, y = sess.run([train, totalLoss, W, b, predictedY], feed_dict = {X: batchX, Y: batchY})
		
	startPoint = (startPoint + batchSize) % len(trainData)
			
	if((i + 1)%epochSize == 0):
		epochTrainLoss.append(totalLoss.eval(feed_dict = {X: trainData, Y: trainTarget}))
		epochTrainAccuracy.append(accuracy.eval(feed_dict = {X: trainData, Y: trainTarget}))
		epochValidLoss.append(totalLoss.eval(feed_dict = {X: validData, Y: validTarget}))
		epochValidAccuracy.append(accuracy.eval(feed_dict = {X: validData, Y: validTarget}))
	
	testAccuracy = accuracy.eval(feed_dict = {X: testData, Y: testTarget})
	if(testAccuracy > bestTestAccuracy):
		bestTestAccuracy = testAccuracy
		print("Test Accuracy: " + str(testAccuracy))

val = 0
print('Initial Train Loss: %0.3f, Initial Valid Loss: %0.3f' % (epochTrainLoss[val], epochValidLoss[val]))
print('Initial Train Accuracy: %0.3f, Initial Valid Accuracy: %0.3f' % (epochTrainAccuracy[val], epochValidAccuracy[val]))

val = len(epochTrainLoss) - 1
print('Final Train Loss: %0.3f, Final Valid Loss: %0.3f' % (epochTrainLoss[val], epochValidLoss[val]))
print('Final Train Accuracy: %0.3f, Final Valid Accuracy: %0.3f' % (epochTrainAccuracy[val], epochValidAccuracy[val]))

	
print("Best Test Accuracy Obtained: " + str(bestTestAccuracy))

plt.figure(1)

plt.subplot(211)
plt.plot(epochTrainLoss, label = ("Training Loss"))
plt.plot(epochValidLoss, label = ("Valid Loss"))

plt.legend()
plt.title("Best Training and Validation Loss, Learning Rate = " + str(learningRate))

plt.subplot(212)
plt.plot(epochTrainAccuracy, label = ("Training Accuracy"))
plt.plot(epochValidAccuracy, label = ("Valid Accuracy"))

plt.legend()
plt.title("Best Accuracy and Validation Accuracy, Learning Rate = " + str(learningRate))
plt.show()
