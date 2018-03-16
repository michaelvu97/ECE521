import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def data_segmentation(data_path, target_path, task):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task
	data = np.load(data_path)/255
	data = np.reshape(data, [-1, 32*32])
		
	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
	data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
	data[rnd_idx[trBatch + validBatch+1:-1],:]
	
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
	target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
	target[rnd_idx[trBatch + validBatch + 1:-1], task]
	
	return trainData, validData, testData, trainTarget, validTarget, testTarget

#To ease computation, indices with right label are matched to 1, rest are 0.
def one_hot_encoding(data):
	encoding = np.zeros((len(data), 6))
	encoding[np.arange(len(data)), data] = 1
	return encoding

trainData, validData, testData, trainTarget, validTarget, testTarget = \
		data_segmentation('data.npy', 'target.npy', 0)

#Reshaping data into a tensor		
trainData = np.reshape(trainData, (trainData.shape[0], 1024)) 
validData = np.reshape(validData, (validData.shape[0], 1024))
testData = np.reshape(testData, (testData.shape[0], 1024))

#Encode the labels using one hot encoding
trainTarget = one_hot_encoding(trainTarget)
validTarget = one_hot_encoding(validTarget)
testTarget = one_hot_encoding(testTarget)

#Values we will be using for training model -
valLambda = 0.001
numIterations = 10000
batchSize = 300
learningRate = 0.005
numBatches = int(trainData.shape[0]/batchSize) 
epochSize = 3 #900 Images in total

#Graph Input Variables
X = tf.placeholder(tf.float32, [None, 1024]) #32 * 32 = 1024
Y = tf.placeholder(tf.float32, [None, 6]) #6 classes

#Set model weights
W = tf.Variable(tf.truncated_normal(shape = [1024,6], dtype = tf.float32, stddev = 0.5))
b = tf.Variable(tf.zeros([6]))

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
		
	#Calculate losses and accuracy for training and validation sets
	if((i + 1)%epochSize == 0):
		epochTrainLoss.append(totalLoss.eval(feed_dict = {X: trainData, Y: trainTarget}))
		epochTrainAccuracy.append(accuracy.eval(feed_dict = {X: trainData, Y: trainTarget}))
		epochValidLoss.append(totalLoss.eval(feed_dict = {X: validData, Y: validTarget}))
		epochValidAccuracy.append(accuracy.eval(feed_dict = {X: validData, Y: validTarget}))
			
	testAccuracy = accuracy.eval(feed_dict = {X: testData, Y: testTarget})
	if(testAccuracy > bestTestAccuracy):
		bestTestAccuracy = testAccuracy

		
		
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
plt.title("Training & Validation Loss, Learning Rate = " + str(learningRate) + ", Lambda = " + str(valLambda))
plt.xlabel("Number of Epochs")
plt.ylabel("Cross-Entropy Loss")

plt.subplot(212)
plt.plot(epochTrainAccuracy, label = ("Training Accuracy"))
plt.plot(epochValidAccuracy, label = ("Valid Accuracy"))

plt.legend()
plt.title("Training & Validation Accuracy, Learning Rate = " + str(learningRate) + ", Lambda = " + str(valLambda))
plt.xlabel("Number of Epochs")
plt.ylabel("Classification Accuracy")

plt.show()

