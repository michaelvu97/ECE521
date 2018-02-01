import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def data_segmentation(data_path, target_path, t):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task 
	task = 0
	if t:
		task = 1 
	data = np.load(data_path)/255.0
	data = np.reshape(data, [-1, 32*32])
	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
	data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
	target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
	target[rnd_idx[trBatch + validBatch + 1:-1], task]
	return trainData, validData, testData, trainTarget, validTarget, testTarget

def PairwiseEuclidian(X,Y):
    return tf.reduce_sum((tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))**2, 2)

def PickKNearest(distMatrix, k):
	dists, indices_k = tf.nn.top_k(-distMatrix, k)
	return indices_k

def MSE(Ytest, Ynew):
	numEntries = tf.shape(validY)[0]
	return tf.reduce_mean(tf.reduce_sum((Ynew - Ytest)**2,1))



def predict(R, Y):
	return(tf.matmul(R, tf.cast(Y, tf.float32)))

def predictClass(entries, index):
	out, bad, count = tf.unique_with_counts(entries[index])
	return tf.gather(out, tf.argmax(count))
################################################################################

"""

Takes a [(32^2)] float32 shape array for a grayscale image and plots it using 
matplotlib.

"""
def DrawImage(imgArray):

	imgArray = tf.reshape(imgArray, [32,32,1])

	# Convert the dimensions from grayscale to RGB
	imgArray = tf.tile(imgArray, [1,1,3])

	# print(imgArray.eval())
	imgplot = plt.imshow(imgArray.eval())
	plt.show()
	
	return 0

################################################################################

names = ['Lorraine Bracco', 'Gerard Butler', 'Peri Gilpin', 'Angie Harmon',\
		'Daniel Radcliffe', 'Michael Vartan']

genders = ['Male', 'Female']

init = tf.global_variables_initializer()


sess = tf.InteractiveSession()
sess.run(init)

# Selects gender instead of name
use_genders = True
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy', 'target.npy', use_genders)

Y = tf.constant(trainTarget)
targetData = tf.constant(validTarget)

# Keep track of min K value and error
minError = tf.constant([10000], tf.float32)
minK = 1

ks = [1, 5, 10, 25, 50, 100, 200]
for k in ks:

	# Get the indices of the k-nearest training data -> new input
	indices = PickKNearest(PairwiseEuclidian(validData, trainData), k)

	# Get the classes of the k-nearest training data
	entries = tf.gather(Y, indices)

	# Initialize error to 0
	error = tf.constant([0], tf.float32)

	# Make a prediction for every input
	failureFound = False;
	for i in range(0, (validTarget.shape)[0]):

		# Predicted label
		guess = predictClass(entries, i)

		# The true label
		actual = targetData[i]
		error = tf.add(error,tf.to_float(tf.not_equal(guess, actual)))
		"""
		if k == 10 and not failureFound and tf.reduce_any(tf.not_equal(guess, actual)).eval():
			# Find the first failure
			failureFound = True
			
			# Display the failure case
			if use_genders:
				plt.title("{} misclassified as {}" \
						.format(genders[actual.eval()], genders[guess.eval()]))
			else:
				plt.title("{} misclassified as {}" \
						.format(names[actual.eval()], names[guess.eval()]))

			DrawImage(tf.cast(validData[i], tf.float32));

			badIndices = PickKNearest(PairwiseEuclidian(tf.expand_dims(validData[i], 0), trainData), k)[0]

			#print(sess.run(badIndices))

			# print(badIndices.shape[0])

			for j in range(0, (badIndices.shape)[0]):
				badImage = tf.gather(trainData, badIndices[j])
				#print(sess.run(badImage))
				plt.title("Nearest images to the misclassified image")
				DrawImage(tf.cast(badImage, tf.float32))
		"""

			

	# Update the minimum error and k values
	if sess.run(error) < sess.run(minError):
		mink = k
	minError = tf.where(tf.less(error, minError), error, minError)
	




print("The minimum error occurs when k = %d" %minK)

# Now compute the error in the test set with this minimum K
targetData = tf.constant(testTarget)

# Get the indices of the k-nearest training data -> new input
indices = PickKNearest(PairwiseEuclidian(testData, trainData), minK)

# Get the classes of the k-nearest training data
entries = tf.gather(Y, indices)

# Initialize error to 0
error = tf.constant([0], tf.float32)

# Make a prediction for every input
for i in range(0, (testTarget.shape)[0]):
	guess = predictClass(entries, i)
	actual = targetData[i]
	error = tf.add(error,tf.to_float(tf.not_equal(guess, actual)))

print("Number wrong in the test set when k = %d is %f" %(minK, sess.run(error)))


ks1 = [10]
for k in ks1:

	# Get the indices of the k-nearest training data -> new input
	indices = PickKNearest(PairwiseEuclidian(testData, trainData), k)

	# Get the classes of the k-nearest training data
	entries = tf.gather(Y, indices)

	# Initialize error to 0
	error = tf.constant([0], tf.float32)

	# Make a prediction for every input
	failureFound = False;
	for i in range(0, (testTarget.shape)[0]):

		# Predicted label
		guess = predictClass(entries, i)

		# The true label
		actual = targetData[i]
		error = tf.add(error,tf.to_float(tf.not_equal(guess, actual)))

		if k == 10 and not failureFound and tf.reduce_any(tf.not_equal(guess, actual)).eval():
			# Find the first failure
			failureFound = True
			
			# Display the failure case
			if use_genders:
				plt.title("{} misclassified as {}" \
						.format(genders[actual.eval()], genders[guess.eval()]))
			else:
				plt.title("{} misclassified as {}" \
						.format(names[actual.eval()], names[guess.eval()]))

			DrawImage(tf.cast(testData[i], tf.float32));

			badIndices = PickKNearest(PairwiseEuclidian(tf.expand_dims(testData[i], 0), trainData), k)[0]

			#print(sess.run(badIndices))

			# print(badIndices.shape[0])

			for j in range(0, (badIndices.shape)[0]):
				badImage = tf.gather(trainData, badIndices[j])
				#print(sess.run(badImage))
				plt.title("Nearest images to the misclassified image")
				DrawImage(tf.cast(badImage, tf.float32))
