import numpy as np
import tensorflow as tf

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
	print(sess.run(indices_k))
	index = tf.expand_dims(index, 1)
	

	#This prepares the tensor for the element wise comparison
	indices_k = tf.expand_dims(indices_k, 1)

	#This creates a matrix of 0s and 1s that represent whether or not a training point is used
	#then divides it by k to obtain the correct tensor for each new test input
	#The responsibility vectors are row vectors
	responsibilites = tf.reduce_sum(tf.to_float(tf.equal(index, indices_k)), 2)/tf.to_float(k)

	return responsibilites


################################################################################


init = tf.global_variables_initializer()


sess = tf.InteractiveSession()
sess.run(init)

#the format of the input matrix is the same as the output of part 1, where X is the new data
#and Z is the trained data

X = tf.constant([[1,2],[3,4],[9,3]])
Z = tf.constant([[8,2],[3,4],[3,6],[3,3],[12,34],[0,0]])

#print(sess.run(X))
#print(sess.run(Z))
print(sess.run(PairwiseEuclidian(X,Z)))
print(sess.run(PickKNearest(PairwiseEuclidian(X,Z), 3)))
