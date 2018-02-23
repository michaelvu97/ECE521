import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

################################################################################

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

################################################################################

lambda_weight_penalty = tf.constant(0.01, dtype=tf.float64)
n_iterations = 5000
batch_size = 500
learningRates = [.005, .001, .0001]
epoch_size = 7

################################################################################

trainData = np.reshape(trainData, [3500, -1 ])

dimension = trainData.shape[1]

X = tf.placeholder(tf.float64); # [Nxd]
Y = tf.placeholder(tf.float64); # [N]

for learning_rate in learningRates:

    one = tf.constant([1], dtype=tf.float64)

    w = tf.get_variable("weights", shape=[dimension, 1], dtype=tf.float64, \
            initializer=tf.zeros_initializer)

    b = tf.get_variable(shape=[1], dtype=tf.float64, name="bias", \
            initializer=tf.zeros_initializer)


    # Xw + b, no sigmoid since that is handled in the cross entropy function
    yhat = tf.add(tf.matmul(X, w), b)

    Loss_Data = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))

    Loss_Weights = (lambda_weight_penalty / tf.constant(2.0, dtype=tf.float64)) * tf.reduce_sum(w**2)

    Loss = Loss_Data + Loss_Weights

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(Loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    epochLoss = []

    training_set = {
        X: trainData,
        Y: trainTarget
    }

    start_point = 0

    for iteration in range(n_iterations):

        # Once step of optimization

        batch = {
            X: trainData[start_point : start_point + batch_size],
            Y: trainData[start_point : start_point + batch_size]
        }

        sess.run(optimizer, feed_dict=batch)

        start_point = (start_point + batch_size) % len(trainData)

        # Check the result of this epoch
        if ((iteration + 1) % epoch_size == 0):
            epochLoss.append(sess.run(Loss, feed_dict=training_set))

    plt.plot(np.arange(len(epochLoss)), epochLoss, \
            label=("loss with training rate " + str(learning_rate)))

    print(epochLoss[-1])

plt.legend()
plt.show()
