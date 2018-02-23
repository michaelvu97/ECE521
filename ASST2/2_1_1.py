import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
learningRates = [.01, .001, .005]
epoch_size = 7

################################################################################

trainData = np.reshape(trainData, [3500, -1])
validData = np.reshape(validData, [100, -1])

dimension = trainData.shape[1]

X = tf.placeholder(tf.float64); # [Nxd]
Y = tf.placeholder(tf.float64); # [N]

best_learning_rate = learningRates[0]
minimum_loss = np.inf
best_learning_epoch_training_loss = []
best_learning_epoch_validation_loss = []

for learning_rate in learningRates:

    w = tf.Variable(tf.zeros([dimension,1], dtype=tf.float64), name="weights", dtype=tf.float64)

    b = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64, name="bias")

    # Xw + b, no sigmoid since that is handled in the cross entropy function
    yhat = tf.add(tf.matmul(X, w), b)

    Loss_Data = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))

    Loss_Weights = (lambda_weight_penalty / tf.constant(2.0, dtype=tf.float64)) * tf.reduce_sum(w**2)

    Loss = Loss_Data + Loss_Weights

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(Loss)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch_training_loss = []
    epoch_validation_loss = []

    training_set = {
        X: trainData,
        Y: trainTarget
    }

    validation_set = {
        X: validData,
        Y: validTarget
    }

    start_point = 0

    for iteration in range(n_iterations):

        # Once step of optimization

        batch = {
            X: trainData[start_point : start_point + batch_size],
            Y: trainTarget[start_point : start_point + batch_size]
        }

        sess.run(optimizer, feed_dict=batch)

        start_point = (start_point + batch_size) % len(trainData)

        # Check the result of this epoch
        if ((iteration + 1) % epoch_size == 0):
            epoch_training_loss.append(sess.run(Loss, feed_dict=training_set))
            epoch_validation_loss.append(sess.run(Loss, feed_dict=validation_set))

    if epoch_validation_loss[-1] < minimum_loss:
        minimum_loss = epoch_validation_loss[-1]
        best_learning_rate = learning_rate
        best_learning_epoch_training_loss = epoch_training_loss
        best_learning_epoch_validation_loss = epoch_validation_loss


plt.plot(np.arange(len(best_learning_epoch_training_loss)), best_learning_epoch_training_loss, \
            label=("Training loss achieved with best training rate " + str(best_learning_rate)))

plt.plot(np.arange(len(best_learning_epoch_validation_loss)), best_learning_epoch_validation_loss, \
            label=("Validation loss achieved with best training rate " + str(best_learning_rate)))

plt.legend()
plt.show()
