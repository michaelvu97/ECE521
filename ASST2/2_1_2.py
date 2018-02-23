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
learning_rate = 0.001
epoch_size = 7

################################################################################

trainData = np.reshape(trainData, [3500, -1])
validData = np.reshape(validData, [100, -1])
testData = np.reshape(testData, [testData.shape[0], -1])

dimension = trainData.shape[1]

X = tf.placeholder(tf.float64); # [Nxd]
Y = tf.placeholder(tf.float64); # [N]

training_set = {
    X: trainData,
    Y: trainTarget
}

validation_set = {
    X: validData,
    Y: validTarget
}

test_set = {
    X: testData,
    Y: testTarget
}

################################################################################
"""
ADAM OPTIMIZER
"""
################################################################################

w = tf.Variable(tf.zeros([dimension,1], dtype=tf.float64), name="weights", dtype=tf.float64)

b = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64, name="bias")

# Xw + b, no sigmoid since that is handled in the cross entropy function
yhat = tf.add(tf.matmul(X, w), b)

Loss_Data = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))

Loss_Weights = (lambda_weight_penalty / tf.constant(2.0, dtype=tf.float64)) * tf.reduce_sum(w**2)

Loss = Loss_Data + Loss_Weights



Accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, tf.round(tf.sigmoid(yhat))), tf.float64))

optimizers = [
    tf.train.AdamOptimizer(learning_rate).minimize(Loss),
    tf.train.GradientDescentOptimizer(learning_rate).minimize(Loss)
]

optimizer_names = ["ADAM", "SGD"]

epoch_training_losses = [[] for i in range(len(optimizers))]

optimizer_index = 0

for optimizer in optimizers:

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

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
            epoch_training_losses[optimizer_index] \
                    .append(sess.run(Loss, feed_dict=training_set))

    optimizer_index = optimizer_index + 1

plt.figure(1)

for i in range(len(optimizers)):
    plt.plot(epoch_training_losses[i], label=("Training loss: " + str(optimizer_names[i])))

plt.legend()
plt.title("Best Training Loss, Learning Rate = " + str(learning_rate))
plt.show()
