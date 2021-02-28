import tensorflow as tf
import numpy as np


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        # Initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward propagate the inputs
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output


# TensorFlow has actually implemented this dense
# layer, so we don't need to do that from scratch
# instead, we can just call like this:
# 2 outputs.

# layer = tf.keras.layers.Dense(units=2)

# n1 = 5
# n2 = 65
# model = tf.keras.Sequential([
#    tf.keras.layers.Dense(n1),
#    tf.keras.layers.Dense(n2),
#    #
#    # hidden layers
#    #
#    tf.keras.layers.Dense(2)  # 2 output neurons.
# ])
# #################################### #
# Example Problem: Will I pass this class? Simple two feature model
# Data: x1 := number of lectures I attended.
#       x2 := hours spent on the final project.

# My x vector iz [4, 5]. But this is not enough to train a model. I need data of a lot of students.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(2)
])
# This is a binary classification problem. We can use what is called a soft max cross entropy loss
# cross entropy bw/ two prob. distr., it measures how far apart the ground truth prob. distr. is
# from the predicted prob. distr.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predicted))
# If I want to find out the grade, I will use another loss func. because the output type changed.
# Mean Squared Error Loss
loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted))) # OR...
loss = tf.keras.losses.MSE(y, predicted)

# Training NNs
# find the weights of the nn that will min. the loss of our data set.
# Loss optimization, Gradient Descent:

weights = tf.Variable([tf.random.normal()])

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights) # backpropagation

    weights = weights - lr * graident
