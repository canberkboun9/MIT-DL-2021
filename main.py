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

# Gradient Descent Algos:
# SGD, Adam, Adadelta, Adagrad, RMSProp
# tf.keras.optimizers.SGD, Adam, etc..
# ruder.io/optimizers-gradient-descent/

# Putting it together

model = tf.keras.Sequential([...])

optimizer = tf.keras.optimizer.SGD()

while True:
    prediction = model(x)

    with tf.GradientTape() as tape:

        loss = compute_loss(y, prediction)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Regularization
# to aviod over fitting. types: Dropout, Early Stopping
tf.keras.layers.Dropout(p=0.5)
# LECTURE 2

# RNN Intuition (pseudo code ofc)

my_rnn = RNN()  # Initialize my RNN
hidden_state = [0,0,0,0]  # Init my hidden state

sentence = ["I","love","recurrent","neural"]

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction  # >>> "networks!"

# RNNs from Scratch

class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        # Initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, input_dim])

        # Initialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])

    def call(self, x):
        # Update the hidden state
        self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)

        # Compute the output
        output = self.W_hy * self.h

        # Return the current output and hidden state
        return output, self.h
# This gives a sense of breaks down how we define the forward pass
# through an RNN in code using TF, but conveniently TF has already implemented
# these types of RNN Cells for us


tf.keras.layers.SimpleRNN(rnn_units)

