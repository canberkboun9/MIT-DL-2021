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
layer = tf.keras.layers.Dense(units=2)
n1 = 5
n2 = 65
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n1),
    tf.keras.layers.Dense(n2),
    #
    # hidden layers
    #
    tf.keras.layers.Dense(2)  # 2 output neurons.
])

