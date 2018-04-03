import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optionmizer=tf.train.AdagradOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._init_weight()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        mat = tf.matmul(self.x + self.scale * tf.random_uniform(n_input,), self.weights['w1'])

        self.n_hidden = self.transfer(tf.add(mat, self.weights['b1']))
        self.reconstruction = tf.add(tf.hidden, self.weights['w2'], self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))




def vaxvier_init(fan_in, fan_out, constant = 1):
    low = constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
