#coding=utf-8

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

#create data
x_train = np.linspace(-1, 1, 1000)[:, np.newaxis]
y_train = np.square(x_train)



def add_layer(inputs, in_feature_size, out_feature_size, activate_function = None):
    Weights = tf.Variable(tf.random_normal([in_feature_size,out_feature_size]))
    biases = tf.Variable(tf.ones([1, out_feature_size]))
    y = tf.matmul(inputs,Weights) + biases

    if activate_function is None:
        outputs = y
    else:
        outputs = activate_function(y)

    return outputs


#create graph start
    #set hidden layer 1 features


#create graph end