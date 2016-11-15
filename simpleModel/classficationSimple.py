#coding=utf-8

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


#define add layer function
def add_layer(inputs, in_features, out_features, layer_name, activate_function = None, keep_prob = None):
    """

    :param inputs: inputs data, numpy.ndarray
    :param in_features: inputs data feature size
    :param out_features: outputs data feature size
    :param layer_name: this layer name to show
    :param activate_function: set activate function
    :param keep_prob: if there is drop out layer, set keep probability
    :return: outputs data and this layer weight
    """
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([in_features, out_features]))
        biases = tf.Variable(tf.zeros([1, out_features]))
        y = tf.matmul(inputs, Weights) + biases

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        if keep_prob is None:
            return outputs,Weights
        else:
            outputs = tf.nn.dropout(outputs, keep_prob)
            return outputs,Weights

#define calculate accuracy
def get_accuracy(predictions, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32))

#create data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

#create graph
    #set layer 1 features
inputs_features = 784
outputs_features = 10

xp = tf.placeholder(tf.float32, [None, inputs_features])
outputs,Weights = add_layer(xp, inputs_features, outputs_features, "outputs", tf.nn.softmax)

#set outputs features
yp = tf.placeholder(tf.float32, [None, outputs_features])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(yp - outputs),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init = tf.initialize_all_variables()
#create graph end

with tf.Session() as sess:
    sess.run(init)

    for i in xrange(10001):
        train_x, train_y = mnist.train.next_batch(120)
        sess.run(train_step, feed_dict={ xp : train_x, yp : train_y})

        if i % 50 == 0 :
            test_x, test_y = mnist.test.next_batch(3000)
            outs,los =  sess.run([outputs, loss], feed_dict={xp : test_x, yp : test_y})
            print "%d loss:%.6f, accuracy:%.6f" % (i, los, sess.run(get_accuracy(outs, test_y)))
