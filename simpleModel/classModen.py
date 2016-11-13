#coding = utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_features, out_features, layer_name, activate_function = None):
    Weight = tf.Variable(tf.random_normal([in_features, out_features]))
    biases = tf.Variable(tf.zeros([1,out_features]))
    y = tf.matmul(inputs, Weight) + biases

    if activate_function is None:
        outputs = y
    else:
        outputs = activate_function(y)
    return outputs,Weight

def get_accuracy(predictions, lables):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(lables, 1)), tf.float32))


#create data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)


#create graph
    #set hidden layer 1 features
inputs_features = 784
outputs_features = 10

xp = tf.placeholder(tf.float32, [None, inputs_features])
outputs,Weights = add_layer(xp, inputs_features, outputs_features, "outputs")

yp = tf.placeholder(tf.float32,[None, outputs_features])
#define loss function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(outputs),reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

    #use regularization to loss
# loss = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(outputs),reduction_indices=[1])) + 0.01 * tf.nn.l2_loss(Weights)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, yp) + 0.01 * tf.nn.l2_loss(Weights))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


# loss = tf.reduce_mean(tf.reduce_sum(tf.square(yp-outputs),reduction_indices=1))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()


sess = tf.Session()
try:
    sess.run(init)
    for i in xrange(10001):
        train_x, train_y = mnist.train.next_batch(120)
        sess.run([train_step], feed_dict={ xp : train_x, yp : train_y})
        if i % 50 == 0:
            test_x, test_y = mnist.test.next_batch(2000)
            test_result = sess.run(outputs, feed_dict={ xp : test_x})
            print i,sess.run(get_accuracy(test_result, test_y))



finally:
    sess.close()