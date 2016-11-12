#coding=utf-8

import tensorflow as tf
import numpy as np


#define a add layer function
def add_layer(inputs, in_feature_size, out_feature_size, activate_function = None):
    Weights = tf.Variable(tf.random_normal([in_feature_size,out_feature_size]))
    biases = tf.Variable(tf.ones([1,out_feature_size]))
    y = tf.matmul(inputs, Weights) + biases

    if activate_function is None:
        outputs = y
    else:
        outputs = activate_function(y)

    return outputs





#create data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True )


#create graph start
    #set outputs layer features
hl1_features = 10
inputs_features = 784

xp = tf.placeholder(tf.float32,[None,inputs_features])

outputs = add_layer(xp, inputs_features, hl1_features, tf.nn.softmax)


yp = tf.placeholder(tf.float32,[None, hl1_features])

#define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(outputs),reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

init = tf.initialize_all_variables()
#create graph end


sess = tf.Session()
sess.run(init)
for i in xrange(10000):
    batch_x, batch_y = mnist.train.next_batch(120)
    sess.run(train_step, feed_dict= {xp : batch_x, yp : batch_y})
    if i % 50 == 0:
        # print compute_accuacy(mnist.test.images, mnist.test.labels)
        ypre = sess.run(outputs, feed_dict={xp: mnist.test.images})
        correct_prediction = tf.equal(tf.argmax(ypre, 1), tf.argmax(yp, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype= tf.float32))
        print sess.run(accuracy, feed_dict={xp: mnist.test.images, yp : mnist.test.labels })


sess.close()