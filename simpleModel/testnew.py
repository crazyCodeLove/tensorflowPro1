#coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28

#define add layer function
def add_layer(inputs, in_features, out_features, layer_name, keep_prob = -1.0, activate_function = None):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.truncated_normal([in_features, out_features], stddev=(np.sqrt(1./(img_size*img_size)))))
        biases = tf.Variable(tf.zeros([out_features]))

        y = tf.matmul(inputs, Weights) + biases
        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        if keep_prob == -1:
            return outputs,Weights
        else:
            outputs = tf.nn.dropout(outputs, keep_prob)
            return outputs,Weights

def get_accuracy(outputs, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(outputs,1),tf.arg_max(labels,1)),dtype=tf.float32))

#create data
mnist = input_data.read_data_sets("MNIST_DATA/",one_hot=True)

#create graph
    #set hidden layer 1 features
keep_prob = tf.placeholder(tf.float32)
xp = tf.placeholder(tf.float32, [None, (img_size*img_size)])
hl1_features = img_size * img_size

hidden1,Weights1 = add_layer(xp, img_size*img_size, hl1_features, "hidden1", keep_prob = keep_prob, activate_function=tf.nn.relu)

    #set outputs features
outputs_features = 10
outputs,Weights2 = add_layer(hidden1, hl1_features, outputs_features, "outputs", keep_prob=keep_prob)

    #set loss placeholder
yp = tf.placeholder(tf.float32)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, yp)) \
       + 0.001*(tf.nn.l2_loss(Weights1) + tf.nn.l2_loss(Weights2))

train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
#setart training
for i in xrange(10001):
    train_x,train_y = mnist.train.next_batch(120)
    sess.run(train_step,feed_dict={xp:train_x, yp:train_y, keep_prob:0.6})
    if i % 50 == 0:
        test_x,test_y = mnist.test.next_batch(5000)
        outs = sess.run(outputs,feed_dict={xp:test_x, yp:test_y, keep_prob:-1})
        print "%5d staps, accuracy is %.5f"% (i,sess.run(get_accuracy(outs, test_y)))


sess.close()




