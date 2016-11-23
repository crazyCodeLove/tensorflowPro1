#coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#set global variable
img_size = 28
img_depth = 1
batch_size = 20
patch_size = 3


#create data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)


def get_accuracy(prediction, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction, 1), tf.arg_max(labels, 1)), tf.float32))


def add_conv_layer(inputs, in_depth, out_depth, activate_function = None):
    #kernal is 3*3 block
    Weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, in_depth, out_depth],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_depth]))
    strides = [1,1,1,1]
    y = tf.nn.conv2d(inputs, Weights, strides, padding='SAME') + biases

    if activate_function is None:
        outputs = y
    else:
        outputs = activate_function(y)

    return outputs


def max_pool(inputs):
    return tf.nn.max_pool(inputs,ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")


def add_fc_layer(inputs, in_features, out_features, layer_name=None, activate_function=None, keep_prob=-1):
    Weights = tf.Variable(tf.truncated_normal([in_features, out_features],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[out_features]))
    y = tf.matmul(inputs, Weights) + biases

    if activate_function is None:
        outputs = y
    else:
        outputs = activate_function(y)

    if keep_prob == -1:
        return outputs
    else:
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        return outputs



#create graph start
    #set conv layer 1 features
xp = tf.placeholder(tf.float32, [None,img_size*img_size*img_depth])
xp_reshape = tf.reshape(xp, [-1, img_size, img_size, img_depth])

yp = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


cl1_depth = 8
#conv_layer1 is 28*28*8
conv_layer1 = add_conv_layer(xp_reshape, img_depth, cl1_depth, activate_function=tf.nn.relu)

    #set conv layer 2 features
cl2_depth = 16
#conv_layer2 is 28*28*16
conv_layer2 = add_conv_layer(conv_layer1,cl1_depth,cl2_depth,activate_function=tf.nn.relu)
#pool_layer2 is 14*14*16
pool_layer2 = max_pool(conv_layer2)

cl3_depth = 32
#conv_layer3 is 14*14*32
conv_layer3 = add_conv_layer(pool_layer2,cl2_depth,cl3_depth,activate_function=tf.nn.relu)

cl4_dipth = 64
#conv_layer4 is 14*14*64
conv_layer4 = add_conv_layer(conv_layer3,cl3_depth,cl4_dipth,activate_function=tf.nn.relu)
#pool_layer4 is 7*7*64
pool_layer4 = max_pool(conv_layer4)


#fully connected layer
    #set fc layer 1 features
fcl1_in_features = 7*7*64
fcl1_inputs = tf.reshape(pool_layer4,[-1,fcl1_in_features])
fcl1_features = 1024


fc_layer1 = add_fc_layer(fcl1_inputs,fcl1_in_features, fcl1_features,activate_function=tf.nn.relu,keep_prob=keep_prob)

#set fc layer2 features
outputs_features = 10
outputs = add_fc_layer(fc_layer1,fcl1_features,outputs_features,keep_prob=keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs,yp))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    for i in xrange(1000):
        train_x,train_y = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={xp:train_x, yp:train_y, keep_prob:0.6})

        if i % 50 == 0:
            test_x,test_y = mnist.test.images,mnist.test.labels
            rs = sess.run(outputs,feed_dict={xp:test_x, keep_prob:-1})
            print sess.run(get_accuracy(rs,test_y))








