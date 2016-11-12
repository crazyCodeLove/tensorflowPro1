#coding=utf-8

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

#create data
x_train = np.linspace(-1, 1, 400)[:, np.newaxis]
noise = np.random.normal(0,0.05, x_train.shape)
y_train = np.square(x_train) - 0.5 + noise




def add_layer(inputs, in_feature_size, out_feature_size, activate_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
            Weights = tf.Variable(tf.random_normal([in_feature_size,out_feature_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.ones([1, out_feature_size]),name='b')
        with tf.name_scope('y'):
            y = tf.matmul(inputs,Weights) + biases

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        return outputs


#create graph start
#set hidden layer 1 features
inputs_features = 1
hl1_features = 10
with tf.name_scope('trainX'):
    xp = tf.placeholder(tf.float32,[None,inputs_features],"inputs")

with tf.name_scope('hidden1'):
    hidden_layer1 = add_layer(xp, inputs_features, hl1_features, activate_function = tf.nn.relu)

#set hidden layer 2 features
# hl2_features = 5
# hidden_layer2 = add_layer(hidden_layer1, hl1_features, hl2_features, activate_function= tf.nn.relu)

#set outputs features
outputs_features = 1
with tf.name_scope('trainY'):
    yp = tf.placeholder(tf.float32,[None,outputs_features])

with tf.name_scope('outputs'):
    outputs = add_layer(hidden_layer1,hl1_features, outputs_features)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(outputs - yp), reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
#create graph end

#start training
sess = tf.Session()
logdir = "/home/allen/work/tensorlog/"
writer = tf.train.SummaryWriter(logdir,sess.graph)
sess.run(init)



#end training