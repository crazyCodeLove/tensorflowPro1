#coding=utf-8

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

#create data
x_train = np.linspace(-1, 1, 400)[:, np.newaxis]
noise = np.random.normal(0,0.05, x_train.shape)
y_train = np.square(x_train) - 0.5 + noise




def add_layer(inputs, in_feature_size, out_feature_size,layer_num, activate_function = None):
    layer_name = "layer%d" % layer_num
    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weights = tf.Variable(tf.random_normal([in_feature_size,out_feature_size]),name='W')
            tf.histogram_summary(layer_name+"/Weights", Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.ones([1, out_feature_size]),name='b')
            tf.histogram_summary(layer_name + "/biases", biases)
        with tf.name_scope('y'):
            y = tf.matmul(inputs,Weights) + biases

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        tf.histogram_summary(layer_name+"/outputs",outputs)
        return outputs


#create graph start
#set hidden layer 1 features
inputs_features = 1
hl1_features = 10
with tf.name_scope('trainX'):
    xp = tf.placeholder(tf.float32,[None,inputs_features],"inputs")

hidden_layer1 = add_layer(xp, inputs_features, hl1_features,layer_num = 2 , activate_function = tf.nn.relu)

#set hidden layer 2 features
# hl2_features = 5
# hidden_layer2 = add_layer(hidden_layer1, hl1_features, hl2_features, activate_function= tf.nn.relu)

#set outputs features
outputs_features = 1
with tf.name_scope('trainY'):
    yp = tf.placeholder(tf.float32,[None,outputs_features])

outputs = add_layer(hidden_layer1,hl1_features, outputs_features, layer_num = 3)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(outputs - yp), reduction_indices=[1]))
    tf.scalar_summary("loss",loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

merged = tf.merge_all_summaries()
init = tf.initialize_all_variables()
#create graph end

#start training
sess = tf.Session()
logdir = "/home/allen/work/tensorlog/"



writer = tf.train.SummaryWriter(logdir,sess.graph)
sess.run(init)

for i in xrange(1000):
    sess.run(train_step,feed_dict={ xp : x_train, yp : y_train})
    if i % 40 == 0 :
        result = sess.run(merged, feed_dict={ xp : x_train, yp : y_train})
        writer.add_summary(result,i)


#end training