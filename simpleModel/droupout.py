#coding=utf-8

import tensorflow as tf
import numpy as np

baseLogDir = "/home/allen/work/tensorlog"


#create data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot= True)





def add_layer(inputs, in_featurs, out_features, layer_name, activate_function = None):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([in_featurs,out_features]), name= "W")
        biases = tf.Variable(tf.ones([1,out_features]),name="b")
        y = tf.matmul(inputs, Weights) + biases
        y = tf.nn.dropout(y, keep_prob)

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        tf.histogram_summary(layer_name+"/outputs",outputs)
        return outputs


#create graph
    #set hidden layer 1 features
inputs_features = 784
hl1_features = 400

xp = tf.placeholder(tf.float32,[None, inputs_features])
keep_prob = tf.placeholder(tf.float32)


hidden_layer1 = add_layer(xp, inputs_features, hl1_features, "hidden1", activate_function = tf.nn.tanh)

    #set hidden layer 2 features
hl2_features = 100
hidden_layer2 = add_layer(hidden_layer1, hl1_features, hl2_features, "hidden2", activate_function= tf.nn.tanh)




    #set outputs layer features
outputs_features = 10
outputs = add_layer(hidden_layer2, hl2_features, outputs_features, "outputs", activate_function= tf.nn.softmax)

#define loss function
yp = tf.placeholder(tf.float32,[None,outputs_features])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(outputs),reduction_indices=[1]))
tf.scalar_summary("loss",cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
merged = tf.merge_all_summaries()

sess = tf.Session()


#summary writer define here
train_writer = tf.train.SummaryWriter(baseLogDir+"/train",sess.graph)
test_writer = tf.train.SummaryWriter(baseLogDir+"/test",sess.graph)

init = tf.initialize_all_variables()
sess.run(init)
#create graph end
try:
    for i in xrange(1001):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={ xp : batch_x, yp : batch_y, keep_prob : 1})
        print sess.run(cross_entropy, feed_dict={ xp : batch_x, yp : batch_y, keep_prob : 1})

        if i % 50 == 0:
            batch_x, batch_y = mnist.train.next_batch(50)
            train_result = sess.run(merged, feed_dict= {xp : batch_x, yp : batch_y, keep_prob : 1})
            test_result = sess.run(merged, feed_dict= {xp : mnist.test.images, yp: mnist.test.labels, keep_prob : 1})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)
finally:
    sess.close()
