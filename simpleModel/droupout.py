#coding=utf-8
"""
udacity tutorial regularization question answer


"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

#set global variable

baseLogDir = "/home/allen/work/tensorlog"
beta_regul = 1e-3
batch_size = 120
num_steps = 5001
picture_size = 28


#define add layer function
def add_layer(inputs, in_features, out_features, layer_name, activate_function = None, keep_prob = None):
    """

    :param inputs: inputs data, numpy.ndarray
    :param in_features: inputs data feature size
    :param out_features: outputs data feature size
    :param layer_name: this layer name to show
    :param activate_function: set activate function
    :param keep_prob: if there is drop out layer, set keep probability.
                        if is -1, means in training;
                        if 0-1, means test or validate
    :return: outputs data and this layer weight
    """
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.truncated_normal([in_features, out_features],
                                                  stddev=np.sqrt(1./(picture_size*picture_size))))
        biases = tf.Variable(tf.zeros([1, out_features]))
        y = tf.matmul(inputs, Weights) + biases

        if activate_function is None:
            outputs = y
        else:
            outputs = activate_function(y)

        tf.histogram_summary(layer_name + "/outputs", outputs)

        if -1 == keep_prob:
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
inputs_features = picture_size * picture_size
hl1_features = 784


keep_prob = tf.placeholder(tf.float32)

xp = tf.placeholder(tf.float32, [None, inputs_features])
hidden_layer1,Weights1 = add_layer(xp, inputs_features, hl1_features,
                            "hidden1",activate_function=tf.nn.relu,  keep_prob=keep_prob)

    #set hidden layer 2 features
hl2_features = 784
hidden_layer2,Weights2 = add_layer(hidden_layer1, hl1_features, hl2_features,
                            "hidden2", activate_function=tf.nn.relu, keep_prob=keep_prob)

#set outputs layer features
outputs_features = 10
outputs,Weights3 = add_layer(hidden_layer2, hl2_features, outputs_features, "outputs", keep_prob=keep_prob)



#set outputs features
yp = tf.placeholder(tf.float32, [None, outputs_features])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, yp) )\
       +beta_regul * (tf.nn.l2_loss(Weights1) + tf.nn.l2_loss(Weights2)+ tf.nn.l2_loss(Weights3))
tf.scalar_summary("loss", loss)

train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
merged_summary = tf.merge_all_summaries()
init = tf.initialize_all_variables()
#create graph end

with tf.Session() as sess:
    # summary writer define here
    # train_writer = tf.train.SummaryWriter(baseLogDir + "/train", sess.graph)
    test_writer = tf.train.SummaryWriter(baseLogDir + "/test", sess.graph)

    sess.run(init)

    for i in xrange(num_steps):
        train_x, train_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={ xp : train_x, yp : train_y, keep_prob : 0.6})

        if i % 50 == 0 :
            test_x, test_y = mnist.test.next_batch(3000)
            feed_dict = {xp : test_x, yp : test_y, keep_prob : -1}
            test_result,outs,los =  sess.run([merged_summary,outputs, loss], feed_dict=feed_dict)
            print "%5d   loss:%.6f   accuracy:%.6f" % (i, los, sess.run(get_accuracy(outs, test_y)))
            test_writer.add_summary(test_result,i)
