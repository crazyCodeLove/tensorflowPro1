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
inputs_features = 1
hl1_features = 5
xp = tf.placeholder(tf.float32,[None,inputs_features],"inputs")

hidden_layer1 = add_layer(xp, inputs_features, hl1_features, tf.nn.relu)


#set outputs features
outputs_features = 1
yp = tf.placeholder(tf.float32,[None,outputs_features])

outputs = add_layer(hidden_layer1,hl1_features, outputs_features)

init = tf.initialize_all_variables()
#create graph end

#start training
sess = tf.Session()
sess.run(init)


sess.close()
#end training