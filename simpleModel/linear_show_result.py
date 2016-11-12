#coding=utf-8

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

#create data
x_train = np.linspace(-1, 1, 400)[:, np.newaxis]
noise = np.random.normal(0,0.05, x_train.shape)
y_train = np.square(x_train) - 0.5 + noise


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_train, y_train)
plt.ion()   #pyplot show过后防止show 后程序暂停

plt.show()



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
hl1_features = 10
xp = tf.placeholder(tf.float32,[None,inputs_features],"inputs")

hidden_layer1 = add_layer(xp, inputs_features, hl1_features, activate_function = tf.nn.relu)

#set hidden layer 2 features
# hl2_features = 5
# hidden_layer2 = add_layer(hidden_layer1, hl1_features, hl2_features, activate_function= tf.nn.relu)

#set outputs features
outputs_features = 1
yp = tf.placeholder(tf.float32,[None,outputs_features])

outputs = add_layer(hidden_layer1,hl1_features, outputs_features)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(outputs - yp), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
#create graph end

#start training
sess = tf.Session()
sess.run(init)

for i in xrange(5000):
    if i % 100 == 0:
        print sess.run(loss, feed_dict={xp : x_train, yp : y_train})
        try:
            ax.lines.remove(lines[0])
        except Exception,e:
            pass

        prediction_value = sess.run(outputs, feed_dict={xp: x_train})
        lines = ax.plot(x_train, prediction_value, 'r-', lw=2)
        plt.pause(0.1)

    sess.run(train_step, feed_dict={xp : x_train, yp : y_train})


#end training