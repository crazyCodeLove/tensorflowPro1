#coding=utf-8

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from tools import getdatafromgnt
from PIL import Image

traindirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1trn_gnt"
testdirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1tst_gnt"

save_file = "/home/allen/work/variableSave/simplecnn4layer/simplecnn4layer.ckpy"

#set global features
train_step_nums = 500000000001
character_class = 3755
batch_nums = 20

img_width = 32
img_depth = 1

des_acc = 0.8


keep_rate = 0.7
keep_prob = tf.placeholder(tf.float32)

learning_rate = 1e-4
alpha = 1e-2

def get_accurate(prediction,labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.arg_max(labels,1)),dtype=tf.float32))


def add_fc_layer(inputs, inFeatures, outFeatures, layerName="layer", activateFunc=None, keepProb=-1):
    with tf.name_scope(layerName):
        Weights = tf.Variable(tf.truncated_normal([inFeatures, outFeatures], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outFeatures]))

        y = tf.matmul(inputs,Weights) + biases
        if activateFunc is None:
            outputs = y
        else:
            outputs = activateFunc(y)

        if keepProb != -1:
            outputs = tf.nn.dropout(outputs, keepProb)
        return outputs,Weights

def add_conv_layer(inputs, kernalWidth, inDepth, outDepth, layername="layer", activateFunc=None,keep_prob=-1, stride=[1, 1, 1, 1]):
    with tf.name_scope(layername):
        Weights = tf.Variable(tf.truncated_normal([kernalWidth, kernalWidth, inDepth, outDepth], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outDepth]))

        y = tf.nn.conv2d(inputs,Weights,stride,padding='SAME') + biases
        if activateFunc is None:
            outputs = y
        else:
            outputs = activateFunc(y)

        if keep_prob!=-1:
            outputs = tf.nn.dropout(outputs,keep_prob=keep_prob)

        return outputs

def add_pool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.max_pool(inputs,kernal,strides=kernal,padding='SAME')

def conv2fc(inputs):
    conv_out_shape = inputs.get_shape().as_list()
    fcl_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    fcl_inputs = tf.reshape(inputs, [-1, fcl_in_features])
    return fcl_inputs,fcl_in_features


#create graph start
    #set conv net layer 1 features
in_features = img_width*img_width*img_depth

xp = tf.placeholder(tf.float32,shape=[None,in_features])
xp_reshape = tf.reshape(xp,[-1,img_width,img_width,img_depth])

yp = tf.placeholder(tf.float32,[None,character_class])

cl1_depth = 30
kernal_width = 3
    #conv_layer1 is 32*32*cl1_depth
conv_layer1 = add_conv_layer(xp_reshape,kernal_width,img_depth,cl1_depth,keep_prob=keep_prob)

    #set pool layer1 features
    #pooling layer 1 is 16*16*cl1_depth
pool_layer1 = add_pool_layer(conv_layer1)



    #set conv layer 2 features
cl2_depth = 60
kernal_width = 2
    #conv_layer2 is 32*32*cl2_depth
conv_layer2 = add_conv_layer(pool_layer1,kernal_width,cl1_depth,cl2_depth,keep_prob=keep_prob)

    #set pooling layer 2 features
    #pool_layer2 is 8*8*cl2_features
pool_layer2 = add_pool_layer(conv_layer2)

    #set conv layer 3 features
cl3_depth = 90
kernal_width = 2
    #conv_layer3 is 16*16*cl3_depth
conv_layer3 = add_conv_layer(pool_layer2,kernal_width,cl2_depth,cl3_depth,keep_prob=keep_prob)

    #set pooling layer 3 features
    #pool_layer3 is 8*8*cl3_depth
pool_layer3 = add_pool_layer(conv_layer3)

    #set conv layer 4 features
cl4_depth = 120
kernal_width = 2
    #conv_layer4 is 8*8*cl4_depth
conv_layer4 = add_conv_layer(pool_layer3,kernal_width,cl3_depth,cl4_depth,keep_prob=keep_prob)

    #set pooling layer 4 features
    #pool_layer4 is 4*4*cl4_depth
pool_layer4 = add_pool_layer(conv_layer4)

    #set conv layer 5 features
cl5_depth = 150
kernal_width = 2
    #conv_layer5 is 4*4*cl5_depth
conv_layer5 = add_conv_layer(pool_layer4,kernal_width,cl4_depth,cl5_depth,keep_prob=keep_prob)


    #set fully connected layer 1 features

fcl1_inputs,fcl1_in_features = conv2fc(conv_layer5)

# conv_out_shape = conv_layer5.get_shape().as_list()
# fcl1_in_features = conv_out_shape[1]*conv_out_shape[2]*conv_out_shape[3]
# fcl1_inputs = tf.reshape(conv_layer5,[-1,fcl1_in_features])
fcl1_features = 4000
    #fc_layer1 features is
fc_layer1,Weights1 = add_fc_layer(fcl1_inputs,fcl1_in_features,fcl1_features,keepProb=keep_prob)

    #set fully connected layer 2 features
# fcl2_features = 10000
# fc_layer2 = add_fc_layer(fc_layer1,fcl1_features,fcl2_features,activateFunc=tf.nn.relu,keepProb=keep_prob)

    #set outputs features
outputs_features = 3755
outputs,Weights2 = add_fc_layer(fc_layer1,fcl1_features,outputs_features,keepProb=keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs,yp)) \
       + alpha*(tf.nn.l2_loss(Weights1) + tf.nn.l2_loss(Weights2))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()
#create graph end

#start training

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

with tf.Session() as sess:
    cur_acc = 0
    i = 0

    # sess.run(init)

    saver.restore(sess,save_file)

    train_gen = getdatafromgnt.next_batch(batch_nums,traindirname)
    test_gen = getdatafromgnt.next_batch(100,testdirname)



    while cur_acc<des_acc:
        train_x, train_y = train_gen.next()
        #
        # chars = np.reshape(train_x,[-1,32,32])
        # for each in chars:
        #     plt.figure()
        #     plt.imshow(each)
        #     plt.show()


        lo,ou,_ = sess.run([loss,outputs,train_step],feed_dict={xp:train_x,yp:train_y,keep_prob:keep_rate})
        i += 1
        if i % 500 == 0:

            cur_acc = sess.run(get_accurate(ou,train_y))
            print "train:%5d th, loss is %.5f, accurate is %.5f"%(i,lo,cur_acc)

        # if i % 500 == 0:
        #     test_x,test_y = test_gen.next()
        #     ou = sess.run(outputs,feed_dict={xp:test_x,yp:test_y,keep_prob:-1})
        #     print "test:%5d th, accurate is %.5f"%(i,sess.run(get_accurate(ou,test_y)))

        if i % 5000 == 0:
            saver.save(sess,save_file)

        if i%10000000 == 0:
            i = 0














