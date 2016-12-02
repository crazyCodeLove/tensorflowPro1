#coding=utf-8

import tensorflow as tf
import numpy as np

from tools import getdatafromgnt

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


keep_rate = 0.6
keep_prob = tf.placeholder(tf.float32)

learning_rate = 1e-4


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
        return outputs

def add_conv_layer(inputs, kernalWidth, inDepth, outDepth, layername="layer", activateFunc=None, stride=[1, 1, 1, 1]):
    with tf.name_scope(layername):
        Weights = tf.Variable(tf.truncated_normal([kernalWidth, kernalWidth, inDepth, outDepth], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outDepth]))

        y = tf.nn.conv2d(inputs,Weights,stride,padding='SAME') + biases
        if activateFunc is None:
            outputs = y
        else:
            outputs = activateFunc(y)

        return outputs

def add_pool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.max_pool(inputs,kernal,strides=kernal,padding='SAME')


#create graph start
    #set conv net layer 1 features
in_features = img_width*img_width*img_depth

xp = tf.placeholder(tf.float32,shape=[None,in_features])
xp_reshape = tf.reshape(xp,[-1,img_width,img_width,img_depth])

yp = tf.placeholder(tf.float32,[None,character_class])

cl1_depth = 10
kernal_width = 3
    #conv_layer1 is 32*32*10
conv_layer1 = add_conv_layer(xp_reshape,kernal_width,img_depth,cl1_depth,activateFunc=tf.nn.relu)

    #set conv layer 2 features
cl2_depth = 100
kernal_width = 3
    #conv_layer2 is 32*32*128
conv_layer2 = add_conv_layer(conv_layer1,kernal_width,cl1_depth,cl2_depth,activateFunc=tf.nn.relu)

    #set pooling layer 1 features
    #pool_layer1 is 16*16*100
pool_layer1 = add_pool_layer(conv_layer2)

    #set conv layer 3 features
cl3_depth = 300
kernal_width = 3
    #conv_layer3 is 16*16*256
conv_layer3 = add_conv_layer(pool_layer1,kernal_width,cl2_depth,cl3_depth,activateFunc=tf.nn.relu)

    #set conv layer 4 features
cl4_depth = 300
kernal_width = 3
    #conv_layer4 is 16*16*256
conv_layer4 = add_conv_layer(conv_layer3,kernal_width,cl3_depth,cl4_depth,activateFunc=tf.nn.relu)

    #set pool layer 2 features
    #pool_layer2 is 8*8*256
pool_layer2 = add_pool_layer(conv_layer4)


    #set fully connected layer 1 features

pl2_shape = pool_layer2.get_shape().as_list()
fcl1_in_features = pl2_shape[1]*pl2_shape[2]*pl2_shape[3]
fcl1_inputs = tf.reshape(pool_layer2,[-1,fcl1_in_features])
fcl1_features = 5000
    #fc_layer1 features is 30000
fc_layer1 = add_fc_layer(fcl1_inputs,fcl1_in_features,fcl1_features,activateFunc=tf.nn.relu,keepProb=keep_prob)

    #set fully connected layer 2 features
# fcl2_features = 10000
# fc_layer2 = add_fc_layer(fc_layer1,fcl1_features,fcl2_features,activateFunc=tf.nn.relu,keepProb=keep_prob)

    #set outputs features
outputs_features = 3755
outputs = add_fc_layer(fc_layer1,fcl1_features,outputs_features,keepProb=keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs,yp))
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

        lo,ou,_ = sess.run([loss,outputs,train_step],feed_dict={xp:train_x,yp:train_y,keep_prob:keep_rate})
        i += 1
        if i % 50 == 0:

            cur_acc = sess.run(get_accurate(ou,train_y))
            print "train:%5d th, loss is %.5f, accurate is %.5f"%(i,lo,cur_acc)

        # if i % 500 == 0:
        #     test_x,test_y = test_gen.next()
        #     ou = sess.run(outputs,feed_dict={xp:test_x,yp:test_y,keep_prob:-1})
        #     print "test:%5d th, accurate is %.5f"%(i,sess.run(get_accurate(ou,test_y)))

        if i % 2000 == 0:
            saver.save(sess,save_file)














