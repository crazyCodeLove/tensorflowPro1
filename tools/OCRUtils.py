#coding=utf-8
"""
MyLog is to log result
learning_rate_down is to down learning rate

"""
import logging
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm


class MyLog(object):
    logfile = ""
    logger = None

    def __init__(self,logfile):
        self.logfile = logfile
        filehandler = logging.FileHandler(filename=logfile,encoding='utf-8')
        fmter = logging.Formatter(fmt="%(asctime)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
        filehandler.setFormatter(fmter)
        loger = logging.getLogger(__name__)
        loger.addHandler(filehandler)
        loger.setLevel(logging.DEBUG)
        self.logger = loger

    def log_message(self,msg):
        self.logger.debug(msg)



def get_accurate(prediction,labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.arg_max(labels,1)),dtype=tf.float32))


def get_test_right_num(prediction,labels):
    return tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.arg_max(labels,1)),dtype=tf.float32))

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



def add_conv_layer(inputs, kernalWidth, inDepth, outDepth,
                   layername="layer", activateFunc=None,
                   keep_prob=-1, stride=[1, 1, 1, 1]):
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

        return outputs,Weights

def add_BN_conv_layer(inputs, kernalWidth, inDepth, outDepth,
                      is_training_ph, scope = None, layername="layer",
                      activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):

    with tf.name_scope(layername):
        Weights = tf.Variable(tf.truncated_normal([kernalWidth, kernalWidth, inDepth, outDepth], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outDepth]))

        y = tf.nn.conv2d(inputs, Weights, stride, padding='SAME') + biases


        outputs = tf.cond(is_training_ph,
                           lambda: batch_norm(y,decay=0.92, is_training=True,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope),
                           lambda: batch_norm(y,decay=0.92, is_training=False,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope,
                                              reuse=True))

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

def down_learning_rate(test_acc, lr):

    if test_acc >=0.8 and lr>5e-4:
        lr /= 5.0
    elif test_acc>0.8 and lr>5e-5:
        lr *= 0.8
    elif test_acc>0.9:
        lr *= 0.9

    return lr


def test():
    # loger = MyLog("/home/allen/work/temp/test.txt")
    # loger.log_message("nice to meet you")
    num = 0

    lr = 2e-3
    acc = 0.85
    while True:
        num += 1
        lr = down_learning_rate(acc,lr)
        print num,lr
        if lr < 1e-4:
            acc = 0.95

        if lr < 1e-5:
            acc = 0.96




if __name__ == "__main__":
    test()


