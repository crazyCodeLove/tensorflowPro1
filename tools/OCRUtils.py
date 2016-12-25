#coding=utf-8
"""
MyLog is to log result
learning_rate_down is to down learning rate

all character
test data all 3755 class, character number is:   533675
train data all 3755 class, character number is: 2144749

test data 100 class, character number is:  14202
train data 100 class, character number is: 56987

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

def add_carriage_zero_pad(same_width_block_num,inputs, kernalWidth, inDepth,
                              is_training_ph, scope=None, layername="layer",
                              activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    由一个dimension increase block和building_block_serial组合在一起,形成一个carriage
    注意：在添加carriage时，会有一个max_pooling层，注意不要超过最大可以pooling的层数
    the depth of outputs is 2*inDepth


    :param same_width_block_num: the number of same width building bolck
    :param inputs:
    :param kernalWidth:
    :param inDepth:
    :param is_training_ph:
    :param scope:
    :param layername:
    :param activateFunc:
    :param stride:
    :return:
    """
    if scope is None:
        raise ValueError('scope should be a string')
    depth = inDepth

    tscope =scope + "pad"
    y1 = building_block_zero_pad(inputs,kernalWidth,depth,
                                 is_training_ph,scope=tscope,layername=layername,
                                 activateFunc=activateFunc,stride=stride)
    tscope = scope + "ser"
    depth *= 2
    outputs = add_building_block_serial(same_width_block_num,y1, kernalWidth, depth,
                              is_training_ph, scope=tscope, layername=layername,
                              activateFunc=activateFunc, stride=stride)

    return outputs

def add_carriage_proj(same_width_block_num,inputs, kernalWidth, inDepth,
                              is_training_ph, scope=None, layername="layer",
                              activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    由一个dimension increase block和building_block_serial组合在一起
    形成一个carriage
    注意：在添加carriage时，会有一个max_pooling层，注意不要超过最大可以pooling的层数
    the depth of outputs is 2*inDepth

    :param same_width_block_num: the number of same width building bolck
    :param inputs:
    :param kernalWidth:
    :param inDepth:
    :param is_training_ph:
    :param scope:
    :param layername:
    :param activateFunc:
    :param stride:
    :return:
    """
    if scope is None:
        raise ValueError('scope should be a string')
    depth = 2*inDepth

    tscope =scope + "pad"
    y1 = building_block_project(inputs,kernalWidth,inDepth,
                                is_training_ph,scope=tscope,layername=layername,
                                activateFunc=activateFunc,stride=stride)

    tscope = scope + "ser"
    outputs = add_building_block_serial(same_width_block_num,y1, kernalWidth, depth,
                              is_training_ph, scope=tscope, layername=layername,
                              activateFunc=activateFunc, stride=stride)

    return outputs



def add_building_block_serial(nums,inputs, kernalWidth, inDepth,
                              is_training_ph, scope=None, layername="layer",
                              activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    将一组same width的building_bolck组合在一起,形成一串
    the depth of outputs is same as inDepth

    :param nums:
    :param inputs:
    :param kernalWidth:
    :param inDepth:
    :param is_training_ph:
    :param scope:
    :param layername:
    :param activateFunc:
    :param stride:
    :return:
    """
    if scope is None:
        raise ValueError('scope should be a string')

    outputs = inputs
    for it in range(nums):
        tscope = scope + "block" + str(it)
        outputs = building_block_same_width(outputs,kernalWidth,inDepth,
                                            is_training_ph,scope=tscope,layername=layername,
                                            activateFunc=activateFunc,stride=stride)
    return outputs





def building_block_same_width(inputs, kernalWidth, inDepth,
                              is_training_ph, scope=None, layername="layer",
                              activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    feature width相同,两个卷积层,一个identity short cut connection层组成
    short cut connection是identity short cut
    the depth of outputs is same as inDepth

    :param inputs:
    :param kernalWidth:
    :param inDepth:
    :param is_training_ph:
    :param scope:
    :param layername:
    :param activateFunc:
    :param stride:
    :return:
    """
    if scope is None:
        raise ValueError('scope should be a string')
    depth = inDepth
    tscope = scope + "layer1"

    y1 = add_BN_conv_layer(inputs, kernalWidth, depth, depth,
                           is_training_ph, tscope,activateFunc=activateFunc)

    tscope = scope + "layer2"
    y2 = add_BN_conv_layer(y1, kernalWidth, depth, depth,
                           is_training_ph, tscope, activateFunc=None)

    hx = tf.add(y2,inputs)
    outputs = tf.nn.relu(hx)
    return outputs

def building_block_zero_pad(inputs,kernalWidth,inDepth,
                            is_training_ph,scope=None, layername="layer",
                            activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    首先是一层max pooling层,两层卷积层,和zero padding short cut connection 层组成
    在增加维度时short cut connection使用zero padding
    the depth of outputs is 2*inDepth

    :param inputs:
    :param kernalWidth:
    :param inDepth:inputs depth
    """
    if scope is None:
        raise ValueError('scope should be a string')
    depth = 2*inDepth
    pool_layer = add_pool_layer(inputs)

    pad_zero = tf.constant(0,dtype=tf.float32,shape=pool_layer.get_shape().as_list())


    tscope = scope + "layer1"
    y1 = add_BN_conv_layer(pool_layer,kernalWidth,inDepth,depth,
                           is_training_ph,tscope,activateFunc=activateFunc)

    tscope = scope + "layer2"
    y2 = add_BN_conv_layer(y1,kernalWidth,depth,depth,
                           is_training_ph,tscope,activateFunc=None)

    hx = tf.add(y2,tf.concat(3,[pool_layer,pad_zero]))
    outputs = tf.nn.relu(hx)
    return outputs


def building_block_project(inputs,kernalWidth,inDepth,
                            is_training_ph,scope=None, layername="layer",
                            activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    首先是一层max pooling层,,两层卷积层,和project short cut connection 层组成
    在增加维度时short cut connection使用projection
    the depth of outputs is 2*inDepth

    """
    if scope is None:
        raise ValueError('scope should be a string')
    depth = 2*inDepth
    pool_layer = add_pool_layer(inputs)

    tscope = scope+"proj"
    proj_layer = add_BN_conv_layer(pool_layer,kernalWidth,inDepth,depth,
                                   is_training_ph,tscope,activateFunc=None)

    tscope = scope + "layer1"
    y1 = add_BN_conv_layer(pool_layer,kernalWidth,inDepth,depth,
                           is_training_ph,tscope,activateFunc=activateFunc)

    tscope = scope + "layer2"
    y2 = add_BN_conv_layer(y1,kernalWidth,depth,depth,
                           is_training_ph,tscope,activateFunc=None)

    hx = tf.add(y2,proj_layer)
    outputs = tf.nn.relu(hx)
    return outputs




def add_BN_conv_layer(inputs, kernalWidth, inDepth, outDepth,
                      is_training_ph, scope , layername="layer",
                      activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):

    with tf.name_scope(layername):
        Weights = tf.Variable(tf.truncated_normal([kernalWidth, kernalWidth, inDepth, outDepth], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outDepth]))

        y = tf.nn.conv2d(inputs, Weights, stride, padding='SAME') + biases


        outputs = tf.cond(is_training_ph,
                           lambda: batch_norm(y,decay=0.94, is_training=True,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope),
                           lambda: batch_norm(y,decay=0.94, is_training=False,
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
        lr *= 0.85
    elif test_acc>0.9 and lr>1e-6:
        lr *= 0.95
    elif test_acc>0.95:
        lr *= 0.99

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
        print "%d %f"%(num,lr)
        if lr < 1e-4:
            acc = 0.95

        if lr < 1e-5:
            acc = 0.96






if __name__ == "__main__":
    test()


