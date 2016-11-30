#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

import os
import struct

mnist = input_data.read_data_sets("MNIST_DATA/",one_hot=True)

def fun1():
    for j in range(20):

        train_x,train_y = mnist.train.next_batch(20)

        fig = plt.figure()
        for i in range(1,7):
            img = np.reshape(train_x[i],[28,28])
            lab = np.argmax(train_y[i],0)

            plt.subplot(3,2,i)
            plt.title(lab)
            plt.imshow(img)

        plt.show()


def fun2():
    a = u'啊'

    autf = repr(a.encode("utf-8"))
    print "utf-8:%r"%autf


    ag = repr(a.encode("gbk"))
    print "gbk:%r"%ag

    auni = repr(a)
    print "unicode:%r"%auni

    myasc = "\\xb0\\xa1"
    print myasc


def fun3():
    dirname = "/home/allen/work/data/HWDB1.1orign/HWDB1.1trn_gnt"
    filenames = os.listdir(dirname)
    filenames.sort()

    filename = os.path.join(dirname,filenames[0])

    with file(filename,mode="rb") as fobj:
        for n in xrange(3):

            try:
                sampleSize= struct.unpack('<I',fobj.read(4))[0]
                charGBK = fobj.read(2)

                width = struct.unpack("<H",fobj.read(2))[0]
                height = struct.unpack("<H",fobj.read(2))[0]

                char_img = np.zeros([height,width],dtype=np.float32)

                for i in xrange(height):
                    for j in xrange(width):
                        pixel = struct.unpack('<B',fobj.read(1))[0]
                        char_img[i,j] = pixel


                title = "sample size is %s, "%sampleSize
                title += "%r %s\n"% (charGBK,charGBK.decode('gbk'))
                title += "height is %s  "%height
                title += "width is %r"%width

                print title


                plt.figure()
                plt.imshow(char_img)
                plt.title(title)

                plt.show()

            except Exception,e:
                print "error:",e



def test():
    fun3()

if __name__ == "__main__":
    test()




