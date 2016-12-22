# coding=utf-8

"""
test data all character number is:      223991
test data 1000 class,character number is 59688

train database,all character number is 897758
train 1000 class, character number is  239064



item_length is a single character size. from start to end(bytes)

"""

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cv
import cv2



charWidth = 64
itemLength = charWidth * charWidth + 2


def next_batch(batchnum, dirname, charWidth, character_class, charTagcodeMapFile):
    itemLength = charWidth*charWidth+2
    filenames = sorted(os.listdir(dirname))
    filenum = -1
    batch_x = []
    batch_y = []

    # with open(characterTagcodeMapFile) as fobj:
    #     tagmap = pickle.load(fobj)

    while True:
        filenum += 1
        filenum = filenum % len(filenames)

        filename = filenames[filenum]
        filename = os.path.join(dirname, filename)

        # print filename

        with open(filename, mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start < contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []

                fetchnum = batchnum - len(batch_x)
                end = start + fetchnum * itemLength

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = end
                    batch_x, batch_y = fromList2Stand(batch_x, batch_y, character_class)

                    features = extractFeatures(batch_x,charWidth)
                    batch_x /= 255

                    yield batch_x, batch_y,features

                else:
                    end = contentlength
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = contentlength


def extractFeatures(data,charWidth,feature_num=10):
    batchnum = data.shape[0]
    result = np.zeros([batchnum,1280],dtype=np.float32)

    surf = cv2.SURF(600)

    for it in xrange(batchnum):
        pic = np.reshape(data[it,:],[charWidth,charWidth])

        pic = pic.astype(np.ubyte)

        kp,descript = surf.detectAndCompute(pic,None)

        if descript is not None:
            respDespMap = sorted([{k.response:v} for k,v in zip(kp,descript)],key=lambda k:k.keys()[0],reverse=True)


            if len(respDespMap)>feature_num:
                itresult = [itmp.values()[0] for itmp in (respDespMap[:feature_num])]
                result[it] = np.reshape(itresult,[1,-1])
            else:
                itresult = [itmp.values()[0] for itmp in respDespMap]
                itresult = np.reshape(itresult,[1,-1]).flatten()
                result[it,:len(itresult)] = itresult


    return result





def fromList2Stand(batch_x, batch_y, character_class):
    """

    :param batch_x:是一个ndarray,shape 是batch_num*item_length
    :param batch_y: 是各个字符的索引列表
    :param character_class: 该数据库的字符类别数目
    :return:
    """
    out_x = 255 - np.array(batch_x).astype(np.float32)

    out_y = np.zeros([len(batch_y), character_class], dtype=np.float64)
    for i in xrange(len(batch_y)):
        out_y[i, batch_y[i]] = 1.0

    return out_x, out_y


def data2list(data, start, end, batch_x, batch_y, itemLength, characterTagcodeMapFile):
    """

    :param data: 文件内容
    :param start: 该批数据的开始位置
    :param end: 该批数据的结束位置，可能在文件中间，也可能是文件末尾
    :param batch_x: 存储x数据缓存区
    :param batch_y: 存储y数据的缓存区
    :param itemLength: a single character size. from start to end(bytes)
    :param characterTagcodeMapFile: 存储要识别类别字符的对照表
    :return:
    """
    length = (end - start) / itemLength

    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for i in xrange(length):
        substart = i * itemLength + start
        tagcode = data[substart:substart + 2]
        bitmap = data[substart + 2:substart + itemLength]

        bitmap = [struct.unpack('<B', pixcel)[0] for pixcel in bitmap]

        batch_y.append(tagmap.index(tagcode))
        batch_x.append(bitmap)




def test():
    global charWidth, itemLength

    traindirname = "/home/allen/work/data/HWDB1.1des64/train1000class"
    testdirname = "/home/allen/work/data/HWDB1.1des64/test1000class"

    characterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des64/1000class.pkl"

    character_class = 1000

    number = 6
    gen = next_batch(number, traindirname, charWidth, character_class, characterTagcodeMapFile)

    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for j in xrange(10):
        x, y,features = gen.next()
        plt.figure()

        for i in xrange(number):
            pic = x[i, :]
            label = y[i, :]
            index = np.argmax(label)
            print tagmap[index].decode('gbk'),

            pic = np.reshape(pic, [charWidth, charWidth])
            plt.subplot(2, 3, i + 1)
            plt.imshow(pic)
        print

        plt.show()


def fun2():
    batch_num = 10

    traindirname = "/home/allen/work/data/HWDB1.1des64/train1000class"
    testdirname = "/home/allen/work/data/HWDB1.1des64/test1000class"
    character_class = 1000
    charTagcodeMapFile = "/home/allen/work/data/HWDB1.1des64/1000class.pkl"

    train_gen = next_batch(batch_num,traindirname,64,character_class,charTagcodeMapFile)

    with open(charTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)


    zeronum = 0
    for itemnum in xrange(1000):
        x,y,features = train_gen.next()

        for i in xrange(batch_num):
            pic = np.reshape(x[i,:],[charWidth,charWidth])


            surf = cv2.SURF(700)
            pic = pic.astype(np.ubyte)



            kp,d = surf.detectAndCompute(pic,None)

            print kp[0].response

            despic = cv2.drawKeypoints(pic,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(pic)
            plt.title('0')

            plt.subplot(1,2,2)
            plt.imshow(despic)
            plt.title(str(len(kp)))
            plt.show()



            # if len(kp) ==0:
            #     zeronum +=1

    print "total zero feature number is %d"%zeronum



def fun3():
    global charWidth, itemLength

    traindirname = "/home/allen/work/data/HWDB1.1des64/train100class"
    testdirname = "/home/allen/work/data/HWDB1.1des64/test100class"

    characterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des64/100class.pkl"

    character_class = 100

    number = 50
    gen = next_batch(number, traindirname, charWidth, character_class, characterTagcodeMapFile)

    for i in xrange(500):
        x,y,f = gen.next()
        print i,f.shape





if __name__ == "__main__":
    # test()
    # fun2()
    fun3()
