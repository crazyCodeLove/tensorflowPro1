#coding=utf-8

"""
test data all character number is:      223991
test data 1000 class,character number is 59688

train database,all character number is 897758
train 1000 class, character number is  239064


"""

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import pickle


traindirname = "/home/allen/work/data/HWDB1.1des64/train1000class"
testdirname = "/home/allen/work/data/HWDB1.1des64/test1000class"

characterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des64/1000class.pkl"


charWidth = 64
itemLength = charWidth*charWidth + 2



def next_batch(batchnum, dirname, itemLength, character_class, charTagcodeMapFile):
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

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start<contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []

                fetchnum = batchnum - len(batch_x)
                end = start+ fetchnum * itemLength

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = end
                    batch_x,batch_y = fromList2Stand(batch_x,batch_y,character_class)
                    yield batch_x,batch_y




                    # for each in batch_y:
                    #     print tagmap[each].decode('gbk'),
                    #
                    # print ""
                    # for each in batch_x:
                    #     each = np.array(each).astype(dtype=np.float32).reshape([32,32])
                    #
                    #     each = Image.fromarray(each)
                    #     each.show()
                    #     plt.figure()
                    #     plt.imshow(each)
                    #     plt.show()






                else:
                    end = contentlength
                    data2list(content, start, end, batch_x, batch_y, itemLength, charTagcodeMapFile)
                    start = contentlength

def fromList2Stand(batch_x,batch_y,character_class):
    out_x = 255 - np.array(batch_x).astype(np.float32)

    out_y = np.zeros([len(batch_y),character_class],dtype=np.float64)
    for i in xrange(len(batch_y)):
        out_y[i,batch_y[i]] = 1.0

    return out_x,out_y


def data2list(data,start,end,batch_x,batch_y,itemLength,characterTagcodeMapFile):
    length = (end-start) / itemLength


    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for i in xrange(length):
        substart = i * itemLength + start
        tagcode = data[substart:substart+2]
        bitmap = data[substart+2:substart + itemLength]

        bitmap = [struct.unpack('<B',pixcel)[0] for pixcel in bitmap]

        batch_y.append(tagmap.index(tagcode))
        batch_x.append(bitmap)





def test():
    global charWidth,itemLength,characterTagcodeMapFile
    character_class = 1000

    number = 6
    gen = next_batch(number, testdirname,itemLength, character_class,characterTagcodeMapFile)

    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)
    
    for j in xrange(10):
        x,y = gen.next()
        plt.figure()


        for i in xrange(number):
            pic = x[i,:]
            label = y[i,:]
            index = np.argmax(label)
            print tagmap[index].decode('gbk'),

            pic = np.reshape(pic,[charWidth,charWidth])
            plt.subplot(2,3,i+1)
            plt.imshow(pic)
        print

        plt.show()






if __name__ == "__main__":
    test()
