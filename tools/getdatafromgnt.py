#coding=utf-8
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import pickle


traindirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1trn_gnt"
testdirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1tst_gnt"

charSize = 32*32+2

characterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des32/tagindexmap.pkl"


def next_batch(batchnum,dirname):
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

        print filename

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            contentlength = len(content)
            start = 0

            while start<contentlength:
                if len(batch_y) == batchnum:
                    batch_x = []
                    batch_y = []

                fetchnum = batchnum - len(batch_x)
                end = start+fetchnum*charSize

                if end <= contentlength:
                    data2list(content, start, end, batch_x, batch_y)
                    start = end
                    batch_x,batch_y = fromList2Stand(batch_x,batch_y)
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
                    data2list(content, start, end,batch_x, batch_y)
                    start = contentlength

def fromList2Stand(batch_x,batch_y):
    out_x = np.array(batch_x).astype(np.float32)/255
    out_x = out_x.reshape([-1,1024])

    out_y = np.zeros([len(batch_y),3755],dtype=np.float32)
    for i in xrange(len(batch_y)):
        out_y[i,batch_y[i]] = 1.0

    return out_x,out_y


def data2list(data,start,end,batch_x,batch_y):
    length = (end-start) / charSize


    with open(characterTagcodeMapFile) as fobj:
        tagmap = pickle.load(fobj)

    for i in xrange(length):
        substart = i*charSize +start
        tagcode = data[substart:substart+2]
        bitmap = data[substart+2:substart+charSize]

        bitmap = [struct.unpack('<B',pixcel)[0] for pixcel in bitmap]

        batch_y.append(tagmap.index(tagcode))
        batch_x.append(bitmap)





def test():

    number = 4
    gen = next_batch(number, testdirname)
    
    for i in xrange(10):
        x,y = gen.next()
        plt.figure()
        for i in xrange(number):
            pic = x[i,:]
            pic = np.reshape(pic,[32,32])
            plt.subplot(2,2,i+1)
            plt.imshow(pic)
        plt.show()






if __name__ == "__main__":
    test()
