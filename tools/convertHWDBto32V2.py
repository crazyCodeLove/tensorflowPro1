#coding=utf-8

"""
character count is 3755

convert source file to destination file
format of destination file is pickle file
[{tagcode:bitmap},...]

item,        type,             length,                instance,                        comment
Tag code(GBK),    char,             2B,                    "阿"=0xb0a1 stored as 0xa1b0
bitmap,      unsighed char,    width*height bytes,                                  ,  stored row by row




"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image
import pickle

oridirname = "/home/allen/work/data/HWDB1.1orign/HWDB1.1trn_gnt"
desdirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1trn_pkl"


tag_buffer = []
bitmap_buffer = []
desfilename = ""


def fromSrc2Des():
    global tag_buffer,bitmap_buffer,desfilename,oridirname,desdirname

    orifilenames = sorted(os.listdir(oridirname))

    numOfFile = 1

    # if len(orifilenames)!=0:
    for filename in orifilenames:
        # filename = orifilenames[0]

        orifilename = os.path.join(oridirname,filename)

        filename = filename.replace(".gnt",".pkl")

        desfilename = os.path.join(desdirname,filename)

        readFileContent(orifilename)
        write2DesFile(desfilename)

        print "%5dth file convert completed"%(numOfFile)
        numOfFile += 1







def readFileContent(filename):
    global tag_buffer,bitmap_buffer
    with open(filename, mode='rb') as src_fobj:

        while src_fobj.read(1) !="":
            src_fobj.seek(-1,1)
            sampleSize = struct.unpack('<I', src_fobj.read(4))[0]
            # tag code is gbk
            tagCode = src_fobj.read(2)
            width = struct.unpack('<H', src_fobj.read(2))[0]
            height = struct.unpack('<H', src_fobj.read(2))[0]

            bitmap = np.zeros([height, width], dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', src_fobj.read(1))[0]
                    bitmap[i, j] = pixel

            des = convertTo32(bitmap)

            tag_buffer.append(tagCode)
            bitmap_buffer.append(des)

            # print "%s"%tagCode.decode('gbk')


def write2DesFile(filename):
    global tag_buffer, bitmap_buffer

    data = [{tag:bm} for tag,bm in zip(tag_buffer,bitmap_buffer)]
    with open(filename,mode='wb') as fobj:
        pickle.dump(data, fobj)




def convertTo32(ori):
    imgdata = Image.fromarray(ori)
    imgdata = imgdata.resize([32,32],Image.BILINEAR)
    des = np.array(imgdata)
    des = np.reshape(des,[32,32])
    return des.astype(np.ubyte)


def fun2():
    with open('test.ocr',mode='w') as fobj:
        fobj.write('nice to meet you')

    with open('/home/allen/work/data/HWDB1.1orign/HWDB1.1trn_gnt/1001-c.gnt',mode='rb') as fobj:
        size = fobj.read(4)
        print size
        tag = fobj.read(2)
        print '%r'%tag


def showOriImage():
    orifilenames = sorted(os.listdir(oridirname))

    filename = orifilenames[0]
    filename = os.path.join(oridirname, filename)

    with open(filename,mode='rb') as fobj:
        for i in xrange(5):
            sampleSize = struct.unpack('<I',fobj.read(4))[0]
            tagcode = fobj.read(2)
            width = struct.unpack('<H', fobj.read(2))[0]
            height = struct.unpack('<H', fobj.read(2))[0]

            bitmap = np.zeros([height,width],dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', fobj.read(1))[0]
                    bitmap[i, j] = pixel

            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()


def showDesImage():
    desFilenames = sorted(os.listdir(desdirname))

    filename = desFilenames[0]
    filename = os.path.join(desdirname, filename)


    with open(filename,mode='rb') as fobj:
        data = pickle.load(fobj)
        for i in xrange(5):

            item = data[i]
            tagcode = item.keys()[0]
            bitmap = item[tagcode].astype(np.float32)


            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()



def calculatCharCount():
    orifilenames = sorted(os.listdir(oridirname))

    charSet = set()
    numOfFile = 1


    for filename in orifilenames:
        filename = os.path.join(oridirname, filename)

        with open(filename, mode='rb') as src_fobj:

            while src_fobj.read(1) != "":
                src_fobj.seek(-1, 1)
                src_fobj.read(4)
                # tag code is gbk
                tagCode = src_fobj.read(2)
                width = struct.unpack('<H', src_fobj.read(2))[0]
                height = struct.unpack('<H', src_fobj.read(2))[0]

                for i in xrange(height):
                    for j in xrange(width):
                        pixel = src_fobj.read(1)

                if tagCode not in charSet:
                    charSet.add(tagCode)
        print "%3d now character size is "%numOfFile,len(charSet)
        numOfFile += 1

def test():
    fromSrc2Des()
    # calculatCharCount()
    # showOriImage()
    # showDesImage()

if __name__ == "__main__":
    test()