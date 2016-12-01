#coding=utf-8

"""
convert source file to destination file
format of destination file is
item,        type,             length,                instance,                        comment
Tag code,    char,             2B,                    "阿"=0xb0a1 stored as 0xa1b0
bitmap,      unsighed char,    width*height bytes,                                  ,  stored row by row

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image
import pickle


trainoridirname = "/home/allen/work/data/HWDB1.1orign/HWDB1.1trn_gnt"
traindesdirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1trn_gnt"

testoridirname = "/home/allen/work/data/HWDB1.1orign/HWDB1.1tst_gnt"
testdesdirname = "/home/allen/work/data/HWDB1.1des32/HWDB1.1tst_gnt"

characterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des32/tagindexmap.pkl"

tag_buffer = []
bitmap_buffer = []
desfilename = ""


def fromSrc2Des(oridirname,desdirname):
    global tag_buffer,bitmap_buffer,desfilename

    orifilenames = sorted(os.listdir(oridirname))

    numOfFile = 1

    # if len(orifilenames)!=0:
    for filename in orifilenames:
        # filename = orifilenames[0]

        orifilename = os.path.join(oridirname, filename)
        desfilename = os.path.join(desdirname, filename)

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

    with open(filename,mode='wb') as des_fobj:
        while len(tag_buffer) !=0:

            tagcode = tag_buffer.pop(0)
            bitmap = bitmap_buffer.pop(0)
            des_fobj.write(tagcode)
            for i in xrange(32):
                for j in xrange(32):
                    pixcel = struct.pack('<B', bitmap[i, j])
                    des_fobj.write(pixcel)

            # print "%s"%tagcode.decode('gbk')

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
    orifilenames = sorted(os.listdir(trainoridirname))

    filename = orifilenames[0]
    filename = os.path.join(trainoridirname, filename)

    print filename

    with open(filename,mode='rb') as fobj:
        for i in xrange(10):
            sampleSize = struct.unpack('<I',fobj.read(4))[0]
            tagcode = fobj.read(2)
            width = struct.unpack('<H', fobj.read(2))[0]
            height = struct.unpack('<H', fobj.read(2))[0]

            bitmap = np.zeros([height,width],dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', fobj.read(1))[0]
                    bitmap[i, j] = pixel

            print tagcode.decode('gbk')

            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()


def showDesImage():
    desFilenames = sorted(os.listdir(traindesdirname))

    filename = desFilenames[0]
    filename = os.path.join(traindesdirname, filename)

    print filename

    with open(filename,mode='rb') as fobj:
        for i in xrange(10):

            tagcode = fobj.read(2)
            width = 32
            height = 32

            bitmap = np.zeros([height,width],dtype=np.float32)

            for i in xrange(height):
                for j in xrange(width):
                    pixel = struct.unpack('<B', fobj.read(1))[0]
                    bitmap[i, j] = pixel

            print tagcode.decode('gbk')
            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()



def calculatCharCount(dirname):
    """
    计算原文件夹下所有不同的字符个数
    :return:
    """
    orifilenames = sorted(os.listdir(dirname))

    charSet = set()
    numOfFile = 1


    for filename in orifilenames:
        filename = os.path.join(dirname, filename)

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

def calculateAllCharacterCount(dirname):
    """
    计算目标文件夹下所有文件字符总数
    显示单个文件字符数，总字符数
    :return:
    """
    filenames = os.listdir(dirname)

    allCharacterCount = 0
    filenum = 0

    chSize = 2+32*32

    # if len(filenames) != 0:
    #     filename = filenames[0]
    #     filename = os.path.join(desdirname,filename)

    for each in filenames:
        filename = os.path.join(dirname, each)

        with open(filename,mode='rb') as fobj:
            content = fobj.read()
            filenum += 1
            fileCharacterCount = len(content)/chSize

            allCharacterCount += fileCharacterCount

            print "%d th file, character count is %d, all count is %d"%(filenum, fileCharacterCount, allCharacterCount)


def createTagIndexMap():
    """
    将每个汉字编码存放到列表中，按照读入的先后顺序添加
    :return:
    """
    charlist = []
    filenames = sorted(os.listdir(traindesdirname))

    numOfFile = 0
    itemlength = 32*32+2

    while len(charlist) < 3755:
        filename = filenames[numOfFile]
        filename = os.path.join(traindesdirname, filename)

        with open(filename,mode='rb') as fobj:
            numOfFile += 1
            content = fobj.read()

            charnums = len(content)/itemlength
            for i in xrange(charnums):

                start = i * (itemlength)
                end = start + 2
                tagcode = content[start:end]

                if tagcode not in charlist:
                    charlist.append(tagcode)

    with open(characterTagcodeMapFile,mode='w') as fobj:
        pickle.dump(charlist,fobj)

    print "create tagcode index map file done"


def fun3():
    """
    read characterTagcodeMapFile

    :return:
    """
    with open(characterTagcodeMapFile) as fobj:
        data = pickle.load(fobj)
        print len(data)
        for i in xrange(10):
            print data[i].decode('gbk')





def test():
    # fun3()
    # createTagIndexMap()
    # calculateAllCharacterCount()
    # fromSrc2Des()
    calculatCharCount(testoridirname)
    # showOriImage()
    # showDesImage()

if __name__ == "__main__":
    test()