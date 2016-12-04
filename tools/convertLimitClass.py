#coding=utf-8

"""
convert source file to destination file
format of destination file is
item,        type,             length,                instance,                        comment
Tag code,    char,             2B,                    "阿"=0xb0a1 stored as 0xa1b0
bitmap,      unsighed char,    width*height bytes,                                  ,  stored row by row

all character class is 3755
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image
import pickle


trainoridirname = "/home/allen/work/data/HWDB1.1des64/HWDB1.1trn_gnt"
traindesdirname = "/home/allen/work/data/HWDB1.1des64/train10class"

testoridirname = "/home/allen/work/data/HWDB1.1des64/HWDB1.1tst_gnt"
testdesdirname = "/home/allen/work/data/HWDB1.1des64/test10class"

characterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des64/10class.pkl"
oricharacterTagcodeMapFile = "/home/allen/work/data/HWDB1.1des64/tagindexmap.pkl"

buffer=[]
class_num = 10
charWidth = 64
item_length = 2+charWidth*charWidth


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
    global buffer,class_num,characterTagcodeMapFile,item_length

    with open(characterTagcodeMapFile) as fobj:
        tagcodeMap = pickle.load(fobj)

    with open(filename,mode='rb') as fobj:
        content = fobj.read()

    content_length = len(content)
    start = 0

    while start<content_length:
        tagcode = content[start:start+2]

        if tagcode in tagcodeMap:
            end = start + item_length
            buffer.append(content[start:end])

        start += item_length



def write2DesFile(filename):
    global buffer
    with open(filename,'wb') as fobj:
        content = ''.join(buffer)

        # char = buffer[0][:2]
        # print "buffer 1 item",char.decode('gbk')
        # char = content[:2]
        # print "content 1 item",char.decode('gbk')


        fobj.write(content)
    buffer = []



def limitDesClass():
    """
    convert oricharacterTagcodeMapFile to des limit class characterTagcodeMapFile

    :return:
    """
    global class_num
    global oricharacterTagcodeMapFile,characterTagcodeMapFile

    with open(oricharacterTagcodeMapFile) as fobj:
        data = pickle.load(fobj)
    data = data[:class_num]

    with open(characterTagcodeMapFile,mode='wb')as fobj:
        pickle.dump(data,fobj)


def showDesImage(desdirname):
    global charWidth
    desFilenames = sorted(os.listdir(desdirname))

    filename = desFilenames[20]
    filename = os.path.join(desdirname, filename)

    print filename

    with open(filename,mode='rb') as fobj:
        itemSize = 2+charWidth*charWidth
        content = fobj.read()
        for i in xrange(10):

            start = i*itemSize
            end = (i+1)*itemSize
            tagcode = content[start:start+2]
            bitmap = content[start+2:end]
            bitmap = [struct.unpack('<B',each)[0] for each in bitmap]
            bitmap = np.array(bitmap).astype(np.ubyte).reshape([charWidth,charWidth])

            print tagcode.decode('gbk')
            plt.figure()
            img = Image.fromarray(bitmap)
            plt.imshow(img)

            plt.show()



def calculateAllCharacterCount(dirname):
    """
    计算目标文件夹下所有文件字符总数
    显示单个文件字符数，总字符数
    :return:
    """
    global item_length
    filenames = sorted(os.listdir(dirname))

    allCharacterCount = 0
    filenum = 0

    chSize = item_length

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




def test():
    # limitDesClass()
    showDesImage(traindesdirname)
    # calculateAllCharacterCount(traindesdirname)
    # fromSrc2Des(trainoridirname,traindesdirname)

if __name__ == "__main__":
    test()