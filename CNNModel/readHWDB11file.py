#coding=utf-8

"""
读取CASIA_HWDB1.1 *.gnt 格式文件内容
文件格式是以c语言结构体给出，所以用python解析的时候需要使用struct
来解析
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image

dirname = "/home/allen/work/data/HWDB1.1orign/HWDB1.1trn_gnt"

def fun1():
    """
    read file content, then show it
    :return:
    """
    filename = sorted(os.listdir(dirname))[0]
    filename = os.path.join(dirname,filename)

    with file(filename,mode='rb') as fobj:
        try:
            for n in xrange(5):
            # while True:

                sampleSize = struct.unpack('<I',fobj.read(4))[0]

                #tag code is gbk
                tagCode = fobj.read(2)
                width = struct.unpack('<H',fobj.read(2))[0]
                height = struct.unpack('<H',fobj.read(2))[0]

                bitmap = np.zeros([height,width],dtype=np.float32)

                for i in xrange(height):
                    for j in xrange(width):
                        pixel = struct.unpack('<B',fobj.read(1))[0]
                        bitmap[i,j] = pixel

                title = "sample size:%s bytes, tag code:%r\n"% (sampleSize,tagCode)
                title += "height:%s, width:%s"% (height, width)

                plt.figure()
                print title
                newbitmap = Image.fromarray(bitmap)

                plt.imshow(newbitmap)
                plt.title(title)
                plt.show()


                # newimg = Image.fromarray(bitmap)
                newimg = newbitmap.resize([32,32], Image.BILINEAR)


                plt.figure()

                plt.imshow(newimg)
                plt.show()

        except EOFError:
            pass


def Matrix2Img(matrix):
    img = Image.fromarray(matrix)
    return img

def Img2Matrix(imgName):
    img = Image.open(imgName)
    width,height = img.size
    img = img.convert('L')
    imgdata = img.getdata()
    imgdata = np.matrix(imgdata,dtype=np.float32)
    imgdata = np.reshape(imgdata,[width,height])
    return imgdata



def test():
    fun1()


if __name__ == "__main__":
    test()

