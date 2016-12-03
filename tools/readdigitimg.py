#coding=utf-8

import numpy as np

from PIL import Image
import os
import pickle

dirname = "/home/allen/work/data/digit"
digitfilepkl = "/home/allen/work/data/digitpkl/digit.pkl"

def savedata():
    filenames = sorted(os.listdir(dirname))
    batch_x = []
    batch_y = []


    for i in range(len(filenames)):
        y = np.zeros([10],dtype=np.float64)
        y[i] = 1
        batch_y.append(y)

        each = os.path.join(dirname,filenames[i])
        img = Image.open(each)
        imgdata = 1 - np.array(img,dtype=np.float32).reshape([-1,])/255
        batch_x.append(imgdata)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    with open(digitfilepkl,mode='w') as fobj:
        pickle.dump((batch_x,batch_y),fobj)






def test():
    savedata()


if __name__ == "__main__":
    test()
