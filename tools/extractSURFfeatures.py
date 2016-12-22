#coding=utf-8
"""
extract SURF feature from des64 picture
save to data/SURFFeatures/train1000class

"""

import cv2
import pickle
import os


def  convert2desdir(srcDir,desDir):
    filenames = sorted(os.listdir(srcDir))
    for eachfile in filenames:
        srcFilename = os.path.join(srcDir,eachfile)
        desFilename = os.path.join(desDir,eachfile)



