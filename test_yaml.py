import yaml
from yaml import load, dump
import cv2
import numpy as np

fname = './data/config.yaml'
fh = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)

fn = fh.getNode('camera_matrix')
print (type(fn.mat()))

print (fh.getNode('image_width').real())