# Test essential/fundamental matrix

from matplotlib import lines
import numpy as np
import cv2
# from cv2 import imread, cvtColor
import matplotlib.pyplot as plt
import random
from random import randint as rd
from utils import *
import time
import sys
import yaml
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter

# cam matrix
config_fname = './data/config/config.yaml'
fh = cv2.FileStorage(config_fname, cv2.FILE_STORAGE_READ)
cam_mat = fh.getNode('camera_matrix').mat()
img_w = int(fh.getNode('image_width').real())
img_h = int(fh.getNode('image_height').real())
print (cam_mat.shape)

print (cam_mat)

# get file name for target
fname, mat_method = sys.argv[1], sys.argv[2]

target = plt.imread(fname)
target = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
target = gaussian_filter(target, sigma = 2)
print (target.shape)
target_show = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

print ('Now showing target')
plt.imshow(target_g)
plt.show()

cap = cv2.VideoCapture(-1)
font = cv2.FONT_HERSHEY_SIMPLEX

s_time, e_time = 0, 0

while True:
    
    _, test = cap.read()

    if test[0, 0, 0] == 0: continue

    plt.imshow(test)
    plt.show()

    test_g = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    cor1, cor2 = get_matches(test_g, target_g)

    # cor1, cor2 = np.array(cor1).T, np.array(cor2).T

    print (f'After processing total number: {len(cor1)}')

    if len(cor1) < 10: 
        print ('WAITING: waiting for more points')
        time.sleep(0.1)
        continue
    else:
        print ('Done getting matches')
        # Essential matrix
        if mat_method == 'e': 
            mat, mask = cv2.findEssentialMat(
                                            points1 = cor1, 
                                            points2 = cor2, 
                                            cameraMatrix = cam_mat,
                                            method = cv2.RANSAC,
                                            prob = 0.999,
                                            threshold = 0.1
                                            )
        # Fundamental matrix
        if mat_method == 'f':
            mat, mask = cv2.findFundamentalMat(
                                            points1 = cor1, 
                                            points2 = cor2, 
                                            method = cv2.FM_RANSAC,
                                            ransacReprojThreshold = 3,
                                            confidence = 0.99,
                                            maxIters = 1000
                                            )
        
        print ('Done finding the matrix')

        cor1 = cor1[mask.ravel() == 1]
        cor2 = cor2[mask.ravel() == 1]

        print (mat)
        break


corner_len = len(cor1)
cor1, cor2 = np.array(cor1).T, np.array(cor2).T
cor1_homog = np.concatenate((cor1, np.ones(corner_len).reshape((1, -1))), axis = 0)
cor2_homog = np.concatenate((cor2, np.ones(corner_len).reshape((1, -1))), axis = 0)

plot_epipolar_lines(test, target_show, mat, cor1_homog, cor2_homog)
