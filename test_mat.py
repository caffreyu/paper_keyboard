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

config_fname = './data/config.yaml'
fh = cv2.FileStorage(config_fname, cv2.FILE_STORAGE_READ)
cam_mat = fh.getNode('camera_matrix').mat()
img_w = int(fh.getNode('image_width').real())
img_h = int(fh.getNode('image_height').real())

print (cam_mat)

# get file name for target
fname, mat_method = sys.argv[1], sys.argv[2]

target = plt.imread(fname)
target_show = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

print ('Now showing target')
plt.imshow(target)
plt.show()

cap = cv2.VideoCapture(-1)
font = cv2.FONT_HERSHEY_SIMPLEX

s_time, e_time = 0, 0

while True:
    
    _, test = cap.read()

    test = cv2.resize(test, (img_w, img_h))
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
        if mat_method == 'f': 
            mat, mask = cv2.findEssentialMat(
                                            points1 = cor1, 
                                            points2 = cor2, 
                                            cameraMatrix = cam_mat,
                                            method = cv2.RANSAC,
                                            prob = 0.999,
                                            threshold = 100
                                            )
        # Fundamental matrix
        if mat_method == 'e':
            mat, mask = cv2.findFundamentalMat(
                                            points1 = cor1, 
                                            points2 = cor2, 
                                            method = cv2.FM_RANSAC,
                                            ransacReprojThreshold = 3,
                                            confidence = 0.99)
        
        print ('Done finding the matrix')

        cor1 = cor1[mask.ravel() == 1]
        cor2 = cor2[mask.ravel() == 1]

        print (mat)
        # print (mat)
        break

def plot_epipolar_lines(img1, img2, F, cor1, cor2):
    """Plot epipolar lines on image given image, corners

    Args:
        img1: Image 1.
        img2: Image 2.
        F:    Fundamental matrix
        cor1: Corners in homogeneous image coordinate in image 1 (3xN)
        cor2: Corners in homogeneous image coordinate in image 2 (3xN)
    """
    
    assert cor1.shape[0] == 3
    assert cor2.shape[0] == 3
    assert cor1.shape == cor2.shape
    
    """ ==========
    YOUR CODE HERE
    ========== """
    
    # print (cor1.shape, cor2.shape)
    
    n = cor1.shape[1]
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    x = np.array([0, w1 - 1])  
    # x = np.array([0, 50])  
    fig = plt.figure(figsize=(12, 9))

    print (img1.shape, img2.shape, x)
    
    ax1 = fig.add_subplot(221)
    ax1.imshow(img1)
    ax1.axis('off')

    ax2 = fig.add_subplot(222)
    ax2.imshow(img2)
    ax2.axis('off')
    
    for i in range(n):
        x1, x2 = cor1[:, i], cor2[:, i]
        # l1, l2 = np.dot(F.T, x2), np.dot(F, x1)
        color = np.random.rand(3,)

        x1_n, x2_n = deepcopy(x1), deepcopy(x2)

        x1_n[0], x1_n[1] = x1_n[0] / h1, x1_n[1] / w1
        x2_n[0], x2_n[1] = x2_n[0] / h2, x2_n[1] / w2
        print (x2_n.T @ F @ x1_n)

        l1, l2 = np.dot(F.T, x2_n), np.dot(F, x1_n)

        y1, y2 = -x * l1[0] / l1[1] - l1[2] / l1[1], -x * l2[0] / l2[1] - l2[2] / l2[1]
        print ('First image: ', x, y1)
        print ('Second image: ', x, y2)

        ax1.plot(x, -x * l1[0] / l1[1] - l1[2] / l1[1], c = color)
        ax1.scatter(x1[0], x1[1], s = 35, edgecolors = 'b', facecolors = color)
        ax2.plot(x, -x * l2[0] / l2[1] - l2[2] / l2[1], c = color)
        ax2.scatter(x2[0], x2[1], s = 35, edgecolors = 'b', facecolors = color)
        
        # x1[0], x1[1] = x1[0] / h1, x1[1] / w1
        # x2[0], x2[1] = x2[0] / h2, x2[1] / w2
        # print (x2.T @ F @ x1)

    
    plt.show()

# plot_epipolar_lines(test, target_show, mat, cor1, cor2)

# print (len(cor1), np.shape(np.array(cor1)))

corner_len = len(cor1)
cor1, cor2 = np.array(cor1).T, np.array(cor2).T
cor1_homog = np.concatenate((cor1, np.zeros(corner_len).reshape((1, -1))), axis = 0)
cor2_homog = np.concatenate((cor2, np.zeros(corner_len).reshape((1, -1))), axis = 0)

plot_epipolar_lines(test, target_show, mat, cor1_homog, cor2_homog)


# plot_epipolar_lines(test, target_show, mat, cor1_homog, cor2_homog)

# lines1 = cv2.computeCorrespondEpilines(cor2.reshape(-1, 1, 2), 2, mat)
# lines1 = lines1.reshape(-1, 3)

# print (lines1)

# for r in lines1:
#     x0,y0 = map(int, [0, -r[2]/r[1] ])
#     x1,y1 = map(int, [img_h, -(r[2]+r[0]*img_h)/r[1] ])
#     print (x0, y0, x1, y1)

# img1, img2 = drawlines(test_g, target_g, lines1, cor1, cor2)

# fig = plt.figure(figsize=(12, 9))

# ax1 = fig.add_subplot(221)
# ax1.imshow(img1)
# ax1.axis('off')

# ax2 = fig.add_subplot(222)
# ax2.imshow(img2)
# ax2.axis('off')

# plt.show()

