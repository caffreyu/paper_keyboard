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

img1, img2 = cv2.imread('./images/dino0.png'), cv2.imread('./images/dino1.png')
# print (img1.shape)

# cor1, cor2 = get_matches(img1, img2)
# print (cor1, cor1.shape)

cor1 = np.load("./data/"+'npy'+"/cor1.npy")
cor2 = np.load("./data/"+'npy'+"/cor2.npy")

length = cor1.shape[1]
pts1, pts2 = [], []

for i in range(length):
    x, y = cor1[: 2, i]
    pts1.append([x, y])
    x, y = cor2[: 2, i]
    pts2.append([x, y])

pts1, pts2 = np.array(pts1), np.array(pts2)
print (pts1, pts1.shape)
# cv2.findFundamentalMat()

mat, mask = cv2.findFundamentalMat(
                                    points1 = pts1, 
                                    points2 = pts2, 
                                    method = cv2.FM_RANSAC,
                                    ransacReprojThreshold = 0.1,
                                    confidence = 0.99,
                                    maxIters = 1000
                                    )

# print (mat)

corner_len = len(cor1)
cor1, cor2 = np.array(cor1).T, np.array(cor2).T
cor1_homog = np.concatenate((cor1, np.ones(corner_len).reshape((1, -1))), axis = 0).T
cor2_homog = np.concatenate((cor2, np.ones(corner_len).reshape((1, -1))), axis = 0).T

print (cor1_homog, cor1_homog.shape)

plot_epipolar_lines(img1, img2, mat, cor1_homog, cor2_homog)


