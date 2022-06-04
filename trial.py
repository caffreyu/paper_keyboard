import numpy as np
import cv2
# from cv2 import imread, cvtColor
import matplotlib.pyplot as plt
import random
from random import randint as rd
from utils import *
from time import time
import sys

# def get_matches(img1, img2):
#     sift = cv2.SIFT_create()
#     cor1, cor2 = [], []

#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k = 2)

#     good = []

#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append(m)
#             # print (type(m))
    
#     for match in good:
#         cor1.append(kp1[match.queryIdx].pt)
#         cor2.append(kp2[match.trainIdx].pt)
    
#     # cor1, cor2 = np.array(cor1).reshape(-1, 2), np.array(cor2).reshape(-1, 2)
    
#     return cor1, cor2


# test, target = plt.imread('testing.jpg'), plt.imread('target_2.jpg')
# test_g = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)

fname = sys.argv[1]

target = plt.imread(fname)
target_show = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# cor1, cor2 = get_matches(test, target)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

s_time, e_time = 0, 0

while True:
    
    _, test = cap.read()
    # test_show = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

    # cv2.imshow('output', test)
    test_g = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)

    cor1, cor2 = get_matches(test_g, target_g)

    if len(cor1) < 3 or len(cor2) < 3: 
        e_time = time()
        fps = str(int(1 / (e_time - s_time)))
        cv2.putText(test, fps, (10, 35), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('output', test)
        s_time = e_time
    else:
        e_time = time()
        fps = str(int(1 / (e_time - s_time)))
        output = create_matching_image(test, target_show, cor1, cor2)
        cv2.putText(output, fps, (10, 35), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('output', output)
        s_time = e_time

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
        break



# print (cor1)

# plt.imshow(test)
# plt.plot(*zip(*cor1), marker='o', color='r', ls='')
# plt.show()

# cv2.imshow('image',test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print (type(good[0]))


# plt.imshow(test)
# plt.show()
