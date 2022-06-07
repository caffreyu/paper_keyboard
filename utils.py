import numpy as np
import cv2
# from cv2 import imread, cvtColor
import matplotlib.pyplot as plt
import random
from random import randint as rd
from copy import deepcopy

def get_matches(img1, img2):
    sift = cv2.SIFT_create()
    cor1, cor2 = [], []

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print (len(kp1), len(kp2))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            # print (type(m))
    
    print ('Done getting corners from two images')
    print (f'Before processing total number: {len(good)}')
    
    for match in good[: 30]:
        cor1.append(kp1[match.queryIdx].pt)
        cor2.append(kp2[match.trainIdx].pt)
    
    cor1, cor2 = np.array(cor1).reshape(-1, 2), np.array(cor2).reshape(-1, 2)
    
    return cor1, cor2


def create_matching_image(img1, img2, corners1, corners2):
    
    # grey
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    height = max(h1, h2)
    width = w1+w2
    matching_img = np.zeros((height, width, 3), dtype=img1.dtype)
    
    matching_img[: h1, : w1] = img1[:, :]
    matching_img[: h2, w1 : w1 + w2] = img2[:, :]
    
    for i in range(corners2.shape[0]):
        p1, p2 = (int(corners1[i, 0]), int(corners1[i, 1])), (int(corners2[i, 0]) + w1, int(corners2[i, 1]))
        R, G, B = rd(0, 255), rd(0, 255), rd(0, 255)
        cv2.line(matching_img, p1, p2, (R, G, B), thickness = 2)

    return matching_img


def cv_show(img):
    cv2.imshow('output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    _,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        # img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        # img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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

        # x1_n[0], x1_n[1] = x1_n[0] / h1, x1_n[1] / w1
        # x2_n[0], x2_n[1] = x2_n[0] / h2, x2_n[1] / w2
        # print (x2_n.T @ F)
        print (x2_n.T @ F @ x1_n)

        l1, l2 = np.dot(F.T, x2_n), np.dot(F, x1_n)

        # y1, y2 = -x * l1[0] / l1[1] - l1[2] / l1[1], -x * l2[0] / l2[1] - l2[2] / l2[1]
        # print ('First image: ', x, y1)
        # print ('Second image: ', x, y2)

        ax1.plot(x, -x * l1[0] / l1[1] - l1[2] / l1[1], c = color)
        ax1.scatter(x1[0], x1[1], s = 35, edgecolors = 'b', facecolors = color)
        ax2.plot(x, -x * l2[0] / l2[1] - l2[2] / l2[1], c = color)
        ax2.scatter(x2[0], x2[1], s = 35, edgecolors = 'b', facecolors = color)
        
        # x1[0], x1[1] = x1[0] / h1, x1[1] / w1
        # x2[0], x2[1] = x2[0] / h2, x2[1] / w2
        # print (x2.T @ F @ x1)

    
    plt.show()
