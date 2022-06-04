import numpy as np
import cv2
# from cv2 import imread, cvtColor
import matplotlib.pyplot as plt
import random
from random import randint as rd

def get_matches(img1, img2):
    sift = cv2.SIFT_create()
    cor1, cor2 = [], []

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    good = []

    for m, n in matches:
        if m.distance < 0.60 * n.distance:
            good.append(m)
            # print (type(m))
    
    print ('Done getting corners from two images')
    print (f'Before processing total number: {len(good)}')
    
    for match in good[: 20]:
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

