from ctypes import c_short
import cv2
import mediapipe as mp
import time
import pyautogui as gui
import numpy as np
import pickle as pk
import sys
import csv

if len(sys.argv) > 1:
    WAIT_TIME = int(sys.argv[1]) # time waited until asking for keyboard input
else: 
    WAIT_TIME = 5

fname = 'record.csv'
fh = open(fname, 'w', newline = '')
csvwriter = csv.writer(fh)

# Load pickle file
f = open('./data/pickle/stored_pattern.pickle', 'rb')
dic = pk.load(f)

time.sleep(int(WAIT_TIME))

# Finger recognition initilization
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(-1)

p_time = 0
c_time = 0

p_stamp = 0
c_stamp = 0

p_press = 0
c_press = 0

p_x, p_y = 0, 0
p_char = ''

# strength = {'S' : (8, 50), 'W' : (10, 40), 'D' : ((0, 5), (10, 20)), 'A' : (10, 50), 'space': ((0, 5), (10, 20))}

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[8]
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            csvwriter.writerow([cx, cy])
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            for k, v in dic.items():
                lu, rd = k # lu: left up; rd: right bottom
                xs, ys = lu
                xe, ye = rd

                if xs < cx < xe and ys < cy < ye:

                    if p_char == '': p_char = v
                    elif p_char != v: 
                        p_stamp = time.time()
                        p_x, p_y = 0, 0
                        p_char = v

                    cv2.putText(img, '(' + str(cx) + ', ' + str(cy) + ')', (cx - 10, cy - 10), 
                                cv2.FONT_HERSHEY_PLAIN, 1.5,
                                (255, 0, 255), 2)

                    c_stamp = time.time()

                    if c_stamp - p_stamp < 0.1: break

                    if p_x == p_y == 0:
                        p_time = time.time()
                        p_x, p_y = cx, cy
                        break
                    else:
                        # low_s, high_s = strength[v]
                        # if low_s <= abs(p_x - cx) + abs(p_y - cy) <= high_s:
                        if p_x - 10 <= cx <= p_x + 10 and p_y + 8 <= cy <= p_y + 20:
                            c_press = time.time()
                            if c_press - p_press < 0.1: break
                            print (v)
                            gui.press(v)
                            p_x, p_y = 0, 0
                            p_press = c_press
                        else:
                            c_time = time.time()
                            if c_time - p_time > 0.3:
                                p_x, p_y = 0, 0
                        break

    cv2.imshow('Image', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break