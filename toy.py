import cv2
import mediapipe as mp
import time
import pyautogui as gui
import numpy as np
import pickle as pk
import sys

WAIT_TIME = int(sys.argv[1]) # time waited until asking for keyboard input

# Load pickle file
f = open('stored_pattern.pickle', 'rb')
dic = pk.load(f)

time.sleep(int(WAIT_TIME))

# Finger recognition initilization
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(-1)

p_time = 0
c_time = 0

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[12]
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            c_time = time.time()

            if c_time - p_time <= 0.3: continue

            for k, v in dic.items():
                lu, rd = k # lu: left up; rd: right bottom
                xs, ys = lu
                xe, ye = rd

                if xs < cx < xe and ys < cy < ye:
                    print (v)
                    gui.press(v)

                    # Update time
                    p_time = c_time
                    
    cv2.imshow('Image', img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



