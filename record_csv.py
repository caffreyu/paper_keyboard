import csv
import mediapipe as mp
import tty
import sys
import termios
import cv2
import matplotlib.pyplot as plt

# csv
fname = sys.argv[1]
index = 0

# cv2 cap
cap = cv2.VideoCapture(-1)

# raw input
orig_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)

# finger recognition initilization
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while 1:
    x = sys.stdin.read(1)[0]
    
    if x == 'y': 
        print ('start recording')
        record_pos = []

        while 1: 

            success, img = cap.read()

            if img[0, 0, 0] == 0: continue
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            # csv_writer.writerow(['wtf'])

            # curr_fname = './data/txt/' + fname + str(index) + '.txt'
            # fh = open(curr_fname, 'w', newline = '\n')

            if results.multi_hand_landmarks:

                for handLms in results.multi_hand_landmarks:
                    lm = handLms.landmark[12]
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print ([cx, cy])
                    record_pos.append([cx, cy])
                    # csv_writer.writerow([cx, cy])
                    # fh.flush()
                    # fh.write(' '.join([str(cx), str(cy)]))
                    # fh.write(str(cx))
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                # csv_writer.writerows(record_pos)
                print (record_pos)
                curr_fname = './data/csv/' + fname + str(index) + '.csv'
                with open(curr_fname, 'w', newline = '\n') as fh:
                    csv_writer = csv.writer(fh)
                    csv_writer.writerows(record_pos)
                fh.close()
                index += 1
                break

        
    elif x == 'n':
        print ('Do you want to record another one? \nPress "y" to record another file. ')
        judge = sys.stdin.read(1)[0]
        print (judge)
        if judge != 'y': break

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings) 