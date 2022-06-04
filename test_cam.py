import cv2
import sys
import random

cap = cv2.VideoCapture(-1, cv2.CAP_V4L) # video capture source camera (Here webcam of laptop) 
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# ret,frame = cap.read() # return a single frame in variable `frame`

while(True):

    num = str(random.randint(0, 100000))
    fname = 'images/logi/auto_off/c' + num + '.png'
    ret, frame = cap.read()
    cv2.imshow('img1',frame) # display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): # save on pressing 'y' 
        cv2.imwrite(fname,frame)
        cv2.destroyAllWindows()
        print ('got it')
    elif cv2.waitKey(1) & 0xFF == ord('n'): # save on pressing 'y' 
        cv2.destroyAllWindows()
        break

cap.release()