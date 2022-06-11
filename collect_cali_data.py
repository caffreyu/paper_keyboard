import cv2
import sys
import random

cap = cv2.VideoCapture(-1, cv2.CAP_V4L) # video capture source camera (Here webcam of laptop) 
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# ret,frame = cap.read() # return a single frame in variable `frame`
directory = sys.argv[1]

while(True):

    num = str(random.randint(0, 100000))
    # fname = 'images/logi/auto_off/c' + num + '.png'
    fname = directory + num + '.png'
    ret, frame = cap.read()
    cv2.imshow('Image View: Move the Cheesboard Around if Necessary',frame) # display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): # save on pressing 'y' 
        cv2.imwrite(fname,frame)
        cv2.destroyAllWindows()
        print ('got it')
    elif cv2.waitKey(1) & 0xFF == ord('n'): # quit on pressing 'n' 
        cv2.destroyAllWindows()
        break

cap.release()