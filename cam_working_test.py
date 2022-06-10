# Test if camera is still working

import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)

# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
font = cv2.FONT_HERSHEY_SIMPLEX

# print (major_ver)

while True:
    _, img = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = str(int(fps))

    cv2.putText(img, fps, (10, 35), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
 
    cv2.imshow("Webcam", img)

    # print (fps)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


# while True:
#     _, img = cap.read()
#     cv2.imshow("Webcam", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         cv2.destroyAllWindows()
#         break