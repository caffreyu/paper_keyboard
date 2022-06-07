# Script for users to load parameters

import cv2
from copy import deepcopy
import tkinter.simpledialog
import tkinter as tk
import tkinter.messagebox
import pickle
import sys

pts = []
inp = []
RED = (0, 0, 255)

def mark_current(fname):

    cap = cv2.VideoCapture(-1)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.imwrite(fname, frame)
            cv2.destroyAllWindows()
            break
    
    cap.release()
        

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        pts.append((x, y))
        curr_img = deepcopy(clone)

        length = len(pts)

        if length > 1: 
            for i in range(0, length, 2):
                if i + 1 >= length: break
                prev, curr = pts[i], pts[i + 1]
                curr_img = cv2.rectangle(curr_img, prev, curr, RED, 5)
        
        curr_img = cv2.circle(curr_img, (x, y), radius = 1, color = RED, thickness = 5)
        cv2.imshow('image', curr_img)

        if length % 2 == 0:

            while 1: 
                window = tk.Tk()
                window.withdraw()
                answer = tkinter.simpledialog.askstring('Input ', 'Type the value for this box:  ')
                
                if answer: 
                    inp.append(answer)
                    break
                else:
                    tkinter.messagebox.showinfo('Warning', 'Space is not a valid value!')
                    # greetings = tk.Label(text='Type the value for this box: ')
                    # greetings.pack()

                window.update()
            
    
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:

        curr_img = deepcopy(clone)

        if pts: pts.pop(-1)

        if pts:
            length = len(pts)
            
            if length > 1: 
                for i in range(0, length, 2):
                    if i + 1 >= length: break
                    prev, curr = pts[i], pts[i + 1]
                    curr_img = cv2.rectangle(curr_img, prev, curr, RED, 5)

            curr_pt = pts[-1]
            curr_img = cv2.circle(curr_img, curr_pt, radius = 1, color = RED, thickness = 5)

            if len(inp) != 0 and len(inp) * 2 != len(pts): inp.pop(-1)
    
        cv2.imshow('image', curr_img)

        # if len(inp) > 0 and len(pts) % 2 == 0: inp.pop(-1)
 
    # print (pts, inp)
 
if __name__ == "__main__":
 
    # reading parameters
    method, fname = sys.argv[1], sys.argv[2]
    
    # read from image
    if method == 'read' or method == 'r': 
        img = cv2.imread(fname)
    
    # create image
    if method == 'create' or method == 'c': 
        mark_current(fname)
        img = cv2.imread(fname)

    clone = deepcopy(img)
 
    # displaying the image
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
 
    # closing and shutdown
    k = cv2.waitKey(0)
    if k == 'q': cv2.destroyAllWindows()

    dic = {}

    for i in range(len(inp)):
        if 2 * i + 1 <= len(pts) - 1: dic[(pts[2 * i], pts[2 * i + 1])] = inp[i]
        else: break
    
    print (dic)

    with open('stored_pattern.pickle', 'wb') as f:
        pickle.dump(dic, f)