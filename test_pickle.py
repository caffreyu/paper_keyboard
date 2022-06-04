import pickle as pk
import pyautogui as gui
import time

f = open('stored_pattern.pickle', 'rb')
dic = pk.load(f)

print (dic)
time.sleep(3)

arr = [(500, 200), (500, 200), (500, 200), (500, 400)]

for x, y in arr:
    print (x, y)
    judge = True
    for k, v in dic.items():
        lu, rd = k
        xs, ys = lu
        xe, ye = rd
        if xs < x < xe and ys < y < ye:
            print ('Current input is ' + v)
            gui.press(v)
            judge = False
            break
    if judge: print ('No input found')
    time.sleep(0.2)