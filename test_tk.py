import tkinter as tk
import tkinter.simpledialog

window = tk.Tk()

answer = tkinter.simpledialog.askstring('Input ', 'Type the value for this box:  ')

# def get_answer():
#     answer = tkinter.simpledialog.askstring('Input ', 'Type the value for this box:  ')

# button = tk.Button(window, text = 'Bounding Box Input', command = get_answer())

# greeting = tk.Label(text="Type the value for this box: ")
# button.pack()

print (answer)

window.mainloop(1)
