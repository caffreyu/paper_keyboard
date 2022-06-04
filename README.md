# paper_keyboard
Convert a piece of paper to a keyboard.

UCSD ECE285 Introduction to Visual Learning (Spring 2022)
Member: Hao Yu; Weilai Wei

# Instructions on Installation and Usage

__Installation__

Create a new directory and push this repo to that directory. Install the required packages. Give permission to the shell script file. 

**This package is tested on Ubuntu 20.04. It should work on Ubuntu 18.04 as well.**

```
mkdir -p ~/paper_keyboard
cd ~/paper_keyboard
git init
git pull https://github.com/caffreyu/paper_keyboard.git
pip3 install -r requirements.txt
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

__Usage__

Pull out a piece of paper. Draw some characters on it. Or actually anything. ;)

Put that paper under a camera. It can be a webcame. It can be your laptop's camera. 

```
cd ~/paper_keyboard
```

___If this is your first time using it___

```
python3 test_bbox.py create test_keyboard.png
```

___If you already used it and wanted to use a keyboard you used before___

```
python3 test_bbox.py read $PATH_TO_YOUR_IMAGE$
```

When the window of the camera view popped up, try your best to make sure every character is on sight of the camera. 

Right click on the image and right click again to draw bounding box for characters on the screen. A screen will pop up for 
to enter information you entered in that box area. 

If you messed up, no worries! Just left click your mouse. The process will be reverted and you can reselect the end point for the 
bounding box. If you messed up with the initial selection, left click the mouse again. 

<br>

**First Iteration (a toy implementation)**

```
python3 toy.py
```

