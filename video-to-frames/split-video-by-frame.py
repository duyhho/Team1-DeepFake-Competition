'''
Using OpenCV takes a mp4 video and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('example2.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(currentFrame<50):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#!/usr/bin/python
from PIL import Image
import os, sys

path = "./data/"
try:
    if not os.path.exists('./data-resized/data'):
        os.makedirs('./data-resized/data')
except OSError:
    print ('Error: Creating directory of data')
resize_ratio = 0.5
def resize_aspect_fit():
    dirs = os.listdir(path)
    for item in dirs:
        print(item)
        if item == '.jpg':
            continue
        if os.path.isfile(path+item):
            newpath = "./data-resized/data/"
            newfile_path = newpath + item
            file_path, extension = os.path.splitext(newfile_path)
            resized_filename = file_path + "_resized" + extension
            image = Image.open(path+item)

            # new_image_height = int(image.size[0] / (1/resize_ratio))
            # new_image_length = int(image.size[1] / (1/resize_ratio))
            #
            # image = image.resize((new_image_height, new_image_length), Image.ANTIALIAS)
            image = image.resize((64, 64), Image.ANTIALIAS)
            image = image.convert("RGB")
            image.save(resized_filename, 'JPEG', quality=90)
            # if os.path.isfile(resized_filename) == False:
            #     # print("SAVED!")
            #     image.save(resized_filename, 'JPEG', quality=90)
            # else:
            #     print("EXISTED!")


resize_aspect_fit()