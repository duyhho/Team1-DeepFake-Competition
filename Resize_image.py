#!/usr/bin/python
from PIL import Image
import os, sys

path = "./video-to-frames/data/"
try:
    if not os.path.exists('./video-to-frames/data-resized/data'):
        os.makedirs('./video-to-frames/data-resized/data')
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
            newpath = "./video-to-frames/data-resized/data/"
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