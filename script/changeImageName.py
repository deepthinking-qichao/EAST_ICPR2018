'''
#input
image_dir = "./image_test/*.jpg"                         #orignial images name(perhaps abnormal)
#output
imgs_save_dir = "./image_test_change"                    #renamed images(e.g. img_1.jpg)
'''

import os
import path
import glob
from PIL import Image  
from PIL import ImageDraw


# original images directory
image_dir = "./image_test/*.jpg"
imgDirs = []
imgLists = glob.glob(image_dir)

# where to save the images with ground truth boxes
imgs_save_dir = "./image_test_change"

for item in imgLists:
    imgDirs.append(item)

count = 0;
for img_dir in imgDirs:
    img = Image.open(img_dir)
    count +=1
    count_s = str(count)
    count_s_img = "img_" + count_s + ".jpg"
    img.save(os.path.join(imgs_save_dir, count_s_img))
