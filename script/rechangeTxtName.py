'''
#input
image_dir = "./image_test/*.jpg"     #original images directory 
gt_text_dir = "./txt_test"           #renamed txt directory e.g. txt_1
#output
gt_text_dir = "./txt_test"           #inversely rename txt directory e.g. T1cMkaFMFcXXXXXXXX_!!0-item_pic.jpg> but not <img_***.jpg
'''

import os
import path
import glob
from PIL import Image  
from PIL import ImageDraw

# output txt directory e.g img_***
gt_text_dir = "./txt_test"

# original images directory
image_dir = "./image_test/*.jpg"

imgDirs = []
imgLists = glob.glob(image_dir)

for item in imgLists:
    imgDirs.append(item)

count = 0;
for img_dir in imgDirs:
    img = Image.open(img_dir)  

    img_basename = os.path.basename(img_dir)
    (img_name, temp2) = os.path.splitext(img_basename)

    img_gt_text_name = img_name + ".txt"
    print (img_gt_text_name)

    count +=1
    count_s = str(count) 
    count_s_txt = "img_" + count_s + ".txt"
    os.rename(os.path.join(gt_text_dir,count_s_txt),os.path.join(gt_text_dir,img_gt_text_name))

