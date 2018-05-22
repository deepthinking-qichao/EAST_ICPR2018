'''
#input
gt_text_dir = "./txt_test"             #ground truth directory
image_dir = "./image_test/*.jpg"       #original images directory
#output
imgs_save_dir = "./processImageTest"   #where to save the images with ground truth boxes
'''

#modify:1,how to read picture,read txt in chinese,draw and save picture
#date:2018-5-3

import os
import path
import glob
from PIL import Image  
from PIL import ImageDraw

# ground truth directory
gt_text_dir = "./txt_test"

# original images directory
image_dir = "./image_test/*.jpg"
imgDirs = []
imgLists = glob.glob(image_dir)

# where to save the images with ground truth boxes
imgs_save_dir = "./processImageTest"

for item in imgLists:
    imgDirs.append(item)

for img_dir in imgDirs:
    img = Image.open(img_dir)
    dr = ImageDraw.Draw(img)    

    img_basename = os.path.basename(img_dir)
    (img_name, temp2) = os.path.splitext(img_basename)
    
    # open the ground truth text file
    img_gt_text_name = img_name + ".txt"
    print (img_gt_text_name)
    bf = open(os.path.join(gt_text_dir, img_gt_text_name),encoding='utf-8').read().splitlines()

    for idx in bf:
        rect = []
        spt = idx.split(',')
        rect.append(float(spt[0]))
        rect.append(float(spt[1]))
        rect.append(float(spt[2]))
        rect.append(float(spt[3]))
        rect.append(float(spt[4]))
        rect.append(float(spt[5]))
        rect.append(float(spt[6]))
        rect.append(float(spt[7]))

        # draw the polygon with (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
        dr.polygon((rect[0], rect[1], rect[2], rect[3], rect[4], rect[5], rect[6], rect[7]), outline="red")

    img.save(os.path.join(imgs_save_dir, img_basename))
