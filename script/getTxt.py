'''
gt_text_dir="./txt_9000"                   #original ground true labels
image_dir = "./image_9000/*.jpg"           #original image(Assume that the picture is in jpg format. 
                                            If the picture is in another format, please change the suffix of the picture.
revised_text_dir = "./trainData"           #Rename txt for EAST and make the coordinate of detected text block in txt clockwise
imgs_save_dir = "./trainData"              #Renane image for EAST which must be in 3 channels
'''
import os
import path
import glob
from PIL import Image  
from PIL import ImageDraw

# ground truth directory
gt_text_dir = "./txt_9000"

# revised txt directory
revised_text_dir = "./trainData"

# original images directory
image_dir = "./image_9000/*.jpg"

imgDirs = []
imgLists = glob.glob(image_dir)

# revised txt directory
imgs_save_dir = "./trainData"

for item in imgLists:
    imgDirs.append(item)

count = 0;
for img_dir in imgDirs:
    img = Image.open(img_dir)
    #if you want to draw output images with groud true labels,you can remove the comment below
    #dr = ImageDraw.Draw(img)    

    # get images
    img_basename = os.path.basename(img_dir)
    (img_name, temp2) = os.path.splitext(img_basename)

    # open the ground truth text file
    img_gt_text_name = img_name + ".txt"
    print (img_gt_text_name)

    #bf = open(os.path.join(gt_text_dir, img_gt_text_name)).read().decode("utf-8-sig").encode("utf-8").splitlines()
    bf = open(os.path.join(gt_text_dir, img_gt_text_name),encoding='utf-8').read().splitlines()

    #rename images original name to img_***,e.g img_1
    count +=1
    count_s = str(count)
    count_s_txt = "img_" + count_s + ".txt"
    count_s_img = "img_" + count_s + ".jpg" 
    f_revised = open(os.path.join(revised_text_dir, count_s_txt),mode = 'w',encoding='utf-8') 

    #rename and clockwise orgianl txt
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

        #dr.polygon((rect[0], rect[1], rect[2], rect[3], rect[4], rect[5]), outline="red")

        #clockwise adjustment
        #(x2-x1)*(y3-y2)-(y2-y1)*(x3-x2)
        clockwiseFlag = (rect[2]-rect[0])*(rect[5]-rect[3])-(rect[3]-rect[1])*(rect[4]-rect[2])
        if clockwiseFlag < 0:
            tmp_x2 = rect[2]
            tmp_y2 = rect[3]
            rect[2] = rect[6]
            rect[3] = rect[7]
            rect[6] = tmp_x2
            rect[7] = tmp_y2
            tmp_x2_s = spt[2]
            tmp_y2_s = spt[3]
            spt[2] = spt[6]
            spt[3] = spt[7]
            spt[6] = tmp_x2_s
            spt[7] = tmp_y2_s

        #write txt for clockwise adjustment
        sep= ','
        s1 = sep.join(spt)
        s1= s1 + '\n'
        f_revised.write(s1)

        # draw the polygon with (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
        #if you want to draw output images with groud true labels,you can remove the comment below
        #dr.polygon((rect[0], rect[1], rect[2], rect[3], rect[4], rect[5], rect[6], rect[7]), outline="red")
        #dr.polygon((rect[0], rect[1], rect[2], rect[3], rect[4], rect[5]), outline="green")

    img.save(os.path.join(imgs_save_dir, count_s_img))
