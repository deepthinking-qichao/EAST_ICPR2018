'''
#input
gt_text_dir = "./traintxt9000/"      # ground truth directory
#output
test_text_dir = "./output/"          # test directory 
'''

import os
import path
import sys
import cv2
import numpy as np
import math

def distance(box1,box2):
	tmp1 = (box1[0][0]-box2[0][0])**2 + (box1[0][1]-box2[0][1])**2
	tmp2 = (box1[1][0]-box2[1][0])**2 + (box1[1][1]-box2[1][1])**2
	tmp3 = (box1[2][0]-box2[2][0])**2 + (box1[2][1]-box2[2][1])**2
	tmp4 = (box1[3][0]-box2[3][0])**2 + (box1[3][1]-box2[3][1])**2
	dist = tmp1**0.5 + tmp2**0.5 + tmp3**0.5 + tmp4**0.5
	return dist

# ground truth directory
gt_text_dir = "./traintxt9000/"

# test directory
test_text_dir = "./output/"

gtList = []
testList = []

gtFiles = os.listdir(gt_text_dir)
simi = 0
distAll = 0
countAll = 0
for file in gtFiles:
    print(file)
    gt_f = open((gt_text_dir+file),encoding='utf-8').read().splitlines()
    test_f = open((test_text_dir+file),encoding='utf-8').read().splitlines()
    print(gt_text_dir+file)
    print(test_text_dir+file)
    distAll1 = 0
    countAll1 = 0
    for idx in gt_f:
        rect = []
        spt = idx.split(',')
        rect.append(int(float(spt[0])))
        rect.append(int(float(spt[1])))
        rect.append(int(float(spt[2])))
        rect.append(int(float(spt[3])))
        rect.append(int(float(spt[4])))
        rect.append(int(float(spt[5])))
        rect.append(int(float(spt[6])))
        rect.append(int(float(spt[7])))

        cnt_gt = np.array([[rect[0],rect[1]],[rect[2],rect[3]],[rect[4],rect[5]],[rect[6],rect[7]]])
        #print(cnt_gt)
        #cnt_test_gt =np.array([[1,1],[2,2],[3,3],[4,4]])
        rect_small_gt = cv2.minAreaRect(cnt_gt)
        #print(cnt_test_gt)
        box_gt = cv2.boxPoints(rect_small_gt)

        distMin = sys.float_info.max
        for idxTest in test_f:
            rectTest = []
            sptTest = idxTest.split(',')
            rectTest.append(int(float(sptTest[0])))
            rectTest.append(int(float(sptTest[1])))
            rectTest.append(int(float(sptTest[2])))
            rectTest.append(int(float(sptTest[3])))
            rectTest.append(int(float(sptTest[4])))
            rectTest.append(int(float(sptTest[5])))
            rectTest.append(int(float(sptTest[6])))
            rectTest.append(int(float(sptTest[7])))

            cntTest = np.array([[rectTest[0],rectTest[1]],[rectTest[2],rectTest[3]],[rectTest[4],rectTest[5]],[rectTest[6],rectTest[7]]])
            rect_small_test = cv2.minAreaRect(cntTest)
            box_test = cv2.boxPoints(rect_small_test)
            
            dist = distance(box_gt,box_test)
            if dist<distMin:
            	distMin = dist
        #print(distMin)
        distAll+= distMin
        countAll=countAll + 1
        distAll1 += distMin
        countAll1 = countAll1 + 1
        print(distAll1/countAll1)
        tmp_simi = math.exp((-0.01)*distAll1/countAll1)*100
        print(tmp_simi)
        simi += tmp_simi
print(countAll)
print(distAll)
simi /= countAll
print(distAll/countAll)
print(simi)





