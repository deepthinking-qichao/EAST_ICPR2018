'''
#input 
output_dir = "./output/"        #original output dir(txt and picture)
#output 
output_dir = "./output/"        #processed output dir(only txt)
'''

import os
import path
import glob

#original output dir(txt and picture)
output_dir = "./output/"
gtFiles = os.listdir(output_dir)

for file in gtFiles:
    print(file)
    (txt_name, temp) = os.path.splitext(file)
    if (temp==".jpg"):
        os.remove(output_dir+file)

