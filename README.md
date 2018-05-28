# EAST_ICPR2018: EAST for ICPR MTWI 2018 Challenge II (Text detection of network images)

### Introduction
This is a repository forked from [argman/EAST](https://github.com/argman/EAST) for the [ICPR MTWI 2018 Challenge II](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409391.333.2.7cb749ecEmHnUn&raceId=231651).
<br>Origin Repository: [argman/EAST - EAST: An Efficient and Accurate Scene Text Detector](https://github.com/argman/EAST). It is a tensorflow re-implementation of EAST: An Efficient and Accurate Scene Text Detector.
<br>Origin Author: [argman](https://github.com/argman)

This repository also refers to [HaozhengLi/EAST_ICPR](https://github.com/HaozhengLi/EAST_ICPR)
<br>Origin Repository: [HaozhengLi/EAST_ICPR](https://github.com/HaozhengLi/EAST_ICPR). 
<br>Origin Author: [HaozhengLi](https://github.com/HaozhengLi).

Author: [Qichao Wu](https://github.com/deepthinking-qichao)
<br>Email: 467736413@qq.com or 13660479414@163.com

### Contents
1. [Dataset and Transform](#dataset-and-transform)
2. [Models](#models)
3. [Demo](#demo)
3. [Train](#train)
4. [Test](#test)
5. [Results](#results)

### Dataset and Transform
the dataset for model training include [ICDAR 2017 MLT (train + val)](http://rrc.cvc.uab.es/?ch=8&com=downloads), [RCTW-17 (train)](http://www.icdar2017chinese.site:5080/dataset/) and [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4ec66a80qvIKLc&raceId=231651). 
Among them, [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4ec66a80qvIKLc&raceId=231651) include 9000 train data <[ICPR_text_train_part2_20180313](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.bc556a80HmESyP&raceId=231651)> and 1000 validate data <[(update)ICPR_text_train_part1_20180316](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.bc556a80HmESyP&raceId=231651)>.

Some data in the dataset is abnormal for [argman/EAST](https://github.com/argman/EAST), just like [ICPR_text_train_part2_20180313](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.bc556a80HmESyP&raceId=231651) or [(update)ICPR_text_train_part1_20180316](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.bc556a80HmESyP&raceId=231651).
Abnormal means that the ground true labels are anticlockwise, or the images are not in 3 channels. Then errors like ['poly in wrong direction'](https://github.com/argman/EAST/issues?utf8=%E2%9C%93&q=poly+in+wrong+direction) will occur while using [argman/EAST](https://github.com/argman/EAST).

Images and ground true labels files must be renamed as <img_1>, <img_2>, ..., <img_xxx> and <txt_1>, <txt_2>, ..., <txt_xxx> while using argman/EAST to train or test
Because Names of the images and txt in [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4ec66a80qvIKLc&raceId=231651) are abnormal. Like <T1cMkaFMFcXXXXXXXX_!!0-item_pic.jpg> but not <img_***.jpg>. Then errors will occur while using [argman/EAST#test](https://github.com/argman/EAST#test).

So I wrote a python program to check and transform the dataset. The program named <[getTxt.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/getTxt.py)> is in the folder 'script/' and its parameters are descripted as bellow:
```
#input
gt_text_dir="./txt_9000"                   #original ground true labels 
image_dir = "./image_9000/*.jpg"           #original image which must be in 3 channels(Assume that the picture is in jpg format. If the picture is in another format, please change the suffix of the picture.
#output
revised_text_dir = "./trainData"           #Rename txt for EAST and make the coordinate of detected text block in txt clockwise
imgs_save_dir = "./trainData"              #Rename image for EAST 
```

Before you run [getTxt.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/getTxt.py) to transform the dataset for [argman/EAST](https://github.com/argman/EAST), you should make sure that the original images are all in 3 channels. I write a cpp file to selete the abnormal picture(not in 3 channels) from the dataset. 
The program named <[change_three_channels.cpp](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/change_three_channels.cpp)> is in the folder 'script/' and its parameters are descripted as bellow:
```
string dir_path = "./image_9000/";             //original images which include abnomral images
string output_path = "./output/";              //abnormal images which is in three channels 
```
When you get the output abnormal images from [getTxt.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/getTxt.py), please transform them to normal ones through other tools like [Format Factory](http://www.pcfreetime.com/) (e.g. Cast to jpg format in Format Factory)

I have changed [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4ec66a80qvIKLc&raceId=231651) for [EAST](https://github.com/argman/EAST). Their names are [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA) which include 9000 train images+txt and [ICPR2018_validation](https://pan.baidu.com/s/1ZFT0qjqAWrsRlvJFPUjQYA) which include 1000 validate images+txt.
<br>I have also changed [ICDAR 2017 MLT (train + val)](http://rrc.cvc.uab.es/?ch=8&com=downloads) for [EAST](https://github.com/argman/EAST). Their names are [ICDAR2017_training](https://pan.baidu.com/s/1MbUPjRfeizYFIB_l9cV-Ag) which include 1600 train images+txt and [ICDAR2017_validation](https://pan.baidu.com/s/1CIWwq7mprDh2x1HUpvsw1Q) which include 400 images+txt.
<br>I have changed [RCTW-17 (train)](http://www.icdar2017chinese.site:5080/dataset/) but it's too large to upload so maybe you change yourself.

### Models
1. Use [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA) and 0.0001 learning rate to train Resnet_V1_50 model which is pretrained by ICDAR 2013 (train) + ICDAR 2015 (train). The pretrained model is provided by [argman/EAST](https://pan.baidu.com/s/1jHWDrYQ?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=), it is trainde by 50k iteration.
<br>The 100k iteration model is [50net-100k](https://pan.baidu.com/s/1hi7dsTTLSlDS0QfvRoGjxA), 270k iteration model is [50net-270k](https://pan.baidu.com/s/1d2ajQiWyxSpJfmIzu6VBvg), 900k iteraion model is [50net-900k](https://pan.baidu.com/s/1N3ljQkAB-T2ZuEeysjSt7w)
2. Use [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA), [ICDAR2017_training](https://pan.baidu.com/s/1MbUPjRfeizYFIB_l9cV-Ag), [ICDAR2017_validation](https://pan.baidu.com/s/1CIWwq7mprDh2x1HUpvsw1Q), [RCTW-17 (train)](http://www.icdar2017chinese.site:5080/dataset/) and 0.0001 learing rate to train Resnet_V1_101 model. The pretrainede model is [slim_resnet_v1_101](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) provided by tensorflow slim.
<br>The 230k iteration model is [101net-mix-230k](https://pan.baidu.com/s/1nBvduv129-lKkx-jHK9G7g)
3. Use [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA), [ICDAR2017_training](https://pan.baidu.com/s/1MbUPjRfeizYFIB_l9cV-Ag), [ICDAR2017_validation](https://pan.baidu.com/s/1CIWwq7mprDh2x1HUpvsw1Q), [RCTW-17 (train)](http://www.icdar2017chinese.site:5080/dataset/) and 0.001 learing rate to train Resnet_V1_101 model. The pretrainede model is [101net-mix-230k](https://pan.baidu.com/s/1nBvduv129-lKkx-jHK9G7g).
<br>The 330k iteration model is [101net-mix-10*lr-330k](https://pan.baidu.com/s/1NF1cnCIsoL2ItQmUetYoOA)
4. Use [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA) and 0.0001 learing rate to train Resnet_V1_101 model. The pretrainede model is [mix-10lr-330k](https://pan.baidu.com/s/1NF1cnCIsoL2ItQmUetYoOA).
<br>The 460k iteration model is [101net-460k](https://pan.baidu.com/s/1DUaT9zHPEtHvhqqAMshbPQ)
5. Use [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA) and 0.0001 learing rate to train Resnet_V1_101 model. The pretrainede model is [101net-mix-230k](https://pan.baidu.com/s/1nBvduv129-lKkx-jHK9G7g).
<br>The 300k iteration model is [101net-300k](https://pan.baidu.com/s/1uGyLN946lpU1ImHdswUfAA), 400k iteration model is [101net-400k](https://pan.baidu.com/s/1kp_efup3QCTqnO8g0fViFg), 500k iteration model is [101net-500k](https://pan.baidu.com/s/12DJRMYAM4Q28wPaPaXm1eA), 550k iteraion model is [101net-550k](https://pan.baidu.com/s/1AwsbwWEgoKrh0XxmzObifg)
6. Use [ICPR2018_training](https://pan.baidu.com/s/1TnWta5-I1dWOWBqEaCVUsA) and 0.0001 learing rate with data argument to train Resnet_V1_101 model. The pretrainede model is [101net-550k](https://pan.baidu.com/s/1AwsbwWEgoKrh0XxmzObifg).
<br>The 700k iteration model is [101net-arg-700k](https://pan.baidu.com/s/1UpgxGQtHswmLDT8q1xhxrg), 1000k iteration model is [101net-arg-1000k](https://pan.baidu.com/s/1qrQiPBugDcLD13n-hulVGw)

### Demo
Download the pre-trained models and run:
```
python run_demo_server.py --checkpoint-path models/east_icpr2018_resnet_v1_50_rbox_100k/
```
Then Open http://localhost:8769 for the web demo server, or get the results in 'static/results/'.
<br>***Note: See [argman/EAST#demo](https://github.com/argman/EAST#demo) for more details.***

### Train
Prepare the training set and run:
```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=/tmp/resnet_v1_50.ckpt
```
***Note 1: Images and ground true labels files must be renamed as <img_1>, <img_2>, ..., <img_xxx> while using [argman/EAST](https://github.com/argman/EAST). Please see the examples in the folder 'training_samples/'.
<br>Note 2: If ```--restore=True```, training will restore from checkpoint and ignore the ```--pretrained_model_path```. If ```--restore=False```, training will delete checkpoint and initialize with the ```--pretrained_model_path``` (if exists).
<br>Note 3: If you want to change the learning rate during training, your setting learning rate in the command line is equal to the learning rate which you want to set in current step divided by the learning rate in current step times original learing rate setted in the command line 
<br>Note 4: See [argman/EAST#train](https://github.com/argman/EAST#train) for more details.***

when you use Resnet_V1_101 model, you should modify three parts of code in [argman/EAST](https://github.com/argman/EAST).
1.model.py
```
with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
    # logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
    logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')
```
2.nets/resnet_v1.py
```
if __name__ == '__main__':
    input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
    with slim.arg_scope(resnet_arg_scope()) as sc:
        # logits = resnet_v1_50(input)
        logits = resnet_v1_101(input)
```
3.nets/resnet_v1.py
```
try:
    # end_points['pool3'] = end_points['resnet_v1_50/block1']
    # end_points['pool4'] = end_points['resnet_v1_50/block2']
    end_points['pool3'] = end_points['resnet_v1_101/block1']
    end_points['pool4'] = end_points['resnet_v1_101/block2']
except:
    #end_points['pool3'] = end_points['Detection/resnet_v1_50/block1']
    #end_points['pool4'] = end_points['Detection/resnet_v1_50/block2']
    end_points['pool3'] = end_points['Detection/resnet_v1_101/block1']
    end_points['pool4'] = end_points['Detection/resnet_v1_101/block2']
```

when you use data argument, you should add two parts of code [argman/EAST](https://github.com/argman/EAST).

1.nets/resnet_v1.py
```
#add before resnet_v1 function
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise/250
```

2.nets/resnet_v1.py
```
with slim.arg_scope([slim.batch_norm], is_training=is_training):
	inputs=gaussian_noise_layer(inputs,1)								#add gaussian noise data argument
	inputs=tf.image.random_brightness(inputs,32./255)                   #add brightness data argument
	inputs=tf.image.random_contrast(inputs,lower=0.5,upper=1.5)         #add contrast data argument
	net = inputs
```

### Test
when you use [argman/EAST](https://github.com/argman/EAST) for testing, Names of the images in [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4ec66a80qvIKLc&raceId=231651) are abnormal. Like <T1cMkaFMFcXXXXXXXX_!!0-item_pic.jpg> but not <img_***.jpg>.
Then errors will occur while using [argman/EAST#test](https://github.com/argman/EAST#test).
<br> So I wrote a python programs to rename and inversely rename the dataset. Before evaluating, run the program named <[changeImageName.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/changeImageName.py)> to make names of the images normal. This program is in the folder 'script/' and its parameters are descripted as bellow:
```
#input
image_dir = "./image_test/*.jpg"                         #orignial images name(perhaps abnormal e.g <T1cMkaFMFcXXXXXXXX_!!0-item_pic.jpg>)
#output
imgs_save_dir = "./image_test_change"                    #renamed images(e.g. <img_1.jpg>)
```

After evaluating, the output file folder contain images with bounding boxes and txt. If I want to get the original name of txt, we should delete the images in the output file folder and inversely rename the txt.
<br> So I wrote two python programs to get the original name of txt. First, run the program named <[deleteImage.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/deleteImage.py)> to delete the images in folder. This program is in the folder 'script/' and its parameters are descripted as bellow:
```
#input 
output_dir = "./output/"        #original output file folder(txt and images)
#output 
output_dir = "./output/"        #processed output file folder(only txt)
```
Second, run the program named <[rechangeTxtName.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/rechangeTxtName.py)> to inversely rename the txt in output folder. This program is in the folder 'script/' and its parameters are descripted as bellow:
```
#input
image_dir = "./image_test/*.jpg"     #original images  
gt_text_dir = "./txt_test"           #the folder which contain renamed txt e.g. <txt_1>
#output
gt_text_dir = "./txt_test"           #the folder which contain inversely renamed txt e.g. <T1cMkaFMFcXXXXXXXX_!!0-item_pic.jpg> but not <img_1.jpg>
```

If you want to see the output result on the image, you can draw the output bounding boxes on the origanl image.
<br> So I wrote a python programs to read picture and txt coompatibel with Chinese, then draw and save images with output bounding boxes. This program named <[check.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/check.py)> is in the folder 'script/' and its parameters are descripted as bellow:
#input
gt_text_dir = "./txt_test"             #output labels(bounding boxes) folder
image_dir = "./image_test/*.jpg"       #original images folder
#output
imgs_save_dir = "./processImageTest"   #where to save the images with output bounding boxes. This program is in the folder 'script/' and its parameters are descripted as bellow:

I wrote a python programs to evaluate the output performance. The program named <[getACC.py](https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/script/getACC.py)> is in the folder 'script/' and its parameters are descripted as bellow:
```
#input
gt_text_dir = "./traintxt9000/"      # ground truth directory
#output
test_text_dir = "./output/"          # output directory 
```

Finally, If you want to compress the output txt in order to submit, you can run the command 'zip -r sample_task2.zip sample_task2' to get the .zip file

### Results
Here are some results on [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.4ec66a80qvIKLc&raceId=231651):
<br><div align=center><img width="800" height="800" src="https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/show_image/T1..U9FjReXXXXXXXX_!!0-item_pic.jpg.jpg"/></div>
<br><div align=center><img width="800" height="800" src="https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/show_image/T1.0RlFShdXXXXXXXX_!!0-item_pic.jpg.jpg"/></div>
<br><div align=center><img width="800" height="800" src="https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/show_image/T1.5LCXXxnXXaUXTcW_024647.jpg.jpg"/></div>
<br><div align=center><img width="800" height="800" src="https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/show_image/T1.kCuXiR2XXXaiR6a_091915.jpg.jpg"/></div>
<br><div align=center><img width="800" height="800" src="https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/show_image/T1Akq3XhFmXXc9IAza_121724.jpg.jpg"/></div>
<br><div align=center><img width="800" height="800" src="https://github.com/deepthinking-qichao/EAST_ICPR2018/blob/master/show_image/T1SJ96Xm0uXXXXXXXX_!!0-item_pic.jpg.jpg"/></div>

# *Hope this helps you*



