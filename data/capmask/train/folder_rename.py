from os.path import join
from glob import glob
import os, sys
import cv2

d="capyesmaskyes"

dataset_dir = "F:/ObjectDetection/darknet/darknet_master4/build/darknet/x64/data/capmask/train/capmask2/" + d

newdir = d+"/"

if not os.path.exists(newdir):
    os.makedirs(newdir)

train_input_names = []
for ext in ('*.gif', '*.png', '*.jpg'):
    train_input_names.extend(glob(join(dataset_dir, ext)))

i=0
for name in train_input_names:
    print(name)
    newnam= newdir+str(i).zfill(4) + "_"+d+".jpg"
    img = cv2.imread(name)
    img=cv2.resize(img,(32,32))
    cv2.imwrite(newnam,img)
    i+=1


