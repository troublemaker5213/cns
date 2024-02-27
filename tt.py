
import os
import cv2
import time

paths = ['videos/0628','videos/HP-FCN','videos/IIDNet','videos/STTNet']

for path in paths:

    videolist = os.listdir(path)
    for video in videolist:
        imglist = os.listdir(path+'/'+video)
        for imgname in imglist:
            print(path+'/'+video+'/',imgname)
            im = cv2.imread(path+'/'+video+'/'+imgname)
            
            im = im[:224,:,:]
            cv2.imwrite(path+'/'+video+'/'+imgname,im)
