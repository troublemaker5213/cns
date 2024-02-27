import cv2
import os
# 指定目录下所有jpg文件名

import os
import imageio
from PIL import Image
img_dir = "C://Users/26906/Desktop/video/6a37a91708"
import os
import cv2
import time


imglist="C:/Users/26906/Desktop/video/258b3b33c6"
imglist=os.listdir(imglist)
fps = 8
    # file_path='saveVideo.avi' # 导出路径MJPG
    # file_path='saveVideo'+str(int(time.time()))+'.mp4' # 导出路径DIVX/mp4v
save_path='save_orign_video/A-258b3b33c6.mp4' # 导出路径DIVX/mp4v


size=(432,240)

    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4

videoWriter = cv2.VideoWriter(save_path,fourcc,fps,size)

    # 这种情况更适合于照片是从"1.jpg" 开始，然后每张图片名字＋1的那种
    # for i in range(8):
    #     frame = cv2.imread(img_root+str(i+1)+'.jpg')
    #     videoWriter.write(frame)
print(imglist)
for item in imglist:
    if item.endswith('.jpg'):   #判断图片后缀是否是.jpg
        item ="C:/Users/26906/Desktop/video/258b3b33c6"+ '/' + item
        img = cv2.imread(item) #使V用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            # print(type(img))  # numpy.ndarray类型

        videoWriter.write(img)        #把图片写进视频
videoWriter.release() #释放