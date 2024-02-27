import cv2
from facenet_pytorch import MTCNN
from os import makedirs
from skimage.io import imsave
from os.path import join, exists
from PIL import Image
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')

training_videos_folder = r"../videos/deepfakes/"

dest_fram_folder = r"../videos/copy_match/"

real_video_file = r"../videos/test/"

mtcnn = MTCNN(

    margin=400,

    select_largest=False,

    post_process=False,

    device="cuda:0",image_size=512

)


videos_path = glob.glob(join(training_videos_folder, "*.mp4"))



m=0


for video_path in videos_path:
    m=m+1
    # print("video", video_path, "is doing", "----", str(m / 10), "%")
    cap1 = cv2.VideoCapture(video_path)


    if not exists( dest_fram_folder+ "/"+ str(m) ):

        makedirs(dest_fram_folder+ "/"+ str(m) )


    picId=0
    while cap1.isOpened() :
        frameId1 = cap1.get(1)  # current frame number
        if picId>17:
            break

        ret1, frame1 = cap1.read()

        if not ret1 :
            break

        if frameId1%2!=0:
            continue

        filename1 = (

            dest_fram_folder + "/"+ str(m)

            + "/image_"

            + str(int(picId) + 1)

            + ".png"

        )
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1 = Image.fromarray(frame1)
        frame1 = mtcnn(frame1)
        try:
            imsave(filename1, frame1.permute(1, 2, 0).numpy().astype(np.uint8))
        except AttributeError:
            continue
        picId=picId+1
    cap1.release()
