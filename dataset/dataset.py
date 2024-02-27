from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch
from utils.lap4_n import lap4_kernel_n
from torchvision import transforms
import zipfile
import io
import numpy as np
import json
import os
import random

class MyDataSet(Dataset):
    def __init__(self, k=4, train=True, transform=None, transform_label=None, dataset_name="youtubevos"):
        self.transform = transform
        self.trainform_label = transform_label
        self.dataset_name = dataset_name
        path = f"data/{self.dataset_name}/train" if train else f"data/{self.dataset_name}/test"
        path = os.path.join(os.getcwd(), path)
        self.dataset = get_dataset(path, k=k)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        frames = data['frames']
        mask = data['mask']
        video_name = data['video_name']

        #label = Image.open(mask)
        label = cv2.imread(mask, flags=cv2.IMREAD_GRAYSCALE)  # [240, 432]
        imgs = []
        laps = []
        for i in frames:
            #img = Image.open(i)
            img = cv2.imread(i)
            lap = lap4_kernel_n(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [240,432,3]
            #img = self.transform(img)*2.0-1.0  # (3,240,432)
            laps.append(lap)
            imgs.append(img)

        img = np.stack(imgs, axis=2)
        img = img.reshape(img.shape[0], img.shape[1], -1)
        img = self.transform(img) * 2.0 - 1.0
        img = torch.reshape(img, (-1, 3, img.shape[1], img.shape[2]))
        #img = torch.stack(imgs, 0)

        lap = np.stack(laps, axis=2)
        lap = self.transform(lap) * 2.0 - 1.0

        label = self.label_transform(label)
        label = torch.Tensor(label)  # [240,432]
        label = torch.unsqueeze(label, dim=0) # [1, 240, 432]
        # lable = label /255.0
        return {'img': img, 'seg': label, 'lap': lap, 'name': video_name}

    # 未修复区域像素为0，修复区域像素为1
    def label_transform(self, label):
        label_array = np.array(label)
        _, label_01 = cv2.threshold(label_array, 0, 1, cv2.THRESH_BINARY)
        return label_01

    def imread(self, path, image_name):
        zfile = zipfile.ZipFile(path, 'r')
        data = zfile.read(image_name)
        im = Image.open(io.BytesIO(data))
        return im

def get_frames_mask(length, frame_distance):
    pivot = random.randint(0 + frame_distance, length - frame_distance - 1)
    ref_index = [i for i in range(pivot - frame_distance, pivot + frame_distance + 1)]
    return ref_index, pivot


def get_dataset(path, k):
    print("generate dataset list")
    with open(os.path.join(path, 'data_info.json'), 'r') as f:
        video_dict = json.load(f)
        f.close()
    dataset = []
    video_names = list(video_dict.keys())
    for name in video_names:
        info = video_dict[name]
        length = info['length']
        frames = info['frames']
        masks = info['masks']
        for i in range(k, length-k):
            data = dict()
            data['video_name'] = name
            data['mask'] = os.path.join(path, "mask", name, masks[i])
            data['frames'] = []
            for j in range(i-k, i+k+1):
                data['frames'].append(os.path.join(path, "video", name, frames[j]))
            dataset.append(data)
    print("dataset list done")
    print(len(dataset))
    return dataset