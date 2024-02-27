import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import torch
import os
from utils.reshape import reshape
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from model.upernet_convnext_tiny import upernet_convnext_tiny


def label_transform(label):
    label_array = np.array(label)
    _, label_01 = cv2.threshold(label_array, 0, 1, cv2.THRESH_BINARY)
    return label_01


def main(img_path, label_path, name, count):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([transforms.ToTensor()])
    # label = Image.open(mask)
    label = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)  # [240, 432]
    imgs = []
    for i in img_path:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [240,432,3]
        imgs.append(img)
    img = np.stack(imgs, axis=2)
    img = img.reshape(img.shape[0], img.shape[1], -1)
    img = train_transform(img) * 2.0 - 1.0
    img = torch.reshape(img, (-1, 3, img.shape[1], img.shape[2]))

    label = label_transform(label)
    label = torch.Tensor(label)  # [240,432]
    label = torch.unsqueeze(label, dim=0)  # [1, 240, 432]

    img = torch.unsqueeze(img, dim=0)
    label = torch.unsqueeze(label, dim=0)

    model = upernet_convnext_tiny(in_chans=3, out_chans=2).to(device)

    # load model weights
    model_weight_path = "./weights/0616/0616.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        one_hot_labels = one_hot(label, 2)  # [n,1,d,w]-->[n,num_class,d,w]
        img = img.to(device)
        output = model(img)
        output = reshape(output, label.size())

        Dice = DiceMetric()
        dice_metric = Dice(output.cpu(), one_hot_labels, 2)
        print(dice_metric)

        pred = F.softmax(output, dim=1)
        #pred = pred.argmax(dim=1)  # 最大的索引值
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu()
        pred = transforms.ToPILImage()(pred.float())
        save_path = f"/home/hddb/gxd2/pycharm/conv-next-seg/test_result/{name}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        pred.save(save_path + f"/{count:0>4d}.png", "PNG")


def path_get(dir, video=True):
    k = 4
    images = os.listdir(dir)
    images.sort()
    length = len(images)
    if video:
        dataset = []
        for i in range(k, length - k):
            data = []
            for j in range(i - k, i + k + 1):
                data.append(os.path.join(dir, images[j]))
            dataset.append(data)
    else:
        dataset = []
        for i in range(k, length - k):
            path = os.path.join(dir, images[i])
            dataset.append(path)
    return dataset


def test_video(video_name):
    name = video_name
    path_train = f"data/youtubevos/test/video/{name}"
    path_label = f"data/youtubevos/test/mask/{name}"
    path_trains = path_get(path_train, True)
    path_labels = path_get(path_label,  False)
    count = 0
    for i in range(len(path_labels)):
        main(path_trains[i], path_labels[i], name, count)
        count += 1

def test_videos(videos_dir):
    pass

if __name__ == '__main__':
    # test one video
    name = "377db65f60"
    test_video(name)


    # test all video
    # video_path = ""
    # test_videos(video_path)