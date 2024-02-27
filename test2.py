import argparse
import os
from dataset.test_dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from model.upernet_convnext_tiny import upernet_convnext_tiny
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.reshape import reshape
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from utils.Segloss import SegLoss
from utils.iouLoss import softIoULoss
from utils.focal_loss import FocalLoss
from utils.F1 import Metric


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_transform = transforms.Compose([transforms.ToTensor(), ])

    test_dataset = MyDataSet(train=False,
                             transform=test_transform,
                             k=args.k,
                             dataset_name="youtubevos")

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1)

    # 定义模型，损失函数，优化器
    model = upernet_convnext_tiny(in_chans=args.in_chans, out_chans=args.out_chans).to(device)
    model_weight_path = f"./weights/{args.model_name}/{args.model_name}.pth"
    #model_weight_path = f"./weights/0628/0628.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    segloss = SegLoss()
    iouloss = softIoULoss()
    focalloss = FocalLoss(alpha=0.25)
    total_dice = torch.zeros(1,2).to(device)
    total_loss = 0
    number = 0
    total_f1 = 0
    total_iou = 0
    lossset={}
    with tqdm(total=len(test_loader)) as pbar:
        for _, batch_data in enumerate(test_loader):
            images, labels, laps = batch_data['img'], batch_data['seg'], batch_data['lap']
            video_name = batch_data['name'][0]
            one_hot_labels = one_hot(labels,2)  #[n,1,d,w]-->[n,num_class,d,w]

            images = images.to(device, non_blocking=True)
            laps = laps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            one_hot_labels = one_hot_labels.to(device, non_blocking=True)

            #predict
            pred = model(images, laps)
            pred = reshape(pred,one_hot_labels.size())
            iou_loss = iouloss(one_hot_labels, pred)
            seg_loss = segloss(pred, one_hot_labels)
            #focal_loss = focalloss(pred, one_hot_labels.float())
            loss = seg_loss

            #dice
            Dice = DiceMetric()
            dice_metric = Dice(pred,one_hot_labels,2)

            total_dice = torch.add(total_dice, dice_metric)
            total_loss = total_loss + loss.item()
            number = number + 1


            predmask = F.softmax(pred, dim=1)
            predmask = predmask.argmax(dim=1)
            predmask = predmask.cpu()
            labels_ = torch.squeeze(labels, dim=1).cpu()
            metric = Metric(predmask, labels_)

            total_f1 += metric.F1()
            total_iou += metric.IoU()


            pred_ = F.softmax(pred, dim=1)
            pred_ = pred_.argmax(dim=1)  # 最大的索引值
            index = 0
            pred_ = pred_.cpu()
            pred_ = np.array(pred_)
            pred_ = pred_[index] * 255.0
            pred_ = Image.fromarray(pred_)
            label = labels[index]
            #pred_ = transforms.ToPILImage()(pred_.float())
            label = transforms.ToPILImage()(label.float())
            img = Image.new("L", (432, 485), color=255)
            img.paste(pred_, (0, 0, 432, 240))
            img.paste(label, (0, 245, 432, 485))
            save_path = f"/home/hddb/gxd2/pycharm/conv-next-seg/test_result/{args.model_name}/{video_name}"

            if video_name not in lossset:
                lossset[video_name]=loss.item()
            else:
                lossset[video_name]+=loss.item()

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img.save(save_path+f"/{_:>7d}.png", "PNG")

            pbar.update(1)
            pbar.set_description((
                f"val: {loss.item():.3f}")
            )

    epoch_mean_dice = total_dice/number
    epoch_mean_loss = total_loss/number
    epoch_mean_f1 = total_f1 / number
    epoch_mean_iou = total_iou / number
    print("mean_dice", epoch_mean_dice)
    print("mean_loss", epoch_mean_loss)
    print("mean_f1", epoch_mean_f1)
    print("mean_iou", epoch_mean_iou)
    print(lossset)
    with open(f"/home/hddb/gxd2/pycharm/conv-next-seg/test_result/{args.model_name}.txt", "w") as f:
        f.write("mean_loss:  ")
        f.write(str(epoch_mean_loss))
        f.write("\n")
        f.write("mean_f1:  ")
        f.write(str(epoch_mean_f1))
        f.write("\n")
        f.write("mean_iou:  ")
        f.write(str(epoch_mean_iou))
        f.write("\n")

        f.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--out_chans', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_name', type=str, default="0628")
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--data_path', type=str, default="./data/")
    args = parser.parse_args()
    main(args)
