import torch
from utils.reshape import reshape
from tqdm import tqdm
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from utils.Segloss import SegLoss
from utils.iouLoss import softIoULoss
from utils.F1 import Metric
from utils.focal_loss import FocalLoss
import torch.nn.functional as F
import random
from PIL import Image
from torchvision import transforms
import numpy as np

def train_one_epoch(model, optimizer, data_loader, device, epoch,num_classes,step):
    model.train()
    segloss = SegLoss()
    iouloss = softIoULoss()
    #focalloss = FocalLoss(alpha=0.25)
    total_dice = torch.zeros(1, num_classes).to(device)
    total_loss = 0
    number = 0
    with tqdm(total=len(data_loader)) as pbar:
        for _, batch_data in enumerate(data_loader):
            step = step + 1
            images, labels, laps = batch_data['img'], batch_data['seg'], batch_data['lap']  #label : [0,1,2,3,4][0.0000, 0.0039, 0.0078]),
            one_hot_labels = one_hot(labels,num_classes)  #[n,1,d,w]-->[n,num_class,d,w]

            images = images.to(device,non_blocking=True)
            laps = laps.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            one_hot_labels = one_hot_labels.to(device,non_blocking=True)

            #梯度清0
            optimizer.zero_grad()

            #predict and loss
            pred = model(images, laps)
            pred = reshape(pred,one_hot_labels.size())
            iou_loss = iouloss(one_hot_labels, pred)
            seg_loss = segloss(pred, one_hot_labels)
            #focal_loss = focalloss(pred, one_hot_labels.float())
            loss = seg_loss

            # dice
            Dice = DiceMetric()
            dice_metric = Dice(pred,one_hot_labels,num_classes)

            #反向传播
            loss.backward()
            optimizer.step()

            #show
            total_dice = torch.add(total_dice, dice_metric)
            total_loss = total_loss + loss.item()
            number = number + 1

            if step % 20 == 0:
                pred_ = F.softmax(pred, dim=1)
                pred_ = pred_.argmax(dim=1)  # 最大的索引值
                index = random.randint(0, len(pred_) - 1)
                pred_ = pred_.cpu()
                pred_ = np.array(pred_)
                pred_ = pred_[index] * 255.0
                pred_ = Image.fromarray(pred_)
                label = labels[index]
                #pred_ = transforms.ToPILImage()(pred_)
                label = transforms.ToPILImage()(label)
                img = Image.new("L", (432, 485), color=255)
                img.paste(pred_, (0, 0, 432, 240))
                img.paste(label, (0, 245, 432, 485))
                save_path = f"/home/hddb/gxd2/pycharm/conv-next-seg/train_show/train/{epoch}_{step}.png"
                img.save(save_path, "PNG")

            pbar.update(1)
            pbar.set_description((
                f"train: {loss.item():.3f}")
            )

        # if step % 10 == 0 :
            # print("---------step:{} ,  dice:{} ------ ".format(step,dice_metric[1].item()))
            # print("---------step:{} ,  dice:{}  {}  ------ ".format(step,dice_metric[0].item()\
            #           ,dice_metric[1].item()))
    epoch_mean_dice = total_dice/number
    epoch_mean_loss = total_loss/number
    
    # print('train---------epoch:{} , loss: {} , dice:  {}  '\
    #     .format(epoch,epoch_mean_loss,epoch_mean_dice[0][1].item()))

    # print('train---------epoch:{} , loss: {} , dice: {}  {} '\
    #     .format(epoch,epoch_mean_loss,epoch_mean_dice[0][0].item(),epoch_mean_dice[0][1].item()))
    
    return epoch_mean_loss, epoch_mean_dice