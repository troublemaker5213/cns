import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from utils.reshape import reshape
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from utils.Segloss import SegLoss
from utils.iouLoss import softIoULoss
from utils.focal_loss import FocalLoss
from utils.F1 import Metric

def evaluate(model, data_loader, device, epoch,num_classes):
    model.eval()

    segloss = SegLoss()
    iouloss = softIoULoss()
    focalloss = FocalLoss(alpha=0.25)
    total_dice = torch.zeros(1,num_classes).to(device)
    total_loss = 0
    number = 0
    total_f1 = 0
    total_iou = 0
    with tqdm(total=len(data_loader)) as pbar:
        for _, batch_data in enumerate(data_loader):
            images, labels, laps = batch_data['img'], batch_data['seg'], batch_data['lap']
            one_hot_labels = one_hot(labels,num_classes)  #[n,1,d,w]-->[n,num_class,d,w]

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
            dice_metric = Dice(pred,one_hot_labels,num_classes)

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

            if _ % 1 == 0:
                pred_ = F.softmax(pred, dim=1)
                pred_ = pred_.argmax(dim=1)  # 最大的索引值
                index = random.randint(0, len(pred_) - 1)
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
                save_path = f"/home/hddb/gxd2/pycharm/conv-next-seg/train_show/val/{epoch}_{_}.png"
                img.save(save_path, "PNG")

            pbar.update(1)
            pbar.set_description((
                f"val: {loss.item():.3f}")
            )

    epoch_mean_dice = total_dice/number
    epoch_mean_loss = total_loss/number
    epoch_mean_f1 = total_f1 / number
    epoch_mean_iou = total_iou / number
    
    # print('evaluate---------epoch:{} , loss: {} , dice:  {}  {}  {}  {}'\
    #     .format(epoch,epoch_mean_loss,epoch_mean_dice[0][1].item(),epoch_mean_dice[0][2].item(),epoch_mean_dice[0][3].item(),epoch_mean_dice[0][4].item()))


    # print('evaluate---------epoch:{} , loss: {} , dice: {}  {} '\
    #     .format(epoch,epoch_mean_loss,epoch_mean_dice[0][0].item(),epoch_mean_dice[0][1].item()))

    return epoch_mean_loss, epoch_mean_dice, epoch_mean_f1, epoch_mean_iou
