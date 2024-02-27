#from pyexpat import model
import torch 
import argparse
import glob
from tqdm import tqdm
import os
from dataset.dataset import MyDataSet
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from utils.evaluate_one_epoch import evaluate
from utils.train_one_epoch import train_one_epoch
#from unet_model.unet_model import UNet_monai
from model.upernet_convnext_tiny import upernet_convnext_tiny



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义transform
    train_transform = transforms.Compose([transforms.ToTensor(),])
                                         #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    val_transform = transforms.Compose([transforms.ToTensor(),])
                                       #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    # 实例化训练数据集
    train_dataset = MyDataSet(train=True,
                              transform=train_transform,
                              k=args.k,
                              dataset_name="youtubevos")
    # 实例化验证数据集
    val_dataset = MyDataSet(train=False,
                            transform=val_transform,
                            k=args.k,
                            dataset_name="youtubevos")

   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8)
    
    #定义模型，损失函数，优化器
    model = upernet_convnext_tiny(in_chans=args.in_chans,out_chans=args.out_chans).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)
    writer = SummaryWriter(f"./logs/{args.model_name}")

    #epoch训练
    best_dice = torch.zeros(1, args.out_chans).to(device)
    best_val = 10
    step = 0
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            #train
            train_loss, train_dice = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    num_classes=args.out_chans,
                                                    step = step)

            writer.add_scalar("train loss", train_loss, epoch)
            # writer.add_scalar("train dice 0", train_dice[0][0], epoch)
            # writer.add_scalar("train dice 1", train_dice[0][1], epoch)

            #validate
            val_loss, val_dice, f1, iou = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch,
                                        num_classes=args.out_chans)

            writer.add_scalar("val loss", val_loss, epoch)
            writer.add_scalar("val f1", f1, epoch)
            writer.add_scalar("val iou", iou, epoch)

            if best_dice[0].sum(dim=0) < val_dice[0].sum(dim=0):
                torch.save(model.state_dict(), f"./weights/{args.model_name}/{args.model_name}.pth")
                best_dice = val_dice
            if epoch % 1 == 0:
                torch.save(model.state_dict(), f"./weights/{args.model_name}/{epoch}.pth")
            # if val_loss < best_val:
            #     torch.save(model.state_dict(), "./weights/0617/0617.pth")
            #     best_val = val_loss

            #scheduler.step(val_loss)
            scheduler.step()

            pbar.update(1)
            pbar.set_description((
                f"总进度：train: {train_loss:.3f}; val: {val_loss:.3f}")
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_chans', type=int,default=3)
    parser.add_argument('--out_chans', type=int,default=2)
    parser.add_argument('--epochs', type=int,default=50)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--model_name', type=str, default="model_k3")
    parser.add_argument('--lr', type=float,default=1e-2)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--data_path', type=str,default="./data/")
    args = parser.parse_args()
    main(args)
