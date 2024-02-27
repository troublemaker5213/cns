from mimetypes import init
import torch
import torch.nn as nn
from model.backbone_ConvNeXt import ConvNeXt, LayerNorm
from model.decoder_UPerhead import UPerHead

import numpy as np

class upernet_convnext_tiny(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv_3d = Conv_3d()
        self.SRM = SRMConv()
        self.enhancement = Enhancement()
        self.backbone = ConvNeXt(in_chans=in_chans, dims=[96, 192, 384, 768])
        self.lap_layer = ConvNeXt(in_chans=5, dims=[96, 192, 384, 768])
        self.decoder = UPerHead(in_channels=[96, 192, 384, 768],  # [96, 192, 384, 768], #[16, 32, 64, 128], #tiny的参数
                        in_index=[0, 1, 2, 3],
                        pool_scales=(1, 2, 3, 6),
                        channels=out_chans,
                        dropout_ratio=0.1,
                        num_classes=out_chans,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        align_corners=False,
                        loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        
    def forward(self, x, lap):
        x = self.conv_3d(x)
        x = self.SRM(x)
        lap_out = self.lap_layer(lap)
        backbone_out = self.backbone(x)
        out = self.add_out(lap_out, backbone_out)
        out = self.decoder(out)
        return out

    def add_out(self, a, b):
        a1, a2, a3, a4 = a
        b1, b2, b3, b4 = b
        return tuple([a1+b1, a2+b2, a3+b3, a4+b4])


class Conv_3d(nn.Module):
    def __init__(self, T = 5):
        super().__init__()
        t0 = 1 - 1 / T
        t = 1 / T
        filter1 = [
            [t0, t0, t0],
            [t0, t0, t0],
            [t0, t0, t0]
        ]
        filter2 = [
            [t, t, t],
            [t, t, t],
            [t, t, t]
        ]
        filter1 = np.asarray(filter1, dtype=float)
        filter2 = np.asarray(filter2, dtype=float)
        filters = []
        for i in range(T):
            if i == T // 2:
                filters.append(filter1)
            else:
                filters.append(filter2)
        filters_ = [[filters, filters, filters], [filters, filters, filters], [filters, filters, filters]]
        filters_ = np.asarray(filters_)  # shape=(3,3,5,5)
        # filters = np.transpose(filters, (2, 3, 1, 0))  # shape=(5,5,3,3)
        filters_ = torch.Tensor(filters_)
        self.conv = torch.nn.Conv3d(3, 3, kernel_size=(T, 3, 3), stride=(1, 1, 1), bias=False, padding=(0, 1, 1))
        self.conv.weight.data = filters_


    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = x.squeeze(2)
        return x

class SRMConv(nn.Module):
    def __init__(self):
        super().__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
        # filters = np.transpose(filters, (2, 3, 1, 0))  # shape=(5,5,3,3)
        filters = torch.Tensor(filters)

        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, bias=False, padding='same')
        self.conv.weight.data = filters

    def forward(self, x):
        x = self.conv(x)
        return x

class Enhancement(nn.Module):
    def __init__(self):
        super().__init__()
        BatchNorm = nn.BatchNorm2d
        # self.normal_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        # self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(15, 32, 3, stride=2, padding=1, bias=False)
        # self.bn1 = BatchNorm(32)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        # self.bn2 = BatchNorm(64)

        self.normal_conv = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.pf_conv = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv2d(160, 256, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(256)

        self.pf_list = self.get_pf_list()
        self.reset_pf()

    def constrained_weights(self, weights):
        weights = weights.permute(2, 3, 0, 1)
        # Scale by 10k to avoid numerical issues while normalizing
        #weights = weights * 10000

        # Set central values to zero to exlude them from the normalization step
        weights[2, 2, :, :] = 0

        # Pass the weights
        filter_1 = weights[:, :, 0, 0]
        filter_2 = weights[:, :, 0, 1]
        filter_3 = weights[:, :, 0, 2]

        # Normalize the weights for each filter.
        # Sum in the 3rd dimension, which contains 25 numbers.
        filter_1 = filter_1.reshape(1, 1, 1, 25)
        filter_1 = filter_1 / filter_1.sum(3).reshape(1, 1, 1, 1)
        filter_1[0, 0, 0, 12] = -1

        filter_2 = filter_2.reshape(1, 1, 1, 25)
        filter_2 = filter_2 / filter_2.sum(3).reshape(1, 1, 1, 1)
        filter_2[0, 0, 0, 12] = -1

        filter_3 = filter_3.reshape(1, 1, 1, 25)
        filter_3 = filter_3 / filter_3.sum(3).reshape(1, 1, 1, 1)
        filter_3[0, 0, 0, 12] = -1

        # Prints are for debug reasons.
        # The sums of all filter weights for a specific filter
        # should be very close to zero.
        # print(filter_1)
        # print(filter_2)
        # print(filter_3)
        # print(filter_1.sum(3).reshape(1,1,1,1))
        # print(filter_2.sum(3).reshape(1,1,1,1))
        # print(filter_3.sum(3).reshape(1,1,1,1))

        # Reshape to original size.
        filter_1 = filter_1.reshape(1, 1, 5, 5)
        filter_2 = filter_2.reshape(1, 1, 5, 5)
        filter_3 = filter_3.reshape(1, 1, 5, 5)

        # Pass the weights back to the original matrix and return.
        weights[:, :, 0, 0] = filter_1
        weights[:, :, 0, 1] = filter_2
        weights[:, :, 0, 2] = filter_3

        weights = weights.permute(2, 3, 0, 1)
        return weights

    def get_pf_list(self):
        pf1 = np.array([[0, 0, 0],
                        [0, -1, 0],
                        [0, 1, 0]]).astype('float32')

        pf2 = np.array([[0, 0, 0],
                        [0, -1, 1],
                        [0, 0, 0]]).astype('float32')

        pf3 = np.array([[0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]]).astype('float32')

        return [torch.tensor(pf1).clone(),
                torch.tensor(pf2).clone(),
                torch.tensor(pf3).clone(),
                torch.tensor(pf1).clone(),
                torch.tensor(pf2).clone(),
                torch.tensor(pf3).clone(),
                torch.tensor(pf1).clone(),
                torch.tensor(pf2).clone(),
                torch.tensor(pf3).clone()
                ]

    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf

    def forward(self, x):
        self.bayar_conv.weight.data = self.constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        normal_x = self.normal_conv(x)
        pf_x = self.pf_conv(x)
        x = torch.cat([normal_x, bayar_x, pf_x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UpSampling(nn.Module):
    def __init__(self, in_channels=768, out_size=(240, 432), class_nums=2):
        super().__init__()
        # self.in_channels = in_channels
        # self.out_size = out_size
        # self.class_nums = class_nums
        #
        # self.upsample_layer = nn.Sequential(
        #     torch.nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size=4, stride=4),  # torch.Size([1, 384, 28, 52])
        #     LayerNorm(self.in_channels // 2, eps=1e-6, data_format="channels_first"),
        #     torch.nn.ConvTranspose2d(self.in_channels // 2, self.in_channels // 4, kernel_size=4, stride=4),  # torch.Size([1, 192, 112, 208])
        #     LayerNorm(self.in_channels // 4, eps=1e-6, data_format="channels_first"),
        #     torch.nn.Conv2d(self.in_channels // 4, self.in_channels // 8, kernel_size=2, stride=2),  # torch.Size([1, 96, 56, 104])
        #     LayerNorm(self.in_channels // 8, eps=1e-6, data_format="channels_first"),
        #     torch.nn.ConvTranspose2d(self.in_channels // 8, self.in_channels // 16, kernel_size=4, stride=4),  # torch.Size([1, 48, 224, 416])
        #     LayerNorm(self.in_channels // 16, eps=1e-6, data_format="channels_first"),
        #     torch.nn.Conv2d(self.in_channels // 16, self.class_nums, kernel_size=2, stride=2)  # torch.Size([1, 2, 112, 208])
        # )
        #


        self.act = nn.GELU()

        self.x4_12 = nn.Conv2d(768, 2, kernel_size=1)
        self.norm_x4_12 = LayerNorm(2, eps=1e-6, data_format="channels_first")

        self.deconv_x4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.norm_x3 = LayerNorm(384, eps=1e-6, data_format="channels_first")
        self.x3_12 = nn.Conv2d(384, 2, kernel_size=1)

        self.deconv_x3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.norm_x2 = LayerNorm(192, eps=1e-6, data_format="channels_first")
        self.x2_12 = nn.Conv2d(192, 2, kernel_size=1)

        self.deconv_x2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.norm_x1 = LayerNorm(96, eps=1e-6, data_format="channels_first")
        self.x1_12 = nn.Conv2d(96, 2, kernel_size=1)

        self.upsample = nn.ConvTranspose2d(2, 2, 8, 4, 2, bias=False)
        self.upsample2 = nn.ConvTranspose2d(2, 2, 8, 2, 2, bias=False)
        self.norm = LayerNorm(2, eps=1e-6, data_format="channels_first")
        self.upsample3 = torch.nn.UpsamplingBilinear2d(size=out_size)  # torch.Size([1, 2, 240, 432])

        self.up = nn.ConvTranspose2d(2, 2, kernel_size=2, stride=1)

    def forward(self, x, x3, x2, x1):
        x4_12 = self.x4_12(x)
        x4_12 = self.norm_x4_12(x4_12)
        x4_12 = self.act(x4_12)

        x4_x3 = self.deconv_x4(x4_12)
        x4_x3 = self.norm_x4_12(x4_x3)
        x4_x3 = self.act(x4_x3)

        x3_norm = self.norm_x3(x3)
        x3_12 = self.x3_12(x3_norm)
        x3_12 = self.norm_x4_12(x3_12)
        x3_12 = self.act(x3_12)

        x4_x3 = self.up(x4_x3)
        x3_x4 = x4_x3 + x3_12

        x43 = self.deconv_x3(x3_x4)

        x2_norm = self.norm_x2(x2)
        x2_12 = self.x2_12(x2_norm)
        x2_12 = self.norm_x4_12(x2_12)
        x2_12 = self.act(x2_12)

        #x43 = self.up(x43)
        x2_x3 = x2_12 + x43

        x32 = self.deconv_x2(x2_x3)

        x1_norm = self.norm_x1(x1)
        x1_12 = self.x1_12(x1_norm)
        x1_12 = self.norm_x4_12(x1_12)
        x1_12 = self.act(x1_12)

        x32 = self.up(x32)
        x2_x1 = x32 + x1_12

        x = self.upsample(x2_x1)
        x = self.norm(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        return x



if __name__ == '__main__':
    from torchsummary import summary
    data = torch.randn(4,3,9,256,256)
    a = upernet_convnext_tiny(3,2)
    s = a(data)
    print(s.shape)