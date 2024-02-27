import os
import shutil
import torch
from torchvision import transforms
import cv2
import numpy as np
# a = np.array(
#     [
#         [[11,12,13],[11,12,13]],
#         [[11,12,13],[11,12,13]]
#      ]
# )
# b = np.array(
#     [
#         [[21,22,23],[21,22,23]],
#         [[21,22,23],[21,22,23]]
#      ]
# )
# c = np.array(
#     [
#         [[31,32,33],[31,32,33]],
#         [[31,32,33],[31,32,33]]
#      ]
# )
# d = np.array(
#     [
#         [[41,42,43],[41,42,43]],
#         [[41,42,43],[41,42,43]]
#      ]
# )
# e = np.array(
#     [
#         [[51,52,53],[51,52,53]],
#         [[51,52,53],[51,52,53]]
#      ]
# )
#
# img = np.concatenate([a,b,c,d,e], axis=2)
#
# s = torch.randn([4, 3, 9, 240, 432])
# print(s.shape)
#
# conv = torch.nn.Conv3d(3, 3, kernel_size=(9, 3, 3), stride=(1, 1, 1), bias=False,padding=(0, 1, 1))
# res = conv(s)
# print(res.shape)

video = [
    '0b6f9105fc',
    '0de4923598',
    '0e9ebe4e3c',
    '0e30020549',
    '01e64dd36a',
    '1a8bdc5842',
    '1acd0f993b',
    '2f680909e6',
    '3c5ff93faf',
    '4ae13de1cd',
    '4cfdd73249',
    '4dfaee19e5',
    '4f8db33a13',
    '5a841e59ad',
    '7b5388c9f1',
    '7f1723f0d5',
    '11ccf5e99d',
    '24ddf05c03',
    '93f623d716',
    '94db712ac8',
    '2465bf515d',
    '5104aa1fea',
    '06840b2bbe',
    '7053e4f41e',
    '9437c715eb',
    '466839cb37',
    'e92b6bfea4',
    'fd6789b3fe',
    'ff97129478',
    '0f202e9852',
    '2ba621c750',
    '2c2c076c01',
    '3d5aeac5ba',
    '5ede4d2f7a',
    '7dd409947e',
    '11a6ba8c94',
    '0044fa5fba',
    '91be106477',
    '962781c601',
    'a2c996e429',
    'b6d65a9eef',
    'b1457e3b5e',
    'c8f6cba9fd',
    'db0968cdd3',
    'e22ddf8a1b',
    'e348377191',
    'eb6992fe02',
    'ed72ae8825',
    'ef65268834',
    'f9681d5103',
    'fbfd25174f'
]

# path = "/home/hddb/gxd2/pycharm/conv-next-seg/data/youtubevos/train"
# path2 = "/home/hddb/gxd2/pycharm/conv-next-seg/data/youtubevos/test"
# names = os.listdir(os.path.join(path, 'video'))
# for i in names:
#     if i in video:
#         shutil.move(os.path.join(path, "video", i), os.path.join(path2, "video"))
#         shutil.move(os.path.join(path, "mask", i), os.path.join(path2, "mask"))
#         print(i)

