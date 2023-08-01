import sys

import torch
import numpy as np
from thop import profile
from thop import clever_format
import os
from shutil import copyfile
import cv2
import os.path as osp
import shutil
import tqdm
import random
from skimage import io
from scipy.ndimage.morphology import distance_transform_edt


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        lr = param_group['lr']
        print(f'lr:{lr}')


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))


def split_train_val_test(root_dir):
    # ori_img = root_dir + '/images'
    # ori_mask = root_dir + '/masks'
    # for dir in ['train', 'val', 'test']:
    #     dir_path = root_dir + '/' + dir
    #     if not os.path.exists(dir_path):
    #         os.mkdir(dir_path)
    #     for _ in ['images', 'masks']:
    #         subdir_path = dir_path + '/' + _
    #         if not os.path.exists(subdir_path):
    #             os.mkdir(subdir_path)
    # count_train = 0
    # count_test = 0
    # count_val = 0
    #
    # a = ori_img
    # print(a)
    # b = sys.argv[0]
    # print(b)  # 获取当前文件路径
    # print(sys.argv)  # 返回的是个列表
    #
    # # 返回当前文件夹下文件的个数
    # c = len(os.listdir(a))
    # print(c)
    #
    # for f in os.listdir(ori_img):
    #     name = f.split('.')[0]
    #     img = cv2.imread(f'{ori_img}/{name}.jpg')
    #     mask = cv2.imread(f'{ori_mask}/{name}.jpg')
    #
    #     while True:
    #         i = random.randint(1, 10)
    #         print(i)
    #         if i % 10 < 8 and count_train < 0.8*c:
    #             cv2.imwrite(f'{root_dir}/train/images/{name}.jpg', img)
    #             cv2.imwrite(f'{root_dir}/train/masks/{name}.png', mask)
    #             count_train += 1
    #             break
    #         elif i % 10 == 8 and count_val < 0.1*c:
    #             cv2.imwrite(f'{root_dir}/test/images/{name}.png', img)
    #             cv2.imwrite(f'{root_dir}/test/masks/{name}.png', mask)
    #             count_val += 1
    #             break
    #         elif i % 10 == 9 and count_test < 0.1*c:
    #             cv2.imwrite(f'{root_dir}/val/images/{name}.png', img)
    #             cv2.imwrite(f'{root_dir}/val/masks/{name}.png', mask)
    #             count_test += 1
    #             break
    ori_mask = root_dir + '/masks'
    out_dir = root_dir + '/GT'
    for f in os.listdir(ori_mask):
        name = f.split('.')[0]
        mask = cv2.imread(f'{ori_mask}/{name}.jpg')
        cv2.imwrite(f'{out_dir}/{name}.png', mask)

def former_trans(root_dir):
    ori_img = root_dir + '/images'
    ori_mask = root_dir + '/masks'

    for f in os.listdir(ori_img):
        name = f.split('.')[0]
        img = io.imread(f'{ori_img}/{f}') # 读取文件名
        img = img / img.max()  # 使其所有值不大于一
        img = img * 255 - 0.001  # 减去0.001防止变成负整型
        img = img.astype(np.uint8)  # 强制转换成8位整型
        mask = cv2.imread(f'{ori_mask}/{f}')

        img_path = f'{root_dir}/image/{name}.png'
        mask_path = f'{root_dir}/mask/{name}.png'

        b = img[:, :, 0]  # 读取蓝通道
        g = img[:, :, 1]  # 读取绿通道
        r = img[:, :, 2]  # 读取红通道
        bgr = cv2.merge([r, g, b])  # 通道拼接

        cv2.imwrite(img_path, bgr)  # 图片存储
        cv2.imwrite(mask_path, mask)
        print(f)


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


def copy_pic():
    # names = ['cju2qz06823a40878ojcz9ccx.png', 'cju5x7iskmad90818frchyfwd.png', 'cju7d4jk723eu0817bqz2n39m.png', 'cju414lf2l1lt0801rl3hjllj.png', 'cju8dqkrqu83i0818ev74qpxq.png']
    names = ['cju2qz06823a40878ojcz9ccx.jpg', 'cju5x7iskmad90818frchyfwd.jpg', 'cju7d4jk723eu0817bqz2n39m.jpg', 'cju414lf2l1lt0801rl3hjllj.jpg', 'cju8dqkrqu83i0818ev74qpxq.jpg'
             , 'cju18gzrq18zw0878wbf4ftw6.jpg', 'cju42nm68lpyo0818xvvqmupq.jpg', 'cju6xifswvwbo0987nibtdr50.jpg', 'cju6uzxk0v83p0801rcwnexdu.jpg', 'cju87nkyrnb970801q84m47yt.jpg']
    # methods = ['ESFP', 'FCB', 'msrf', 'pranet', 'sg', 'ssformer', 'unet', 'unet_plus']
    methods = ['masks']
    copy_dir = 'C:/Users/YYD/Desktop/paper_a/CGMA-Net/figs/dataset/'
    target = 'C:/Users/YYD/Desktop/paper_a/CGMA-Net/figs/dataset/iii'
    for method in methods:
        for name in names:
            img_path = f'{copy_dir}/{method}/{name}'
            print(img_path)
            img = cv2.imread(img_path)
            out_dir = f'{target}/{method}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(f'{out_dir}/{name}', img)



if __name__ == '__main__':
    copy_pic()
    # split_train_val_test('../dataset/kvasir-seg')
    # former_trans("../dataset/CVC-ClinicDB")
    # val_train_set()
