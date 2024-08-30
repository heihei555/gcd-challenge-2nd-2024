from torchvision import transforms

import torch
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import RandomErasing
import random
import numpy as np
class CutMix:
    def __init__(self, p=0.1, alpha=1.0):
        self.p = p
        self.alpha = alpha

    def __call__(self, img, label):
        if np.random.rand(1) < self.p:
            # 获取一张随机样本
            rand_idx = np.random.choice(len(label))
            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(img.size, lam)
            img[:, bbx1:bbx2, bby1:bby2] = img[rand_idx, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size[-1] * img.size[-2]))
            return img, label, lam
        return img, label, 1.0

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:

        raise NotImplementedError

    return (train_transform, test_transform)

def get_transform2(transform_type='imagenet', image_size=32, args=None):
    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct
        # 确保 resize 的最小尺寸大于或等于裁剪尺寸
        min_size = image_size
        max_size = int(image_size * 1.5)  # 最大缩放尺寸可以根据需求调整

        train_transform = transforms.Compose([
            # 使用随机选择的输入尺度进行Resize
            # RandomResize(min_size, max_size),
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.5, 2.0))
            # CutMix(p=0.1)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    else:

        raise NotImplementedError

    return (train_transform, test_transform)