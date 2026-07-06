"""
DINOv2 repro / data.py
======================
multi-crop 增强【沿用 DINO v1, 已预填】(v1 你已写对)。
新增一个小工具 random_patch_mask, 给 iBOT 生成"哪些 patch 被 mask"的布尔掩码(脚手架写好)。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_CFG = {
    "stl10":   dict(img=96, patch=8, n_classes=10),
    "cifar10": dict(img=32, patch=4, n_classes=10),
}


def make_augmentation(name, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                      local_crops_number=8):
    cfg = DATASET_CFG[name]
    size_g = cfg["img"]
    size_l = max(16, cfg["img"] // 2 + 8)
    return DataAugmentationDINO(global_crops_scale, local_crops_scale,
                                local_crops_number, size_g, size_l), cfg


class DataAugmentationDINO:
    """multi-crop: 2 个 global + N 个 local。(沿用 v1)"""
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,
                 size_global, size_local):
        self.local_crops_number = local_crops_number
        self.flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        from torchvision.transforms import InterpolationMode as IM
        blur = lambda p: transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=p)
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=IM.BICUBIC),
            self.flip_and_color_jitter, blur(1.0), self.normalize])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=IM.BICUBIC),
            self.flip_and_color_jitter, blur(0.1), Solarization(0.2), self.normalize])
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size_local, scale=local_crops_scale, interpolation=IM.BICUBIC),
            self.flip_and_color_jitter, blur(0.5), self.normalize])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        import random
        from PIL import ImageFilter
        self.sigma, self.ImageFilter, self.random = sigma, ImageFilter, random

    def __call__(self, x):
        return x.filter(self.ImageFilter.GaussianBlur(radius=self.random.uniform(*self.sigma)))


class Solarization:
    def __init__(self, p):
        from PIL import ImageOps
        self.p, self.ImageOps = p, ImageOps

    def __call__(self, img):
        import random
        return self.ImageOps.solarize(img) if random.random() < self.p else img


def random_patch_mask(B, N, mask_ratio=0.3, generator=None):
    """🆕 iBOT 用: 随机选约 mask_ratio 比例的 patch 标 True。
    返回 (B, N) bool。DINOv2.md §2.2 里 iBOT 会 mask 掉 student 部分 patch。"""
    if generator is not None:
        prob = torch.rand(B, N, generator=generator)
    else:
        prob = torch.rand(B, N)
    return prob < mask_ratio


def get_loaders(name, root="./data", aug=None, batch_size=64, num_workers=4, collate=None):
    cfg = DATASET_CFG[name]
    plain = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if name == "stl10":
        train_set = datasets.STL10(root, split="unlabeled", download=True, transform=aug)
        ltrain = datasets.STL10(root, split="train", download=True, transform=plain)
        ltest = datasets.STL10(root, split="test", download=True, transform=plain)
    elif name == "cifar10":
        train_set = datasets.CIFAR10(root, train=True, download=True, transform=aug)
        ltrain = datasets.CIFAR10(root, train=True, download=True, transform=plain)
        ltest = datasets.CIFAR10(root, train=False, download=True, transform=plain)
    else:
        raise ValueError(name)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=True, collate_fn=collate)
    return loader, ltrain, ltest
