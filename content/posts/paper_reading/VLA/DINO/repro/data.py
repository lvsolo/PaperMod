"""
tiny_dino / data.py
===================
数据集 + multi-crop 数据增强。

【multi-crop 是 DINO 的关键设计之一】, 我把 helper 增强(flip+color jitter+normalize)写好了,
你要填的是 DataAugmentationDINO: 用 RandomResizedCrop 生成 2 个 global + N 个 local 视角。

对照阅读: VLA/DINO.md §2.2 (有官方 DataAugmentationDINO 代码)。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---- 两种数据集的尺寸/patch 约定(脚手架已定好) ----
DATASET_CFG = {
    "stl10":  dict(img=96,  patch=8, n_classes=10),   # 12x12=144 patches, attention 能看出物体
    "cifar10": dict(img=32, patch=4, n_classes=10),   # 8x8=64 patches, 跑得最快
}


def make_augmentation(name, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                      local_crops_number=8):
    """根据数据集名构造 DataAugmentationDINO。"""
    cfg = DATASET_CFG[name]
    size_g = cfg["img"]
    size_l = max(16, cfg["img"] // 2 + 8)               # local 用更小分辨率
    return DataAugmentationDINO(global_crops_scale, local_crops_scale,
                                local_crops_number, size_g, size_l), cfg


class DataAugmentationDINO:
    """
    对一张图生成多个裁剪视角 = DINO 的 multi-crop (DINO.md §2.2)。

    应生成:
        - 2 个 global crops: RandomResizedCrop(scale=global_crops_scale) -> size_global,
          附加 flip + color jitter + 高斯模糊 (+ STL 上可加 solarization)
        - N 个 local  crops: RandomResizedCrop(scale=local_crops_scale)  -> size_local,
          附加 flip + color jitter + 高斯模糊
    __call__(image) 返回一个 list[Tensor], 长度 = 2 + local_crops_number。
    """

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,
                 size_global, size_local):
        self.local_crops_number = local_crops_number

        # 下面这些 helper 增强已写好, 你在 TODO 里组装即可
        self.flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.gaussian_blur = transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)

        # 保存尺寸/比例参数, 供你在下面 TODO 里用
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.size_global = size_global
        self.size_local = size_local

        # ------------------------------------------------------------------
        # TODO  (DINO.md §2.2)
        # ------------------------------------------------------------------
        # 在这里用 transforms.Compose 组装出:
        #   self.global_transfo1  : RandomResizedCrop(size_global, scale=global_crops_scale,
        #                                              interpolation=BICUBIC)
        #                          -> flip_and_color_jitter -> gaussian_blur(p=1.0) -> normalize
        #   self.global_transfo2  : 同上, 但 blur p=0.1, 再加一个 Solarization(p=0.2) 在 blur 之后
        #   self.local_transfo    : RandomResizedCrop(size_local,  scale=local_crops_scale)
        #                          -> flip_and_color_jitter -> gaussian_blur -> normalize
        # 提示: 官方代码见 DINO.md §2.2。Solarization 类我也给你放在本文件末尾了。
        #       组装好后, 把下面的 raise 删掉。
        # ------------------------------------------------------------------
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            self.flip_and_color_jitter,
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
            self.normalize
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size_global, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            self.flip_and_color_jitter,
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            Solarization(p=0.2),
            self.normalize
        ])
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size_local, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            self.flip_and_color_jitter,
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            self.normalize
        ])
        
    def __call__(self, image):
        """
        返回 list[Tensor]:
            [global_transfo1(image), global_transfo2(image),
             local_transfo(image) x local_crops_number]

        (上面 __init__ 的 TODO 填完后, 这里通常就是这么写; 也可以照抄)
        """
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class GaussianBlur:
    """STL/CIFAR 上用的轻量高斯模糊(参数化)。"""
    def __init__(self, sigma=(0.1, 2.0)):
        import random
        from PIL import ImageFilter
        self.sigma = sigma
        self.ImageFilter = ImageFilter
        self.random = random

    def __call__(self, x):
        sigma = self.random.uniform(*self.sigma)
        return x.filter(self.ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    """官方 DataAugmentationDINO 第二个 global crop 用的过度曝光。"""
    def __init__(self, p):
        from PIL import ImageOps
        self.p = p
        self.ImageOps = ImageOps

    def __call__(self, img):
        import random
        return self.ImageOps.solarize(img) if random.random() < self.p else img


def get_loaders(name, root="./data", aug=None, batch_size=64, num_workers=4, collate=None):
    """返回 (unlabeled_loader, labeled_train, labeled_test)。
    自监督阶段只用 unlabeled_loader(标签忽略); 后续 k-NN/linear eval 用 labeled。
    collate: multi-crop 时需要传入自定义 collate (见 train.collate_crops)。"""
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
                        num_workers=num_workers, drop_last=True, pin_memory=True,
                        collate_fn=collate)
    return loader, ltrain, ltest
