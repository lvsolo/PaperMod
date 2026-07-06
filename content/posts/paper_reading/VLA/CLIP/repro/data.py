"""
CLIP repro / data.py
====================
把分类数据集(CIFAR-100/10, STL-10)变成【图文对】: 用类别名套 prompt 模板造文本。
预填三部分:
  ① 数据集 + 增强(torchvision)
  ② 词级 tokenizer(从所有可能 caption 建词表) + CLIP 风格 prompt 模板
  ③ zero-shot 用的"类文本嵌入候选"(每个类用多模板 ensemble)

为什么用 class-prompt 当文本(而不是真实 caption)?
  CLIP 的招牌是 zero-shot 分类: 拿 N 个类名 prompt 编码成文本, 图像和它们比相似度。
  分类数据集有类别标签 → 直接套 prompt 就是合法图文对, 能完整复现"对比训练 + zero-shot 推理"
  这条链路, 且零额外下载。代价: 文本端只见到类名级语言(不是真实自然语言 caption),
  所以这是"机制复现", 不是"复现 CLIP 的语言泛化"(那要 WIT/LAION 级数据)。
"""
import re
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-100 / CIFAR-10 / STL-10 的类名(torchvision 顺序, index 即 label)
CIFAR100_CLASSES = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear", "pickup truck", "pine tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

DATASET_CFG = {
    "cifar100": dict(img=32, classes=CIFAR100_CLASSES, n=100),
    "cifar10":  dict(img=32, classes=CIFAR10_CLASSES,  n=10),
    "stl10":    dict(img=96, classes=STL10_CLASSES,    n=10),
}

# CLIP 风格 prompt 模板(训练时随机选一个 = 文本增强; zero-shot 时全部 ensemble)
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of a {} in the wild.",
    "a bright photo of a {}.",
    "a centered satellite photo of a {}.",
    "a photo of a {} on a white background.",
]


# -------------------- 图像增强 + 数据集 (预填) --------------------
def make_augmentation(name):
    cfg = DATASET_CFG[name]
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train = transforms.Compose([
        transforms.RandomResizedCrop(cfg["img"], scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    plain = transforms.Compose([transforms.Resize(cfg["img"]), transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
    return train, plain, cfg


def get_loaders(name, root="./data", batch_size=256, num_workers=4):
    train_tf, plain_tf, cfg = make_augmentation(name)
    if name == "cifar100":
        train_set = ImageCaptionDataset(datasets.CIFAR100(root, train=True, download=True, transform=train_tf), cfg["classes"])
        test_set = datasets.CIFAR100(root, train=False, download=True, transform=plain_tf)
    elif name == "cifar10":
        train_set = ImageCaptionDataset(datasets.CIFAR10(root, train=True, download=True, transform=train_tf), cfg["classes"])
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=plain_tf)
    elif name == "stl10":
        # STL-10 只有 labeled train 有类别(用于造 caption); unlabeled 没标签用不了
        train_set = ImageCaptionDataset(datasets.STL10(root, split="train", download=True, transform=train_tf), cfg["classes"])
        test_set = datasets.STL10(root, split="test", download=True, transform=plain_tf)
    else:
        raise ValueError(name)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, pin_memory=True)
    return loader, test_set, cfg


class ImageCaptionDataset(torch.utils.data.Dataset):
    """包装分类数据集: 每次返回 (image, 随机选一个 prompt 模板套类名的 token_ids)。"""
    def __init__(self, base, class_names):
        self.base = base
        self.class_names = class_names

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label   # caption 的 tokenization 在 collate 里做(需要 tokenizer)


# -------------------- 词级 tokenizer (预填) --------------------
class Tokenizer:
    """最简单的词级 tokenizer: 小写 + 去标点 + 按空格切。词表 = 所有可能 caption 的词 + 特殊 token。"""
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, classes, templates, max_len=16):
        self.max_len = max_len
        # 枚举所有"模板×类名"句子, 收集词表
        sentences = [t.format(c) for c in classes for t in templates]
        words = set()
        for s in sentences:
            words.update(self._tokenize(s))
        self.id2tok = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted(words)
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}
        self.pad_id, self.unk_id = self.tok2id[self.PAD], self.tok2id[self.UNK]
        self.vocab_size = len(self.id2tok)

    @staticmethod
    def _tokenize(s):
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())     # 只留字母数字空格
        return [w for w in s.split() if w]

    def encode(self, sentence):
        """句子 → [BOS] + ids + [EOS], pad/truncate 到 max_len。返回 (max_len,) long。"""
        ids = [self.tok2id.get(w, self.unk_id) for w in self._tokenize(sentence)]
        ids = [self.tok2id[self.BOS]] + ids + [self.tok2id[self.EOS]]
        ids = ids[: self.max_len]
        ids = ids + [self.pad_id] * (self.max_len - len(ids))   # 后面 pad
        return torch.tensor(ids, dtype=torch.long)

    def make_collate(self, classes, templates):
        """返回一个 collate_fn: batch 里每张图按其 label 随机套一个模板 → token_ids。"""
        import random

        def collate(batch):
            imgs, labels = zip(*batch)
            caps = [random.choice(templates).format(classes[l]) for l in labels]
            token_ids = torch.stack([self.encode(c) for c in caps], 0)
            return torch.stack(imgs, 0), token_ids, torch.tensor(labels)
        return collate

    def class_text_ids(self, classes, templates):
        """zero-shot 用: 每个类用所有模板造句(ensemble), 返回 (n_classes, n_templates, max_len)。"""
        all_ids = []
        for c in classes:
            caps = [t.format(c) for t in templates]
            all_ids.append(torch.stack([self.encode(x) for x in caps], 0))   # (n_templates, max_len)
        return torch.stack(all_ids, 0)       # (n_classes, n_templates, max_len)
