"""
CLIP repro / data.py  (Flickr8k 主, CIFAR/STL 备)
=================================================
两条数据路径, 共用同一个 Tokenizer + CLIPWrapper:
  ① Flickr8k (推荐, 真 CLIP): 6000 训练图 × 5 句真实 caption, 无假负样本, 文本塔学真语言,
     用标准【图文检索 R@1/5/10】评估。HF: jxie/flickr8k (~1GB, 首次自动下)。
  ② CIFAR-100/10 + STL-10 (备选/零下载): 用类名套 prompt 模板当 caption, zero-shot 分类评估。
     有假负样本问题(同类同文本), 文本塔学不到语言 —— 是退化版 CLIP, 仅作 fallback。

为什么要 Flickr8k: CLIP 的 InfoNCE 假设每张图有【独一无二】的匹配文本, batch 内其余全是负样本。
CIFAR+类名只有 100 句重复 prompt → 同类图共享文本 → 假负样本 + 文本塔学不到语言。
Flickr8k 每图 5 句各不相同的真实 caption → 对比损失按设计 work, 文本塔真学语言。
"""
import io, re
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# -------------------- CIFAR/STL 类名 (备选路径) --------------------
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
PROMPT_TEMPLATES = [
    "a photo of a {}.", "a blurry photo of a {}.", "a photo of the large {}.", "a photo of the small {}.",
    "a photo of a {} in the wild.", "a bright photo of a {}.", "a centered satellite photo of a {}.",
]

CIFAR_CFG = {
    "cifar100": dict(classes=CIFAR100_CLASSES),
    "cifar10":  dict(classes=CIFAR10_CLASSES),
    "stl10":    dict(classes=STL10_CLASSES),
}


# -------------------- 通用词级 tokenizer --------------------
class Tokenizer:
    """词级 tokenizer: 小写+去标点+按空格切。词表从【所有可能出现的句子】构建(预填)。"""
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, sentences, max_len=32):
        self.max_len = max_len
        words = set()
        for s in sentences:
            words.update(self._tokenize(s))
        self.id2tok = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted(words)
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}
        self.pad_id, self.unk_id = self.tok2id[self.PAD], self.tok2id[self.UNK]
        self.vocab_size = len(self.id2tok)

    @staticmethod
    def _tokenize(s):
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        return [w for w in s.split() if w]

    def encode(self, sentence):
        ids = [self.tok2id[self.BOS]] + [self.tok2id.get(w, self.unk_id) for w in self._tokenize(sentence)] + [self.tok2id[self.EOS]]
        ids = ids[: self.max_len]
        return torch.tensor(ids + [self.pad_id] * (self.max_len - len(ids)), dtype=torch.long)

    def encode_batch(self, sentences):
        return torch.stack([self.encode(s) for s in sentences], 0)


# -------------------- 图像增强 --------------------
def build_transforms(img_size, train=True):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])


# ============================================================
# 路径 ① Flickr8k (推荐)
# ============================================================
def to_pil(image_field):
    """HF datasets 的 Image 特征访问时通常已解码为 PIL.Image; 兜底原始 dict{bytes, path}。"""
    if isinstance(image_field, dict):
        return Image.open(io.BytesIO(image_field["bytes"]))
    return image_field   # 已是 PIL.Image


class Flickr8kDataset(Dataset):
    """包装 HF jxie/flickr8k: 每条返回 (PIL图, 随机选一句 caption)。"""
    def __init__(self, hf_split, transform):
        from datasets import load_dataset
        self.ds = load_dataset("jxie/flickr8k", split=hf_split)   # 首次自动下载并缓存(~1GB)
        self.transform = transform
        import random; self._rng = random

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        img = to_pil(row["image"]).convert("RGB")
        caps = [row[f"caption_{k}"] for k in range(5)]
        return self.transform(img), self._rng.choice(caps)


def flickr8k_collate(tokenizer):
    def collate(batch):
        imgs, caps = zip(*batch)
        return torch.stack(imgs, 0), tokenizer.encode_batch(caps)
    return collate


def get_flickr8k(img_size=96, batch_size=64, tokenizer_max_len=32, num_workers=4, root="./data"):
    """返回 (train_loader, test_images_tensor_loader, test_captions_list, tokenizer)。
    检索评估: 1000 test 图 × 5 caption, image→text / text→image R@1/5/10。"""
    train_tf, eval_tf = build_transforms(img_size, train=True), build_transforms(img_size, train=False)
    train_set = Flickr8kDataset("train", train_tf)
    # 从所有训练 caption 建词表
    all_caps = []
    for r in train_set.ds:
        all_caps.extend(r[f"caption_{k}"] for k in range(5))
    tokenizer = Tokenizer(all_caps, max_len=tokenizer_max_len)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, pin_memory=True, collate_fn=flickr8k_collate(tokenizer))
    # 测试集: 原始 ds, 评估时按 1000 图 × 5 caption 展开
    from datasets import load_dataset
    test_ds = load_dataset("jxie/flickr8k", split="test")
    return loader, test_ds, eval_tf, tokenizer


# ============================================================
# 路径 ② CIFAR/STL + class-prompt (备选/零下载)
# ============================================================
class CIFARCaptionDataset(Dataset):
    def __init__(self, base, classes):
        self.base, self.classes = base, classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return img, label


def cifar_collate(tokenizer, classes, templates):
    import random
    def collate(batch):
        imgs, labels = zip(*batch)
        caps = [random.choice(templates).format(classes[l]) for l in labels]
        return torch.stack(imgs, 0), tokenizer.encode_batch(caps), torch.tensor(labels)
    return collate


def get_cifar(name, root="./data", batch_size=256, num_workers=4, img_size=32):
    from torchvision import datasets
    classes = CIFAR_CFG[name]["classes"]
    train_tf, eval_tf = build_transforms(img_size, train=True), build_transforms(img_size, train=False)
    if name == "cifar100":
        train_base = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR100(root, train=False, download=True, transform=eval_tf)
    elif name == "cifar10":
        train_base = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    else:  # stl10
        train_base = datasets.STL10(root, split="train", download=True, transform=train_tf)
        test_set = datasets.STL10(root, split="test", download=True, transform=eval_tf)
    train_set = CIFARCaptionDataset(train_base, classes)
    sentences = [t.format(c) for c in classes for t in PROMPT_TEMPLATES]
    tokenizer = Tokenizer(sentences, max_len=16)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, pin_memory=True, collate_fn=cifar_collate(tokenizer, classes, PROMPT_TEMPLATES))
    return loader, test_set, classes, tokenizer
