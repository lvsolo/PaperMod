---
title: "Focal Loss在分类loss中的应用"
author: "lvsolo"
math: true
date: "2024-10-14"
tags: ["DeepLearning", "Loss", "training", "YOLO", "Experiments"]
---
目录：
- [BCE与CE公式的差别](#bce与ce公式的差别)
  - [Entropy](#entropy)
  - [Cross Entropy](#cross-entropy)
  - [二分类交叉熵损失 Binary Cross Entropy](#二分类交叉熵损失-binary-cross-entropy)
    - [如果将此思路扩展到多分类](#如果将此思路扩展到多分类)
    - [公式解释](#公式解释)
  - [多分类交叉熵损失 Multi-classes Cross Entropy](#多分类交叉熵损失-multi-classes-cross-entropy)
    - [标准的多分类交叉熵损失](#标准的多分类交叉熵损失)
    - [注意事项](#注意事项)
    - [总结](#总结)
- [Focal Loss理解](#focal-loss理解)
- [代码实现两种CE+FocalLoss](#代码实现两种cefocalloss)
- [设计实验](#设计实验)
---

# BCE与CE公式的差别
## Entropy
一个分布中的信息熵：

$$
H(p) = - \sum_{i} p_i \log(p_i)
$$
## Cross Entropy 
两个分布的交叉熵：

$$
\text{Cross-Entropy} = - \sum_{i} p_i \log(q_i)
$$


## 二分类交叉熵损失 Binary Cross Entropy  

在二分类问题中，交叉熵损失函数同时考虑了正类和负类的预测损失。公式如下：

$$
\text{Binary Cross-Entropy Loss}(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

其中，$(1 - y) \log(1 - \hat{y})$ 是对负类（错误分类）的惩罚。

### 如果将此思路扩展到多分类

如果不使用 softmax，而是直接使用模型输出的 logits（未归一化的得分），我们可以借鉴二分类交叉熵的思路，为每个错误类别添加惩罚。具体的损失函数修改为：

$$
\text{Modified Cross-Entropy Loss} = - \left[ q_{i^*} \log(p_{i^*}) + \sum_{i \neq i^*} (1 - q_i) \log(1 - p_i) \right]
$$

其中：
- $i^*$ 是真实类别。
- $p_i$ 是模型对类别 $i$ 的预测值，通常可以直接是 logits，或者通过 Sigmoid 函数将 logits 转化为概率。

### 公式解释

- **第一项** $q_{i^*} \log(p_{i^*})$：与标准的交叉熵相同，计算的是正确类别的预测损失。
- **第二项** $\sum_{i \neq i^*} (1 - q_i) \log(1 - p_i)$：显式地加入了对错误类别的惩罚，如果模型对错误类别的预测值 $p_i$ 较高，则 $\log(1 - p_i)$ 会较大，从而增加损失。


**但是在实际多分类交叉熵计算时，并没有出现负类预测值计算的loss，这是为什么？**



## 多分类交叉熵损失 Multi-classes Cross Entropy 
在多分类模型中，交叉熵损失（Cross-Entropy Loss）衡量模型预测的类别分布与实际类别之间的差异。其计算过程主要关注的是正确分类的概率，而错误分类的预测值（非目标类别的概率）对损失的直接影响相对较小。原因可以从交叉熵损失的定义和计算方式来理解：

在标准的多分类交叉熵中，损失函数仅考虑了正确类别的预测概率，而没有直接处理错误类别的预测值。这是因为我们通常使用 softmax 函数将模型的输出转化为概率分布，错误类别的概率间接影响了正确类别的概率。

### 标准的多分类交叉熵损失

假设：
- $q_i$ 是真实标签的 one-hot 编码（正确类别为 1，其他类别为 0）。
- $p_i$ 是通过 softmax 得到的模型对类别 $i$ 的预测概率。

标准的多分类交叉熵损失为：

$$
\text{Cross-Entropy Loss} = - \sum_{i} q_i \log(p_i)
$$

这里，只有正确类别对应的 $q_i = 1$ 会对损失产生影响，其他错误类别的预测概率 $p_i$ 对损失的直接贡献为 0，因为对应的 $q_i = 0$。

### 注意事项

1. **预测值的问题**：
   如果不使用 softmax，而是直接使用 logits，可能会出现 $p_i$ 不在 $[0, 1]$ 范围内的情况。因此，可以对每个 $p_i$ 使用 **Sigmoid 函数** 来将 logits 转化为概率：

   $$
   p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
   $$

   其中 $z_i$ 是模型对类别 $i$ 的原始输出（logits）。

2. **类别之间的独立性问题**：
   在多分类问题中，使用 softmax 可以保证类别概率总和为 1。如果分别对每个类别使用 Sigmoid，则每个类别的预测将相互独立，无法保证总和为 1。这在某些情况下可能不符合问题的要求，需要根据具体任务来选择。**可以用于多标签训练的场合！！！**

3. **多分类交叉熵损失的常见的公式无法用于多标签和softlabel训练。需要借助sigmoid和Modified CE才行。**

### 总结

在不考虑 softmax 的情况下，我们可以借鉴二分类交叉熵的思路，通过修改损失函数显式地加入对错误预测类别的惩罚项。这种方法的优点是可以更直接地控制错误分类对损失的影响，但同时需要处理 logits 的归一化问题，避免出现不可解释的概率分布。


# Focal Loss理解
**Focal Loss**本身是针对困难样本进行权重倾斜的一种方法，他的实现需要通过根不同的具体的loss函数结合进行，当与不同loss结合时，会有不同的表现形式。

**Focal Loss** 的普适公式可以从正负样本的角度统一表示。普适的 **Focal Loss** 公式可以写成：
<<<<<<< HEAD

$$
\text{Focal Loss} = \alpha (1 - \hat{y}_{i^*})^\gamma \cdot \text{Positive Loss}_{i^*} + \sum_{i \neq i^*}[(1 - \alpha) \hat{y_i}^\gamma \cdot \text{Negative Loss}_i]
$$


其中：
- $i^*$是正类的标签，$\hat{y}_{i^*}$ 是模型对正类的预测概率。
- $(1 - \hat{y}_{i^*})^\gamma$ 是正样本的调节因子，用来放大正类中的困难样本（预测值较低）的损失。
- $\hat{y}_{i \neq i^*}^\gamma$ 是负样本的调节因子，用来放大负类中的困难样本（预测值较高）的损失。
- $\alpha$ 是正样本的权重，$(1 - \alpha)$ 是负样本的权重。
- $Positive Loss_{i^*}$ 是正类的损失。
- $Negative Loss_{i \neq i^*}$ 是负类的损失。

**Focal Loss** 的普适公式可以从正负样本的角度统一表示。它有两个作用：

- **1.调节正样本和负样本的损失权重:通过$\alpha$实现；**

- **2.增强对困难样本的关注，弱化对容易样本的贡献：通过$(1 - \hat{y})^\gamma$实现**

由于需要通过区分正负样本来进行正负样本均衡，所以是需要将loss分为$\text{Positive Loss}$和$\text{Negtive Loss}$加权后求和。但是在多分类交叉熵损失**CE**与**Focal Loss**结合时，普遍给出的公式为：

$$
\text{Focal Loss} = - \alpha \cdot \sum_{i} (1 - \hat{y_i})^\gamma \cdot \log(\hat{y_i}) \cdot y_i 
$$

其中：

- $\alpha$ 是一个平衡因子，控制正负类样本的权重（在多分类问题中可以不使用）。
- $\gamma$ 是调节参数，通常取 2，用来降低对容易分类样本的关注。
- $\hat{y_i}$ 是模型的对第i类的预测概率。
- $y_i$ 是第i类的的gt label，正样本为1，负样本为0。

也可以简化为:

$$
\text{Focal Loss} = - \alpha \cdot (1 - \hat{y})^\gamma \cdot \log(\hat{y}) \cdot y
$$

- $\hat{y}$ 是模型的对样本实际类别的预测概率。
- $y$ 是样本实际类别的gt label，正样本为1，负样本为0。
  
如果不考虑**soft label**的情况， $y$恒等于1，进一步简化为：

$$
\text{Focal Loss} = - \alpha \cdot (1 - \hat{y})^\gamma \cdot \log(\hat{y}) \cdot y
$$

通常在代码中$y$就是```gt_label_one_hot```的形式，整体代码是：


```
fl = torch.sum(-alpha * (1 - preds) ** gamma * preds.log() * gt_label_one_hot, dim=1)
```
其中的sum是因为preds是一个向量，表示对多个分类的预测概率；由于sum之前乘上了```gt_label_one_hot```，所以最终的loss就只是正样本的loss。

此处通常有几处疑问：
- 1.为什么没有将负样本的loss也考虑进去？
- 2.为什么在多分类问题中可以不使用alpha？
  
我认为这两个问题其实是一个问题，按照目前我的理解，不使用alpha是因为没有按照FL的普适公式，分别拆开求正、负样本的Loss，因而也就没法通过alpha来对正负样本做平衡。因此其实只存在问题1，为什么不考虑负样本的loss？

**因为在多分类CE Loss中，使用 softmax 函数将模型的输出转化为概率分布，然后再计算交叉熵损失。在正样本的prediction做softmax操作的过程中，已经用到了其他错误分类的预测值；当我们对正样本的prediction做梯度下降时，其实也包含了对负样本预测值的优化，只不过现在这里的alpha是无效的，起不到平衡正负样本的作用，因此可以删除。**

因此，完全套用普适公式的话，参考CE的完整公式，结合Focal Loss原理之后应该是：

$$
\text{Modified Focal Loss} = - \left[ \alpha \cdot（1-\hat{y}_{i^*})^\gamma \cdot \text{PositiveLoss}_{i^*} + (1-\alpha) \cdot \sum_{i \neq i^*} (\hat{y}_i^\gamma \cdot \text{NegtiveLoss}_i) \right]
$$

在Loss采用**CrossEntropyLoss**时，对正负Loss进行加权求和的公式为

$$
\text{Modified Focal Loss} = - \left[ \alpha \cdot（1-\hat{y}_{i^*})^\gamma \cdot (y_{i^*}=1) \cdot \log(\hat{y}_{i^*}) + (1-\alpha) \cdot \sum_{i \neq i^*} [\hat{y}_i^\gamma \cdot ((1 - y_i)=1) \cdot \log(1 - \hat{y}_i) ] \right]
$$

- 其中$i^*$代表样本的实际分类label。

# 代码实现两种CE+FocalLoss

```
def focal_loss(inputs, targets, alpha=0.25):
    alpha = alpha
    gamma = 2.0
    eps = 1e-7
    y_pred = torch.softmax(inputs, dim=1)
    gt_cls_one_hot = F.one_hot(targets, num_classes=20).permute(0, 2, 1)
    ce = -1 * torch.log(y_pred+eps) * gt_cls_one_hot
    floss = torch.pow((1-y_pred), gamma) * ce
    floss = torch.mul(floss, alpha)
    floss = torch.sum(floss,dim=1)
    return floss
def focal_loss_with_negtive_label_loss_added(inputs, targets, alpha=0.25, one_hot=False):
    alpha = alpha
    gamma = 2.0
    eps = 1e-7
    #1.此处的softmax还是包含了负标签预测值的影响，因为后续有了负标签的直接loss计算，这里可以改为sigmoid；
    #2.此处可以改为对每个元素做sigmoid，这样可以用到多标签训练任务中；
    y_pred = torch.softmax(inputs, dim=1)
    if not one_hot:
        gt_cls_one_hot = F.one_hot(targets, num_classes=20).permute(0, 2, 1)
    else:
        gt_cls_one_hot = targets
    floss = -alpha * torch.pow((1-y_pred), gamma) * torch.log(y_pred+eps) * gt_cls_one_hot  - \
          (1-alpha) * torch.pow(y_pred, gamma) * torch.log(1-y_pred+eps) * (1 - gt_cls_one_hot)
    floss = torch.sum(floss,dim=1)
    return floss

#softmax替换为sigmoid
def focal_loss_sigmoid_with_negtive_label_loss_added(inputs, targets, alpha=0.25, one_hot=False):
    alpha = alpha
    gamma = 2.0
    eps = 1e-7
    y_pred = torch.sigmoid(inputs)
    if not one_hot:
        gt_cls_one_hot = F.one_hot(targets, num_classes=20).permute(0, 2, 1)
    else:
        gt_cls_one_hot = targets
    floss = -alpha * torch.pow((1-y_pred), gamma) * torch.log(y_pred+eps) * gt_cls_one_hot  - \
          (1-alpha) * torch.pow(y_pred, gamma) * torch.log(1-y_pred+eps) * (1 - gt_cls_one_hot)
    floss = torch.sum(floss,dim=1)
    return floss
```


# 设计实验

| 实验条件 |  |
|--|--|
|project|new_yolov1_pytorch|
|commit id||
|backbone| res18|
|neck| FPN|
|loss| confidence_score, location loss:BCE(xy),(wh), classes loss:CE|


置信度的loss采用MSE VS BCEWithLogitsLoss：
|Condition|mAP|
|--|--|
|conf_score loss:MSE| 0.7149|
|conf_score loss:nn.BCEWithLogitsLoss(prediction经过sigmoid，负标签贡献loss的二分类BCE)|0.675|
**置信度loss采用MSE更好**

类别loss的loss采用CE+FocalLoss VS Modified BCE+FocalLoss：
|Condition|mAP|
|--|--|
|class loss :prediction经过softmax，负标签不直接贡献loss的CE+FocalLoss| 0.7121|
|class loss :prediction经过softmax，负标签直接贡献loss的Modified CE+FocalLoss|0.7138 |
|class loss :prediction经过sigmoid，负标签直接贡献loss的Modified CE+FocalLoss| 0.7195|

**类别loss的loss采用Modified CE+FocalLoss** 表现更好。
