---
title: "Paper Realization on No Anchor-based YOLO Pipeline"
author: "lvsolo"
date: "2024-10-10"
tags: ["DeepLearning", "training", "YOLO", "Paper"]
---

Code Project: new-YOLOv1_PyTorch
BaseLine：原代码中的初始模型

改进方式一：FPN
1. 加入FPN模块，在多个尺度的特征图上进行预测；
2. BiFPN实现
3. AugFPN实现

改进方式二：Loss
1. FocalLoss实现+应用: 

    [Focal Loss在分类loss中的应用](/content/posts/experiments_for_papers/focalloss.md)

    1.1 PolyFocalLoss

    1.2 VariFocalLoss:

2. IoULoss:

   [IoU及其改进形式](https://learnopencv.com/iou-loss-functions-object-detection/#iou-based-loss-functions)

    2.1 GIoULoss



改进方式三：Attention


改进方式四：layer、算子结构优化


其他改进方式：
1. EMA
2. DHSA

FCOS?ATSS?CenterNet



