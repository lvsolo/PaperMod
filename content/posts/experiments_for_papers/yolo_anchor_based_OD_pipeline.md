---
title: "Paper Realization on No Anchor-based YOLO Pipeline"
author: "lvsolo"
date: "2024-10-10"
tags: ["DeepLearning", "training", "YOLO", "Paper"]
---

Code Project: yolov3-pytorch
BaseLine：原代码中的初始模型

# Target Assigner

## 1. IoU Aware OD

https://readpaper.com/pdf-annotate/note?pdfId=4545076807438327809&noteId=2569501418004823040

### 解决问题：

1) iou 作为重要的检测指标没有直接引导模型训练；
2) nms操作时只考虑cls score或者conf score的问题，加入iou预测分支，与cls/conf score相乘作为联合分数作nms排序；

### 做法： 
除了常规的cls loss和location loss之外，引入了一个iou prediction分支：

1) 训练过程中该分支的gt是每次对模型输出进行解码得到prediction的bbox，跟真实的gt进行计算得到iou，作为iou prediction的gt;
2) 将iou prediction输出的iou值与cls score相乘，作为nms排序时使用的分数。



## 2. PAA：用于anchor based模型

[Probabilistic Anchor Assignment with IoU Prediction for Object Detection](https://readpaper.com/pdf-annotate/note?pdfId=4544162912848060417&noteId=2575494594179433728)

沿用了[IoU Aware OD]中的IoU prediction分支，将cls score和iou prediction分支的输出相乘作为最终的score，进行GMM概率分布估计。

PAA（Probabilistic Anchor Assignment，概率锚框分配）是用于基于锚框（Anchor-based）物体检测方法中的一种优化技术，旨在提高锚框分配的准确性和效率。基于锚框的方法，如 Faster R-CNN、YOLO 和 SSD，依赖于预定义的锚框来预测目标的边界框（bounding box）。然而，传统的锚框分配策略（例如，最大重叠（IoU）分配或是固定的锚框匹配规则）可能存在一些问题，如误匹配或过度依赖于单一的评分标准。

PAA的背景与问题

在基于锚框的物体检测中，常见的做法是通过计算预测锚框与真实目标框之间的**交并比（IoU, Intersection over Union）**来进行匹配。对于每个锚框，会选择与之重叠度最高的真实框进行分配，这种策略称为硬分配（hard assignment）。

然而，这种方法有以下几个问题：

过度依赖IoU阈值：使用硬IoU阈值进行锚框分配，可能会导致误匹配或错过一些具有较低IoU但仍然能正确预测的目标。
锚框的多样性不足：一个固定的锚框策略可能无法涵盖目标物体在不同尺度、形状等方面的多样性。
负样本问题：传统方法容易将一些与真实目标无关的锚框当作负样本，导致训练时的负样本过多，从而影响学习效率。

PAA的优化思路

PAA（概率锚框分配）优化通过引入概率模型来代替传统的硬分配方式，从而提供更灵活的锚框分配方式。其核心思想是根据预测框与真实框之间的匹配度（如IoU）来概率性地分配锚框，而不是直接将锚框分配为正样本或负样本。

具体来说，PAA的优化策略如下：

基于概率的锚框分配：PAA不再使用固定的阈值来进行锚框的硬分配，而是通过计算每个锚框与目标框的匹配概率来进行软分配。这种方法能够更好地处理那些IoU值较低但仍然能够准确检测到目标的情况。

概率分配机制：对于每个锚框，PAA根据它与每个真实目标框之间的匹配程度（通常是IoU或其他匹配度度量）计算一个匹配概率，并将锚框以概率的方式分配给不同的目标。这种分配方式可以缓解传统硬分配方法中由于误匹配带来的影响。

改进的负样本处理：在传统的锚框分配中，负样本（即没有与任何目标框重叠的锚框）往往会影响模型的训练，特别是当锚框的负样本数量过多时。PAA通过对负样本的概率分配，使得负样本对模型的影响更小，从而提高了训练效率。

增强模型鲁棒性：通过引入概率分配，PAA使得模型在面对复杂的场景时能够更加鲁棒，避免了传统硬分配方法可能导致的误匹配问题，从而提高了检测精度。

PAA优化的优势

更加灵活的分配方式：PAA引入了概率分配机制，能够更精准地分配锚框，避免了硬性分配的误差。
减少误匹配：由于采用了概率模型，PAA能够减少那些IoU较低的锚框误分配为负样本或错过正样本的情况。
提升检测精度：通过软分配，PAA优化了正负样本的平衡，提高了目标检测的精度，尤其是在复杂场景中，能够更好地处理目标物体之间的重叠和干扰。
高效训练：由于负样本的概率分配机制，PAA能够有效避免负样本对模型训练的干扰，使训练更加高效。
总结
PAA（Probabilistic Anchor Assignment）是一种改进的锚框分配策略，通过引入概率分配机制来优化传统锚框分配方法，解决了传统硬分配方法中的误匹配和训练效率低下等问题。PAA不仅提升了模型在复杂场景下的鲁棒性，还提高了检测精度，并且能够更高效地处理负样本。它为基于锚框的目标检测提供了一种新的思路，特别适用于处理多尺度、多重重叠的目标检测任务。


## 3.多尺度对比anchor学习方法
