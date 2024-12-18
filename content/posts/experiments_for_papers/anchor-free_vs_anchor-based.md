---
title: "Anchor-free VS Anchor-based"
author: "lvsolo"
math: true
date: "2024-11-05"
tags: ["DeepLearning", "training", "OD", "YOLO"]
---

# Anchor-free VS Anchor-based
| 特征                  | **Anchor-based**                      | **Anchor-free**                     |
|----------------------|--------------------------------------|------------------------------------|
| **核心思想**          | 使用预定义的锚框，回归目标的类别和边界框 | 无需锚框，直接回归目标位置、尺寸或关键点 |
| **目标分配**          | 通过 IoU 匹配和锚框选择正负样本        | 通过中心点、角点或其他关键点进行目标分配 |
| **计算复杂度**        | 高，尤其是多尺度、多锚框的情况下       | 相对较低，避免了锚框生成和匹配过程   |
| **训练难度**          | 锚框匹配过程可能导致正负样本不平衡     | 训练过程更简洁，但可能对目标密集或重叠时有挑战 |
| **精度**              | 通常较高，尤其是在多尺度和大物体检测中 | 通常较好，但在一些任务中(1.密集重叠；2.超大超小目标；)精度略低     |
| **适应性**            | 需要手动设计锚框，通过统计学方法（如kemeans等）设计锚框尺寸，适应性较差           | 更强的适应性，尤其在目标形状、大小多样的任务中 |
| **实时性**            | 较低，计算量较大                      | 更高，适合实时检测任务              |

# anchor-free + IOULoss实验中的问题
在anchor-free的YOLOV1模型中，进行IOULoss相关的实验时，发现IoULoss与xywhLoss相比，精度差非常大，IoUloss的改进形式GIoU，DIoU，CIoU几乎完全无法收敛，经过调试发现，在生成FPN对应的的GT targets时，有以下问题：
- 1.不同尺度的featuremap的GT targets，在hw维度squeeze成一维向量后，顺序会发生变化，在采样率较低的尺度上较为明显；
- 2.不同尺度中，被分配了对应特征点的gt targets数量不同，导致在有些尺度上，有些目标没有被预测，在采样率较大的尺度上较为明显；
在通常的认知中，
- 3.anchor-free的模型在重叠密集、特大特小尺寸的目标检测上精度会明显低于有优秀anchor设计的anchor-based模型

这三个问题的原因是什么？

## 1.不同尺度的featuremap的GT targets，在hw维度squeeze成一维向量后，顺序为何会发生变化
一图看懂
|||
|--|--|
|<span style="color: red;">1|<span style="color: red;">2|
|<span style="color: red;">3|<span style="color: red;">4|

|||||
|--|--|--|--|
|0 | 0 |  <span style="color: red;">2 | 0 |
|<span style="color: red;">1 | 0 | 0 | 0 |
| 0 | 0 | 0 |  <span style="color: red;">4 |
|<span style="color: red;">3 | 0 | 0 | 0 |

|||||||||
|--|--|--|--|--|--|--|--|
| 0 | 0 | 0 | 0 |0 | 0 | <span style="color: red;">2 | 0 |
| 0 | 0 | 0 | 0 |0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|<span style="color: red;">1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 |  0 | 0 | 0 | 0 |  <span style="color: red;">4 |
| 0 | 0 | 0 |  0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|<span style="color: red;">3 | 0 | 0 | 0 |0  | 0 | 0 | 0 |


## 2+3.不同尺度中，被分配了特征点的gt targets数量不同，导致在有些尺度上，有些目标没有被预测
因为在下采样过程中，一些重叠目标会被分配到相同的特征点上，优于一个特征点只能对应一个目标的设计，导致在某些尺度上，有些目标没有被预测到。因此会出现问题3中提到的问题，导致anchor-free模型对于密集重叠的场景中，精度较差。
反之，anchor-based的设计，可以通过设置每个特征点上的anchor数量来使得每个特征点上可以对应的目标大于1，在密集重叠的场景中，可以减少由于共用特征点导致的有些目标没有被分配到特征点的问题。

# YoloV1的anchor-free模型在IOULoss实验时遇到的问题

原因就是在生成GT targets时，会有分配到同一个特征点上的目标，按照先后顺序后分配的会覆盖先分配的，但是后分配的目标并不一定是更接近该特征点的目标，这就导致模型无法收敛，在此情况下添加了限制，优先分配中心点距离特征点更近的目标后，问题解决。