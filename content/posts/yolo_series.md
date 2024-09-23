---
title: "YOLO Series"
author: "lvsolo"
date: "2024-09-16"
tags: ["DeepLearning", "training", "YOLO"]
---

YOLOv2 对于 YOLOv1 的改进主要包括以下几个方面：

1. **Batch Normalization（批归一化）**：在每个卷积层后添加了批归一化，提高了模型的收敛速度，减少了对其他正则化方法的依赖，提升了模型的泛化能力，使得 mAP 提升了约 2%  。

2. **High Resolution Classifier（高分辨率分类器）**：在 ImageNet 数据集上使用更高分辨率（448x448）的图片进行预训练，使得模型在检测数据集上微调之前已经适应了高分辨率输入，提升了 mAP 约 4%  。

3. **Convolutional With Anchor Boxes（带 Anchor Boxes 的卷积）**：借鉴了 Faster R-CNN 的思想，引入了 Anchor Boxes，使用卷积层预测边界框，提高了目标的定位准确率  。

4. **Dimension Clusters（维度聚类）**：通过 K-means 聚类自动确定 Anchor Boxes 的尺寸，使得模型更容易学习预测好的检测结果  。

5. **Direct location prediction（直接位置预测）**：改进了坐标的预测方式，使得模型在预测时更加直接和准确 。

6. **Fine-Grained Features（细粒度特征）**：引入了 passthrough 层来利用更精细的特征图，有助于检测小物体，提升了模型性能约 1%  。

7. **Multi-Scale Training（多尺度训练）**：在训练过程中，每隔一定的迭代次数后改变模型的输入图片大小，增强了模型对不同尺寸图片的鲁棒性，提升了 mAP  。

这些改进使得 YOLOv2 在保持原有速度优势的同时，显著提高了检测精度。

