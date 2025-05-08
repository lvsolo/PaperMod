---
title: "Center-based 3D Object Detection and Tracking"
date: "2024-01-23T18:00:00+08:00"
author: "lvsolo"
---
paper:[CenterPoint: Center-based 3D Object Detection and Trackin](https://readpaper.com/pdf-annotate/note?pdfId=4512589404061732865&noteId=1572645379722562304)

# 1. CenterPoint & History

CenterNet(Objects as Points) --> CenterPoint 二者的联系

| 分支              | CenterNet (2D)    | CenterPoint (3D)    |
| ----------------- | ----------------- | ------------------- |
| Heatmap           | ✅ 类别中心点热图 | ✅ 类别中心点热图   |
| Offset            | ✅ x,y 偏移量     | ✅ x,y 偏移量       |
| Size              | ✅ w,h            | ✅ l,w,h            |
| Height            | ❌                | ✅ z 值             |
| Rotation          | ❌（不需要）      | ✅ 使用 sin, cos    |
| Velocity          | ❌                | ✅（用于 tracking） |
| Output 特征图维度 | H×W（2D）        | H×W（BEV）         |

区别在于Centeroint是一个两阶段的检测器，而centernet是一个单阶段的


# 2.

# 3.Loss
