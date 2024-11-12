---
title: "IoU Loss及其改进"
author: "lvsolo"
math: true
date: "2024-11-11"
tags: ["DeepLearning", "Loss", "training", "YOLO", "Experiments"]
---

based on new-yolov1-pytorch project

# 2.4 IoU Loss
IoU Loss

| Model | mAP(07test) | LogFile |
|---|---|---|
| FPN/MultiHeadFPN | 0.7149 | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142 |
|IoULoss replace origin txtytwth loss|0.558 |log_myYOLOWithFPNMultiPredWithIoULoss_iouweight10_yolo_160|
|添加中心点距离最近的target assign机制，避免按label顺序匹配gt object对应的特征点，IoULoss replace origin txtytwth loss|0.571 |log_myYOLOWithFPNMultiPredWithIoULoss_iouweight10_targetassian_by_min_dist_yolo_160|

GIoULoss
| Model | mAP(07test) | LogFile |
|---|---|---|
| FPN/MultiHeadFPN | 0.7149 | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142 |
|GIoULoss replace origin txtytwth loss|0.6674 |log_myYOLOWithFPNMultiPredWithGIoULoss_SGD_iouweight1_tvgiouloss_sum_target_assign_by_min_dist_yolo_130|

DIoULoss

CIoULoss

SIoULoss

EIoULoss

Luxury IoU Loss:
|||
|---|---|
|condition|mAP|
|txtytwth_iou_weightsum_loss = txtytwth_loss + giou_loss * 1||

