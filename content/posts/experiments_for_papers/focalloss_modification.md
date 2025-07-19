---
title: "Focal Loss及其改进"
author: "lvsolo"
math: true
date: "2024-10-24"
tags: ["DeepLearning", "Loss", "training", "YOLO", "Experiments"]
---
based on new-yolov1-pytorch project

### 2.1 FocalLoss: PaperMode blog

- MultiHeadFPN+FocalLoss

| Model                                 | mAP(07test) | LogFile                                                          |
| ------------------------------------- | ----------- | ---------------------------------------------------------------- |
| FPN/MultiHeadFPN                      | 0.7149      | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142 |
| Loss/MultiHeadFPNFocalloss alpha=0.75 | 0.6742      | eval_log/                                                        |
| Loss/MultiHeadFPNFocalloss alpha=0.5  | 0.6964      | eval_log/log_myYOLOWithFPNMultiPredFocalLoss_yolo_143_new        |
| Loss/MultiHeadFPNFocalloss alpha=0.25 | 0.7121      | eval_log/log_myYOLOWithFPNMultiPredFocalLoss_alpha0.25_yolo_154  |

- BiFPN+FocalLoss

| Model                                   | mAP(07test) | LogFile                                                           |
| --------------------------------------- | ----------- | ----------------------------------------------------------------- |
| FPN/MultiHeadFPN                        | 0.7149      | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142  |
| Loss/MultiHeadBiFPNFocalLoss alpha=0.5  | 0.6975      | eval_log/log_myYOLOWithBiFPNMultiPredFocalLoss_yolo_152           |
| Loss/MultiHeadBiFPNFocalLoss alpha=0.25 | 0.7201      | eval_log/log_myYOLOWithBiFPNMultiPredFocalLoss_alpha0.25_yolo_141 |

- AugFPN+FocalLoss

| Model                                    | mAP(07test) | LogFile                                                            |
| ---------------------------------------- | ----------- | ------------------------------------------------------------------ |
| FPN/MultiHeadFPN                         | 0.7149      | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142   |
| Loss/MultiHeadAugFPNFocalLoss alpha=0.5  | 0.6922      | eval_log/log_myYOLOWithAugFPNMultiPredFocalLoss_yolo_143           |
| Loss/MultiHeadAugFPNFocalLoss alpha=0.25 | 0.7128      | eval_log/log_myYOLOWithAugFPNMultiPredFocalLoss_alpha0.25_yolo_154 |

### 2.2 PolyFocalLoss

| Model                                                | mAP(07test) | LogFile                                                          |
| ---------------------------------------------------- | ----------- | ---------------------------------------------------------------- |
| FPN/MultiHeadFPN                                     | 0.7149      | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142 |
| Loss/MultiHeadFPNPolyLossFL poly_scale=1 poly_pow=1  | 0.7182      |                                                                  |
| Loss/MultiHeadFPNPolyLossFL poly_scale=1 poly_pow=2  | 0.7187      |                                                                  |
| Loss/MultiHeadFPNPolyLossFL poly_scale=1 poly_pow=3  | 0.7199      |                                                                  |
| Loss/MultiHeadFPNPolyLossFL poly_scale=-1 poly_pow=1 | 0.7073      |                                                                  |
| Loss/MultiHeadFPNPolyLossFL poly_scale=-1 poly_pow=2 | 0.7204      |                                                                  |
| Loss/MultiHeadFPNPolyLossFL poly_scale=-1 poly_pow=3 | 0.7212      |                                                                  |

### 2.3 VariFocalLoss

| Model                                          | mAP(07test) | LogFile                                                          |
| ---------------------------------------------- | ----------- | ---------------------------------------------------------------- |
| FPN/MultiHeadFPN                               | 0.7149      | eval_log/log_eval_myYOLOWithFPNMultiPred_with_sam_for_3_head_142 |
| Loss/MultiHeadFPNVariLossFL alpha=0.25 gamma=2 | 0.6729      | eval_log/log_myYOLOWithFPNMultiPredVariFocalLoss_yolo_160        |


### Distribution Focal Loss(DFL)
