---
title: "jishi tips for ultralytics project and python3.8"
author: "lvsolo"
date: "2023-11-27"
tags: ['jishi', 'competation', 'yolov5']

---

## environment building

* 安装tensorrt-8.6.1环境：
    手动下载tensorrt lib8.6.1后上传至服务器安装（https://pypi.nvidia.com/tensorrt-libs/tensorrt_libs-8.6.1-py2.py3-none-manylinux_2_17_x86_64.whl）
    
    安装pycuda前，
    `sudo apt-get install python3.8-dev libxml2-dev libxslt-dev`
    

## tips
pt2onnx 转换生成的如果是static的onnx，那么就无法生成dynamic的engine了

## reference
### TODO
dynamic shape model: https://zhuanlan.zhihu.com/p/299845547 

## training tips
* 💡 Add --cache ram or --cache disk to speed up training (requires significant RAM/disk resources).




## experiments 
### TODO
0.如何将dynamic输入的模型调通？使用不加padding的letterbox得到正确结果

1.ultralytics默认选项设置训练出来的模型，对于resize、letterbox的适应性试验；

2.如果适应很好，可以替换letterbox

3.如果适应不好，需要：
* 1）使用torch重构letterbox；
* 2）使用cuda memcp的stride格式（存在么？）
* 3) 使用cuda memcpy可以从torch tensor中拷贝么？或者直接使用torch tensor的cuda显存？
* 4) 优化目前的myletterbox
* 5）训练一个真正的dynamic输入的模型

4.如何训练出可以接受resize输入，不需要补齐黑边的模型

