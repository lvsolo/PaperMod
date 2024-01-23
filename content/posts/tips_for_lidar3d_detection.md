---
title: "tips for lidar3d detection"
author: "lvsolo"
date: "2024-01-05"
tags: ["lidar", "3dDet", "commensense"]
---

### pypcd install 

using codes below instead of ```pip install pypcd``` which is not well maintained.
```
pip3 install --upgrade git+https://github.com/klintan/pypcd.git
```
大小汽车 90m
摩托车 70m
三轮车 75m
自行车、行人 60m