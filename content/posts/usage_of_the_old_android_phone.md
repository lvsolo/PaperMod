---
title: "Usages for Old Cellphones"
date: "2023-12-10"
author: "lvsolo"
tags: ["old products reuse", "android", "ubuntu", "linux"]
---

不需要root即可在android手机上安装linux，从而部署各种系统、博客等。
更进一步的，能够通过linux调用手机上的摄像头等传感器，使手机丰富的传感器组件得到利用。
1. 手机用andronix+termux+vnc安装ubuntu系统
    不使用桌面的话可以不安装vnc
    电脑连接手机上的termux，使用质感文件管理器可以管理termux中的文件
    https://www.jianshu.com/p/2e6c8152a2ba
    
2. 手机andronix的ubuntu22中安装anaconda
    https://zhuanlan.zhihu.com/p/608147907
    
3.手机中ubuntu安装hugo+papermode theme
    ```bash
    sudo apt install hugo
    hugo new site hugo_themes --format yaml
    cd hugo_themes
    cd themes
    git clone git@github.com:lvsolo/PaperMod.git
    ```
    项目中有crontab定时代码，可以让手机中的博客自动更新github上的最新推送，从而做到随处编辑都可几乎实时更新博客页面。

4.如何使用安装好的ubuntu系统，调用摄像头等传感器？
TODO
