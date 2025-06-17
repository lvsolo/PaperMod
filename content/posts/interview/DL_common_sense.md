title: "DL common sense"
author: "Benboba"
date: "2025-05-18"
tags: ["knowledge review",  "interview"]


### 1.one stage ？  two stage？

two stage：ROI Pooling之前，需要根据Region Proposal Network（RPN提供is object置信度和物体框的粗略坐标）提供的ROI，在featuremap上找到对应的feature区域，然后roi pooling得到固定大小的feature去做分类和（更精确的）物体框回归

    ![0]()

one stage：没有根据ROI找到对应feature这一步，比如yolo因为有固定的anchors，直接就在原图上做回归和分类

目标检测中样本不均衡问题

[https://mp.weixin.qq.com/s/iOAICJege2b0pCVxPkvNiA](https://mp.weixin.qq.com/s/iOAICJege2b0pCVxPkvNiA)

### 2.faster-rcnn fast-rcnn rcnn关系

    ![0]()

### 3.faster rcnn原理

[https://www.cnblogs.com/bile/p/9117253.html](https://www.cnblogs.com/bile/p/9117253.html)

SSD原理

[https://blog.csdn.net/weixin_43869605/article/details/119970953](https://blog.csdn.net/weixin_43869605/article/details/119970953)

yolov3 4 5 X原理:

### 4.为什么一般深度学习检测不适合小目标？如何解决？

### 5.进程与线程是什么？

#### 形象的比喻（形容描述）：进程是列车，线程是被牵引的车厢

链接：[https://www.zhihu.com/question/25532384/answer/411179772](https://www.zhihu.com/question/25532384/answer/411179772)

做个简单的比喻：进程=火车，线程=车厢

* 线程在进程下行进（单纯的车厢无法运行）
* 一个进程可以包含多个线程（一辆火车可以有多个车厢）
* 不同进程间数据很难共享（一辆火车上的乘客很难换到另外一辆火车，比如站点换乘）
* 同一进程下不同线程间数据很易共享（A车厢换到B车厢很容易）
* 进程要比线程消耗更多的计算机资源（采用多列火车相比多个车厢更耗资源）
* 进程间不会相互影响，一个线程挂掉将导致整个进程挂掉（一列火车不会影响到另外一列火车，但是如果一列火车上中间的一节车厢着火了，将影响到所有车厢）
* 进程可以拓展到多机，线程最多适合多核（不同火车可以开在多个轨道上，同一火车的车厢不能在行进的不同的轨道上）
* 进程使用的内存地址可以上锁，即一个线程使用某些共享内存时，其他线程必须等它结束，才能使用这一块内存。（比如火车上的洗手间）－"互斥锁"
* 进程使用的内存地址可以限定使用量（比如火车上的餐厅，最多只允许多少人进入，如果满了需要在门口等，等有人出来了才能进去）－“信号量”

#### 底层本质（名词定义）：

链接：[https://www.zhihu.com/question/25532384/answer/81152571](https://www.zhihu.com/question/25532384/answer/81152571)

首先来一句概括的总论：进程和线程都是一个时间段的描述，是CPU工作时间段的描述。是运行中的程序指令的一种描述，这需要与程序中的代码区别开来。

另外注意这里我说的进程线程概念，和编程语言中的API接口对应的进程/线程是有差异的。

下面细说背景：

CPU+RAM+各种资源（比如显卡，光驱，键盘，GPS, 等等外设）构成我们的电脑，但是电脑的运行，实际就是CPU和相关寄存器以及RAM之间的事情。

一个最最基础的事实：CPU太快，太快，太快了，寄存器仅仅能够追的上他的脚步，RAM和别的挂在各总线上的设备则难以望其项背。那当多个任务要执行的时候怎么办呢？轮流着来?或者谁优先级高谁来？不管怎么样的策略，一句话就是在CPU看来就是轮流着来。而且因为速度差异，CPU实际的执行时间和等待执行的时间是数量级的差异。比如工作1秒钟，休息一个月。所以多个任务，轮流着来，让CPU不那么无聊，给流逝的时间增加再多一点点的意义。这些任务，在外在表现上就仿佛是同时在执行。

一个必须知道的事实：执行一段程序代码，实现一个功能的过程之前 ，当得到CPU的时候，相关的资源必须也已经就位，就是万事俱备只欠CPU这个东风。所有这些任务都处于就绪队列，然后由操作系统的调度算法，选出某个任务，让CPU来执行。然后就是PC指针指向该任务的代码开始，由CPU开始取指令，然后执行。

这里要引入一个概念：除了CPU以外所有的执行环境，主要是寄存器的一些内容，就构成了的进程的上下文环境。进程的上下文是进程执行的环境。当这个程序执行完了，或者分配给他的CPU时间片用完了，那它就要被切换出去，等待下一次CPU的临幸。在被切换出去做的主要工作就是保存程序上下文，因为这个是下次他被CPU临幸的运行环境，必须保存。

串联起来的事实：前面讲过在CPU看来所有的任务都是一个一个的轮流执行的，具体的轮流方法就是：先加载进程A的上下文，然后开始执行A，保存进程A的上下文，调入下一个要执行的进程B的进程上下文，然后开始执行B,保存进程B的上下文。。。。========= 重要的东西出现了========进程和线程就是这样的背景出来的，两个名词不过是对应的CPU时间段的描述，名词就是这样的功能。

* 进程就是上下文切换之间的程序执行的部分。是运行中的程序的描述，也是对应于该段CPU执行时间的描述。
* 在软件编码方面，我们说的进程，其实是稍不同的，编程语言中创建的进程是一个无限loop，对应的是tcb块。这个是操作系统进行调度的单位。所以和上面的cpu执行时间段还是不同的。
* 进程，与之相关的东东有寻址空间，寄存器组，堆栈空间等。即不同的进程，这些东东都不同，从而能相互区别。

线程是什么呢？进程的颗粒度太大，每次的执行都要进行进程上下文的切换。如果我们把进程比喻为一个运行在电脑上的软件，那么一个软件的执行不可能是一条逻辑执行的，必定有多个分支和多个程序段，就好比要实现程序A，实际分成 a，b，c等多个块组合而成。那么这里具体的执行就可能变成：

程序A得到CPU =》CPU加载上下文，开始执行程序A的a小段，然后执行A的b小段，然后再执行A的c小段，最后CPU保存A的上下文。

这里a，b，c的执行是共享了A进程的上下文，CPU在执行的时候仅仅切换线程的上下文，而没有进行进程上下文切换的。进程的上下文切换的时间开销是远远大于线程上下文时间的开销。这样就让CPU的有效使用率得到提高。这里的a，b，c就是线程，也就是说线程是共享了进程的上下文环境，的更为细小的CPU时间段。线程主要共享的是进程的地址空间。到此全文结束，再一个总结：

进程和线程都是一个时间段的描述，是CPU工作时间段的描述，不过是颗粒大小不同。

注意这里描述的进程线程概念和实际代码中所说的进程线程是有区别的。编程语言中的定义方式仅仅是语言的实现方式，是对进程线程概念的物化。

进程

### 6.相机标定原理

世界坐标系---（外参）--->相机坐标系--（内参）--->像素坐标系

[https://blog.csdn.net/honyniu/article/details/51004397](https://blog.csdn.net/honyniu/article/details/51004397)

    ![0]()

张正友标定法：[https://blog.csdn.net/qq_40369926/article/details/89251296](https://blog.csdn.net/qq_40369926/article/details/89251296)

[https://www.cnblogs.com/leoking01/p/13341190.html](https://www.cnblogs.com/leoking01/p/13341190.html)

LM算法：[https://www.cnblogs.com/shhu1993/p/4878992.html](https://www.cnblogs.com/shhu1993/p/4878992.html)

7.EM算法

[https://blog.csdn.net/abcjennifer/article/details/8170378](https://blog.csdn.net/abcjennifer/article/details/8170378)

[https://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html](https://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)

8.卡尔曼滤波

[https://www.zhihu.com/question/23971601/answer/839664224](https://www.zhihu.com/question/23971601/answer/839664224)

[https://blog.csdn.net/u010720661/article/details/63253509](https://blog.csdn.net/u010720661/article/details/63253509)

各种滤波器可视化：[http://www.lifl.fr/~casiez/1euro/InteractiveDemo/](http://www.lifl.fr/~casiez/1euro/InteractiveDemo/)

关键点防抖:一欧元滤波[http://www.lifl.fr/~casiez/publications/CHI2012-casiez.pdf](http://www.lifl.fr/~casiez/publications/CHI2012-casiez.pdf)[](http://www.lifl.fr/~casiez/1euro/InteractiveDemo/)

Deepsort sort JDE

9.Focal Loss

Focal loss 哪些情况下不适用?

数据不干净, 会有很多的噪声，导致将噪声当作hard数据去学习，进而导致精度下降

[https://zhuanlan.zhihu.com/p/49981234](https://zhuanlan.zhihu.com/p/49981234)

Cross ENtropy--->α-B(alanced)C(ross)E(ntropy)---->focal loss+α-BCE

这个损失函数是在标准交叉熵损失基础上修改得到的

10.机器学习常用术语

[https://mp.weixin.qq.com/s/qjcPnEAo4G_BAvShvaWNAg](https://mp.weixin.qq.com/s/qjcPnEAo4G_BAvShvaWNAg)

11.极大似然估计 交叉熵损失关系

二分类问题的极大似然估计 == 交叉熵损失最小

多分类?TODO

[https://blog.csdn.net/cxx654/article/details/113346864](https://blog.csdn.net/cxx654/article/details/113346864)

[https://zhuanlan.zhihu.com/p/445551303](https://zhuanlan.zhihu.com/p/445551303)

[https://zhuanlan.zhihu.com/p/98785902](https://zhuanlan.zhihu.com/p/98785902)

12.数据降维可视化 t-sne算法

[https://blog.csdn.net/sinat_20177327/article/details/80298645](https://blog.csdn.net/sinat_20177327/article/details/80298645)

13.nms softnms

iou-nms实现

import  torch
import numpy as np
def iou(bb1,bb2):
    x11,y11,w1,h1 = bb1
    x21,y21,w2,h2 = bb2

# print(x11,y11,w1,h1)

# print(x21,y21,w2,h2)

    x12,y12 = x11+w1, y11+h1
    x22,y22 = x21+w2, y21+h2

    x_min = min(max(x11,x21),min(x12,x22))
    x_max = max(max(x11,x21),min(x12,x22))
    y_min = min(max(y11,y21),min(y12,y22))
    y_max = max(max(y11,y21),min(y12,y22))

    I = (x_max-x_min) * (y_max-y_min)
    A1 = abs((x11-x12)*(y11-y12))
    A2 = abs((x21-x22)*(y21-y22))
    return I / (A1 +A2 - I)
def NMS(bbs,scores,res_num=10,iou_thr=0.3):
    orders = scores.argsort()[::-1]
    keep_num = 0
    keep_org_ind = []
    keep_bb, keep_score = [], []
    scores = scores[orders]
    bbs = bbs[orders]
    while keep_num < res_num and len(orders):
        keep_org_ind.append(orders[0])
        keep_bb.append(bbs[0])
        keep_score.append(scores[0])
        bbs = bbs[1:]
        scores = scores[1:]
        orders = orders[1:]
        ious = []
        for j in range(len(scores)):
            ious.append(iou(bbs[j],keep_bb[-1]))
        ious = np.array(ious)
        scores = scores[np.where(ious<iou_thr)]
        bbs = bbs[np.where(ious<iou_thr)]
        orders = orders[np.where(ious<iou_thr)]
        keep_num += 1

# print(len(bbs),len(orders), keep_num)

    return keep_bb,keep_score, keep_org_ind

def softNMS(bbs,scores,res_num=10,iou_thr=0.3):#不一定对
    orders = scores.argsort()[::-1]
    keep_num = 0
    keep_org_ind = []
    keep_bb, keep_score = [], []
    scores = scores[orders]
    bbs = bbs[orders]
    while keep_num < res_num and len(orders):
        keep_org_ind.append(orders[0])
        keep_bb.append(bbs[0])
        keep_score.append(scores[0])
        bbs = bbs[1:]
        scores = scores[1:]
        orders = orders[1:]
        ious = []
        for j in range(len(scores)):
            ious.append(iou(bbs[j],keep_bb[-1]))
        ious = np.array(ious)
        scores = scores[np.where(ious<iou_thr)]
        bbs = bbs[np.where(ious<iou_thr)]
        orders = orders[np.where(ious<iou_thr)]
        sigma =1.0
        scores = scores * np.exp(-(ious[np.where(ious<iou_thr)])**2/sigma)
        tmp = scores.argsort()[::-1]
        scores = scores[tmp]
        bbs = bbs[tmp]
        orders = orders[tmp]
        keep_num += 1

# print(len(bbs),len(orders), keep_num)

    return keep_bb,keep_score, keep_org_ind

colors = 'wgrcbyk'
bbs = np.random.rand(100,4)
scores = np.random.rand(100,)
import matplotlib.pyplot as plt
%matplotlib inline
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.xlim(0, 2)
plt.ylim(0, 2)
for i, (bb, score) in enumerate(zip(bbs, scores)):

# print(bb, score)

    rect=plt.Rectangle(
            (bb[0], bb[1]),  # (x,y)矩形左下角
            bb[2],  # width长
            bb[3],  # height宽
            fill = False,
            color=colors[i%len(colors)])
    ax1.add_patch(rect)
plt.show()

##NMS
nmsbb, nmsscore, nms_org_ind = NMS(bbs,scores,10)

import matplotlib.pyplot as plt
%matplotlib inline
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.xlim(0, 2)
plt.ylim(0, 2)
for i, (bb, score) in enumerate(zip(nmsbb, nmsscore)):

# print(bb, score)

    rect=plt.Rectangle(
            (bb[0], bb[1]),  # (x,y)矩形左下角
            bb[2],  # width长
            bb[3],  # height宽
            fill = False,
            color=colors[i%len(colors)])
    ax1.add_patch(rect)
plt.show()

##soft nms 不一定对
nmsbb, nmsscore, nms_org_ind = softNMS(bbs,scores,10)

import matplotlib.pyplot as plt
%matplotlib inline
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.xlim(0, 2)
plt.ylim(0, 2)
for i, (bb, score) in enumerate(zip(nmsbb, nmsscore)):

# print(bb, score)

    rect=plt.Rectangle(
            (bb[0], bb[1]),  # (x,y)矩形左下角
            bb[2],  # width长
            bb[3],  # height宽
            fill = False,
            color=colors[i%len(colors)])
    ax1.add_patch(rect)
plt.show()

    ![0]()

    ![0]()

    ![0]()

14.residual block  inception block  dense block resnext block

    ![0]()

15.常用的移动端 轻量级模型

MobileNet V123 1+resnet = 2 + se block =3

GhostNet：[GhostNet 解读及代码实验（附代码、超参、日志和预训练模型）_TensorSense的博客-CSDN博客](https://blog.csdn.net/u011995719/article/details/105207344?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-105207344-blog-119493239.pc_relevant_multi_platform_featuressortv2removedup&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-105207344-blog-119493239.pc_relevant_multi_platform_featuressortv2removedup&utm_relevant_index=1)

Blaze

EfficientNet 由NAS得到的网络

ShuffleNet

    ![0]()

16.组卷积 深度卷积 深度可分离卷积关系 以及与 15的关系

  空洞卷积（Atrous Convolution）:红色的点为实际要卷积的像素，两个像素之间的距离，就是 dilation rate（扩张率）,dilation rate = 1时就是普通卷积

    ![0](https://note.youdao.com/yws/res/1007/WEBRESOURCEb4957c755cce822b39caf4d49a987ed2)

17.手写卷积伪代码

import numpy as np
h,w,c = 100,100,3
nc = 4
a = np.ones((c,h,w))
kh,kw = 3,3
k = np.ones((nc,c,kh,kw))
stride = 2
nh = (h-kh+1)//stride +1
nw = (w-kw+1)//stride +1
res = []
ind = 0
for cc in range(nc):
    for hh in range(0,h,stride):
        for ww in range(0,w,stride):
            img_pix = []
            k_pix = []
            for kkh in range(-(kh//2), (kh+1)//2):
                for kkw in range(-(kw//2), (kw+1)//2):

# print(ind,cc,hh,ww,kkh,kkw)

    ind+=1
                    if hh+kkh >=0 and hh+kkh < h and ww+kkw >=0 and ww+kkw < w:

# print(a[:,hh+kkh, ww+kkw][:])

    for ccc in range(c):
                            img_pix += [a[ccc,hh+kkh, ww+kkw]]
                            k_pix += [k[cc,ccc,kh//2+kkh, kw//2+kkw]]
            ga = np.array(img_pix).dot(np.array(k_pix))
            res += [ga]
res=np.array(res).reshape(nc,nh,nw)
res=res.transpose(1,2,0)

18.手写AUC的计算代码

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
y    = np.array([1,   0,   0,   0,   1,    0,   1,    0,    0,   1  ])
pred = np.array([0.9, 0.4, 0.3, 0.1, 0.35, 0.6, 0.65, 0.32, 0.8, 0.7])
preds = {}
for i in range(y.size):
    preds[i] = pred[i]
sort_preds = sorted(preds.items(), key=lambda d: -d[1])

sort_ys = []
for i in sort_preds:
    sort_ys += [y[i[0]]]
sort_preds = [s[1] for s  in sort_preds]

# print(sort_preds)

# print(sort_ys)

tp_sum_ = 0
fp_sum_ = 0
pairs = []
for  i in range(len(sort_preds)):
    TP_count =  np.sum(np.array(sort_ys[:i+1])) / np.sum(y)
    FP_count = (i+1- np.sum(np.array(sort_ys[:i+1]))) / (len(y)-np.sum(y))
    pairs.append([TP_count, FP_count])

# for i in range(len(pairs)-1):

# print(pairs[i], pairs[i+1])

# plt.plot(pairs[i], pairs[i+1])

xs = []
ys = []
for i in range(len(pairs)):
    xs.append(pairs[i][1])
    ys.append(pairs[i][0])
plt.plot(xs,ys)
plt.show()

[https://zhuanlan.zhihu.com/p/500806744](https://zhuanlan.zhihu.com/p/500806744)

    ![0](https://note.youdao.com/yws/res/921/WEBRESOURCEabe3a5e6a46d815fc7a0b8910f17333f)

19.dropout原理，会有一个缩放过程，使得mean不变

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)

>>> input[0]
>>> tensor([ 0.5912,  1.2004,  1.3565, -0.0842, -0.8571,  2.0991, -1.3650, -2.0478,
>>> 0.5515, -0.3978, -0.0024, -1.1846, -0.7232, -0.1319,  1.7912, -1.6392])
>>> output[0]
>>> tensor([ 0.7390,  0.0000,  1.6956, -0.1052, -0.0000,  2.6239, -1.7062, -2.5598,
>>> 0.0000, -0.0000, -0.0030, -1.4808, -0.9039, -0.0000,  2.2391, -2.0490])
>>>
>>

# 注意最开始选择了p = 0.2，output进行了缩放，手动计算一下

>>> round(0.5912 * (1 / (1 - 0.2)), 4)
>>> 0.739
>>>
>>

20. 常用的正则化方法

研究者提出和开发了多种适合机器学习算法的正则化方法，如数据增强、L2 正则化、权重衰减（weightdecay某些情况等驾驭L2正则化）、L1 正则化、Dropout、Drop Connect、随机池化和早停等

数据增强

weight_decay momentum normalization :[https://zhuanlan.zhihu.com/p/409346926](https://zhuanlan.zhihu.com/p/409346926)

weight decay  防止过拟合

[https://zhuanlan.zhihu.com/p/339448370](https://zhuanlan.zhihu.com/p/339448370)

weight_decay和L2正则的区别

 SGD 可以等效,在自适应梯度 优化算法 不同

 [https://blog.csdn.net/andyjkt/article/details/107944816](https://blog.csdn.net/andyjkt/article/details/107944816)

[https://zhuanlan.zhihu.com/p/342526154](https://zhuanlan.zhihu.com/p/342526154)

21.常用的匹配策略:

暴力匹配:A中每一个点都与B类每一个点计算距离;取A*B个配对中距离最小的N个作为匹配点;

握手匹配:在暴力匹配的基础上,同时计算B->A和A->B, 两个点互为距离最小的认为是匹配点;

快速近似最近邻匹配:ratio=最近匹配点距离/次近匹配点距离,ratio尽量小才好,相当于匹配的点(类内距)距离小,不匹配的点(类间距)距离大

22.常用的相似度计算方式:

欧式距离

马氏距离

余弦相似度

23.ROI Pooling ROI Align

24.使用梯度下降求一个数的平方根

#求2的平方根
a=2
gt=2#np.sqrt(a)
a0 = a
loss=(a0**2-gt)**2 //可优化的凸函数
for i in range(10000):
    dloss = 2*(a0**2-gt)*2*a0
    a1 = a0 - (0.01) * dloss
    a0 = a1
    print(a0, a0**2)

-25.常用的图像插值方法

    ![0](https://note.youdao.com/yws/res/754/WEBRESOURCE39f1bc1cc35975150e6c22231d396348)

26.EMA模型

[https://blog.csdn.net/weixin_43135178/article/details/122147538](https://blog.csdn.net/weixin_43135178/article/details/122147538)

TP:P/N表示结果, F/T 表示预测 结果正确与否

TPR=TP/P=TP/(TP+FN)=Recall

FPR=FP/N=FP/(FP+TN)=虚警

AUC Area under TPR-FPR

Precise=TP/(TP+FP)=TP/T

Recall=TPR=TP/(TP+FN)=TP/P

Accuracy=(TP+TN)/(TP+TN+FP+FN)=(TP+TN)/(P+N)=Correct / Total

28.排序

class PX():
    def __init(self):
        pass
    def mp(self, l:list):
        for i in range(len(l)):
            for j in range(len(l)-i-1):
                if l[j] > l[j+1]:
                    l[j],l[j+1] = l[j+1],l[j]
                    print(l)

# print(l)

    return l
    def xz(self, l:list):
        for i in range(len(l)):
            max_i = 0
            for j in range(len(l)-i):
                if l[j] > l[max_i]:
                    max_i = j
            l[max_i], l[len(l)-i-1] = l[len(l)-i-1],l[max_i]
            print(l)
        return l
    def gb(self, l):
        def merge(l1,l2):
            if len(l1) == 2:
                if l1[0] > l1[1]:
                    l1[0],l1[1] = l1[1],l1[0]
            if len(l2) == 2:
                if l2[0] > l2[1]:
                    l2[0],l2[1] = l2[1],l2[0]
            if len(l1) > 2:
                l1 = merge(l1[:len(l1)//2], l1[len(l1)//2:])
            if len(l2) > 2:
                l2 = merge(l2[:len(l2)//2], l2[len(l2)//2:])
            i1 = 0
            i2 = 0
            ret = []
            while i1 < len(l1) and i2 < len(l2):
                if l1[i1] <= l2[i2]:
                    ret += [l1[i1]]
                    i1+=1
                else:
                    ret += [l2[i2]]
                    i2+=1
            if i1 == len(l1):
                ret += l2[i2:]
            if i2 == len(l2):
                ret += l1[i1:]
            print('l1,l2:', l1, l2)
            print("ret:", ret)
            return ret
        return merge(l[:len(l)//2], l[len(l)//2:])
    def tong(self,l):
        tmp = [0] * len(l)
        for ll in l:
            tmp[ll] +=1
        ret = []
        for i,tt in enumerate(tmp):
            for _ in range(tt):
                ret += [i]
        print(ret)
        return ret
    def cr(self, l):
        ret = []
        for i,ll in enumerate(l):
            insert_i = 0
            for j, rr in enumerate(ret):
                if rr <= ll:
                    insert_i = j+1
            ret.insert(insert_i, ll)
            print(ret)
        return ret
    def kp(self, ls):
        ret = []
        if len(ls) == 2:
            return [min(ls), max(ls)]
        if len(ls) <=1:
            return ls
        l, m, r = 0, 1, 2
        while m < len(ls):
            if ls[m] <= ls[l]:
                m+=1
            else:
                break
        if m ==  len(ls) -1:
            ls[l], ls[m] = ls[m],ls[l]
            return ls
        else:
            r = m + 1
        while r< len(ls):
            if ls[r]>ls[l]:
                r+=1
            else:
                ls[m], ls[r] = ls[r], ls[m]
                r+=1
                m+=1
        ls[l], ls[m-1] = ls[m-1], ls[l]
        print(ls)
        return self.kp(ls[:m-1]) + [ls[m-1]] +self.kp(ls[m:])
l=[4,3,2,5,6,7,8,1,9,0,10,1,3,8]

# l=[3, 3, 2, 1, 0, 1]

px=PX()

# px.mp(l)

# px.xz(l)

# px.gb(l)

# px.tong(l)

# px.cr(l)

print(l)
l=px.kp(l)
print(l)

29.merge bn, REP merge conv2D merge fuse fusion conv2d avgPool2d

#merge BN
#return conv2d
def MergeConvBN(conv, bn):
    cw = conv.weight
    bw = bn.weight
    bb = bn.bias
    bm = bn.running_mean
    bv = bn.running_var
    bs = (bv+bn.eps).sqrt()
    ret = nn.Conv2d(in_channels=conv.in_channels, out_channels = conv.out_channels,
                    kernel_size=conv.kernel_size, bias=True,
                    stride=conv.stride, padding=conv.padding, groups=conv.groups)
    ret.weight = nn.Parameter(cw * (bw/bs).reshape(conv.out_channels, 1, 1, 1))
    if conv.bias is not None:
        ret.bias = nn.Parameter((conv.bias - bm)*bw/bs + bb)
    else:
        ret.bias = nn.Parameter(bb - bm*bw/bs)
    return ret

#return conv2d
def REPMergeTwoConvs(conv1, conv2):
    stride1 = conv1.stride
    stride2 = conv2.stride
    assert(stride1 == stride2)
    k1 = conv1.kernel_size
    k2 = conv2.kernel_size
    assert(k1[0] == k1[1])
    assert(k2[0] == k2[1])
    k1 = k1[0]
    k2 = k2[0]
    assert(conv1.bias.shape == conv2.bias.shape)
    assert(conv1.groups == conv2.groups)

# print('010101', conv1.groups, conv2.groups)

    w1 = conv1.weight.clone()
    w2 = conv2.weight.clone()
    with torch.no_grad():
        if k1 > k2:
            if conv1.bias is not None or conv2.bias is not None:
                ret = nn.Conv2d(in_channels=conv1.in_channels, out_channels=conv1.out_channels,
                                kernel_size=conv1.kernel_size,stride=stride1,
                                bias=True,
                                padding=(max(conv1.padding[0],conv2.padding[0]),
                                         max(conv1.padding[1],conv2.padding[1])),
                                groups = conv1.groups)
            else:
                ret = nn.Conv2d(in_channels=conv1.in_channels, out_channels=conv1.out_channels,
                                kernel_size=conv1.kernel_size,stride=stride1,
                                bias=False,
                                padding=(max(conv1.padding[0],conv2.padding[0]),
                                         max(conv1.padding[1],conv2.padding[1])),
                               groups = conv1.groups)
            if conv1.bias is not None:
                ret.bias = nn.Parameter(conv1.bias.clone())
                if conv2.bias is not None:
                    ret.bias += nn.Parameter(conv2.bias.clone())
            elif conv2.bias is not None:
                ret.bias = nn.Parameter(conv2.bias.clone())
            ret.weight = nn.Parameter(w1)
            ret.weight[:,:,(k1-k2)//2:k1-(k1-k2)//2,(k1-k2)//2:k1-(k1-k2)//2] += w2
            return ret
        else:
            if conv1.bias is not None or conv2.bias is not None:
                ret = nn.Conv2d(in_channels=conv2.in_channels, out_channels=conv2.out_channels,
                                kernel_size=conv2.kernel_size,stride=stride2,
                                bias=True,
                                padding=(max(conv1.padding[0],conv2.padding[0]),
                                         max(conv1.padding[1],conv2.padding[1])),
                                groups = conv1.groups)
            else:
                ret = nn.Conv2d(in_channels=conv2.in_channels, out_channels=conv2.out_channels,
                            kernel_size=conv2.kernel_size,stride=stride2,
                            bias = False,
                            padding=(max(conv1.padding[0],conv2.padding[0]),
                                     max(conv1.padding[1],conv2.padding[1])),
                               groups = conv1.groups)
            if conv1.bias is not None:
                ret.bias = nn.Parameter(conv1.bias.clone())
                if conv2.bias is not None:
                    ret.bias += nn.Parameter(conv2.bias.clone())
            elif conv2.bias is not None:
                ret.bias = nn.Parameter(conv2.bias.clone())
            ret.weight = nn.Parameter(w2)
            ret.weight[:,:,(k2-k1)//2:k2-(k2-k1)//2,(k2-k1)//2:k2-(k2-k1)//2] += w1
            return ret
right = 0
wrong = 0
for _ in range(10000):
    h, w = int(torch.abs(torch.randn(1))*100+7),int(torch.abs(torch.randn(1))*100+7)
    k11 = int(torch.abs(torch.randn(1)*7+7))%7*2+1
    k21 = int(torch.abs(torch.randn(1)*7+7))%7*2+1
    k1 = np.array((k11,k11))# np.array((7,7))
    k2 = np.array((k21,k21))# np.array((7,7))

    # k2 = np.array((3,3))
    stride1 = 1 if torch.randn(1) < 0 else 2#int(torch.randn(1)*5%3+1)#2
    stride2 = stride1
    in_channels = int(torch.abs(torch.randn(1)))*8+8
    out_channels = int(torch.abs(torch.randn(1)))*8+8
    padding1 = np.array((int(torch.abs(torch.randn(1)*7)%7+1),int(torch.abs(torch.randn(1)*7))%7+1))
    tmp = (2*padding1-k1) // stride1 * stride2
    for i in range(stride1):
        y = (tmp+i+k2) % stride2
        if not y.any():
            padding2 = (tmp+i+k2) // stride2
            break
    if not (padding2 >= 0).all():
        continue
    if h < max(k11,k21) or w < max(k11,k21):
        continue

# assert((padding2 >= 0).all())

    oh1 = (h - k1[0] +2*padding1[0]) // stride1 + 1
    oh2 = (h - k2[0] +2*padding2[0]) // stride2 + 1
    ow1 = (w - k1[1] +2*padding1[1]) // stride1 + 1
    ow2 = (w - k2[1] +2*padding2[1]) // stride2 + 1
    if oh1 != oh2 or ow1 != ow2:

# print("no")

    continue
    conv33 = nn.Conv2d(kernel_size=k1,stride=stride1, padding=padding1,
                       in_channels=in_channels, out_channels=out_channels,
                       bias=True if torch.randn(1)>0 else False,
                      groups = 1).eval()
    BN33 = nn.BatchNorm2d(out_channels, affine=True).eval()
    conv11 = nn.Conv2d(kernel_size=k2,stride=stride2, padding=padding2,
                       in_channels=in_channels, out_channels=out_channels,
                       bias=True if torch.randn(1)>0 else False,
                      groups = 1).eval()
    BN11 = nn.BatchNorm2d(out_channels, affine=True).eval()

    BN11.weight = nn.Parameter(torch.randn(out_channels))
    BN11.bias = nn.Parameter(torch.randn(out_channels))
    BN11.running_mean = nn.Parameter(torch.randn(out_channels),requires_grad=False)
    BN11.running_var = nn.Parameter(torch.abs(torch.randn(out_channels)),requires_grad=False)
    BN33.weight = nn.Parameter(torch.randn(out_channels))
    BN33.bias = nn.Parameter(torch.randn(out_channels))
    BN33.running_mean = nn.Parameter(torch.randn(out_channels),requires_grad=False)
    BN33.running_var = nn.Parameter(torch.abs(torch.randn(out_channels)),requires_grad=False)
    BN11 = BN11.eval()
    BN33 = BN33.eval()

    data = torch.randn((1,in_channels, h, w))

# print(h,w,k1,k2,stride1,in_channels, out_channels, padding1,padding2)

    result_org = 0
    with torch.no_grad():
        data33 = BN33(conv33(data))
        data11 = BN11(conv11(data))
        result_org = data11+data33

    merged33 = MergeConvBN(conv33, BN33).eval()
    merged11 = MergeConvBN(conv11, BN11).eval()
    rep = REPMergeTwoConvs(merged11,merged33).eval()
    result_merge = 0
    with torch.no_grad():
        result_merge = rep(data)
    if not torch.allclose(result_org, result_merge, atol=1e-4):
        wrong+=1
        print(max(torch.abs(result_org- result_merge).flatten()))
        print(h,w,k1,k2,stride1,in_channels, out_channels, padding1, padding2)
    else:
        right+=1
print(right,wrong)

avg = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2,stride=2,bias=False)

data = torch.randn((1, 16, 64, 64))
res_conv = conv(data)
res_avg_conv = avg(res_conv)

merge_convpool = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=4,stride=4,bias=False,padding=0).eval()
with torch.no_grad():
    merge_convpool.weight[:,:,:2,:2] = conv.weight /4.
    merge_convpool.weight[:,:,:2,2:4] = conv.weight   /4.
    merge_convpool.weight[:,:,2:4,:2] = conv.weight    /4.
    merge_convpool.weight[:,:,2:4,2:4] = conv.weight    /4.
res_merge_convpool = merge_convpool(data)
torch.allclose(res_avg_conv, res_merge_convpool, atol=1e-4)
print(max(torch.abs(res_avg_conv - res_merge_convpool).flatten()[:]))

梯度爆炸 梯度消失的原因和应对方法

梯度裁剪 [https://blog.csdn.net/qq_34769162/article/details/111171116](https://blog.csdn.net/qq_34769162/article/details/111171116)

- 正样本特别稀疏该怎么处理？
- 分类问题为什么常用交叉熵loss？
- ImageNet top-1提升技巧有哪些？
- 如何解决对过曝的图像进行目标检测？

深度学习问题合集：[https://zhuanlan.zhihu.com/p/471196226](https://zhuanlan.zhihu.com/p/471196226)

数据探索分析EDA:[https://blog.csdn.net/qq_43519779/article/details/105067691](https://blog.csdn.net/qq_43519779/article/details/105067691)

Transformer：

0.patch embedding

# embed_dim表示切好的图片拉成一维向量后的特征长度

# 图像共切分为N = HW/P^2个patch块

# 在实现上等同于对reshape后的patch序列进行一个PxP且stride为P的卷积操作

# output =^2

# 即output =^2 = (n/P)^2

1.layer normalization

[Transformer中的归一化(五)：Layer Norm的原理和实现 &amp; 为什么Transformer要用LayerNorm - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/492803886)

    ![0](https://note.youdao.com/yws/res/1945/WEBRESOURCEad95fa547defd2885276720accb30f80)

2.FFN

3.GELU：

    ![0](https://note.youdao.com/yws/res/1922/WEBRESOURCE6fdb87209ff8caa2f1cdadcb6f4209d8)

import numpy as np
x=np.linspace(-10,10,100)
y = []
for xx in x:
    y.append(0.5*xx*(1+np.tanh((2/np.pi)**0.5*(xx+0.044715*xx**3))))
plt.plot(x,y)

DCN

y(p0)=pn∈R∑w(pn)⋅x(p0+pn)

Pn

    ![0](https://note.youdao.com/yws/res/1930/WEBRESOURCE818ccb4630d29c1c183014d2c1535630)

y(p0)=pn∈R∑w(pn)⋅x(p0+pn+Δpn)

公式中用Δ p表示偏移量。需要注意的是，该偏移量是针对x的，也就是可变形卷积变的不是卷积核，而是input。

    ![0](https://note.youdao.com/yws/res/1927/WEBRESOURCE979623be6959dd0b45d91c09380f99d0)
