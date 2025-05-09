title: "LSS:lift splat shoot"
author: "lvsolo"
date: "2025-04-10"
tags: ["paper reading", "lidar detection", "BEV"]

# 1.Paper

Lift

![1744337617291](image/LSS/1744337617291.png)

Lift splat architechture

![1744348362367](image/LSS/1744348362367.png)

实验条件超参数

![1744348781445](image/LSS/1744348781445.png)

整体流程

![1744361551642](image/LSS/1744361551642.jpg)

借用一张CADDN的图说明frustum到bev的过程，实际上会是一个对**梯形**的立方体进行裁切成**矩形**的过程

![1746682669036](image/LSS/1746682669036.png)

![1746756430929](image/LSS/1746756430929.png)

# 2.Code

github.com:nv-tlabs/lift-splat-shoot.git

## 2.1 Lift

在lift这一步，生成的点云数量是图像尺寸W\*H与深度分割的网格数量D 的乘积，也就是最终点云的点数量是W\*H\**D

```python

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
```

## 2.2 Splat

cum sum pooling  trick

**不使用max pooling 或者 average pooling，而是使用sum pooling**

文章中提到的“sum Trick”是指在进行求和池化（sum pooling）时的一种高效操作方法，称为“累积和技巧”（cumulative sum trick）。具体来说，这个技巧用于提高训练效率，尤其是在处理由多个相机生成的大型点云时。

在“Lift-Splat”模型中，经过“lift”步骤后，会生成一个包含大量点的点云。为了将这些点转换为固定维度的张量，模型需要对这些点进行池化。在这里，作者选择了求和池化而不是最大池化，因为求和池化能够更好地保留信息。

累积和技巧的具体操作如下：

1. 首先，将所有点按照其所属的柱（pillar）ID进行排序。
2. 接着，对所有特征执行累积和计算，这样可以得到每个柱的特征总和。
3. 最后，通过减去柱边界处的累积和值，来获得每个柱的最终特征表示。

这种方法的优点在于，它避免了对每个柱进行填充（padding），从而减少了内存使用。此外，累积和操作的梯度可以被有效地计算出来，从而加快了自动求导（autograd）的速度，使得训练速度提高了约两倍。

因此，这个“sum Trick”不仅提高了模型的训练效率，也在处理大规模数据时，确保了计算的可行性和效率。

```python
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final
```

```python
def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats

```
