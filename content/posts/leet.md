---
title: "LeetCode"
author: "lvsolo"
date: "2024-11-26"
tags: ['leetcode']
---
# 202411

链表

92

二叉树
543 124 101 236

二叉搜索树：

重要性质：中序遍历为排序好的序列
回溯:DFS,穷举算法，与DP动态规划区别，适用于没有最优子结构的问题
39 78 90 101 112




动态规划：保存子结构的最优解，适用于有最优子结构的问题
198 322 97

背包问题:

https://seramasumi.github.io/docs/Algorithms/mc-%E5%BE%AE%E8%AF%BE%E5%A0%82-%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%98.html

0-1背包

https://www.lintcode.com/problem/92/

```python
    def back_pack(self, m: int, a: List[int]) -> int:
        # write your code here
        mat = []
        for _ in range(len(a)+1):
            tmp = [0]
            for _ in range(m):
                tmp.append(0)
            mat.append(tmp)
        for i in range(m):
            mat[0][i] = 0
        for i in range(1,len(a)+1):
            aa = a[i-1]
            for j in range(1,m+1):
                mat[i][j] = mat[i-1][j]
                if j-aa >=0:
                    mat[i][j] = max(mat[i][j], mat[i-1][j-aa]+aa)
        return mat[len(a)][m]
```

多重背包



完全背包

215 quick select

图
133.DFS和BFS

```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # BFS
        if not node:
            return node
        map = {}
        stack = []
        stack.append(node)
        while len(stack):
            cur = stack.pop(0)
            map[cur] =  Node(cur.val)
            for nb in cur.neighbors:
                if nb not in map:
                    stack.insert(0,nb)
        map1 = {}
        stack = []
        stack.append(node)
        while len(stack):
            cur = stack.pop(0)
            if cur in map1:continue
            map1[cur] =  1
            for nb in cur.neighbors:
                map[cur].neighbors.append(map[nb])
                if nb not in map1:
                    stack.insert(0,nb)
        return map[node]
      
        # # DFS
        # if not node:
        #     return node
        # map = {}
        # stack = []
        # stack.append(node)
        # while len(stack):
        #     cur = stack.pop(0)
        #     map[cur] =  Node(cur.val)
        #     for nb in cur.neighbors:
        #         if nb not in map:
        #             stack.append(nb)
        # map1 = {}
        # stack = []
        # stack.append(node)
        # while len(stack):
        #     cur = stack.pop(0)
        #     if cur in map1:continue
        #     map1[cur] =  1
        #     for nb in cur.neighbors:
        #         map[cur].neighbors.append(map[nb])
        #         if nb not in map1:
        #             stack.append(nb)
        # return map[node]
```
