---
title: "Video or Image Utils"
date: "2025-01-16"
author: "lvsolo"
tags: ["utils"]
---

1.mp4 video stritching: stritch 2 same-size videos.
```python
import os
from tqdm import tqdm
import sys
import glob
'''
python pinjie.py results/split_val/closed_loop_nonreactive_agents/withrule_true/epoch17_ckpt/val14_benchmark_random109_2/videos/ results/split_val/closed_loop_nonreactive_agents/withrule_true/epoch5_ckpt/val14_benchmark_random109/videos/closed_loop_nonreactive_agents.withrule/  test ep5_ep17
'''

dir1 = sys.argv[1]
dir2 = sys.argv[2]
dest_dir = sys.argv[3]
add_text = sys.argv[4]

cmd1 = 'ffmpeg -i {} -i {} -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" {}'
cmd2 = 'ffmpeg -i {} -vf "drawtext=text={}:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:fontsize=30:fontcolor=red:x=50:y=50" {}'
for name1 in os.listdir(dir1):
    if not name1.endswith('.mp4'):
        continue
    #print(name1)
    #print(os.path.join(dir2, "**", name1))
    print(os.path.join(dir2,  name1))
    if  os.path.exists(os.path.join(dir2, name1)):
        print(os.path.join(dir2,  name1))
        path1 = os.path.join(dir1, name1)
        path2 = os.path.join(dir2, name1)
        cmd1.format(path1, path2, dest_dir +'/'+ name1)
        cmd2.format(dest_dir +'/'+ name1, add_text, dest_dir +'/'+ name1)
        os.system(cmd1.format(path1, path2, dest_dir +'/'+ name1))
        #os.system(cmd2.format(dest_dir +'/'+ name1, add_text, dest_dir +'/'+ name1)) 
```
