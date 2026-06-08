import graphviz

# ============================================================
# Sparse4D v3 Tracking vs 传统 Query-based Tracking 对比
# ============================================================
dot = graphviz.Digraph('tracking_comparison', format='png')
dot.attr(rankdir='TB', size='28,36', dpi='150',
         fontname='DejaVu Sans', bgcolor='white',
         label='''Sparse4D v3 端到端 Tracking vs 传统 Query-based Tracking 对比
左: Sparse4D v3 (无跟踪模块, 检测即跟踪)
右: 传统方法 (MOTR/MUTR3D 等, 需要跟踪头+匹配)''',
         labelloc='t', fontsize='20', fontcolor='#333333')
dot.attr('node', fontname='DejaVu Sans', fontsize='11', margin='0.15,0.08')
dot.attr('edge', fontname='DejaVu Sans', fontsize='9')

S4D_C = '#E8F5E9'; S4D_B = '#2E7D32'    # 绿色: Sparse4D
TRAD_C = '#E3F2FD'; TRAD_B = '#1565C0'   # 蓝色: 传统方法
SAME_C = '#FFF9C4'; SAME_B = '#F9A825'    # 黄色: 相同点
DIFF_C = '#FFEBEE'; DIFF_B = '#C62828'    # 红色: 不同点

def nd(name, label, color, border, shape='box', pw='2'):
    dot.node(name, label, shape=shape, style='filled,rounded',
             fillcolor=color, color=border, penwidth=pw,
             fontname='DejaVu Sans', fontsize='11')

# ============================================================
# 左列: Sparse4D v3 Tracking
# ============================================================
with dot.subgraph(name='cluster_s4d') as c:
    c.attr(label='Sparse4D v3 端到端跟踪\n(检测=跟踪, 无额外模块)', style='rounded',
           color=S4D_B, fontsize='16', bgcolor=S4D_C)

    # Frame t-1
    nd('s4d_prev', '第 t-1 帧\n检测输出: 900个实例\n每实例: (anchor, feature, cls)', S4D_C, S4D_B)
    nd('s4d_cache', 'InstanceBank.cache()\n━━━━━━━━━━━━━━━━━━━━\n1. detach (不传梯度)\n2. 置信度衰减: conf = max(conf*0.6, new_conf)\n3. topk(600): 选600个最高置信度\n4. 保存: cached_feature, cached_anchor\n   cached_confidence', S4D_C, S4D_B, pw='2.5')
    nd('s4d_id_save', 'ID 保存到 cached\ninstance_id[:, :600]\n= 当前帧分配的 ID\n(-1 表示无效)', S4D_C, S4D_B)

    # Frame t
    nd('s4d_cur', '第 t 帧\n新初始化: 300个实例\n+ 上一帧传播: 600个实例\n= 900个输入', S4D_C, S4D_B)
    nd('s4d_project', 'InstanceBank.get()\n━━━━━━━━━━━━━━━━━━━━\nanchor: ego motion 投影\nfeature: F_t=F_{t-1} 不变\nembedding: E_t=Ψ(A_t) 重编码\nID: instance_id 沿用 (传播过来的)', S4D_C, S4D_B, pw='2.5')
    nd('s4d_forward', 'Sparse4DHead.forward()\nDecoder x6层精修\n→ 检测结果\n(跟纯检测完全一样!)', S4D_C, S4D_B)

    # ID assignment
    nd('s4d_id_assign', 'get_instance_id()\n━━━━━━━━━━━━━━━━━━━━\n1. 初始化所有 ID 为 -1\n2. 传播实例: 继承上一帧 ID\n   (instance_id[:600] = cached)\n3. 新实例: 分配新 ID\n   new_id = arange(num_new) + prev_id\n   prev_id += num_new\n4. 阈值过滤: conf >= 0.25\n5. 更新传播的 ID: topk(600)', S4D_C, S4D_B, pw='3')
    nd('s4d_output', '输出: 每个检测框带 ID\n(boxes_3d, scores, labels, instance_ids)\nID=-1 的被过滤\n→ 既是检测结果也是跟踪结果', S4D_C, S4D_B, shape='doubleoctagon', pw='3')

c.edge('s4d_prev', 's4d_cache', label='  精修后')
c.edge('s4d_cache', 's4d_id_save')
c.edge('s4d_cache', 's4d_project', label='  传播到下一帧')
c.edge('s4d_id_save', 's4d_project', label='  ID 传递')
c.edge('s4d_project', 's4d_cur')
c.edge('s4d_cur', 's4d_forward')
c.edge('s4d_forward', 's4d_id_assign')
c.edge('s4d_id_assign', 's4d_output')

# ============================================================
# 右列: 传统 Query-based Tracking (MOTR/MUTR3D)
# ============================================================
with dot.subgraph(name='cluster_trad') as c:
    c.attr(label='传统 Query-based Tracking\n(MOTR / MUTR3D / TrackFormer)', style='rounded',
           color=TRAD_B, fontsize='16', bgcolor=TRAD_C)

    # Frame t-1
    nd('trad_prev', '第 t-1 帧\n检测输出: N个实例\n每实例: (box, feature, cls, track_query)', TRAD_C, TRAD_B)
    nd('trad_propagate', 'Track Query 传播\n━━━━━━━━━━━━━━━━━━━━\n1. 选高置信度检测作为 track_query\n2. 新对象: 初始化 new_query\n3. 传播 query + box 到 t 帧\n4. 需要单独的 Embedding 层', TRAD_C, TRAD_B, pw='2.5')

    # Frame t
    nd('trad_cur', '第 t 帧\ntrack_query (来自t-1)\n+ new_query (新初始化)\n→ 传入 Decoder', TRAD_C, TRAD_B)
    nd('trad_forward', 'Transformer Decoder\n━━━━━━━━━━━━━━━━━━━━\nSelf-Attn: query间交互\nCross-Attn: query与图像\n→ 更新后的 queries', TRAD_C, TRAD_B)
    nd('trad_det_head', '检测头\n→ (box, cls, conf)', TRAD_C, TRAD_B)

    # 跟踪匹配
    nd('trad_match', '跟踪匹配 (关键区别!)\n━━━━━━━━━━━━━━━━━━━━\n需要额外的匹配步骤:\n1. 每个 track_query 与检测匹配\n2. 匈牙利匹配或IoU匹配\n3. 匹配成功 → 继承ID\n4. 未匹配 → 标记丢失/新对象\n5. 需要跟踪损失 (ID分类loss)', TRAD_C, TRAD_B, pw='3')

    nd('trad_track_head', '跟踪头 (额外模块)\n━━━━━━━━━━━━━━━━━━━━\nMOTR: 输出 query → 下一帧输入\n  + 跟踪分类loss\nMUTR3D: 3D track query\n  + 运动模型预测\nTrackFormer: track embedding\n  + 关联得分', TRAD_C, TRAD_B, pw='2.5')

    nd('trad_output', '输出: 检测 + 跟踪\n需要后处理:\n1. NMS 去重\n2. 跟踪匹配\n3. 轨迹管理 (生/死/丢失)', TRAD_C, TRAD_B, shape='doubleoctagon', pw='3')

c.edge('trad_prev', 'trad_propagate')
c.edge('trad_propagate', 'trad_cur')
c.edge('trad_cur', 'trad_forward')
c.edge('trad_forward', 'trad_det_head')
c.edge('trad_det_head', 'trad_match')
c.edge('trad_forward', 'trad_track_head', style='dashed', label='  额外模块')
c.edge('trad_track_head', 'trad_match')
c.edge('trad_match', 'trad_output')

# ============================================================
# 底部: 对比总结
# ============================================================
with dot.subgraph(name='cluster_compare') as c:
    c.attr(label='核心对比', style='rounded',
           color=SAME_B, fontsize='16', bgcolor='#FAFAFA')

    nd('same', '相同点\n━━━━━━━━━━━━━━━━━━━━\n1. Query-based: 都用 query/instance 表示目标\n2. 时序传播: 都传播 query/instance 到下一帧\n3. 检测+跟踪端到端: 都不依赖单独的跟踪器(如SORT)\n4. No NMS: 都不需要NMS后处理(在检测层面)',
        SAME_C, SAME_B, pw='3')

    nd('diff', '不同点\n━━━━━━━━━━━━━━━━━━━━\nSparse4D v3:\n  - 无跟踪头/跟踪loss/跟踪匹配\n  - ID 由 InstanceBank 简单分配\n  - 纯靠检测置信度选择传播实例\n  - 训练时无跟踪约束,不需要微调\n  - 跟踪 = 检测 + ID分配 (推理时)\n\n传统方法:\n  - 需要专门的跟踪头/跟踪loss\n  - 需要 query-object 匹配机制\n  - 跟踪分类loss (ID分类) 训练\n  - 可能需要跟踪数据微调\n  - 更复杂但有更多跟踪监督',
        DIFF_C, DIFF_B, pw='3')

    nd('key_insight', 'Sparse4D v3 的核心洞察\n━━━━━━━━━━━━━━━━━━━━\n"训练好的时序检测模型天然具备跟踪能力"\n→ 因为递归传播中每个 instance 天然携带身份信息\n→ 只需在推理时分配/维护 ID 即可\n→ 不需要任何额外的跟踪模块或训练\n→ 代价: 跟踪质量完全依赖检测质量\n   没有显式跟踪监督 → IDS 可能偏高',
        '#E8EAF6', '#3F51B5', pw='3')

c.edge('s4d_output', 'same', style='dashed')
c.edge('trad_output', 'same', style='dashed')
c.edge('same', 'diff')
c.edge('diff', 'key_insight')

out_path = 'tracking_comparison'
dot.render(out_path, cleanup=True)
print(f"OK: {out_path}.png")
