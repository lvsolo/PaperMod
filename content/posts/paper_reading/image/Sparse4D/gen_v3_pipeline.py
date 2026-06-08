import graphviz

dot = graphviz.Digraph('sparse4dv3_full_pipeline_sections', format='png')
dot.attr(rankdir='TB', size='30,52', dpi='150',
         fontname='DejaVu Sans', bgcolor='white',
         label='Sparse4D v3 完整 Pipeline — 各章节(§3.3~§3.7)对应位置标注',
         labelloc='t', fontsize='22', fontcolor='#333333')
dot.attr('node', fontname='DejaVu Sans', fontsize='11', margin='0.12,0.06')
dot.attr('edge', fontname='DejaVu Sans', fontsize='9')

def nd(name, label, color, border, shape='box', pw='2', style='filled,rounded'):
    dot.node(name, label, shape=shape, style=style,
             fillcolor=color, color=border, penwidth=pw,
             fontname='DejaVu Sans', fontsize='11')

def section_title(name, label, color, border):
    dot.node(name, label, shape='box', style='filled,bold,rounded',
             fillcolor=color, color=border, penwidth='4',
             fontname='DejaVu Sans Bold', fontsize='16',
             margin='0.3,0.15')

# 颜色
S33_C = '#E8F5E9'; S33_B = '#2E7D32'   # §3.3 Anchor Embedding
S34_C = '#E3F2FD'; S34_B = '#1565C0'   # §3.4 Decoupled Attention
S35_C = '#FFF3E0'; S35_B = '#E65100'   # §3.5 TID
S36_C = '#FCE4EC'; S36_B = '#C2185B'   # §3.6 Quality Estimation
S37_C = '#F3E5F5'; S37_B = '#7B1FA2'   # §3.7 Tracking
V2_C  = '#ECEFF1'; V2_B  = '#546E7A'   # v2 沿用模块
BASE_C = '#F5F5F5'; BASE_B = '#9E9E9E'
IO_C = '#FFF9C4'; IO_B = '#F9A825'

# ========== 章节标题节点 ==========
section_title('title_s33', '§3.3 Anchor Embedding Encoder\n(Anchor 编码器)', S33_C, S33_B)
section_title('title_s34', '§3.4 Decoupled Attention\n(解耦注意力)', S34_C, S34_B)
section_title('title_s35', '§3.5 Temporal Instance Denoising\n(时序实例去噪 TID)', S35_C, S35_B)
section_title('title_s36', '§3.6 Quality Estimation\n(质量估计)', S36_C, S36_B)
section_title('title_s37', '§3.7 End-to-End Tracking\n(端到端跟踪)', S37_C, S37_B)

# ========== 输入 ==========
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='输入', style='rounded', color=IO_B, fontsize='14', bgcolor=IO_C)
    nd('in_img', '多视图图像\n(6相机)', IO_C, IO_B)
    nd('in_prev', '上一帧传播实例\ncached (600) + ID', IO_C, IO_B)
    nd('in_anchor', '可学习Anchor\n(300) + feature', IO_C, IO_B)

# ========== Backbone ==========
nd('bb', 'Backbone + FPN\n(沿用v2)', V2_C, V2_B)

# ========== §3.7 InstanceBank.get ==========
nd('ib_get', 'InstanceBank.get()\n可学习(300) + temp(600)\nanchor: ego motion投影\nfeature: F_t=F_{t-1}\nID: 沿用传播的ID', S37_C, S37_B, pw='2.5')

# ========== Phase 1: Single-Frame ==========
nd('p1_title', 'Phase 1: Single-Frame\nDecoder (第1层)', BASE_C, BASE_B, pw='2')
nd('p1_eda', 'Deformable Aggregation\n3D→2D投影 + camera_embed\n+ 双线性采样 + 加权融合\n(沿用v2, §2.4+§2.5)', V2_C, V2_B)
nd('p1_refine', 'Refinement', BASE_C, BASE_B)

# ========== InstanceBank.update ==========
nd('ib_update', 'InstanceBank.update()\ntopk(300) + temporal(600)\n= 900实例\n(沿用v2, §2.7)', S37_C, S37_B, pw='2.5')

# ========== Phase 2: Temporal Decoder ==========
with dot.subgraph(name='cluster_p2') as c:
    c.attr(label='Phase 2: Temporal Decoder (第2-6层)', style='rounded',
           color=BASE_B, fontsize='14', bgcolor=BASE_C)

    # §3.4 Decoupled Attention
    with c.subgraph(name='cluster_s34') as s34:
        s34.attr(label='', style='rounded', color=S34_B, bgcolor=S34_C, penwidth='3')

        # §3.4 Temporal Attention
        nd('temp_gnn', '§3.4 Temporal Attention\n(时序交叉注意力)\nQ=当前900, K=V=历史600\nQ=K=FC([F,E]), V=F\n(concat后投影, 非直接add)', S34_C, S34_B, pw='3')

        # §3.4 Self-Attention
        nd('gnn', '§3.4 Self-Attention\n(帧内自注意力)\nQ=K=FC([F,E]), V=F\n(concat后投影)', S34_C, S34_B, pw='2.5')

    # §3.3 Anchor Embedding
    nd('anchor_enc', '§3.3 SparseBox3DEncoder\n12维 anchor → 4组MLP → concat\npos_fc(128d)+size_fc(32d)\n+yaw_fc(32d)+vel_fc(64d)\n= 256d (mode="cat")', S33_C, S33_B, pw='3')

    # EDA (沿用v2)
    nd('eda', 'Deformable Aggregation\n(沿用v2 §2.4+§2.5)\n3D→2D + camera_embed\n+ CUDA kernel融合', V2_C, V2_B)

    # §3.6 Quality Estimation
    with c.subgraph(name='cluster_s36') as s36:
        s36.attr(label='', style='rounded', color=S36_B, bgcolor=S36_C, penwidth='3')
        nd('quality', '§3.6 Quality Estimation\nCenterness = exp(-‖pred-gt_center‖₂)\nYawness = cos(Δyaw) > 0 (二值)\n→ 辅助监督, 推理时:\nfinal_score = cls × sigmoid(centerness)', S36_C, S36_B, pw='3')

    nd('refine', 'Refinement\nanchor残差 + 分类预测', BASE_C, BASE_B)

# ========== §3.7 InstanceBank.cache ==========
nd('ib_cache', 'InstanceBank.cache()\ndetach → 衰减 → topk(600)\n→ 保存 cached + ID\n(沿用v2, §2.7)', S37_C, S37_B, pw='2.5')

# ========== §3.7 ID Assignment ==========
nd('id_assign', '§3.7 get_instance_id()\n① 传播实例: 继承上一帧ID\n② 新实例: prev_max_id++\n③ 阈值过滤: conf≥0.25\n④ ID=-1 被过滤\n→ 检测结果=跟踪结果', S37_C, S37_B, pw='3')

# ========== §3.5 TID ==========
with dot.subgraph(name='cluster_s35') as c:
    c.attr(label='§3.5 Temporal Instance Denoising (仅训练时)', style='rounded,bold',
           color=S35_B, fontsize='15', bgcolor=S35_C, penwidth='3.5')
    nd('tid_noise', '对GT加噪声\n正样本: U(-x,x)\n负样本: U(-2x,-x)∪(x,2x)\n包含时序去噪:\n从上一帧GT出发加噪→传播', S35_C, S35_B, pw='2.5')
    nd('tid_mask', 'Attention Mask\n去噪实例与普通实例之间\n有特定的注意力掩码\n防止信息泄露', S35_C, S35_B)
    nd('tid_loss', '去噪Loss\nλ_dn=5.0\n独立分类+回归loss\n提供密集监督信号', S35_C, S35_B)

# ========== §3.6 Loss ==========
with dot.subgraph(name='cluster_loss') as c:
    c.attr(label='§3.6 Quality Estimation Loss (训练时)', style='rounded',
           color=S36_B, fontsize='14', bgcolor=S36_C, penwidth='2.5')
    nd('loss_cns', 'Centerness Loss\nBCE with Sigmoid\n只对正样本', S36_C, S36_B)
    nd('loss_yns', 'Yawness Loss\nGaussian Focal\n只对正样本', S36_C, S36_B)

# ========== 输出 ==========
nd('output', '输出: 检测+跟踪结果\n(boxes, scores, labels, IDs)', IO_C, IO_B, shape='doubleoctagon', pw='3')

# ========== 边: 主流程 ==========
dot.edge('in_img', 'bb')
dot.edge('in_prev', 'ib_get')
dot.edge('in_anchor', 'ib_get')
dot.edge('bb', 'p1_eda')
dot.edge('ib_get', 'p1_eda', label='  300可学习anchor')
dot.edge('p1_eda', 'p1_refine')
dot.edge('p1_refine', 'ib_update', label='  粗略检测')
dot.edge('ib_get', 'temp_gnn', label='  temp(600)+ID', style='dashed')
dot.edge('ib_update', 'temp_gnn', label='  900(600temp+300cur)')
dot.edge('temp_gnn', 'gnn')
dot.edge('gnn', 'anchor_enc')
dot.edge('anchor_enc', 'eda')
dot.edge('eda', 'quality')
dot.edge('quality', 'refine')
dot.edge('refine', 'anchor_enc', label='  下一层', style='dashed')
dot.edge('refine', 'ib_cache', label='  精修后实例')
dot.edge('ib_cache', 'id_assign')
dot.edge('id_assign', 'output')
dot.edge('ib_cache', 'in_prev', label='  传播下一帧+ID', style='dashed', color=S37_B, penwidth='2')

# ========== 边: TID (训练时) ==========
dot.edge('tid_noise', 'temp_gnn', label='  去噪实例加入Decoder', style='dashed', color=S35_B)
dot.edge('tid_mask', 'temp_gnn', style='dotted', color=S35_B)
dot.edge('refine', 'tid_loss', label='  去噪实例的预测', style='dashed', color=S35_B)

# ========== 边: Quality Loss ==========
dot.edge('quality', 'loss_cns', style='dashed', color=S36_B)
dot.edge('quality', 'loss_yns', style='dashed', color=S36_B)

# ========== 边: 章节标题 → 对应模块 ==========
dot.edge('title_s33', 'anchor_enc', style='dotted', color=S33_B, penwidth='2', arrowhead='none')
dot.edge('title_s34', 'temp_gnn', style='dotted', color=S34_B, penwidth='2', arrowhead='none')
dot.edge('title_s35', 'tid_noise', style='dotted', color=S35_B, penwidth='2', arrowhead='none')
dot.edge('title_s36', 'quality', style='dotted', color=S36_B, penwidth='2', arrowhead='none')
dot.edge('title_s37', 'id_assign', style='dotted', color=S37_B, penwidth='2', arrowhead='none')

out_path = 'sparse4dv3_full_pipeline_sections'
dot.render(out_path, cleanup=True)
print(f"OK: {out_path}.png")
