import graphviz

dot = graphviz.Digraph('Sparse4D_v3_code_level', format='png')
dot.attr(rankdir='TB', size='32,50', dpi='120',
         fontname='DejaVu Sans', bgcolor='white',
         label='''Sparse4D v3 代码级数据流图
类名 -> 论文模块 | Tensor维度 (ResNet50, 256x704输入, batch=1)
颜色: 蓝=config/输入 | 绿=InstanceBank | 黄=AnchorEncoder | 紫=Attention | 红=Deformable | 橙=Refinement | 灰=Loss''',
         labelloc='t', fontsize='18', fontcolor='#333333')
dot.attr('node', fontname='DejaVu Sans', fontsize='11', margin='0.15,0.08')
dot.attr('edge', fontname='DejaVu Sans', fontsize='9')

# 颜色
CFG_C = '#E3F2FD'; CFG_B = '#1565C0'
BANK_C = '#E8F5E9'; BANK_B = '#2E7D32'
ENC_C = '#FFF9C4'; ENC_B = '#F9A825'
ATTN_C = '#F3E5F5'; ATTN_B = '#7B1FA2'
DEFORM_C = '#FFEBEE'; DEFORM_B = '#C62828'
REF_C = '#FFF3E0'; REF_B = '#E65100'
LOSS_C = '#F5F5F5'; LOSS_B = '#616161'
DEC_B = '#333333'

def nd(name, label, color, border, shape='box', pw='2'):
    dot.node(name, label, shape=shape, style='filled,rounded',
             fillcolor=color, color=border, penwidth=pw,
             fontname='DejaVu Sans', fontsize='11')

# ============================================================
# 1. 输入层
# ============================================================
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='1. 输入 (Sparse4D.forward)', style='rounded',
           color=CFG_B, fontsize='14', bgcolor=CFG_C)
    nd('img_input', '多视图图像\n6 x 3 x 256 x 704\n(batch=1, 6路相机)', CFG_C, CFG_B, 'folder')
    nd('backbone', 'img_backbone: ResNet50\n4个stage特征\n[256, 512, 1024, 2048]', CFG_C, CFG_B)
    nd('fpn', 'img_neck: FPN\nnum_levels=4, out_channels=256\n4个尺度特征图', CFG_C, CFG_B)
    nd('feat_maps', 'feature_maps (I)\n{I_s in R^(BxNx256xHsxWs)}\n尺度: x4, x8, x16, x32\nN=6(相机), C=256', CFG_C, CFG_B, pw='3')

dot.edge('img_input', 'backbone')
dot.edge('backbone', 'fpn', label='  [256,512,1024,2048]')
dot.edge('fpn', 'feat_maps', label='  4尺度, 256通道')

# ============================================================
# 2. InstanceBank
# ============================================================
with dot.subgraph(name='cluster_bank') as c:
    c.attr(label='2. InstanceBank.get() - 实例初始化与传播', style='rounded',
           color=BANK_B, fontsize='14', bgcolor=BANK_C)
    nd('new_anchor', '新初始化 Anchor\nnum_anchor=900, dim=11\nK-means聚类中心\n{x,y,z,w,l,h,sin,cos,vx,vy}', BANK_C, BANK_B)
    nd('new_feat', '新初始化 Feature\nR^(Bx900x256)\n随机初始化', BANK_C, BANK_B)
    nd('temp_anchor', '上一帧传播 Anchor\ntemp_instances=600\ndim=11\n经过ego motion投影', BANK_C, BANK_B, pw='2.5')
    nd('temp_feat', '上一帧传播 Feature\nR^(Bx600x256)\nF_t = F_{t-1} 不变!', BANK_C, BANK_B, pw='2.5')
    nd('merged', '合并后\nanchor: R^(Bx1500x11)\ninstance_feature: R^(Bx1500x256)\n(900新 + 600时序)', '#C8E6C9', BANK_B, pw='3')

dot.edge('new_anchor', 'merged')
dot.edge('new_feat', 'merged')
dot.edge('temp_anchor', 'merged', label='  投影后')
dot.edge('temp_feat', 'merged', label='  原样传递')

# ============================================================
# 3. Denoising
# ============================================================
with dot.subgraph(name='cluster_dn') as c:
    c.attr(label='3. Denoising (仅训练时)', style='dashed',
           color='#9E9E9E', fontsize='14', bgcolor='#FAFAFA')
    nd('gt', 'GT: 最多32个\nA^gt in R^(Bx32x11)\ncls in Z_32', '#FAFAFA', '#9E9E9E')
    nd('noisy_anchor', '加噪声: A^noise = A^gt + dA\n5组 (num_dn_groups=5)\n正负样本各1\n-> R^(Bx320x11)', '#FAFAFA', '#9E9E9E')
    nd('merged_dn', '最终输入\nanchor: R^(Bx1820x11)\nfeat: R^(Bx1820x256)\n(1500 + 320去噪)\nattn_mask: R^(1820x1820)', '#FAFAFA', '#9E9E9E')

dot.edge('gt', 'noisy_anchor', style='dashed')
dot.edge('noisy_anchor', 'merged_dn', style='dashed')
dot.edge('merged', 'merged_dn', style='dashed')

# ============================================================
# 4. Anchor Encoder
# ============================================================
with dot.subgraph(name='cluster_encoder') as c:
    c.attr(label='4. SparseBox3DEncoder - Anchor Embedding编码', style='rounded',
           color=ENC_B, fontsize='14', bgcolor=ENC_C)
    nd('enc_input', 'anchor in R^(Bx1820x12)\n[v2: 11维, v3: +vz->12维]', ENC_C, ENC_B)
    nd('pos_mlp', 'pos_fc: [x,y,z] (3d)\n[Linear->ReLU->LN]x4\n-> 128d', ENC_C, ENC_B)
    nd('size_mlp', 'size_fc: [w,l,h] (3d)\n[Linear->ReLU->LN]x4\n-> 32d', ENC_C, ENC_B)
    nd('yaw_mlp', 'yaw_fc: [sin,cos] (2d)\n[Linear->ReLU->LN]x4\n-> 32d', ENC_C, ENC_B)
    nd('vel_mlp', 'vel_fc: [vx,vy,vz] (3d)\n[Linear->ReLU->LN]x4\n-> 64d', ENC_C, ENC_B)
    nd('anchor_embed', 'anchor_embed in R^(Bx1820x256)\n= concat(128+32+32+64)\n[v2: mode=add, output_fc有]\n[v3: mode=cat, 无output_fc]', '#FFECB3', ENC_B, pw='3')
    nd('temp_anchor_embed', 'temp_anchor_embed in R^(Bx600x256)\n同样Encoder处理上一帧anchor', '#FFECB3', ENC_B, pw='2')

dot.edge('merged_dn', 'enc_input')
dot.edge('enc_input', 'pos_mlp')
dot.edge('enc_input', 'size_mlp')
dot.edge('enc_input', 'yaw_mlp')
dot.edge('enc_input', 'vel_mlp')
for n in ['pos_mlp', 'size_mlp', 'yaw_mlp', 'vel_mlp']:
    dot.edge(n, 'anchor_embed')
dot.edge('temp_anchor', 'temp_anchor_embed')

# ============================================================
# 5. Decoder 循环
# ============================================================
with dot.subgraph(name='cluster_decoder') as c:
    c.attr(label='5. Decoder - x6层迭代 (Sparse4DHead.forward)', style='rounded',
           color=DEC_B, fontsize='14', bgcolor='#FAFAFA')

    c.node('loop_note', 'Layer 1: Single-Frame (无temp_gnn)\nLayer 2-6: Multi-Frame (有temp_gnn)\n每层独立参数, 循环6次',
           shape='plaintext', fillcolor='white', color='white', fontsize='12')

    # 5a. Decoupled Attention
    nd('fc_before', 'fc_before: Linear(256->512)\n(v3 Decoupled; v2为Identity)\nvalue = fc_before(instance_feature)\n-> R^(Bx1820x512)', ATTN_C, ATTN_B)
    nd('decoupled_concat', 'Decoupled Attention [v3]:\nquery = concat(F, E) = R^(Bx1820x512)\nkey = concat(F, E) = R^(Bx1820x512)\nvalue = fc_before(F) = R^(Bx1820x512)', ATTN_C, ATTN_B, pw='2.5')
    nd('gnn', 'MultiheadAttention\nembed_dims=512, num_heads=8\n(v2: 256, v3: 512)\nbatch_first, dropout=0.1', ATTN_C, ATTN_B)
    nd('fc_after', 'fc_after: Linear(512->256)\n(v3; v2为Identity)\n-> instance_feature R^(Bx1820x256)', ATTN_C, ATTN_B)

    # 5b. Temporal Cross-Attention
    nd('temp_gnn', 'temp_gnn: MultiheadAttention\nQ=当前帧instance\nK=V=上一帧instance\nquery_pos=anchor_embed\nkey_pos=temp_anchor_embed', '#E1BEE7', ATTN_B, pw='2.5')

    # 5c. Norm
    nd('norm1', 'LayerNorm(256)', '#FAFAFA', DEC_B)
    nd('norm2', 'LayerNorm(256)', '#FAFAFA', DEC_B)
    nd('norm3', 'LayerNorm(256)', '#FAFAFA', DEC_B)

    # 5d. Deformable Aggregation
    nd('kps_gen', 'SparseBox3DKeyPointsGenerator\nfix_scale=7(中心+6面心)\nlearnable=6(网络预测)\n-> 13个3D Keypoints\nR^(Bx1820x13x3)', DEFORM_C, DEFORM_B)
    nd('dFA', 'DeformableFeatureAggregation\nembed_dims=256, num_groups=8\nnum_levels=4, num_cams=6\nattn_drop=0.15\nuse_deformable_func=True (CUDA)', DEFORM_C, DEFORM_B, pw='3')
    nd('dFA_detail', '内部操作:\n1) 投影3D->2D (相机内外参)\n2) 双线性插值采样: R^(Bx1820x13x6x4x256)\n3) 预测权重加权求和\n4) CUDA kernel: 采样+融合一步完成\n-> R^(Bx1820x256)', '#FFCDD2', DEFORM_B)

    # 5e. FFN
    nd('ffn', 'AsymmetricFFN\nin=256x2=512, hidden=256x4=1024\npre_norm=LN, act=ReLU\nnum_fcs=2, dropout=0.1\n-> R^(Bx1820x256)', '#FAFAFA', DEC_B)

    # 5f. Refinement
    nd('refine', 'SparseBox3DRefinementModule\nfeature = instance_feature + anchor_embed\n-> R^(Bx1820x256)', REF_C, REF_B, pw='3')
    nd('reg_head', '回归头: [Linear->ReLU->LN]x2 + Linear(256->11)\n+ Scale\n输出: d_anchor (残差)\nanchor_new = anchor + d\n-> R^(Bx1820x11)', REF_C, REF_B)
    nd('cls_head', '分类头: [Linear->ReLU->LN]x1 + Linear(256->10)\n输入: 只用 instance_feature!\n-> R^(Bx1820x10)', REF_C, REF_B)
    nd('quality_head', '质量头 (v3): [Linear->ReLU->LN]x1 + Linear(256->2)\n输入: feature (F+E add后)\n-> centerness(1d) + yawness(1d)\n-> R^(Bx1820x2)', REF_C, REF_B)

# Decoder 内部连接
dot.edge('anchor_embed', 'decoupled_concat', label='  E (query_pos)')
dot.edge('fc_before', 'decoupled_concat', label='  value')
dot.edge('decoupled_concat', 'gnn', label='  Q,K,V in R^(Bx1820x512)')
dot.edge('gnn', 'fc_after')
dot.edge('temp_anchor_embed', 'temp_gnn', label='  key_pos')
dot.edge('temp_feat', 'temp_gnn', label='  K, V (上一帧)')
dot.edge('fc_after', 'norm1')
dot.edge('temp_gnn', 'norm1', label='  +残差')
dot.edge('norm1', 'kps_gen', label='  instance_feature')
dot.edge('kps_gen', 'dFA', label='  13个keypoints')
dot.edge('feat_maps', 'dFA', label='  多视图特征')
dot.edge('anchor_embed', 'dFA', label='  用于生成权重')
dot.edge('dFA', 'dFA_detail')
dot.edge('dFA', 'norm2', label='  +残差')
dot.edge('norm2', 'ffn')
dot.edge('ffn', 'norm3', label='  +残差')
dot.edge('norm3', 'refine')
dot.edge('refine', 'reg_head')
dot.edge('refine', 'cls_head')
dot.edge('refine', 'quality_head')

# ============================================================
# 6. Loss
# ============================================================
with dot.subgraph(name='cluster_loss') as c:
    c.attr(label='6. Loss计算 (SparseBox3DLoss)', style='rounded',
           color=LOSS_B, fontsize='14', bgcolor=LOSS_C)
    nd('hungarian', '匈牙利匹配\nSparseBox3DTarget.sample()\n-> 正样本mask', LOSS_C, LOSS_B)
    nd('loss_cls', 'loss_cls: FocalLoss\nγ=2.0, α=0.25\n全部实例 (正+负)', LOSS_C, LOSS_B)
    nd('loss_box', 'loss_box: SmoothL1Loss\n只对正样本\nweight=[2,2,2,0.5,0.5,0.5,0,0,1,1]', LOSS_C, LOSS_B)
    nd('loss_cns', 'loss_cns: BCE (use_sigmoid)\nGT: exp(-||pred-gt||_2)\n只对正样本', LOSS_C, LOSS_B)
    nd('loss_yns', 'loss_yns: GaussianFocalLoss\nGT: (cos_sim > 0).float()\n只对正样本', LOSS_C, LOSS_B)
    nd('loss_total', 'Total Loss (每层decoder):\nloss_cls + loss_box + loss_cns + loss_yns\n+ denoising losses (x5.0)', '#E0E0E0', LOSS_B, pw='3')

dot.edge('cls_head', 'hungarian')
dot.edge('reg_head', 'hungarian')
dot.edge('hungarian', 'loss_cls')
dot.edge('hungarian', 'loss_box')
dot.edge('quality_head', 'loss_cns')
dot.edge('quality_head', 'loss_yns')
for n in ['loss_cls', 'loss_box', 'loss_cns', 'loss_yns']:
    dot.edge(n, 'loss_total')

# ============================================================
# 7. 输出
# ============================================================
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='7. 输出与传播 (SparseBox3DDecoder)', style='rounded',
           color=BANK_B, fontsize='14', bgcolor=BANK_C)
    nd('score', 'score = cls_sigmoid x centerness_sigmoid\n-> R^(Bx900x10)\n按置信度排序, 取top-K', BANK_C, BANK_B, pw='3')
    nd('track_id', '跟踪: 维护instance_id\n来自InstanceBank\n新目标分配新ID', BANK_C, BANK_B)
    nd('output', '最终输出:\n每帧 ~300-500个检测框\n(类别, 3D框, 置信度, ID)', BANK_C, BANK_B, 'doubleoctagon', pw='3')
    nd('propagate', '传播到下一帧:\nInstanceBank.cache()\n选择top-600高置信度实例\n-> 下一帧的temp_instances', BANK_C, BANK_B, pw='2.5')

dot.edge('reg_head', 'score')
dot.edge('quality_head', 'score', label='  centerness')
dot.edge('score', 'track_id')
dot.edge('score', 'output')
dot.edge('score', 'propagate')

# 渲染
out_path = 'sparse4dv3_code_flow'
dot.render(out_path, cleanup=True)
print(f"OK: {out_path}.png")
