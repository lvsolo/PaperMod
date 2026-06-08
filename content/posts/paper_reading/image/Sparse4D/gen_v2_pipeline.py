import graphviz

dot = graphviz.Digraph('sparse4dv2_full_pipeline_sections', format='png')
dot.attr(rankdir='TB', size='28,50', dpi='150',
         fontname='DejaVu Sans', bgcolor='white',
         label='Sparse4D v2 完整 Pipeline',
         labelloc='t', fontsize='24', fontcolor='#333333')
dot.attr('node', fontname='DejaVu Sans', fontsize='11', margin='0.12,0.06')
dot.attr('edge', fontname='DejaVu Sans', fontsize='9')

def nd(name, label, color, border, shape='box', pw='2', style='filled,rounded'):
    dot.node(name, label, shape=shape, style=style,
             fillcolor=color, color=border, penwidth=pw,
             fontname='DejaVu Sans', fontsize='11')

# 章节标题节点 — 大字、显眼
def section_title(name, label, color, border):
    dot.node(name, label, shape='box', style='filled,bold,rounded',
             fillcolor=color, color=border, penwidth='4',
             fontname='DejaVu Sans Bold', fontsize='16',
             margin='0.3,0.15')

S23_C = '#C8E6C9'; S23_B = '#2E7D32'
S24_C = '#BBDEFB'; S24_B = '#1565C0'
S25_C = '#FFCDD2'; S25_B = '#C62828'
S26_C = '#FFE0B2'; S26_B = '#E65100'
S27_C = '#E1BEE7'; S27_B = '#7B1FA2'
BASE_C = '#EEEEEE'; BASE_B = '#757575'
IO_C = '#FFF9C4'; IO_B = '#F9A825'

# ========== 章节标题节点 (在图中直接显示) ==========
section_title('title_s23', '§2.3 Recurrent Temporal Fusion\n(递归时序融合)', S23_C, S23_B)
section_title('title_s24', '§2.4 Efficient Deformable Aggregation\n(高效可变形聚合)', S24_C, S24_B)
section_title('title_s25', '§2.5 Camera Parameter Encoding\n(相机参数编码)', S25_C, S25_B)
section_title('title_s26', '§2.6 Depth Supervision\n(稠密深度辅助监督)', S26_C, S26_B)
section_title('title_s27', '§2.7 Instance Propagation\n(实例传播机制)', S27_C, S27_B)

# ========== 输入 ==========
nd('in_img', '多视图图像\n(6相机)', IO_C, IO_B)
nd('in_prev', '上一帧传播实例\ncached (600)', IO_C, IO_B)
nd('in_anchor', '可学习Anchor\n(900) + feature', IO_C, IO_B)

# ========== Backbone ==========
nd('bb', 'Backbone + FPN\n→ 多尺度特征图', BASE_C, BASE_B)

# ========== §2.7 InstanceBank.get ==========
nd('ib_get', 'InstanceBank.get()\n可学习(900) + temp(600)\ntemp_anchor: ego motion投影', S27_C, S27_B, pw='2.5')

# ========== Phase 1 ==========
nd('p1_title', 'Phase 1: Single-Frame\nDecoder (第1层)', BASE_C, BASE_B, pw='2')
nd('p1_eda', '3D Keypoint → 2D投影\n→ 双线性采样 → 加权融合\n(无camera_embed, 无时序)', S24_C, S24_B)
nd('p1_refine', 'Refinement\nanchor残差+分类', BASE_C, BASE_B)

# ========== §2.7 InstanceBank.update ==========
nd('ib_update', 'InstanceBank.update()\ntopk(300) + temporal(600)\n= 900实例', S27_C, S27_B, pw='2.5')

# ========== Phase 2: §2.3 ==========
nd('temp_gnn', 'temp_gnn: 时序交叉注意力\nQ=当前900, K=V=历史600', S23_C, S23_B, pw='2.5')
nd('gnn', 'gnn: 帧内自注意力\nQ=K=V, 900实例', S23_C, S23_B)

# ========== Phase 2: §2.4 EDA ==========
nd('p2_anchor', 'Anchor Embedding Encoder\nanchor→embed', S24_C, S24_B)
nd('p2_kps', '3D Keypoint → 2D投影', S24_C, S24_B)
nd('p2_weights', 'weights_fc→softmax\n融合权重预测', S24_C, S24_B, pw='2.5')
nd('p2_cuda', 'CUDA Kernel\n采样+融合一步完成', S24_C, S24_B)

# ========== §2.5 Camera Encoding ==========
nd('cam_enc', 'Camera Encoder MLP\nprojection_mat[:,:,:3]→12d→256d\n→ 广播+ 注入权重', S25_C, S25_B, pw='3')

# ========== §2.6 Depth Supervision ==========
nd('depth_pred', 'DenseDepthNet\nFPN特征→深度预测\ndepth × focal/equal_focal', S26_C, S26_B, pw='2.5')
nd('depth_gt', 'MultiScaleDepthMap\nLiDAR→投影→多尺度深度GT', S26_C, S26_B)
nd('depth_loss', 'L1 Loss\nloss_weight=0.2', S26_C, S26_B)

# ========== §2.7 InstanceBank.cache ==========
nd('ib_cache', 'InstanceBank.cache()\ndetach → 衰减→topk(600)\n→ 保存cached', S27_C, S27_B, pw='2.5')

# ========== 输出 ==========
nd('output', '输出: 检测结果', IO_C, IO_B, shape='doubleoctagon', pw='3')

# ========== 边: 主流程 ==========
dot.edge('in_img', 'bb')
dot.edge('in_prev', 'ib_get')
dot.edge('in_anchor', 'ib_get')
dot.edge('bb', 'p1_eda')

dot.edge('ib_get', 'p1_eda', label='  900可学习anchor')
dot.edge('p1_eda', 'p1_refine')
dot.edge('p1_refine', 'ib_update', label='  粗略检测')

dot.edge('ib_get', 'temp_gnn', label='  temp(600)', style='dashed')
dot.edge('ib_update', 'temp_gnn', label='  900(600temp+300cur)')
dot.edge('temp_gnn', 'gnn')
dot.edge('gnn', 'p2_anchor')
dot.edge('p2_anchor', 'p2_kps')
dot.edge('p2_anchor', 'cam_enc')
dot.edge('cam_enc', 'p2_weights')
dot.edge('p2_kps', 'p2_cuda')
dot.edge('p2_weights', 'p2_cuda')
dot.edge('p2_cuda', 'output')
dot.edge('output', 'ib_cache')
dot.edge('ib_cache', 'in_prev', label='  传播下一帧', style='dashed', color=S27_B, penwidth='2')

# ========== 边: §2.6 Depth分支 ==========
dot.edge('bb', 'depth_pred')
dot.edge('depth_pred', 'depth_loss')
dot.edge('depth_gt', 'depth_loss')

# ========== 边: 章节标题 → 对应模块 (虚线关联) ==========
dot.edge('title_s23', 'temp_gnn', style='dotted', color=S23_B, penwidth='2', arrowhead='none')
dot.edge('title_s24', 'p1_eda', style='dotted', color=S24_B, penwidth='2', arrowhead='none')
dot.edge('title_s25', 'cam_enc', style='dotted', color=S25_B, penwidth='2', arrowhead='none')
dot.edge('title_s26', 'depth_pred', style='dotted', color=S26_B, penwidth='2', arrowhead='none')
dot.edge('title_s27', 'ib_get', style='dotted', color=S27_B, penwidth='2', arrowhead='none')

out_path = 'sparse4dv2_full_pipeline_sections'
dot.render(out_path, cleanup=True)
print(f"OK: {out_path}.png")
