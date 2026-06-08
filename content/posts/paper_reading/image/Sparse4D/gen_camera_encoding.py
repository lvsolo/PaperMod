import graphviz

dot = graphviz.Digraph('camera_encoding_position', format='png')
dot.attr(rankdir='TB', size='24,30', dpi='150',
         fontname='DejaVu Sans', bgcolor='white',
         label='Camera Parameter Encoding 在 Sparse4D v2 Pipeline 中的位置\n(红色虚线框 = Camera Encoding 发生的位置)',
         labelloc='t', fontsize='18', fontcolor='#333333')
dot.attr('node', fontname='DejaVu Sans', fontsize='11', margin='0.15,0.08')
dot.attr('edge', fontname='DejaVu Sans', fontsize='9')

PIPE_C = '#F3E5F5'; PIPE_B = '#7B1FA2'   # 紫色: pipeline 背景
DEC_C = '#E8F5E9'; DEC_B = '#2E7D32'      # 绿色: decoder 层
EDA_C = '#E3F2FD'; EDA_B = '#1565C0'      # 蓝色: EDA 模块
CAM_C = '#FFEBEE'; CAM_B = '#C62828'      # 红色: Camera Encoding
IMG_C = '#FFF9C4'; IMG_B = '#F9A825'      # 黄色: 输入/输出

def nd(name, label, color, border, shape='box', pw='2', style='filled,rounded'):
    dot.node(name, label, shape=shape, style=style,
             fillcolor=color, color=border, penwidth=pw,
             fontname='DejaVu Sans', fontsize='11')

# ========== 顶部: 输入 ==========
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='输入', style='rounded', color=IMG_B, fontsize='14', bgcolor=IMG_C)
    nd('input_img', '多视图图像\n(6 个相机)', IMG_C, IMG_B)
    nd('input_prev', '上一帧传播实例\n(cached_feature, cached_anchor)', IMG_C, IMG_B)
    nd('input_proj', 'projection_mat\n(B, 6, 4, 4)\n内外参矩阵', IMG_C, IMG_B, pw='2.5')

# ========== Backbone ==========
with dot.subgraph(name='cluster_backbone') as c:
    c.attr(label='Backbone + FPN', style='rounded', color=PIPE_B, fontsize='14', bgcolor=PIPE_C)
    nd('backbone', 'ResNet-50 + FPN\n→ 多尺度特征图\n(B, 6, 256, H/4, W/4)\n(B, 6, 256, H/8, W/8)\n(B, 6, 256, H/16, W/16)\n(B, 6, 256, H/32, W/32)', PIPE_C, PIPE_B)

# ========== Decoder ==========
with dot.subgraph(name='cluster_decoder') as c:
    c.attr(label='Decoder × 6 层 (每层执行以下操作)', style='rounded', color=DEC_B, fontsize='14', bgcolor=DEC_C)

    # Anchor encoding
    nd('anchor_enc', 'Anchor Embedding Encoder\nanchor (900, 11) → anchor_embed (900, 256)',
       DEC_C, DEC_B)

    # Temporal attention (Phase 2 only)
    nd('temp_attn', 'temp_gnn / gnn\n时序注意力 + 帧内自注意力', DEC_C, DEC_B)

    # EDA module (重点)
    with c.subgraph(name='cluster_eda') as eda:
        eda.attr(label='Deformable Aggregation (EDA)', style='rounded,dashed',
                 color=EDA_B, fontsize='13', bgcolor=EDA_C, penwidth='2.5')

        nd('kps_gen', 'Keypoint Generator\nanchor → 3D keypoints (13个)', EDA_C, EDA_B)
        nd('proj_2d', '3D → 2D 投影\nkpt × projection_mat → 2D 坐标', EDA_C, EDA_B)

        # ★ Camera Encoding (核心)
        with eda.subgraph(name='cluster_cam') as cam:
            cam.attr(label='★ Camera Parameter Encoding', style='rounded,bold,dashed',
                     color=CAM_B, fontsize='13', bgcolor=CAM_C, penwidth='3.5')
            nd('cam_input', 'projection_mat (B, 6, 4, 4)\n内外参矩阵', CAM_C, CAM_B, pw='2')
            nd('cam_extract', '取前3行 + reshape\n→ (B, 6, 12)', CAM_C, CAM_B)
            nd('cam_mlp', 'Camera Encoder MLP\n[Linear→ReLU→LN] × 2\n12d → 256d', CAM_C, CAM_B, pw='3')
            nd('cam_output', 'camera_embed\n(B, 6, 256)', CAM_C, CAM_B, pw='2')
            nd('cam_inject', '广播相加注入权重预测\nfeature[:,:,None] + cam[:,None]\n→ (B, 900, 6, 256)', CAM_C, CAM_B, pw='3')

        nd('weight_pred', 'weights_fc → softmax\n融合权重预测\n(已包含 camera_embed 信息)', EDA_C, EDA_B, pw='2.5')
        nd('sample', '双线性采样\ngrid_sample at 2D points', EDA_C, EDA_B)
        nd('fuse', '加权融合\nweights × sampled → instance_feature', EDA_C, EDA_B)

    nd('ffn', 'FFN\n特征变换', DEC_C, DEC_B)
    nd('refine', 'Refinement\nanchor 残差更新 + 分类预测', DEC_C, DEC_B)

# ========== 边 ==========
dot.edge('input_img', 'backbone', label='  图像')
dot.edge('input_prev', 'temp_attn', label='  传播实例')
dot.edge('backbone', 'sample', label='  多尺度特征图')

dot.edge('anchor_enc', 'temp_attn')
dot.edge('temp_attn', 'kps_gen')

dot.edge('kps_gen', 'proj_2d')
dot.edge('proj_2d', 'sample')
dot.edge('anchor_enc', 'weight_pred')

# Camera encoding 流程
dot.edge('input_proj', 'cam_input', label='  projection_mat', style='bold', color=CAM_B, penwidth='2.5')
dot.edge('cam_input', 'cam_extract')
dot.edge('cam_extract', 'cam_mlp')
dot.edge('cam_mlp', 'cam_output')
dot.edge('cam_output', 'cam_inject')
dot.edge('cam_inject', 'weight_pred', label='  注入', color=CAM_B, penwidth='2')

dot.edge('weight_pred', 'fuse')
dot.edge('sample', 'fuse')
dot.edge('fuse', 'ffn')
dot.edge('ffn', 'refine')
dot.edge('refine', 'anchor_enc', label='  下一层', style='dashed')

out_path = 'camera_encoding_position'
dot.render(out_path, cleanup=True)
print(f"OK: {out_path}.png")
