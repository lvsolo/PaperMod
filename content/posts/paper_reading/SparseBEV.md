BEVFormer这种query based模型由于query通过DeformableAttention通过ref pts采样时，ref pts的数量太少（默认4个），导致bev特征过于稀疏的问题，会造成大量的计算冗余，如何解决？

BEVDet系列的模型bev特征是从图像像素映射到3D空间再到BEV空间，所以bev特征相对稠密。
