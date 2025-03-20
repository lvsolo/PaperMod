1.算子支持：

1.1 不支持unknown维度算子tensor，一般在使用torch.where等操作时会出现，替换成multiply

1.2 不支持nn.embedding的bpu计算，需要转到cpu上计算，解决方法可以使用nn.Parameter采用nn.Embedding的权重init一下，可以实现可学习的同时保留位置信息；

1.3 Fourier Embedding中的nn.Embedding可以替换为Linear

1.4 NeighborAttention转换trt有问题，J6M存疑
