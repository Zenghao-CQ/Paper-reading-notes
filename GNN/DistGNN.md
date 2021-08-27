在CPU集群上利用DGL实现全批度的训练

对于大图，设备内存往往比较有限，通常用两种方法解决：
1. 用mini-batch进行分批次的训练
2. 聚合整个集群的内存容量
而mini-batch在某些情况下的准确率会比全批次学习更低，DistGNN在这片文章中研究了分布式内存的方案，并计划在后续工作中解决

full-batch的困难：
1. 大量信息交流：点特征和参数梯度
2. 较低的浮点计算密度、训练操作的连续性(sequential nature)：难以将通信与计算重叠
3. 高字节-浮点比：聚合操作成为通信瓶颈

contribution：
1. 为单个CPU设计优化的SpMM操作
2. 全新的分割算法来减少通信
3. 在特征聚合时使用新的延迟算法，达到了新的通信避免

高效聚合原语：
1. 在聚合点的特征时，每次循环中，边的特征是连续的且只用到一次，可以被组织为大的内存块，并且成为内存带宽绑定的流式访问；而一个点的特征是稀疏分布的，加载到cache后被用过一次就废弃了，所以常需要从内存中获取点特征。DistGNN尝试了进行

模型分析 pytorch.autograd.profiler
分布式jiqxx torch.distribute