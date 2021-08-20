# 8/10文献阅读
## A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters ([OSDI’2020](https://arxiv.org/pdf/1807.08887.pdf))
---
BytePs，一种在异构的GPU/CPU集群上加速深度学习训练的架构，可以利用集群中的**闲置的CPU和带宽**来加速**GPU集群**上的DNN训练。
* 通信框架：最优和统一的，通过拆分参数优化器的功能实现
* Summation service：**求和服务**抽象，适用于各种优化器(optimizer)的**梯度求和**，可以用CPU的AVX并行指令集加速
* DNN模型相关的优化器算法在GPU上进行加速


两种分布式训练架构，均采用数据并行思想，在一次iteration中：
* all-reduce：仅使用GPU，计算参数梯度后聚合，在只有GPU的集群中带宽最优
* Parameter server：梯度发送到运行在CPU上的PS并聚合，理论上更优，实践中表现不佳
* 提出Summation service，逼近PS的理论极限。将optimizer分为两阶段：
    * gradient aggregation：梯度聚合，CPU运行
    * parameter update：参数更新，GPU运行
同时结合流水线调度和优先级思路

背景：
1. 分布式DNN训练：常用**数据并行**，对数据进行分割由GPU计算，而每个GPU都有完整的模型
2. All-Reduce：在每个GPU更新本地参数之前，以局部的形式以聚合的形式更新全部GPU的梯度。
3. Parameter Server：分为worker（在GPU上执行FP或BP，将梯度发给PS），PS（聚合work的梯度并更新参数）两种组件，同时又两种放置策略：
* 非同位模式（noncolocated），CPU机器与GPU机器分离（要求CPU在GPU数量两倍以上由于All-Reduce）
* 同位模式（colocated），无单独的CPU机器，利用GPU机器上空闲的CPU资源，通讯时间与All-Reduce相同
PS的GPU可以做异步训练，但是模型收敛速度更慢

问题在于集群中有大量的GPU机器运行着分布式的任务，它们的CPU和带宽资源是空闲的

BytePS架构：
* Coumunication Service：在内部对本地GPU的tensor进行同步，与SS进行通信
* Summation Service：比PS简单，通常PS运行神经网络算子，而SS只接受来自CS的tensor进行加和然后发回（这里是梯度）

