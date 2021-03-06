## intro
**首先，显然的节点的embedding是一个稀疏化的表示**
* 使用同步训练方法(synchronous training approach）
* 特点：高质量、轻量级的图分割算法，多个平衡性约束，以减少通信（复制halo节点、稀疏嵌入更新）并且平衡计算

首先，GNN的问题在于：
* CV和NLP这样的传统机器学习任务的样本通常是互相独立的，而图数据则要内在的表示样本之间的内在关系（点和边）。其每个batch需要包含其中起到决定作用的样本，而这样的样本数量与要用到的邻居节点的hop数成**指数关系**，因此需要用**采样算法**来将GNN扩展到大规模图上，来保证在精确获取GNN节。
* 此外与**传统分布式机器学习**不同，由于获取节点特征时候需要大量多个hop内的邻居节点信息，传输梯度数据会占用大量的**网络流量**。
* 此外，分布式训练框架常用同步SGD算法，需要分布式GNN框架生成包含数量均衡的节点、边、网络以及传输数据的batch，而一个图中的子图结构复杂，很难做到均衡。
* 此前的框架方向的困境：
    * full-batch SGD，全批次样本更新梯度，进行图分割来适应多设备的聚合内存
    * 获取邻居节点数据需要大量网络带宽
    * 此前的分布式学习框架通过网络传输来交换梯度数据，而因为GNN的节点依赖问题，获取邻居节点数据带来的网络带宽本就是GNN训练的一大瓶颈，故传统分布式框架不适用。

遵循同步训练方式，允许自我中心网络(ego-network)形成mini-batch来包含非本地节点；同时采用METIS来分割图（以最小边割集）；使用多约束分区和2层的工作负载拆分来进行load balance；同时通过复制分区图结构的的halo节点来减少采样过程的网络通信

## background：
* 消息传播模型
* mini-batch：典型的GNN上的小批量训练如下：
    * 采样：从训练集中均匀随机的采样目标节点(target vertices)
    * 对**每个**目标节点随机选择K个邻居(称作fan-out)
    * 在fan-out上进行消息传播过程
    * 如上的采样策略得到一个小的图来作为mini-batch[fig1]，此外还有很多不同的采样方法
![](./pics/distgnn.png)
## DistDGL架构概览：
* 分布式训练架构[fig2]: 采用随机梯度下降(SGD)，每台机器计算自己的mini-batch的梯度，同步梯度，更新本地的模型副本。构成部件有：
    * 若干sampler采样器，获取mini-batch子图
    * KVstore，分布式存储点、边数据，也可以保留点的embedding信息
    * 若干训练器trainer，通过mini-batch计算参数的梯度。在每个iteration，从sampler获取mini-batch子图，从KVstore获取相应点、边的特征，进行前向计算和反向传播获取参数梯度，梯度将分别处理：
        * 对于稠密参数（稠密矩阵？）的梯度，将发送到稠密模型更新组件
        * 稀疏的embedding的梯度将送到KVstore
    * 稠密模型更新组件dense model update component，用于聚合(aggregate)稠密GNN参数来执行同步的SGD，这部分可能是All-reduce（当后端用pytorch），或者paramer server（用tf）

为了减少通信，采用owner-compute规则，尽量把计算安排给相应数据的拥有者。通常是在每台机器上起sampler和KVstore来为本地分割数据服务，tainer负责机器上的本地分割数据的计算

* 可训练参数类型：
    * 稠密参数(dense paramer)：聚合(aggregating)、消息传播(message)的以及节点表示更新的参数，作为由所有节点共享的模型参数
    * 稀疏参数(sparse paramer)：有的GNN需要学习节点的嵌入信息(直推式学习)，这不是由全部节点共享的，每次在mini-batch中只包含部分的embedding
* 图分割：训练前的预处理，要求各个分区之间的边尽量的少，同样的使用METIS进行分区，此外分区相关联的边也纳入分区中，这会有部分点重复[fig3]，其中METIS分配的称为核心点(Core vertices)，因为分配边而重复的称为光晕点(Halo vertices).默认情况下的METIS仅仅保证了核心点的平衡，不足以产生相同的batch数目和大小，这里将问题描述为多分区约束问题。
* 分布式的KVstore：管理点、边的特征以及点的嵌入信息。此外，访问点、边的数据占用通信的一大部分。KVstore使用了共享存储，因为通常数据和计算通常再同一机器上，所以使用了共享存储而不是进程间通信(IPC)。
此外KVstore专门设计了对**稀疏embedding**存储的支持，在mini-batch的训练中只有一小部分的embedding被计算和更新，而现有分布式机器学习框架缺乏对分布式稀疏更新的支持
* 分布式采样器：使用DGL原有的采样API。迭代开始，trainer用当前mini-batch的目标点(target vertices)发起采样请求，请求分分配到相应的机器(由图分割算法产生的核心顶点分配决定)，采样器收到请求后在本地的分区上进行采样并返回给trainer进程，trainer即将结果拼凑成mini-batch。
distGNN可以为一个trainer生成多个sampler来，并行产生mini-batch
* Mini-batch trainer：因为partition生成平衡的划分(点、边数目几乎是平衡的)，在这些分区上均匀随机采样，所以总体生成的mini-beatch理论上也是均匀随机的。最后通过训练样本的ID划分并安排的机器上，尽量保证样本的范围和分区得到的范围重合。而这里面有一些不重合的部分，需要在负载均衡与数据局部性之间均衡，这一点很小。如果有多个trainer实例，进一步随机拆分数据集。对于参数同步，使用同步SGD算法；对于稀疏的节点嵌入embedding，用异步SGD算法Hogwild，只更新全局的embedding的一小部分，因为不同trainer用到的embedding不同，所以同步更新几乎没有冲突。

**一个提升是**
**需要预处理用METIS来手动划分固定数量的sub-graph**
**sampler、trainer、update，三者之间副本更新与平衡，流水线，自动伸缩？**
**机器性能差距大时**->性能，trainer、sampler副本个数的调整
一个epoch中可能会等待不同机器的完成

**显然trainer需要大量GPU资源、而sampler未必，可以看一下具体实现，根据资源利用情况进行分配**

工作方向：
* 采样算法：将大图拆分为子图
* 获取节点特征时候需要大量多个hop内的邻居节点信息，此时交换模型的梯度参数需要占用大量网络流量
* 目前一个节点只有一个server和若干backup server

采样梯度时、从单个batch计算梯度，聚合，各自更新