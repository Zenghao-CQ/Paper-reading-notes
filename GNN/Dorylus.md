### intro

解决两个问题：训练GNN需要大量的GPU服务器，价格昂贵；随数十亿条边的图GPU显存有限，用GPU机器训练缺乏可伸缩性
关键指导思想：Computation separation，构建深度、有界的异步流水线。
真实世界的图边数可能达到billion数量级，解决方式：1）用对内存要求不严格的CPU，但是不能并行计算2）用采样的方法，但是会影响精度
dorylus的目标是低成本，高效(接近GPU)，高准确率(高于采样)，因此才用了无服务计算，如AWS Lambda、Google、Cloud Function等。但是Lambda通常用来执行轻量级线程，计算能力和通信带宽都很有限。

#### 问题一：解决Lambda算力有限的问题

不是所有GNN训练的操作需要Lambda资源，GNN训练分为邻域传播图计算(Gather，scatter)，点和边的NN操作(Apply)，此外GNN的大图训练占主要的部分是**图计算**而不是可以由SIMD优化的张量计算、所以根据处理的数据类型将pipeline划分为细粒度任务：
* 在图结构上操作的任务(消息融合？)属于图并行路径graph-parallel path，由CPU资源处理
* 处理张量数据的属于task-parallel path，由Lambda计算

由于图结构不再以张量的形式表达(不用矩阵表达，由于消息传播范式)，所以张量数据和计算显著变小，每个task-parallel任务可以在少量数据上跑简单的线性代数运算，保证Lambda可以快速执行，Lambda**非常适合**处理GNN中的向量计算，而使用CPU资源会提高金钱成本，因为用户还需要对CPU以外的存储等资源付费。

#### 问题二：最小化Lambda的网络延迟

Lambda会用1/3的时间通信，Dorylus使用有界流水线异步计算bounded pipeline asynchronous computation (BPAC)，不同任务相互重叠，需要由计算分离来支持。Dorylus有两条并行路径，在参数更新（tensor并行），从邻居节点聚合数据(graph并行)。为了避免减缓收敛，Dorylus限制了并行程度，前者使用权重隐藏weight stashing，后者使用有界陈腐bounded staleness。实验证明使用采样技术达到同样准确率会更慢。

之前的工作证明用Lambda训练DNN只能到次优性能，而Dorylus使用缺少经费的小机构和训练超大图的情况

#### 架构概览
Graph Server，Lambda Thread，Parameter Server
输入经过图分割存储在GS上，图计算分为点并行(gather)，边并行(Scatter)。在GGNN中，点特征以二维数组保存，边特征用单独的数组保存。每个GS有一个ghost buffer，保存分散在远程服务器的数据，只有在scatter时GS之间才需要通信

#### 任务和流水线
首先，根据数据和计算类型，涉及输入图邻接矩阵计算(乘法)的由GS，而**只有**向量数据的由Lambda计算
**前向计算**，分为四个部分：GA，SC在GS上并行；AV，AE，只涉及特征和权重相乘以及激活函数，在Lambda上并行。对AV，从GS获取点特征，从PS获取权重，计算结果用RELU激活送回GS作为scatter的输入。当AV返回时SC沿着分区之间的边发送数据。对于AE，要从GS获取源点和目标点以及边的数据，
**反向传播**，反向任务与前向任务相对应，要么沿着图中边的反向传递信息、或者计算梯度和权值。同时还要进行参数更新WU（由PS执行），聚合来自各个PS的梯度。其中${\triangledown}AE$、${\triangledown}AD$与AE、AD传播信息(梯度)的方向相反，计算NN的更新
**流水线**从图任务开始训练你，将分区中的点划分为多个部分(如mini-batch)，**张量计算**的数量由点(AV)和边(AE)的数量决定，而图计算的数量主要由**边数**决定。同样采用图分割算法来保证负载均衡，每个部分(mini-batch)有数量相同的点和数量相近的内部边，由task处理。GS维护任务队列，当任务输入可用时入队，维护线程池，数量等于vCPU数，GS任务的输出被喂给Lambda上的张量计算任务。在反向传播中，${\triangledown}AE$、${\triangledown}AV$计算梯度并发给PS以更新性能权值。

#### 有界的异步计算
Dorylus使用有界异步bounded asynchronous训练保证worker在大多数情况下不需要等待的更新，这很重要因为Lambda的运行环境动态中总有掉队者。此外同步策略会延缓收敛速度，因为一些发展快的mini-batch可能使用过期权重。有界陈旧bounded staleness可以通过轻量级的同步来环节收敛问题，但是Dorylus由两个同步点：(1)WU任务的权重同步(2)每个GA同步来自邻居的激活数据

##### 数据更新阶段的有界异步性Bounded Asynchrony
用PipeDream的权值存储技术来控制异步程度，完全的异步的问题在于用v0版本的权值前向计算并生成梯度，但是在v1版本的权值上更新。权值存储让inerval用最新的权重前向计算并存储下来用于相应的反向传播。权重存储在PS中应用，PS存储全部层的权重矩阵(常规PS只存储一层)，因为Lambda可能在任何阶段任何层使用PS，也因为GNN层数少。而一个interval的不同task可以在不同Lambda执行，如果每个Lambda可以访问全部PS，则PS需要存储全部interval使用的参数，不可能实现。故不会在PS中保存全部的权值存储，权值存储只在其在该epoch中访问的第一个PS中

##### 控制Gather的异步性
异步计算，无需等待更新，可从邻居获得**过时**stale的激活向量。GNN的层数N固定，以传递N-hop的信息。问题在于，异步是否会改变GNN的语义，即(1)节点能否收到N-hop的影响？能，一个快的节点或许在当前epoch用了邻居节点的就特征，但是会在后续的epoch中收到(2)是否可能收到大于N-hop的信息？不会。Dorylus使用bounded staleness，运行快速的部分与最慢的的部分只允许差S个epoch。

#### Lambda的管理
GS上运行Lambda控制器，lambda运行OpenBLAS(AVX指令集)库，用ZeroMQ和GS、PS通信，运行在私有云(VPS)上，Lambda初始化后被提供GS和PS的地址，与之通信获得点、边、权重。由于Lambda一直在使用中，重用已经部署代码的容器而不是冷启动。

##### Lambda优化：
克服带宽问题，AWS的每条Lambda的带宽随着数量增加而减少，这是由于这些"Lambda"由同一用户创建，在同一台机器上调度，共享网络。
* 任务融合：最后一层的AV与${\triangledown}AV$直接连接，可以和为一个任务，可以减少Lambda和传输
* 重现张量tensor rematerialization：通常，DNN框架会存储前向计算的中间结果用于反向传播，但是在中Dorylus中需要从Lambda传输回GS，故通过发起更多Lambda来重现张量更高效
* Lambda内部流：如果让Lambda处理一个数据块，则另Lambda检索前半部分，计算后半部分。

#### 自动控制Lambda数量
静态控制Lambda数目不可行，某些图计算任务(如SC)依赖于张量任务(如AV)，lambda不足则不能生成足够多的线程发挥CPU性能，另一方面Lambda太多则CPU任务超出GS能力