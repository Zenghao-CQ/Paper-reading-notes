task parallel&actor parallel
结合现有的Clipper和TF serveing等模型（它们在模型部署管理训练等方面很成熟）
训练training：分布式SGD，使用all-reduce或者paramater server聚合
serving：减小延迟、最大化每秒钟决策数量

在强化学习任务中；training，serving，simulation三者紧密相连，有一定的延迟需求（在有监督学习中可以分离），在现有的框架中是分离的

两个点：1）要处理每秒百万个task 2）不需要从头实现深度学习框架

## 编程模型：
将模型组织为在执行过程中演化的动态图
远程函数运行在不可变的对象上，并且是无状态和没有副作用的，单独的由输入决定

## 计算模型
使用动态任务图，task和actor都在其input可用的时候被系统自动调度。将task构建成计算图结构，同时通过有状态边可以将actor映射为无状态图

## 架构
* 应用层
    * Driver：执行用户程序的进程
    * Worker：无状态进程，执行由driver或者worker调用的远程函数task（由**系统层进自动分配**）
    * Actor：有状态进程，需要worker或者driver显示的实例化
* 系统层
    * Global Control Store(GCS)：key-value数据库（redis）保存全局状态信息，用分片机制实现伸缩性，用分片链副本实现分区可容忍性。对于allreduce这样的分布式训练来说，让schduler参与每个对象的传输代价高昂，因为allreduce对通信强度和延迟敏感。所以Ray将对象的元数据存放在GCS中，使得任务放置（dispatch）与调度（schedule）解耦。
    * scheduler调度器：要求每秒数百万个，毫秒级别，global scheduler+local scheduler(Raylet)。
        1. task在节点上创建并且首先被提交到local scheduler，由其在本地节点上进行调度
        2. 不能在本地调度则提交给global scheduler
        3. global scheduler考虑节点负载和任务约束，通过：标注满足任务资源需要的节点集，选择预估等待时间最短的节点
        4. 预估等待时间计算：在节点上任务队列等待的时间+任务远程输入的预估传输时间，global scheduler通过心跳从GCS获取节点信息
    * 内存分布式对象存储：用Apache Arrow，实现节点上的共享内存（同一个节点task和对象存储之间共享），使得同一个节点上的task之间零数据复制；如果task的输入不在本地，则在执行前将其复制到本地。当节点失效，通过重新执行lineage来恢复对象；此外ray不支持分布式对象，每个对象仅适用一个节点




在分配任务时候，考虑把由边的分配到同一节点？直接索引？