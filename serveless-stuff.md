#### SoCC'20
---
* Wukong: a scalable and locality-enhanced framework for serverless parallel computing
* 无服务并行计算框架，将DAG任务移植到服务服务平台。分布式调度，执行复杂、突发并行的DAG细粒度任务()，需要高吞吐的快速scale和调度并减少数据迁移。分散式的调度增强数据局部性、减少网络I/O、自动资源弹性和提高成本效益。Faas框架不能控制任务的位置，不需要传统集中式调度器(由Faas服务商管理server，且提供几乎无限的临时资源)