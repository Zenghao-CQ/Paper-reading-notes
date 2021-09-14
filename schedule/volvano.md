为kubernetes构建，面向批计算
基于[kube-batch](https://github.com/kubernetes-sigs/kube-batch)

* Queue：用于容纳podgroup的队列
  * weight：软约束，表示再集群资源划分中的相对比重(类似request)，集群资源空闲时可以超出
  * capability：硬约束，queue中所有podgroup的资源上限总和 
  * reclaimable：queue资源超出应得份额(weight)时是否可以被其他queue回收，默认为true

* PodGroup：一组强关联pod集合，主要用于**批处理**工作负载的情况，如tensorflow中的ps核worker
  * minMember，表示该podgroup下最少需要运行的pod或任务数量。如果集群资源不满足，则全部不调度
  



* VolcanoJob