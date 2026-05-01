# 笔记：NADDOD — 分布式训练通信原语

原文：[In-Depth Understanding of AI Distributed Training Communication Primitives](https://naddod.medium.com/in-depth-understanding-of-ai-distributed-training-communication-primitives-eb3b5fcc1f07)（约 8 min）

---

## 1. 原语是啥、为啥要在意

- **原语**不是某个框架的固定 API 名，而是各种分布式里反复出现的那几类**换数据、对齐状态**的抽象；PyTorch/TF/JAX、数据/模型/混合并行，底下多半都是这些在撑。
- 文里一点：**通信能跑多快，训练上限往往就被它限住**；卡性能/卡挂时，先想是不是 collective、链路有没有吃满。

## 2. 八种：模式速记

| 原语 | 模式 | 一句话 |
|------|------|--------|
| **Broadcast** | 一对多 | 一头发**同一份**到所有人；例如初始化、AllReduce 里某段、PS master→worker。 |
| **Scatter** | 一对多（和 Broadcast 不同） | 总数据**切开**各拿一瓣。记清楚：**Broadcast=人人全样，Scatter=各拿一块**；和 Gather 相反。 |
| **Gather** | 多对一 | 多瓣收齐到一头；常和 ReduceScatter 类组合放一起想。 |
| **AllGather** | 多对多（可脑补 Gather+再广播） | **人人最后都有完整份**；模型并行里 forward 前常要把分片参数凑全。和 ReduceScatter 对看。 |
| **Reduce** | 多对一 | SUM/MAX/… 归到一处；要快依赖硬件算子。 |
| **ReduceScatter** | 多对多 | **先全局 reduce，再分片发回**；DP/MP 里都会见；和 AllGather 成对。 |
| **AllReduce** | 多对多 | 每卡都有完整归约结果；典型是**梯度同步**；可看成 Reduce+Broadcast 或 **ReduceScatter+AllGather**。 |
| **All-to-All** | 多对多 | 不是「每人都拿别人的同一份」，而是**各收不同块**，偏转置；联系模型并行、DP↔MP 换格。 |

**嘴碎版**：同一份找 Broadcast，不同瓣找 Scatter；往少数点收 = Gather/Reduce；带 All* 的多半是「**人手一份**」或「**全员都参与**」。

## 3. NCCL 放哪

- **NCCL**：GPU 间 collective、拓扑感知、框架会调；原语名和上表**大致对得上**，还有点对点，方便**自己拼** Scatter/Gather/AllToAll。
- 文里对比：老 CUDA 常拷贝+kernel 拼；NCCL 往往把单次 collective **裹成**通信+本地算一坨（**以官方文档和实际版本为准**）。

## 4. 文尾结论

- 训练侧**一般不动**底网，是 NCCL/各厂 CCL、MPI、NVSHMEM 在做同步、规约；换机柜/调参**别忘**还有这一层。
- 同一份代码**换集群**差很多，以太网/IB/RoCE/NVLink 混用，库要**按拓扑**选路——**不全是**代码写呲了。

## 5. 待跟进

- 当前 job：**AllReduce/通信**先满还是算子/显存先满？可 profiler 或 NCCL 日志**扫一眼**。
- Ring vs Tree AllReduce：[单独一篇](notes-ring-tree-allreduce.md)；延迟/带宽假设别和通信原语那篇糊在一起。
