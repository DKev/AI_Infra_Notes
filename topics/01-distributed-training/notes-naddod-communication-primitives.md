# 我读 NADDOD：分布式训练通信原语

我读的这篇：[In-Depth Understanding of AI Distributed Training Communication Primitives](https://naddod.medium.com/in-depth-understanding-of-ai-distributed-training-communication-primitives-eb3b5fcc1f07)（标了大概 8 min）

---

## 1. 我记一下：原语是什么、我为啥要搞懂

- **我理解的「原语」**：不是 PyTorch 里某一个函数名，而是各种分布式里反复出现的、挺抽象的**换数据、对齐状态**那几种套路；**我**用 PyTorch/TF/JAX、数据并行或模型并行，底下多半都是这些在撑。
- **文里我带走的一句**：**通信多快，我分布式训练能快到哪，很大程度上就被它卡住了**；我以后卡性能、卡挂，**我先想想**我其实在跑哪种 collective、网有没有吃满。

## 2. 我给自己留的一张表：八种、一对多/多对一/多对多

| 原语 | 模式 | 我一句话 |
|------|------|----------|
| **Broadcast** | 一对多 | 一头发**同一份**复制到大家；**我**会想到初始化、AllReduce 里某段、PS 里 master 甩给 worker。 |
| **Scatter** | 一对多（和 Broadcast 不一样） | 一块总数据**劈开**各拿一瓣；**我记**：Broadcast=人人拿全样，Scatter=各拿一块；和 Gather 相反。 |
| **Gather** | 多对一 | 我收到所有瓣拼回一头；**我常和** ReduceScatter 那类组合放一起想。 |
| **AllGather** | 多对多（我脑补成先 Gather 再 Broad） | **最后人人手里都是完整版**；模型并行我 forward 前**可能要**把分片参数先凑全再算。和 ReduceScatter 对着记。 |
| **Reduce** | 多对一 | 多路变一路，SUM/MAX/… **归到一处**；想快**得**有硬件算子。 |
| **ReduceScatter** | 多对多 | **先全局 reduce 再分片发回去**；数据并行/模型并行里**我都会遇见**；和 AllGather 成对。 |
| **AllReduce** | 多对多 | **每卡**最后都有完整归约结果；我常想到**梯度同步**；实现可以是 Reduce+Broadcast 或 **ReduceScatter+AllGather**。 |
| **All-to-All** | 多对多 | 人人互相给，**但我要的是「不同人给我不同块」**那种，有点像转置；**我**联系模型并行、DP/MP 之间换格。 |

**我自己嘟囔版：**  
发同一份找 Broadcast，发**不同瓣**找 Scatter；往少数点收是 Gather/Reduce；带 All* 的往往是「**人人都有一份**」或「**人人都得参与**」。

## 3. 我记 NCCL 在我脑子里的位置

- **我**：NCCL=GPU 之间搞 collective 的库，**会看拓扑**，框架能调；名字和我上面那张表**基本对得上**，还有点对点方便我**自己拼** Scatter/Gather/AllToAll 那种玩法。
- **文里我记住的一点**：老 CUDA 里可能拷贝+kernel 拼；**我**用 NCCL 时相当于**一坨**把通信和本地算**裹在一起**（**细节我信官方文档和我用的版本**）。

## 4. 我从结尾带走什么

- **我提醒自己**：写训练代码时**我很少直接**摸网；**是** NCCL/各家 CCL、MPI、NVSHMEM 这类在干同步和规约；我调参/换环境**要想到这一层**。
- **我经历的困惑终于有个说法**：同一份代码**换集群**天差地别，因为以太网/IB/RoCE/NVLink 混着，库要**按拓扑**选路——**不全是**我程序写坏了。

## 5. 我接下来要问自己的

- 我现在这个任务，**是 AllReduce/通信先顶满**还是算子/显存先爆？**我**会试着用 profiler 或 NCCL 日志**瞟一眼**。
- 文里**没细讲** Ring vs Tree AllReduce；**我**要另找一篇**专门记**延迟和带宽假设，别和这篇搅在一坨。
