# 笔记：vLLM 官宣文 —— PagedAttention 与 KV cache

原文：[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)（Berkeley / LMSYS，2023-06）

---

## 1. 文在解决什么问题

在线 decode（自回归生成）里，每层存的 **KV cache** 占显存大头；序列长度可变 → **预留与碎片**严重，显存利用率差 → **batch 上不去**，吞吐卡住。这篇的核心卖点：**PagedAttention** 管 KV，宣称可把浪费压到很低，从而塞更多并发序列、拉高吞吐。

---

## 2. KV cache 为啥难搞（文里的三条）

- **大**：举例 LLaMA-13B，单序列 KV cache 可到 **GB 量级**。
- **动态**：长度随请求变，事先不好估。
- **浪费**：称既有系统常因 **fragmentation + over-reservation** 丢掉 **约 60%–80%** 内存（数字来自原文实验设定，当作量级直觉即可）。

瓶颈归结为：**memory**，而不是「算力的名义 FLOPs」一句话能说清的那种。

---

## 3. PagedAttention 是啥（类比 OS 分页）

- 把每条序列的 KV cache **切成固定大小的 block**（块里是若干 token 的 K/V）。
- **逻辑上连续**的块，可以映射到 **物理显存上不连续**的块 → 类似虚拟内存：**逻辑页 ↔ 物理页**，中间一张 **block table**。
- **按需分配**：decode 多长就往后追加物理块，减少大块预留带来的空洞。
- 文称：**浪费主要在每条序列最后一个未满 block**，整体接近「够用就好」，宣称 unused **可到 ~4% 以下**（原文措辞）。

Attention 计算时 kernel **按 block 取数**，不要求整块 KV 在内存里连续摆放——这是和传统「一长条连续 tensor」思路的根本差别。

---

## 4. 附带红利：共享物理块（并行采样 / beam）

同一 prompt **多条输出**（parallel sampling）时，**prompt 前缀**对应的 KV **可共用物理 block**。block table 里多条序列映射到同一物理块即可。

- 用 **引用计数** 管生命周期。
- **Copy-on-write**：要写分叉时再拷贝块，避免瞎共享。

文里数字：**并行采样、beam** 等场景显存可降 **~55%**，吞吐可达 **~2.2×**（仍是原文设定下的对比）。

---

## 5. 和「吞吐数字」的关系（读的时候别只看倍数）

博文对比对象：**HF Transformers**、**TGI**；数据集：**ShareGPT** 上抽 I/O 长度；机型：**A10G + LLaMA-7B**、**A100 40GB + LLaMA-13B**。宣称相对 HF **最高约 24×**、相对 TGI **约 3–4×** 量级。

启示：**Serving 论文/博文里的倍数高度依赖 trace、并发、长度分布和 baseline 版本**——带走机制（分页 KV、少浪费、多 batch），别把倍数当自家机器的承诺。

---

## 6. 延伸（这篇没说细的）

- **Continuous batching**：实操里常和 vLLM 一起出现；这篇正文重点是 **PagedAttention + 内存**，batching 调度细节可查 [文档](https://docs.vllm.ai/) 或后续博文。
- **正式论文**：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（OSDI’24），arXiv：[2309.06180](https://arxiv.org/abs/2309.06180)。

---

## 7. 待做

- [ ] 本地起一个 `--max-model-len` 很小的 toy，粗看 **并发上来之后**显存曲线是否更符合「塞得下更多序列」的预期。
- [ ] 若做推理容量估算：**KV block size × 层数 × max concurrency** 这条链子要自己按模型维度推一遍（别只用博文举例数字）。
