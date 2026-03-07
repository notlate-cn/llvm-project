## 业界优秀实践：Add+Reduce 融合与 Tiling 切分

### 一、融合策略的业界主流分类

业界把 Add+Reduce 这类融合统一称为 **"Vertical Fusion"（生产者-消费者融合）**，核心问题是：**Reduce 的输入（Add 的输出）是否需要物化到全局内存**。

有三种主流答案：

------

#### 1. 内联融合（Inline Fusion）——Ansor/TVM 的标准做法

Ansor 的 Sketch 生成有一条明确规则：**"always fuse elementwise ops into the output stage"**。

具体机制：对融合子图 `Add → ReduceSum`，Ansor 在生成 Sketch 时把 Add 直接内联进 ReduceSum 的 reduction 循环 body，不产生中间 buffer。这与今天 demo 里 `linalg-fuse-elementwise-ops` 的结果完全一致。

Ansor 的关键创新在于：它把这个融合当做 **Sketch（骨架）**，然后只对 tile sizes 做搜索——搜索空间是 `(tile_M, tile_N_reduction)` 的组合，通过 evolutionary search 枚举，最终用 cost model 打分选最优。

```
Sketch（固定）：
  for m_outer (parallel):
    for m_inner:
      acc = 0
      for n (reduction):          ← Add 内联在此
        acc += A[m] + B[m,n]
      E[m_outer*T + m_inner] = acc

Annotation（搜索）：
  m_outer_size × m_inner_size × n_unroll_factor
```

这是你们 C++ 方案里"降级为 R 参与者"的学术表述——**Ansor 通过 Sketch 规则系统化实现了这一点**。

------

#### 2. Online Reduction（在线规约）——FlashAttention 的核心思想

FlashAttention 解决的是 `Softmax(Q·Kᵀ)·V` 这个计算，本质是 **Elementwise(Matmul) + Reduce(max/sum) + Elementwise(除法)** 的链式融合，而且 Reduce 的结果还要反过来修正之前的 Elementwise。

核心技术是 **Online Softmax**：利用数学上的递推关系，在 Tile 级别维护滚动的 `(max, sum)` 统计量，使得 Reduce 和 Elementwise 可以在单次遍历中完成：

```
# 标准 Softmax（需要两次遍历，不能融合）
m = max(x)          # 第一遍：全局 Reduce
y = exp(x - m)      # 第二遍：Elementwise
s = sum(y)          # 第二遍：Reduce
out = y / s

# Online Softmax（单次遍历，可按 Tile 融合）
for tile in tiles:
    m_new = max(m_old, max(tile))
    s_new = s_old * exp(m_old - m_new) + sum(exp(tile - m_new))
    # 输出在最后一个 tile 完成后修正：out *= exp(m_old - m_final)
```

这个思路的**工程本质**：把"需要全局 Reduce 结果才能计算的 Elementwise"变成"只需要局部统计量的增量更新"，从而使 Tile 粒度的融合成为可能。

对你们的场景有直接参考价值：**如果 Add 的上游有依赖 Reduce 结果的 Elementwise（比如 LayerNorm = Elementwise ÷ Reduce(mean)），就需要类似的 Online 算法才能融合**。

------

#### 3. 优先级融合 + 代价模型——XLA 的工程做法

XLA GPU 在 2024 年初默认启用了 **Priority-based Fusion**，核心架构：

- **不穷举所有融合组合**，而是用代价模型对每个候选融合打分
- 分数 = `unfused_cost - fused_cost`，即融合能节省多少 kernel 运行时间
- 按分数排优先队列，贪心地选最高收益的融合先做
- 代价模型分别估算 compute time 和 memory bandwidth，取较大者（roofline）

对于 Add+Reduce 这类情形，XLA 的判断是：融合后消除了中间 buffer 的 GM 读写，bandwidth savings 非常显著，因此会被高优融合。

**对你们 C++ 方案的启示**：你们的"轴分组判定 + Runtime 性能模型选 Kernel"与 XLA 的 Priority Fusion 思路是一致的，但 XLA 把这个判定做成了编译期就完成（代价模型），而不是 Runtime 再选。

------

### 二、Tiling 切分策略的业界主流方案

#### 1. Reduction 轴的两种标准切法

业界对 Reduction 轴的切分有两种经典模式，正好对应你们 C++ 方案的"模板A/B"：

**模式一：Row-wise Reduction（Reduce-in-inner）**

```
for m (Parallel, 核间并行):
    acc = 0
    for n (Reduction, 片内完整规约):
        acc += f(m, n)
```

适用条件：`N * dtype_size <= UB_size`，即一行数据能放进片上内存。
硬件映射：每个 AiCore 处理 TB_M 行，每行做完整 N 规约。

**模式二：Split-Reduction（Reduce-outer-split）**

```
for n_outer (Reduction分块, 核间并行):
    for m (Parallel):
        partial_sum[n_outer, m] += f(m, n_outer*T:n_outer*T+T)
# 第二趟：reduce partial_sum along n_outer
```

适用条件：N 很大，单行放不进片上内存；或者 M 很小，Parallel 轴不够填满所有 AiCore。

业界（包括 CUTLASS 的 SplitK、Ascend 的 HGEMM SplitK）都有这两套模板，**Runtime 根据 M/N 比例选择**，这与你们 C++ 方案完全一致，是标准做法。

------

#### 2. IREE 的 Tile+Distribute 策略

IREE 在 MLIR 上构建的 tiling 策略有一个值得借鉴的设计：**两阶段 tiling**。

- **第一阶段（Workgroup Tiling）**：决定如何把工作分配给多个 workgroup（对应你们的 TB 级），这是编译期静态决定的
- **第二阶段（Subgroup/Thread Tiling）**：决定 workgroup 内部如何切分（对应你们的 Tb/t 级），同样编译期静态

关键点：IREE 的 tiling 策略通过 `LoweringConfig`（一个 attribute）附加在 op 上传递给 codegen，**不同策略对应不同的 attribute 值**，这样同一套 codegen 流水线可以处理所有策略变体，只需在入口处改 attribute，而不是换整套代码。

这比你们"每个模板对应一套 Transform 脚本"更结构化——**所有模板共享同一套 lowering 基础设施，只有 tile sizes 和 axis 分配不同**。

------

#### 3. 最新进展：MCFuser（2024/2025，SC'24 收录）

MCFuser 专门针对"多个 Compute-Intensive 算子链的融合 tiling"，与你们的问题最接近：

核心思路：用 **Tiling Expression**（符号化 tiling 参数）来表示搜索空间，每个 tiling 参数对应一个轴的切分方式，然后通过 DAG 分析找出哪些 tiling 组合会产生冗余内存访问，从而裁剪搜索空间。

对比 Ansor 的提升：Ansor 的 Sketch 规则是静态的（预定义规则），对于多个 compute-intensive 算子链的融合空间覆盖不足；MCFuser 用动态 DAG 分析生成搜索空间，在 A100 上比 Ansor 快 5.9x。

------

### 三、直接对应你们方案的建议

| 你们的现有设计                               | 业界最佳实践对标                                  | 差距/改进方向                                                |
| -------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| 轴分组判定融合                               | Ansor Sketch 规则                                 | 可以把判定逻辑表达为 Transform Dialect 的 Sketch 规则，变成数据而非硬编码 |
| 模板A(Reduce-in-inner) / 模板B(Split-Reduce) | CUTLASS SplitK / IREE LoweringConfig              | 两套模板是标准做法，差距在于**模板共享同一套 lowering**，而非两套独立代码 |
| Runtime 性能模型选模板                       | XLA Priority Fusion 代价模型                      | XLA 把选择提前到编译期（Ahead-of-Time 代价估算），避免 Runtime overhead；你们是 JIT 选，更灵活但有 host 开销 |
| 符号化 tiling 参数（tiling_func）            | Ansor Annotation 搜索 / MCFuser Tiling Expression | 你们已经有符号化表达，下一步可以接 Auto-Tuner 做离线搜索     |
| Online Softmax 类算子                        | FlashAttention Online Reduction                   | 若涉及 LayerNorm/Softmax（Reduce 结果反馈给 Elementwise），需要 Online 算法才能融合，这是你们方案目前尚未覆盖的情形 |

最核心的一条结论：**你们 C++ 方案里"枚举 Tiling 模板 + Runtime 选择"的架构本身是正确的，和 Ansor/CUTLASS 的方向一致**。MLIR 方案的优势不在于改变这个架构，而在于用 Transform Dialect + 参数化 Attribute 把模板的表达和搜索做得更结构化、可维护。

Sources:

- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/system/files/osdi20-zheng.pdf)
- [Fused Softmax — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [XLA:GPU Architecture Overview | OpenXLA Project](https://openxla.org/xla/gpu_architecture)
- [XLA:GPU Priority-based fusion pass RFC](https://github.com/openxla/xla/discussions/6407)
- [Cost Models in XLA GPU – Present and Future](https://github.com/openxla/xla/discussions/10065)
- [MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators](https://arxiv.org/html/2506.22169v1)
- [Efficient Low-Precision GEMM on Ascend NPU: HGEMM's Synergy of Pipeline Scheduling, Tiling, and Memory Optimization](https://www.mdpi.com/2073-431X/15/1/39)
- [Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK](https://arxiv.org/html/2402.00025v1)
- [IREE Codegen Passes](https://github.com/iree-org/iree/blob/4546315d004d4cfc4079f3d13164bc92fddf6061/docs/developers/design_docs/codegen_passes.md)
- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)