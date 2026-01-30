# AI编译器融合技术系统化分类总结

## 0. 全篇概述与阅读建议

### 0.1 **全篇概述**

本文旨在深入探讨**编译器融合技术**，特别是如何在硬件架构（如GPU和NPU）上实现高效的计算。我们将从不同的层次和角度分析优化技术，包括算法层面的算子融合（Fusion）、内存布局优化、硬件适配，以及全局优化策略。每一章都旨在帮助读者理解如何通过优化编译器的各个环节来最大化硬件资源的利用，从而提升计算性能。

本文内容的核心思想是：**优化不仅仅是局部算子的改进**，而是**通过跨层次的全局优化策略，解决不同优化手段之间的冲突，以实现最优的硬件适配和性能表现**。特别是随着 AI 处理器（如GPU和Ascend NPU）的发展，编译器需要更加智能地处理算子调度、内存管理、计算与存储的权衡等复杂问题。

### 0.2 **Fusion的本质**

算子**融合（Fusion）**是 AI 编译器优化技术中的一个核心概念。Fusion不仅仅是将多个算子合并为一个操作，它的本质在于通过**改变计算边界、重组执行时序和重映射数据生存期**，以最小化数据传输、最大化硬件利用率，从而提升整体计算效率。

具体来说，Fusion的本质包括：

* **改变计算边界（Compute Boundary）**：通过将多个操作融合到一个计算图中，减少不必要的计算拆分，使得计算边界尽可能靠近硬件资源。
* **重组执行时序（Execution Order）**：通过合理的调度顺序，避免冗余的内存访问和计算，提高计算资源的利用率。
* **重映射数据生存期（Data Lifetime）**：通过延长某些数据的生命周期和优化数据的存储方式，减少不必要的数据传输和缓存读写。

Fusion的核心目标是减少数据移动开销，并最大化硬件的计算能力。这一思想贯穿本文的各个章节，特别是在算子层级和内存优化方面，它是实现高效编译器优化的关键技术之一。

### 0.3 章节结构

* 第1章（图）：先看宏观的图结构，决定谁和谁能连在一起。
* 第2章（环）：进入算子内部，看循环怎么写效率最高。
* 第3章（数）：看数据怎么摆放，算得最顺手。
* 第4章（存）：数据摆好了，怎么在不同速度的存储器（HBM/L2/Reg）之间搬运。
* 第5章（并）：单核算好了，怎么利用多核、多芯片、集群搞并行。
* 第6章（衡）：硬件资源有限（寄存器/专用指令），怎么做取舍（Trade-off）。
* 第7章（变）：形状或流程变了（动态性），怎么通过特化和符号化稳住性能。
* 第8章（全）：以上全是局部招数，最后用全局视野（Cost Model）做最终决策。

---

## 1. 依赖拓扑（Dependency Topology）

作为融合优化的起点，本章立足于**计算图（Computational Graph）** 的宏观视角。

在这一层级，编译器不关注算子内部的具体实现，而是聚焦于算子节点之间的**数据依赖（Data Dependency）**与连接关系。核心目标是通过**重组图结构**，识别并消除冗余的内存读写操作。通过分析生产者-消费者的**垂直链条**、**多分支的水平结构**以及**特定的子图模式**，编译器决定将哪些独立的算子节点"坍缩"为一个内核，从而在算法层面最小化数据搬运的开销。

### 1.1 垂直融合（Vertical Fusion）

垂直融合沿着数据流的生产者-消费者链条进行优化，是最经典的融合模式。

#### 1.1.1 Producer-Consumer Fusion（生产者-消费者融合）

##### 背景

在深度学习模型中，算子之间往往存在严格的数据依赖关系：后一个算子需要前一个算子的输出作为输入。这种依赖链形成了 **"生产者-消费者"**（Producer-Consumer）关系。

传统执行模式下，每个算子独立编译成 kernel，物化（Materialization，中间结果）需要写入全局内存（Global Memory），然后下一个 kernel 再重新读取。这种"物化"过程带来了显著的开销：

- **内存带宽压力**：每个中间张量都需要写回主存再读出
- **Cache利用率低**：物化可能驱逐有用的Cache内容
- **Kernel启动开销**：每个算子单独启动一个Kernel

**核心思想**

**消除中间物化（Intermediate Materialization Elimination）**：将生产者的计算内联到消费者中，使物化仅存活于寄存器/L1 Cache，避免写回全局内存。

```cpp
// 融合前
C = Add(A, B)        // 写入全局内存
D = ReLU(C)          // 从全局内存读取C 

// 融合后
C = ReLU(Add(A, B))  // 物化仅存活于寄存器
```

##### 应用场景

| 场景 | 特征 | 示例 |
|------|------|------|
| 逐元素操作链 | 多个 element-wise 操作串联 | `Add → ReLU → Mul → Sigmoid` |
| 卷积后激活 | 卷积 + 激活函数 | `Conv2d → BiasAdd → ReLU` |
| 归约后处理 | 归约操作 + 后处理 | `ReduceMax → Subtract` (LogSoftmax) |
| 矩阵乘法融合 | GEMM + bias/activation | `MatMul → BiasAdd → Gelu` |

##### 技术原理

Producer-Consumer Fusion 的决策依赖于以下几个关键因素：

1. **依赖链分析**（Dependency Chain Analysis）
   - 构建计算图的 **依赖图**（Dependency Graph）
   - 识别 **合法融合候选**（Legal Fusion Candidates）
   - 检测 **循环依赖**（Cyclic Dependency）

2. **内存代价评估**（Memory Cost Estimation）
   $$
   收益 = \sum(Write_{global} + Read_{global}) - Cost_{recompute} - Cost_{spill}
   $$
   
3. **计算兼容性检查**（Computational Compatibility）

   - **迭代空间一致性**（Iteration Space Alignment）
   - **仿射变换兼容性**（Affine Transform Compatibility）
   - **归约语义保持**（Reduction Semantic Preservation）

##### MLIR 实现方案

MLIR 通过多级 IR（Linalg、Affine、SCF）和模式重写框架实现 Producer-Consumer Fusion：

**核心 Dialect 与 Pass：**

| Dialect/Pass | 作用 | 关键数据结构 |
|--------------|------|--------------|
| `linalg` | 结构化算子定义 | `linalg.generic`, `linalg.matmul` |
| `-linalg-fuse-elementwise-ops` | Element-wise 融合 | `ElementwiseOpFusionPass` |
| `-affine-loop-fusion` | 循环级融合 | `AffineLoopFusion` |
| `scf.for` | 结构化控制流 | `scf::ForOp` |

**实现架构：**

```
┌─────────────────────────────────────────────────────────┐
│                  MLIR Fusion Pipeline                    │
├─────────────────────────────────────────────────────────┤
│  1. 依赖分析 (Dependency Analysis)                       │
│     └── DominanceInfo, PostDominanceInfo                │
├─────────────────────────────────────────────────────────┤
│  2. 融合候选识别 (Fusion Candidate Identification)       │
│     └── linalg.generic 的 producer-consumer 匹配        │
├─────────────────────────────────────────────────────────┤
│  3. 合法性检查 (Legality Check)                          │
│     ├── 迭代空间兼容性 (Iteration Space Compatibility)   │
│     ├── Side-effect 检查                                 │
│     └── 形状推断 (Shape Inference)                       │
├─────────────────────────────────────────────────────────┤
│  4. 融合变换 (Fusion Transform)                          │
│     └── TileAndFuse, FuseIntoContainingOp              │
├─────────────────────────────────────────────────────────┤
│  5. 代码生成 (Code Generation)                          │
│     └── LLVM IR / SPIR-V / CUDA                         │
└─────────────────────────────────────────────────────────┘
```

##### MLIR 示例：Add + ReLU 融合

**融合前的计算图：**

```cpp
// 生产者：逐元素加法
%add = tensor.empty() : tensor<128x128xf32>
%0 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%add : tensor<128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
}

// 消费者：ReLU 激活
%relu = tensor.empty() : tensor<128x128xf32>
%1 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%0 : tensor<128x128xf32>)
    outs(%relu : tensor<128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %zero = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %arg0, %zero : f32
    %result = arith.select %cmp, %arg0, %zero : f32
    linalg.yield %result : f32
}
```

**融合后的单算子：**

```cpp
// Producer-Consumer Fusion: Add + ReLU
%output = tensor.empty() : tensor<128x128xf32>
%2 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,   // A
                   affine_map<(d0, d1) -> (d0, d1)>,   // B
                   affine_map<(d0, d1) -> (d0, d1)>],  // Output
  iterator_types = ["parallel", "parallel"]
} ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%output : tensor<128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    // 原始 Add 操作
    %sum = arith.addf %arg0, %arg1 : f32
    // 原始 ReLU 操作（物化 %sum 无需物化）
    %zero = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %sum, %zero : f32
    %result = arith.select %cmp, %sum, %zero : f32
    linalg.yield %result : f32
}
```

**Pass Pipeline调用：**

```bash
# 使用 MLIR opt 工具执行融合
mlir-opt input.mlir \
  --linalg-fuse-elementwise-ops \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm
```

---

#### 1.1.2 Element-wise Chain Fusion（逐元素操作链融合）

##### 背景

这是垂直融合的一个特化场景，针对 **逐元素操作**（Element-wise Operation）的连续链式结构。这类操作的迭代空间完全一致，且无归约依赖。

例如 Transformer 中的 FFN：`Linear → Gelu → Dropout → Linear`。

##### 技术原理

- **迭代空间对齐**（Identical Iteration Space）：所有操作共享相同的循环结构。
- **索引映射简化**（Simple Indexing Maps）：通常简化为恒等映射（identity）或广播（broadcast）。
- **Kernel Launch 消除**：N 个 Kernel 合并为 1 个。

##### MLIR 实现关键 Pass

```cpp
// mlir/lib/Dialect/Linalg/Transforms/Fusion.cpp
struct ElementwiseOpFusionPattern : public OpRewritePattern<linalg::GenericOp> {
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // 1. 检查是否为 element-wise 操作
    if (!isElementwise(op)) return failure();

    // 2. 查找 producer（当前操作的所有输入）
    for (Value operand : op.getInputs()) {
      if (auto producer = operand.getDefiningOp<linalg::GenericOp>()) {
        // 3. 验证融合合法性
        if (isValidFusionCandidate(producer, op)) {
          // 4. 执行融合
          return fuseOps(producer, op, rewriter);
        }
      }
    }
    return failure();
  }

private:
  bool isElementwise(linalg::GenericOp op) const {
    // 检查所有 iterator_types 是否为 "parallel"
    // 检查 indexing_maps 是否为 permutation（无归约维度）
  }

  bool isValidFusionCandidate(linalg::GenericOp producer,
                              linalg::GenericOp consumer) const {
    // 验证迭代空间兼容性
    // 验证没有 side-effect
    // 验证形状一致性
  }
};
```

##### MLIR 示例：融合 LayerNorm 的逐元素操作部分

LayerNorm 的后半部分（Normalize, Scale, Shift）是典型的逐元素链。

```cpp
// 原始分解实现（多个独立的 linalg.generic）
func.func @layer_norm_unfused(%input: tensor<BxMxNxf32>,
                              %weight: tensor<Nxf32>,
                              %bias: tensor<Nxf32>) -> tensor<BxMxNxf32> {
  // Step 1: 计算 mean (reduction)
  %mean = linalg.generic ... { arith.addf, ... }

  // Step 2: 计算 variance (element-wise + reduction)
  %variance = linalg.generic ... { arith.subf, arith.mulf, ... }

  // Step 3: normalize (element-wise)
  %normalized = linalg.generic ... {
    ^bb0(%x, %m, %v, %w, %b):
      %eps = arith.constant 0.00001 : f32
      %std = arith.sqrt(arith.addf %v, %eps) : f32
      %norm = arith.divf arith.subf(%x, %m), %std
      %scaled = arith.mulf %norm, %w
      %result = arith.addf %scaled, %b
      linalg.yield %result
  }

  return %normalized : tensor<BxMxNxf32>
}

// 融合后
func.func @layer_norm_fused(%input: tensor<BxMxNxf32>,
                            %weight: tensor<Nxf32>,
                            %bias: tensor<Nxf32>) -> tensor<BxMxNxf32> {
  // 假设 Mean 和 Variance 已计算完成

  // 融合：normalize + scale + shift 三个 element-wise 操作
  %output = linalg.generic {
    indexing_maps = [
      affine_map<(b, m, n) -> (b, m, n)>,  // input
      affine_map<(b, m, n) -> (b, m)>,     // mean (broadcast)
      affine_map<(b, m, n) -> (b, m)>,     // variance (broadcast)
      affine_map<(b, m, n) -> (n)>,        // weight (broadcast)
      affine_map<(b, m, n) -> (n)>,        // bias (broadcast)
      affine_map<(b, m, n) -> (b, m, n)>   // output
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input, %mean, %variance, %weight, %bias
        : tensor<BxMxNxf32>, tensor<BxMxf32>, tensor<BxMxf32>,
           tensor<Nxf32>, tensor<Nxf32>)
      outs(%init : tensor<BxMxNxf32>) {
  ^bb0(%x: f32, %m: f32, %v: f32, %w: f32, %b: f32):
    // 融合的计算：一次遍历完成所有 element-wise 操作
    %eps = arith.constant 0.00001 : f32
    %std = arith.sqrt(arith.addf %v, %eps) : f32
    %norm = arith.divf arith.subf(%x, %m), %std : f32
    %scaled = arith.mulf %norm, %w : f32
    %result = arith.addf %scaled, %b : f32
    linalg.yield %result : f32
  }

  return %output : tensor<BxMxNxf32>
}
```

#### 1.1.3 Reduce-to-Elementwise Fusion (Algorithmic Fusion)

> Algorithmic Fusion ≈ 改变计算结构（Algebraic Form），而不仅是合并执行

##### 背景

标准的垂直融合通常在 **归约（Reduction）** 操作处断开。例如 Softmax 或 LayerNorm，传统实现需要多次遍历内存（Pass 1: 求和/均值 $\to$ Pass 2: 归一化）。

**归约-逐元素融合** 旨在打破归约操作的同步屏障，通过**数学算法的重写**（如 Welford 或 Online Softmax），将多次内存扫描合并为一次扫描（One-pass）。

##### 技术原理

1. **Welford 算法 (For LayerNorm)**：
   - **传统**：先遍历一次求 Mean，再遍历一次求 Variance，最后遍历求 Output。
   - **融合**：在单次循环中同时维护 Mean 和 Variance 的迭代更新公式，一次遍历即可得到最终统计量并应用 Normalization。
2. **Online Softmax / Safe Softmax**：
   * **传统**：$max = reduce_{max}(x)$ -> $sum = reduce_{sum}(e^{x - max})$-> $out = \frac{e^{x-max}}{sum}$ (3 Pass)。
   * **融合**：利用数学性质 $\frac{e^{x_i - \max}}{\sum e^{x_j - \max}}$，在一次遍历中动态更新全局 Max 和 Sum，无需预先扫描最大值。

##### 应用场景

| 算子                    | 涉及算法          | 收益                    |
| ----------------------- | ----------------- | ----------------------- |
| **LayerNorm / RMSNorm** | Welford Algorithm | 减少 1-2 次全局内存读写 |
| **Softmax**             | Online Softmax    | 减少 2 次全局内存读写   |
| **Cross Entropy Loss**  | Log-Sum-Exp Trick | 提升数值稳定性与性能    |

这种融合无法通过简单的 `linalg.fuse` 实现，通常需要使用 `scf.for` 携带状态（`iter_args`）来表达复杂的更新逻辑。

##### MLIR 示例

```cpp
// Online Softmax 逻辑结构示例
// scf.for 不仅计算，还通过 iter_args 携带动态更新的 max 和 sum
%final_max, %final_sum = scf.for %i = 0 to %N 
  iter_args(%curr_max = %neg_inf, %curr_sum = %c0) -> (f32, f32) {
  
  %val = load %input[%i]
  // 1. 更新 Max
  %new_max = arith.maxf %curr_max, %val
  // 2. 计算修正因子：exp(old_max - new_max)
  %correction = arith.exp (%curr_max - %new_max)
  // 3. 修正旧的 Sum 并加上新的项
  %term = arith.exp (%val - %new_max)
  %new_sum = arith.addf (arith.mulf %curr_sum, %correction), %term
  
  scf.yield %new_max, %new_sum
}
```

---

### 1.2 水平融合（Horizontal Fusion）

#### 1.2.1 Multi-output Fusion（多输出融合）

##### 背景

**Multi-output Fusion**（也称 **Sibling Fusion**）针对**多个算子共享同一输入**的场景。如果每个消费者独立执行，共享输入会被多次加载。水平融合将这些计算合并，实现"一次读取，多次计算"。

##### 应用场景

| 场景 | 描述 | 性能收益来源 |
|------|------|--------------|
| **Attention QKV**   | 同时计算 Q、K、V                | 内存布局优化 ([3, H, ...] vs [H, 3, ...]) |
| **Gate Projection** | GLU 变体中的 Gate 和 Value 分支 | 合并矩阵乘法                              |
| **Multi-branch**    | Inception 模块 / 多 Loss 计算   | 共享卷积输入                              |

##### 技术原理

Multi-output Fusion 的核心决策因素：

1. **输入共享度**（Input Sharing Degree）
   ```
   Sharing_Score = |Shared_Inputs| / |Total_Inputs|
   融合收益 ∝ Sharing_Score × Input_Size
   ```

2. **寄存器压力**（Register Pressure）
   ```
   融合前：每个 op 使用 R 寄存器
   融合后：同时存储所有物化需要 R × N 寄存器
   寄存器溢出会抵消融合收益
   ```

3. **并行度权衡**（Parallelism Trade-off）
   ```
   融合前：N 个 kernel 并行执行（GPU 上可并发）
   融合后：1 个 kernel，但每个 thread 计算所有输出
   ```

##### MLIR 实现方案

| 方案 | Dialect | Pass | 适用场景 |
|------|---------|------|----------|
| Linalg 融合 | `linalg` | `-linalg-fuse-elementwise-ops` | 结构化算子 |
| Affine 循环融合 | `affine` | `-affine-loop-fusion` | 低层循环优化 |
| IREE Flow 融合 | `flow` | `iree-codegen` | 端到端编译 |

**Linalg 融合机制**

```cpp
// mlir/lib/Dialect/Linalg/Transforms/Fusion.cpp
// 融合策略：识别共享输入的多个 generic op

struct MultiOutputFusionStrategy {
  // 1. 构建输入使用图（Input Use Graph）
  struct UseGraph {
    DenseMap<Value, SmallVector<Operation*>> inputToUsers;
  };

  // 2. 识别可融合的兄弟操作组
  SmallVector<SmallVector<Operation*>> identifyFusionGroups(Operation *root) {
    // 找出共享相同输入的操作集合
    // 验证迭代空间兼容性
    // 评估寄存器压力
  }

  // 3. 生成融合后的 linalg.generic
  linalg::GenericOp buildFusedOp(ArrayRef<Operation*> ops) {
    // 合并 indexing_maps
    // 合并 iterator_types
    // 合并 region（计算逻辑）
  }
};
```

##### MLIR 示例：Attention QKV 融合

**融合策略**：将三个独立的矩阵乘法 $X \times W_q, X \times W_k, X \times W_v$ 合并为一个大的矩阵乘法 $X \times W_{qkv}$。

```cpp
// 融合前
func.func @qkv_projection_unfused(
    %X: tensor<Seq x Hiddenxf32>,        // [Seq, Hidden]
    %W_q: tensor<Hidden x (Heads x Head_Dim)xf32>,
    %W_k: tensor<Hidden x (Heads x Head_Dim)xf32>,
    %W_v: tensor<Hidden x (Heads x Head_Dim)xf32>)
    -> (tensor<Seq x (Heads x Head_Dim)xf32>,
        tensor<Seq x (Heads x Head_Dim)xf32>,
        tensor<Seq x (Heads x Head_Dim)xf32>) {

  // Q = X @ W_q
  %Q = linalg.matmul ins(%X, %W_q : ...) outs(...)
  // K = X @ W_k
  %K = linalg.matmul ins(%X, %W_k : ...) outs(...)
  // V = X @ W_v
  %V = linalg.matmul ins(%X, %W_v : ...) outs(...)

  return %Q, %K, %V
}

// 融合后（Concat-then-Matmul 策略）
func.func @qkv_projection_fused(
    %X: tensor<Seq x Hiddenxf32>,
    %W_qkv: tensor<Hidden x (3 x Heads x Head_Dim)xf32>)  // 权重拼接
    -> tensor<Seq x 3 x Heads x Head_Dimxf32> {

  // 单次矩阵乘法：[Seq, Hidden] @ [Hidden, 3*Heads*Head_Dim]
  // 输出形状：[Seq, 3, Heads, Head_Dim] = [Seq, Q/K/V, Heads, Head_Dim]
  %QKV = linalg.generic {
    indexing_maps = [
      affine_map<(s, h) -> (s, h)>,           // X
      affine_map<(s, h) -> (h, 3, n, d)>,     // W_qkv
      affine_map<(s, h) -> (s, 3, n, d)>      // QKV output
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%X, %W_qkv : ...)
    outs(%init : tensor<Seq x 3 x Heads x Head_Dimxf32>) {
  ^bb0(%x: f32, %w: f32):
    // GEMM 计算
    %prod = arith.mulf %x, %w : f32
    linalg.yield %prod : f32
  }

  // 后续可通过 tensor.extract_slice 分离 Q、K、V
  return %QKV : tensor<Seq x 3 x Heads x Head_Dimxf32>
}
```

**IREE Flow 方言实现**

*注：扩展阅读 [IREE的Flow方言如何高效实现Attention的QKV计算？](https://www.cnblogs.com/notlate-cn/p/19518938)*

---

#### 1.2.2 Batch/SIMD Fusion

##### 背景

将批次（Batch）维度作为并行维度，利用 SIMT（GPU）或 SIMD（CPU）指令并行处理多个独立样本。

##### 应用场景

*   **批量推理**：将多个请求合并为一个 Batch 执行，提升 GPU 利用率。
*   **Ascend NPU**：利用多核并行处理 Batch 维度。
*   **CPU 向量化**：将 Batch 映射到 AVX-512 向量通道。

##### MLIR 实现

```cpp
// Batch 维度提升到迭代空间
func.func @batch_fused_relu(%input: tensor<BxMxNxf32>) {
  %output = linalg.generic {
    indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                     affine_map<(b, m, n) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel"]  // B 也是并行维度
  } ins(%input : ...) outs(...) {
    ^bb0(%x: f32):
      %zero = arith.constant 0.0 : f32
      %cmp = arith.cmpf ogt, %x, %zero : f32
      %result = arith.select %cmp, %x, %zero : f32
      linalg.yield %result
  }
}
```

---

### 1.3 模式融合（Pattern Fusion）

#### 1.3.1 Special Pattern Fusion

##### 背景

针对深度学习中频繁出现的固定子图模式，直接替换为高度优化的实现。

##### 常见模式

| 模式名称 | 操作序列 | 优化机会 |
|----------|----------|----------|
| **Conv-BN-ReLU** | Conv2d → BatchNorm → ReLU | 编译期将 BN 参数折叠进 Conv 权重 |
| **MatMul-Bias-Act** | MatMul → BiasAdd → Activation | Epilogue 融合（详见 6.1.2） |
| **Softmax** | Max → Subtract → Exp → Sum → Div | Online Softmax（详见 1.1.3） |
| **LayerNorm** | Mean → Variance → Normalize | Welford One-pass 算法 |

##### MLIR 示例

使用 `PatternRewriter` 进行图匹配与重写：

```cpp
// 使用 PatternRewriter 进行模式匹配融合
struct ConvBNReLUPattern : public OpRewritePattern<tensor::PackOp> {
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    // 1. 匹配模式：Conv → BatchNorm → ReLU
    auto conv = packOp.getInput().getDefiningOp<linalg::Conv2DNhwcHwcfOp>();
    auto bn = /* 获取 BatchNorm op */;
    auto relu = /* 获取 ReLU op */;

    if (!conv || !bn || !relu) return failure();

    // 2. 转换 BatchNorm 参数到卷积权重
    auto fusedWeights = fuseBatchNormIntoConv(conv, bn);

    // 3. 生成融合后的 Conv + ReLU
    auto fusedOp = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
        conv.getLoc(), conv.getInput(), fusedWeights, ...);

    // 4. 添加 ReLU 到 region
    addReLUToRegion(fusedOp);

    return success();
  }
};
```

---

#### 1.3.2 Transformer Block Fusion (Attention & FFN)

##### 背景

在 LLM 时代，仅仅融合 `MatMul+Bias+Act` 已不足以满足性能需求。**Block Fusion** 旨在打破算子边界，将 Transformer 层中的关键子图（如 `Attention` 块或 `FFN` 块）完全融合为一个或极少数几个 Kernel，以最大化 SRAM 利用率并减少 HBM 访问。

> Transformer 中的 **Multi-Head Attention (MHA)** 是性能瓶颈。
>
> 标准实现（`MatMul(Q,K) -> Softmax -> MatMul(S,V)`）存在两大痛点：
>
> 1.  **$O(N^2)$ 内存复杂度**：中间生成的 Attention Score 矩阵形状为 $[Batch, Heads, Seq, Seq]$。当序列长度 $Seq$ 增长时，该矩阵占用的显存呈平方级增长。
> 2.  **Memory Wall（内存墙）**：在传统执行模式下，这个巨大的 $N \times N$ 矩阵需要被完整写入全局内存（HBM），再重新读回以进行 Softmax 和下一次 MatMul。这种频繁的 HBM 读写导致的延迟远超计算本身的耗时（IO-bound）。
>
> 因此，MHA 融合不仅仅是简单的算子合并，而是一种**IO 感知的算法级融合（IO-aware Algorithmic Fusion）**，旨在利用 Tiling 技术完全消除 $N \times N$ 矩阵对 HBM 的访问。

##### 典型模式

1.  **FlashAttention (v2/v3)**：不仅融合了 Softmax，还进一步将 **Dropout**、**Mask Generation** 甚至 **RoPE (Rotary Embedding)** 融合到 Attention 的 Forward/Backward 循环中。
2.  **SwiGLU Fusion (FFN)**：LLM（如 LLaMA）常用的 FFN 包含三个矩阵乘。融合策略是将两个并行的 MatMul（Gate proj 和 Up proj）合并计算，并在寄存器中直接完成 `SiLU` 和 `Element-wise Mul`，避免中间宽矩阵写回显存。

##### 核心原理

通过 **Tiling（分块）** 和 **Recomputation（重计算）**，将所有计算限制在片上 SRAM 中进行，完全消除 $O(N^2)$ 的物化物化。

```cpp
// 传统实现 (Standard Attention):
1. S = Q @ K^T          (Write S to HBM, size N^2)
2. P = Softmax(S)       (Read S, Write P to HBM, size N^2)
3. O = P @ V            (Read P, Write O to HBM)

// 融合后 (FlashAttention / Memory-Efficient Attention):
Block-wise loop:
  Load block of Q, K, V into SRAM
  Compute block of S = Q_i @ K_j^T (on SRAM)
  Compute block of P = Softmax(S)  (on SRAM, using Online Softmax)
  Compute block of O += P @ V_j    (on SRAM, accumulate to Output)
  (中间矩阵 S 和 P 从未离开过片上 SRAM)
```

##### MLIR 示例：Tiled Attention Logic

在 MLIR 中，这表现为带有 `iter_args` 的双层嵌套循环：

```cpp
func.func @flash_attention_tiled(%Q: tensor<...>, %K: tensor<...>, %V: tensor<...>) {
  // 外层循环：遍历 Query 分块
  %res = scf.for %i = 0 to %SeqLen step %Br 
    iter_args(%O_acc = %init_O, %m_acc = %init_m, %l_acc = %init_l) {
    
    // 加载 Q 到 SRAM (逻辑上)
    %Qi = tensor.extract_slice %Q[...] 

    // 内层循环：遍历 Key/Value 分块
    %O_row, ... = scf.for %j = 0 to %SeqLen step %Bc 
      iter_args(%O_curr = %O_acc, ...) {
      
      // 1. Compute Scores (Q @ K.T)
      %S_ij = linalg.matmul ins(%Qi, %Kj) ...
      
      // 2. Online Softmax Logic (更新 max 和 sum)
      %m_new = arith.maxf %m_curr, %local_max
      %P_ij = ... // exp(S_ij - m_new)
      
      // 3. Accumulate Output (P @ V)
      %O_new = linalg.matmul ins(%P_ij, %Vj) ...
      
      scf.yield %O_new, %m_new, ...
    }
    scf.yield %O_row, ...
  }
  return %res
}
```

**MLIR 示例：SwiGLU Fusion**

```cpp
// SwiGLU 融合示意：Gate_Proj 和 Up_Proj 的后处理融合
// 假设 %gate_out 和 %up_out 是两个 MatMul 的输出 (或者通过 Grouped MatMul 产生)
// 这里的融合消除了两个巨大的中间 Tensor 的 HBM 写回

func.func @swiglu_epilogue_fused(%gate_buf: tensor<?x?xf32>, %up_buf: tensor<?x?xf32>) 
    -> tensor<?x?xf32> {
  
  %res = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>, // Gate input
      affine_map<(d0, d1) -> (d0, d1)>, // Up input
      affine_map<(d0, d1) -> (d0, d1)>  // Output
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%gate_buf, %up_buf : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) {
    
    ^bb0(%g: f32, %u: f32, %out: f32):
      // 1. Swish/SiLU 激活: x * sigmoid(x)
      %sigmoid = math.sigmoid %g : f32
      %act = arith.mulf %g, %sigmoid : f32
      
      // 2. Gated Multiplication: act * up
      %res = arith.mulf %act, %u : f32
      
      linalg.yield %res : f32
  }
  return %res
}
```

*注：扩展阅读 [MLIR如何高效实现Attention？](https://www.cnblogs.com/notlate-cn/p/19522984)*

#### 1.3.3 Optimizer Fusion (Step Fusion / 优化器融合)

##### 背景

在深度学习训练中，反向传播结束后需要执行参数更新（Optimizer Step）。
以主流的 **AdamW** 优化器为例，它包含一系列密集的 Element-wise 操作：计算一阶矩（Momentum）、二阶矩（Variance）、权重衰减（Weight Decay）以及最终的参数更新。
如果这些操作作为独立的 Kernel 执行，将导致严重的**内存带宽浪费**：参数和状态量需要被反复读写 HBM（High Bandwidth Memory），而计算量却很小（Memory-bound）。

##### 技术原理

**Kernel Fusion (Multi-input/Multi-output)**：
编译器将优化器的整个计算逻辑融合为一个单一的 Kernel。该 Kernel 一次性从内存读取 Parameter, Gradient, Momentum, Variance，在寄存器中完成所有数学运算，然后一次性写回更新后的值。

##### MLIR 实现：Fused AdamW

```cpp
// 融合后的 AdamW Update Step
// 输入：Params, Grads, Exp_Avg (m), Exp_Avg_Sq (v)
// 输出：更新后的 Params, m, v
func.func @fused_adamw(%param: tensor<?xf32>, %grad: tensor<?xf32>, 
                       %m: tensor<?xf32>, %v: tensor<?xf32>,
                       %lr: f32, %beta1: f32, %beta2: f32, %eps: f32, %decay: f32) 
                       -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
  
  // 使用 linalg.generic 定义多输入多输出的融合算子
  %res:3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>, // param
      affine_map<(d0) -> (d0)>, // grad
      affine_map<(d0) -> (d0)>, // m
      affine_map<(d0) -> (d0)>, // v
      affine_map<(d0) -> (d0)>, // out_param
      affine_map<(d0) -> (d0)>, // out_m
      affine_map<(d0) -> (d0)>  // out_v
    ],
    iterator_types = ["parallel"]
  } ins(%param, %grad, %m, %v : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) 
    outs(%param, %m, %v : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
    
    ^bb0(%p: f32, %g: f32, %m_in: f32, %v_in: f32, ...):
      // 1. Weight Decay: g = g + p * decay
      %p_decay = arith.mulf %p, %decay : f32
      %g_prime = arith.addf %g, %p_decay : f32

      // 2. Update Momentum (m): m = beta1 * m + (1-beta1) * g
      %m_beta = arith.mulf %m_in, %beta1 : f32
      %one_minus_beta1 = arith.constant ... : f32
      %g_term = arith.mulf %g_prime, %one_minus_beta1 : f32
      %m_out = arith.addf %m_beta, %g_term : f32

      // 3. Update Variance (v): v = beta2 * v + (1-beta2) * g^2
      // ... (省略部分计算细节) ...
      %v_out = ... 

      // 4. Update Parameter
      // p = p - lr * m / (sqrt(v) + eps)
      %p_new = ...

      // 5. 同时返回三个更新后的值
      linalg.yield %p_new, %m_out, %v_out : f32, f32, f32
  }
  return %res:3
}
```

---

## 2. **循环与迭代空间优化（Loop & Iteration Space Optimization）**

在确定了算子间的依赖拓扑（第1章）之后，编译器的优化视角从宏观的数据流图深入到**计算内核（Kernel）内部**的执行逻辑。

本章关注**迭代空间**（Iteration Space）的变换与重组。其核心目标是通过调整**计算时序**（Temporal Order），来提升**时间局部性**（Temporal Locality），减少循环控制开销（Loop Overhead），并最大化指令级并行度（ILP）。编译器通过分析循环嵌套结构和跨迭代的数据依赖，决定如何合并循环范围、在寄存器中复用跨迭代状态，以及如何在指令流水线上掩盖计算延迟。

> 本章所有技术，本质上都是在对 Iteration Space（迭代空间） 做代数变换：
> * 合并（Fusion）
> * 切分（Tiling）
> * 重排（Reordering）
> * 提升维度（Loop Expansion）
> * 消除维度（Reduction）

### 2.1 循环融合（Loop Fusion）

#### 2.1.1 Loop Fusion & Tiling

##### 背景

在底层 IR（如 Affine 或 LLVM IR）中，**循环融合**是指将两个具有相同或兼容**迭代空间**的相邻循环合并为一个循环体的代码变换技术。

与图层面的算子融合不同，循环融合更关注**指令执行流**的优化。其核心收益不仅仅是减少全局内存访问，还包括：
1.  **提升时间局部性（Temporal Locality）**：将数据的**"定义"**与**"使用"**拉近，使其在**寄存器文件**或 **L1/L2 Cache** 中保持活跃（Hot），避免被驱逐。
2.  **减少循环控制开销**：减少了循环计数器的增量指令、条件分支跳转指令（Branch）以及未命中的分支预测惩罚。
3.  **隐式同步消除**：在并行编程（如 OpenMP 或 CUDA）中，两个独立的循环之间通常隐含一个**同步信号（Barrier）**。融合后，这个屏障被消除，减少了线程空转等待。

##### 技术原理

**Loop Fusion (循环融合)** 与 **Tiling (循环分块)** 通常是协同工作的（统称为 **Tile-and-Fuse**），其核心原理涉及以下三个层面：

1.  **最大化时空局部性 (Maximizing Locality)**：
    *   **Temporal Locality (时间局部性)**：融合（Fusion）将“生产者”和“消费者”的距离拉近。在生产者计算出 `A[i]` 后，消费者立刻使用它。如果这个时间间隔足够短，`A[i]` 仍驻留在 **寄存器** 或 **L1 Cache** 中，避免了写入 DRAM 再读回的开销。
    *   **Spatial Locality (空间局部性)**：分块（Tiling）将巨大的迭代空间切割成适应 Cache Line 大小的“小块”。这确保了在处理当前块时，相关的数据能装入 Cache 且不发生**抖动（Thrashing）**。

2.  **变换机制 (Transformation Mechanics)**：
    编译器通常遵循 **"Strip-Mine, Interchange, and Fuse"** 的策略：
    *   **Strip-Mining (条带化)**：将一个大循环 `i` 拆分为外层循环 `i_outer` 和内层循环 `i_inner`。
    *   **Loop Interchange (循环交换)**：调整 `outer` 和 `inner` 循环的顺序，将多个算子的 `outer` 循环对齐。
    *   **Fusion (融合)**：当两个算子的外层循环（Tile Loop）一致时，合并它们的循环体，使得在一个 Tile 内完成一系列操作。

3.  **合法性检查 (Legality Check)**：
    并非所有循环都能融合。编译器（如基于多面体模型的 Polyhedral Compiler）必须构建**依赖图（Dependency Graph）**，检查是否存在：
    *   **RAW (Read-After-Write) 依赖**：融合不能破坏数据的生产-消费顺序。
    *   **负距离依赖 (Negative Distance)**：如果消费者在生产者之前执行（由于错误的 Tiling 方向），会导致计算错误。

##### C++ 示例

```cpp
// 融合前
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++)
    A[i][j] = B[i][j] + C[i][j]; // 写入内存

for (i = 0; i < N; i++)
  for (j = 0; j < M; j++)
    D[i][j] = A[i][j] * 2;       // 读取内存

// 融合后
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++) {
    float temp_a = B[i][j] + C[i][j]; // 临时变量，驻留寄存器
    // A[i][j] = temp_a;              // 可选：若 A 不再被使用，则消除此写操作 (Store Elimination)
    D[i][j] = temp_a * 2;
  }
```

##### MLIR Affine 实现

```cpp
// 融合前
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %a = affine.load %A[%i, %j] : memref<100x100xf32>
    %b = affine.load %B[%i, %j] : memref<100x100xf32>
    %sum = arith.addf %a, %b : f32
    affine.store %sum, %C[%i, %j] : memref<100x100xf32>
  }
}

affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %c = affine.load %C[%i, %j] : memref<100x100xf32>
    %d = arith.mulf %c, %c : f32
    affine.store %d, %D[%i, %j] : memref<100x100xf32>
  }
}

// 融合后（-affine-loop-fusion）
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %a = affine.load %A[%i, %j] : memref<100x100xf32>
    %b = affine.load %B[%i, %j] : memref<100x100xf32>
    %sum = arith.addf %a, %b : f32
    // %C 的 store 操作被消除，物化 %sum 直接用于下一步计算
    %d = arith.mulf %sum, %sum : f32
    affine.store %d, %D[%i, %j] : memref<100x100xf32>
  }
}
```

---

### 2.2 **跨迭代状态优化（Cross-iteration State Optimization）**

此类优化关注跨越时间步（Time Step）或循环迭代（Iteration）的数据状态管理。

#### 2.2.1 Loop-carried Scalar Replacement (Accumulator Fusion)

##### 背景

某些循环带有**循环携带依赖**（Loop-carried Dependency），如求和累加器（Summation）或最大值扫描（Max Scan）。融合时必须显式维护这些状态，防止中间结果溢出到内存。

##### 技术原理

**标量替换（Scalar Replacement of Aggregates, SROA）** 与 **寄存器提升（Register Promotion）** 是实现累加器融合的底层机制：

1.  **消除读-改-写循环（Read-Modify-Write Elimination）**：
    *   *未优化*：在循环的每一次迭代中，执行 `Load(Mem) -> Add -> Store(Mem)`。这会导致 $2N$ 次内存访问。
    *   *优化后*：编译器识别出该变量具有**循环携带依赖（Loop-carried Dependency）**，将其“提升”为寄存器变量。循环变为 `Reg = Reg + Val`，仅在循环结束后执行一次 `Store(Mem)`。内存访问降为 2 次（读初值 + 写终值）。

2.  **SSA 形式与 Phi 节点（SSA & Phi-Nodes）**：
    在现代编译器 IR（如 MLIR/LLVM）中，寄存器状态通过 **SSA（静态单赋值）** 形式建模。
    *   循环的**入口参数（Block Arguments / iter_args）**充当了 **Phi 节点**，负责合并来自“循环入口”的初始值和“循环回边（Back-edge）”的更新值。
    *   后端寄存器分配器（Register Allocator）会直接将这些 SSA 值映射到物理累加寄存器（如 GPU 的 `R` 或 `Accum` 寄存器）。

##### 收益
1.  **消除内存流量**：对于长度为 $N$ 的循环，消除 $2N$ 次内存读写。
2.  **打破延迟链**：通过寄存器直接转发数据，缩短指令间的依赖延迟（Latency Hiding）。

##### MLIR 实现方案

MLIR 的 `scf.for` 和 `affine.for` 引入了 **`iter_args`** 机制，这是显式表达寄存器驻留状态的各种 IR 中的最佳实践。它将可变变量（Mutable Variables）转化为 SSA 值（Immutable Values）的传递。

##### MLIR 示例

```cpp
// 场景：计算点积 Dot(A, B)
// 优化前（概念）：频繁读写内存
%sum_ptr = memref.alloc() : memref<f32>
scf.for %i = 0 to 1024 {
  %old_sum = memref.load %sum_ptr[] : memref<f32>
  %prod = ...
  %new_sum = arith.addf %old_sum, %prod
  memref.store %new_sum, %sum_ptr[] : memref<f32> // 瓶颈：写回内存
}

// 优化后（MLIR iter_args）：全寄存器操作
// %sum_iter 在编译后直接映射为物理寄存器
%final_sum = scf.for %i = 0 to 1024 
  iter_args(%sum_iter = %initial_sum) -> (f32) {
  
  %a = load %A[%i]
  %b = load %B[%i]
  %prod = arith.mulf %a, %b : f32
  
  // 状态更新仅发生在 SSA 值之间
  %sum_next = arith.addf %sum_iter, %prod : f32
  
  // 将新状态传递给下一次迭代
  scf.yield %sum_next : f32
}
```

#### 2.2.2 Temporal Fusion (RNN/LSTM Cell Fusion)

##### 背景
循环神经网络（RNN/LSTM/GRU）包含一个时间维度的循环：$h_t = Cell(x_t, h_{t-1})$。
*   **未融合（Unfused）**：通常实现为“循环展开”或多次 Kernel 启动。每一步计算完 $h_t$ 后，将其写入全局内存，下一步再读出。对于长序列（如 $T=1000$），这会产生巨大的内存带宽开销。
*   **时间融合（Temporal Fusion）**：将整个时间循环编译为一个 Kernel。状态变量 $h_t$ 和 $c_t$ 通过**寄存器（Register）**或**共享内存**在迭代间直接传递，只在最后一步输出结果或在必要时保存历史用于反向传播。

##### 技术原理

**Loop-Carried Dependency Optimization**：
利用 MLIR 的 `scf.for`（结构化控制流循环）中的 `iter_args` 机制，显式地表达跨迭代的数据依赖。后端编译器（如 LLVM 或 NVVM）会将这些 `iter_args` 提升为物理寄存器（Register Promotion），从而消除循环间的 Load/Store。

##### MLIR 实现：LSTM State Persistence

```cpp
// 融合后的 LSTM 时间循环
// 输入：整个输入序列 %seq [SeqLen, Batch, InputDim]
//       初始状态 %h_init, %c_init
// 输出：最终状态 %h_final, %c_final
func.func @lstm_temporal_fused(%seq: tensor<?x?x?xf32>, 
                               %h_init: tensor<?x?xf32>, 
                               %c_init: tensor<?x?xf32>,
                               %weights: tensor<...>) 
                               -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  
  // 获取序列长度 T
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %T = tensor.dim %seq, %c0 : tensor<?x?x?xf32>

  // 核心融合：scf.for 携带 iter_args
  // %iter_h, %iter_c 在循环迭代中驻留在寄存器/L1中，不写回 HBM
  %h_final, %c_final = scf.for %t = %c0 to %T step %c1 
      iter_args(%curr_h = %h_init, %curr_c = %c_init) 
      -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    
    // 1. 读取当前时刻输入 x[t]
    %xt = tensor.extract_slice %seq[%t, 0, 0] [...] : tensor<?x?x?xf32> to tensor<?x?xf32>

    // 2. [Cell Fusion] 融合 LSTM Cell 内部的矩阵乘与激活函数
    // MatMul: gates = x[t] @ W + curr_h @ R + bias
    // 这一步通常也会应用 Element-wise Chain Fusion
    %gates_raw = linalg.matmul ins(%xt, %curr_h, %weights : ...) ...

    // 3. 计算 Gate Activation (Sigmoid/Tanh) 和状态更新
    // next_c = f * curr_c + i * g
    // next_h = o * tanh(next_c)
    %next_c = linalg.generic ... ins(%gates_raw, %curr_c) ... 
    %next_h = linalg.generic ... ins(%gates_raw, %next_c) ...

    // 4. 将更新后的状态 Yield 给下一次迭代
    // 编译器后端将其识别为 Loop-carried variable，分配寄存器
    scf.yield %next_h, %next_c : tensor<?x?xf32>, tensor<?x?xf32>
  }

  return %h_final, %c_final
}
```

##### 代码解析

1.  **`scf.for ... iter_args(%curr_h, %curr_c)`**：这是时间融合的关键。它告诉编译器 `%curr_h` 和 `%curr_c` 是随着循环演进的变量。在生成机器码时，这些变量会被映射到寄存器（如 GPU 的 RF 或 CPU 的 Vector Regs）。
2.  **`linalg.generic` (Cell内部)**：LSTM Cell 内部的复杂门控逻辑（Sigmoid, Tanh, Element-wise Mul/Add）被融合为大的计算块，避免中间结果物化。
3.  **零中间内存**：除了输入的 `%seq` 和权重的 `%weights`，中间产生的 $h_1, h_2... h_{t-1}$ 不需要写回主存（除非是为了训练时的反向传播 Checkpointing）。

#### 2.2.3 Gradient Accumulation & Implicit Transpose (梯度累积与隐式转置)

##### 背景

1.  **梯度累积 (Gradient Accumulation)**：为了在有限显存下模拟大 Batch Size，训练通常将一个大 Batch 切分为多个 Micro-Batch 串行执行，梯度需要跨迭代累加。
2.  **反向转置 (Backward Transpose)**：在计算 Linear 或 Conv 层对输入的梯度时（$dX = dY \cdot W^T$），数学上要求权重的转置。显式执行 `Transpose(W)` 会产生巨大的内存拷贝开销。

##### 技术原理

1.  **Accumulator Persistence (累加器驻留)**：
    编译器将梯度 Buffer 标记为 **Output Stationary**。在循环处理 Micro-Batch 时，将累加结果保留在 **寄存器** 或 **L2 Cache** 中，直到所有 Micro-Batch 处理完毕才刷回 HBM。
2.  **Implicit Transpose (隐式转置)**：
    编译器不生成物理转置，而是通过修改 MatMul 的 **Indexing Map（索引映射）**，指示硬件以“转置的步长”读取原始权重矩阵。

##### MLIR 实现

```cpp
// 1. Gradient Accumulation (跨迭代累加)
// scf.for 循环携带 iter_args (accumulated_grad)
%total_grad = scf.for %i = 0 to %num_micro_batches 
    iter_args(%acc_grad = %zero_grad) -> (tensor<?x?xf32>) {
  
  // 计算当前 Micro-Batch 的梯度
  %mb_grad = call @backward_step(%i)
  
  // 2. Implicit Transpose Fusion (在 MatMul 中融合转置)
  // 计算 dX = dY * W^T
  // 注意 indexing_maps 中 W 的访问顺序是 (d1, d0) 而非 (d0, d1)
  %dx = linalg.generic {
     indexing_maps = [
       affine_map<(d0, d1, d2) -> (d0, d2)>, // dY: [M, K]
       affine_map<(d0, d1, d2) -> (d1, d2)>, // W:  [N, K] (物理上是 [N, K]，逻辑上被视为转置)
       affine_map<(d0, d1, d2) -> (d0, d1)>  // dX: [M, N]
     ],
     iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%dy, %w : ...) outs(...) {
     ^bb0(%a: f32, %b: f32, %c: f32):
       %prod = arith.mulf %a, %b : f32
       %sum = arith.addf %prod, %c : f32
       linalg.yield %sum : f32
  }

  // 累加到总梯度
  %new_acc = arith.addf %acc_grad, %dx : tensor<?x?xf32>
  scf.yield %new_acc : tensor<?x?xf32>
}
```

---

### 2.3 **循环展开与流水线（Loop Unrolling & Pipelining）**

此技术通过重组循环内的指令调度，最大化硬件单元的利用率，主要解决指令流水线气泡和内存延迟问题。

#### 2.3.1 Loop Unrolling (Instruction Level Parallelism)

##### 背景

现代 CPU/GPU 拥有超标量架构（Superscalar），每个时钟周期可发射多条指令。紧凑的循环（Tight Loop）由于频繁的分支跳转检测（Branch Compare & Jump），会打断指令流水线。
**循环展开**通过复制循环体代码，减少跳转次数，并暴露更多的独立指令供硬件调度器进行**指令级并行（ILP）**优化。

##### 技术原理 (通用架构)
循环展开的性能收益主要来自三个微观层面：

1.  **摊薄控制流开销 (Amortizing Control Overhead)**：
    原始循环每执行一次计算，都要执行一次“比较”和“跳转”。展开 $K$ 次后，这 $K$ 次计算共享同一个跳转指令。控制指令在总指令数中的占比大幅下降，显著减轻了 CPU/GPU **分支预测器（Branch Predictor）** 的压力。

2.  **暴露指令级并行 (Exposing ILP)**：
    现代处理器（超标量或 VLIW 架构）拥有多个执行端口（如多个 ALU 和 Load/Store 单元）。展开后，来自不同迭代的独立指令（如 `i` 和 `i+1`）被同时暴露给硬件调度器，可以**同时发射**到不同的执行单元，填满硬件流水线的空槽（Slots）。

3.  **扩大调度窗口 (Expanding Scheduling Window)**：
    展开增加了**基本块（Basic Block）**的大小。这给编译器提供了更大的指令重排空间，使其能够将“高延迟指令（如 Load）”与其“依赖指令（Use）”拉开距离，从而利用乱序执行（Out-of-Order）掩盖内存延迟。

##### 硬件视角：Ascend NPU 的特殊优化

在 **Ascend NPU (达芬奇架构)** 上，循环展开的意义与通用 CPU/GPU 有所不同，它主要为了适配 **Vector Repeat** 机制和掩盖**异构核心通信延迟**：

1.  **Repeat 指令映射 (Hardware Repeater Mapping)**：
    Ascend ISA 支持强大的 **Repeat 机制**。一条 Vector 指令（如 `vadd`）可以通过设置 `repeat_times` 和 `stride`，一次性处理多达 255 个连续数据块。
    *   *编译器策略*：编译器先在 IR 层面展开循环，识别出连续的计算模式，然后在后端代码生成时将其**再折叠（Re-roll）**为单条带 Repeat 的指令。这极大地减少了取指译码（Fetch/Decode）带宽。

2.  **掩盖 Scalar-Vector 依赖 (Hiding Scalar-Vector Latency)**：
    AICORE 是异构多核架构：**Scalar Unit** 负责控制流和地址计算，**Vector Core** 负责密集运算。
    *   *优化逻辑*：通过 Loop Unrolling，编译器可以生成一个“大工作量”的 Vector 指令块（或 Repeat 指令）。当 Vector Core 忙于处理这就绪的数据块时，Scalar Core 可以并行地计算下一批数据的地址。这种**解耦（Decoupling）** 确保了计算流水线永不干涸。

##### 收益

1.  **减少分支开销**：$N$ 次迭代变成 $N/K$ 次跳转。
2.  **向量化机会**：展开后的连续访存指令更容易被合并为向量加载（Vector Load）。

##### MLIR 实现

在 `affine` 或 `scf` 方言中，可以通过属性标记或变换 Pass 显式控制展开因子。

```cpp
// 原始循环
affine.for %i = 0 to 1024 {
  %x = affine.load %A[%i]
  %y = affine.load %B[%i]
  %z = arith.addf %x, %y : f32
  affine.store %z, %C[%i]
}

// 优化后：Unroll Factor = 4
// 1. 控制流指令减少 4 倍
// 2. 暴露出 4 个独立的 Load/Add/Store，利于 SIMD/SIMT 打包
affine.for %i = 0 to 1024 step 4 {
  // Iteration i
  %x0 = affine.load %A[%i]
  %z0 = arith.addf ... 
  
  // Iteration i+1
  %x1 = affine.load %A[%i + 1]
  %z1 = arith.addf ...
  
  // ... i+2, i+3 ...
  
  // Ascend 后端识别：
  // 发现 %x0, %x1... 地址连续，%z0, %z1... 计算逻辑一致
  // -> 可能会合并生成一条 Vector Load 和一条 Vector Add (with Repeat)
}
```

#### 2.3.2 Software Pipelining (Latency Hiding)

##### 背景
在深度学习算子（如 GEMM, Attention）中，从全局内存（HBM）加载数据到片上缓存（SRAM/Register）的延迟极高（数百个时钟周期）。

如果采用 **Load $\to$ Compute** 的串行模式，计算单元在等待数据时会空转。
**软件流水线**（配合双缓冲/多缓冲 Double Buffering）将不同迭代的阶段重叠执行：在计算当前块（Tile $i$）的同时，预取下一块（Tile $i+1$）的数据。

##### 技术原理

流水线变换将循环重构为三个部分：
1.  **Prologue（序言）**：预取第 0 次迭代的数据。
2.  **Steady State（稳态/核）**：同时执行第 $i$ 次计算和第 $i+1$ 次加载。
3.  **Epilogue（尾声）**：完成最后一次迭代的计算。

```
Timeline:
Iter 0: [Load 0]
Iter 1:          [Comp 0] [Load 1]  <-- 并行执行 (Overlap)
Iter 2:                   [Comp 1] [Load 2]
```

##### MLIR 实现 (Async Pipelining)
MLIR 的 `scf.for` 配合 `iter_args` 和异步指令（如 `nvgpu.device_async_copy`）可以完美表达这种模式。

```cpp
// 软件流水线化后的循环结构
// 初始阶段：预加载第 0 个 Tile (Prologue)
%token0 = gpu.async_copy %Global[%c0] to %Shared[%c0] ...

// 循环携带 token 状态 (Stage $i$ 的加载句柄传递给 Stage $i+1$)
scf.for %i = 0 to %N step %TileSize 
  iter_args(%token_prev = %token0) {
  
  // 1. 发起第 $i+1$ 个 Tile 的异步加载 (Prefetch)
  %next_idx = arith.addi %i, %TileSize
  %token_next = gpu.async_copy %Global[%next_idx] to %Shared[%buffer_next] ...
  
  // 2. 等待第 $i$ 个 Tile 加载完成 (Wait)
  gpu.wait %token_prev
  
  // 3. 计算第 $i$ 个 Tile (Compute)
  // 此时计算与步骤 1 的加载是并行的
  linalg.matmul ins(%Shared[%buffer_curr] ...)
  
  // 4. 传递下一轮的 token
  scf.yield %token_next
}

// 尾声：计算最后一个 Tile (Epilogue)
gpu.wait %token_last
linalg.matmul ...
```

---

## 3. 数据布局与表示（Data Layout & Representation）

当计算逻辑（第1、2章）确定后，性能的瓶颈往往转移到数据的**物理组织形式**上。

本章关注张量数据在内存地址空间中的**排布与访问模式**。核心目标是使数据的物理布局与硬件的访问特性（如 SIMD 通道、Cache Line）相匹配，以最大化**空间局部性（Spatial Locality）**。编译器需要从全图范围选择最优的逻辑布局（如 NCHW vs NHWC），并在微观层面执行**数据打包（Packing）**、**对齐（Padding）**和**缓冲区化（Bufferization）**，以消除因格式不匹配导致的昂贵的数据重排开销。

### 3.1 全局布局优化（Global Layout Optimization）

#### 3.1.1 Layout Propagation & Assignment(布局传播与指派)

##### 背景

不同算子对内存布局有不同偏好（例如：Conv2d 在 GPU 上偏好 `NHWC`，在 NPU 上可能偏好 `NC1HWC0`）。
如果在图中频繁插入 `Transpose` 或 `Reshape`，数据搬运开销将抵消计算收益。**布局传播**旨在为整个子图选择统一的最佳布局。

##### 技术原理

布局优化本质上是一个**全局约束满足问题（Global Constraint Satisfaction Problem）**或**最小代价路径搜索**问题。其技术实现通常包含三个步骤：

1.  **约束定义与硬件亲和性（Hardware Affinity）**：
    编译器首先标记每个算子对布局的“偏好”。
    *   **GPU Tensor Core**：偏好 **NHWC**（Channel-last）。原因在于 Tensor Core 计算矩阵乘时，需要 Inner Dimension（通常是 Channel）在内存中连续，以实现合并访问（Memory Coalescing）并利用向量化加载指令（如 `LDG.128`）。
    *   **Ascend Cube Core**：偏好 **Fractal (NC1HWC0/5D)**。这是为了适配矩阵单元内部的 16x16 分块读取逻辑。
    *   **CPU Vector Unit**：可能偏好 **NCHWc**（Blocked Channel），配合 SIMD 宽度。

2.  **双向传播（Bi-directional Propagation）**：
    从对布局敏感的"锚点算子"（如 Conv）开始（Source-Sink Propagate），编译器在计算图上执行前向和后向遍历，传播布局约束。
    
    *   **Forward**：输入数据的布局决定后续算子的布局（如 Input 是 NHWC，则 Conv 也选 NHWC）。
    *   **Backward**：最终输出的要求反推前序算子（如 Output 要求 NCHW，则最后一个 Conv 最好输出 NCHW）。
    *   **Layout Transform Elimination**：利用数学性质消除冗余转换（例如：`Transpose(Transpose(x)) == x`）。
    *   **冲突解决**：当一个算子的输入来自 NHWC，但自身强制要求 NCHW 时，传播受阻。
    
3.  **代价最小化求解（Cost Minimization）**：
    当发生冲突时，编译器需要在图的边（Edge）上插入 **Layout Transform (Transpose/Permute)** 算子。
    优化目标是最小化总代价：
    $$
     \text{Total Cost} = \sum \text{Op\_Execution\_Cost}(\text{Layout}) + \sum \text{Transform\_Overhead}
    $$
    编译器通常使用**动态规划**或**贪心算法**（如 Union-Find）来决定在何处“切一刀”插入转置，使得转置次数最少且算子效率最高。

##### MLIR 实现

在 MLIR 中，这通常发生在 `tosa` 到 `linalg` 的降级过程中，或者通过专门的 `LayoutOptimization` Pass 实现。

```cpp
// 原始图：未指定布局的 Conv2D -> Relu -> MaxPool
// 默认可能是 NCHW，但目标硬件（GPU）偏好 NHWC

func.func @layout_opt(%input: tensor<1x3x224x224xf32>) {
  
  // 1. [Layout Assignment] 
  // 编译器分析发现 Conv2D 在 NHWC 下能使用 TensorCore，收益极高
  // 因此决定插入 Layout Transform，并将后续算子全部染成 NHWC
  
  // 插入 NCHW -> NHWC
  %input_perm = tosa.transpose %input {perms = [0, 2, 3, 1]} 
              : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>

  // 2. [Propagation]
  // Conv2D 选用 NHWC 变体
  %conv = linalg.conv_2d_nhwc_hwcf ins(%input_perm, %weight) ...
  
  // ReLU 自动适配 NHWC (Element-wise 对布局不敏感，随大流)
  %relu = linalg.generic ... ins(%conv) ... 
  
  // MaxPool 选用 NHWC 变体
  %pool = linalg.pooling_nhwc_max ... ins(%relu) ...

  // 3. [Finalize] 如果需要 NCHW 输出，则在最后转回；否则直接输出 NHWC
  return %pool
}
```

##### 补充说明（Ascend NPU）

在 Ascend 场景下，布局传播尤为关键。如果传播失败，导致在计算图中频繁插入 `TransData`（格式转换算子），会严重阻塞流水线。

因此 Ascend 的编译器会极力推导 **DefaultFormat** 为 **5D (Fractal)**，并让尽可能多的算子（包括 ReLU, Add 等）直接在 5D 数据上运行，实现 **Layout-agnostic Fusion**。

---

### 3.2 数据打包与微布局（Data Packing & Micro-layout）

#### 3.2.1 Tensor Packing (Block Layout / Fractal Format)

##### 背景

标准的 **Row-major (行优先)** 或 **Column-major (列优先)** 布局在处理 2D/3D 卷积或矩阵乘法时，往往无法提供最佳的空间局部性。
**Tensor Packing (数据打包)** 指将高维张量的逻辑维度物理重排为**嵌套的块状结构（Blocked Structure）**。这种变换将逻辑上相邻的二维子矩阵（Tile）存储在物理连续的内存地址中。

##### 技术原理

1.  **最大化缓存行利用率 (Cache Line Utilization)**：
    
    **问题**：在 Row-major 中，访问矩阵的一列（Column stride）会导致巨大的内存跳跃，极易引发 Cache Miss。

    **原理**：Packing 将一个 $T_h \times T_w$ 的 Tile 里的数据连续存放。当 GPU/NPU 读取 Tile 的第一个元素时，整个 Cache Line（通常 64 Bytes、128 Bytes 或 256 Bytes）加载进来的数据正好是该 Tile 的后续元素。这使得二维的空间局部性转化为了物理内存的一维连续性。
    
2.  **向量化加载友好 (Vectorization Friendliness)**：
    
    SIMD/SIMT 单元通常一次加载 128/256/512 位。Block Layout 保证了数据的**Inner Dimension**（最内层维度）长度固定且对齐（例如总是 16 或 32），这允许编译器直接生成对齐的向量加载指令（Aligned Vector Load），无需处理边缘情况。

##### 硬件视角：Ascend NPU 的分形格式 (Fractal Format)

在 **Ascend NPU (达芬奇架构)** 上，Tensor Packing 不仅仅是优化，更是驱动 **Cube Unit (矩阵加速单元)** 的**前置条件**。

1.  **分形格式 (Fractal Layout - ZnZ/NC1HWC0)**：
    Ascend 的 Cube Unit 硬件上只能处理 $16 \times 16$ 的微矩阵（Fractal Block）。
    *   **NC1HWC0**: 针对 Activation（Feature Map），逻辑上的 `[N, C, H, W]` 被物理重排为 5D 格式 `[N, C1, H, W, C0]`。其中 `C0` 固定为 16 (FP16)，`C1 = C / 16`。
    *   **ZnZ (Z-in-Z)**: 针对 Weight（权重矩阵）。逻辑上的 2D 矩阵被重排为小“Z”字型嵌套大“Z”字型的结构，以确保 Cube 在计算矩阵乘时，内存读取是连续的。

2.  **强制对齐与补零 (Alignment & Padding)**：
    如果逻辑维度不能被 16 整除，编译器必须在 Packing 阶段隐式补零（Padding）。例如，`Channel=3` 的 RGB 图片在 Ascend 上物理占用实际上是 `Channel=16` 的空间（C0=16），虽然浪费了 13/16 的空间，但换取了 Cube 单元的极致吞吐。

##### MLIR 实现：`tensor.pack` / `tensor.unpack`
MLIR 引入了 `tensor.pack` 和 `tensor.unpack` 算子来显式表达这种布局变换。在 Ascend 编译流中，这通常对应于将 `linalg` 降级为 NPU 专用 Dialect 时的步骤。

```cpp
// 场景：为了适配 Ascend Cube Unit (16x16 FP16)，将 Row-major 转换为 Block Layout
// 逻辑形状：[1024, 1024]
// 物理形状：[64, 64, 16, 16] (Outer_M, Outer_N, Inner_m, Inner_n)

func.func @ascend_packing(%input: tensor<1024x1024xf16>) -> tensor<64x64x16x16xf16> {
  
  // 定义 Packing 策略
  // inner_dims_pos = [0, 1] 表示对两个维度都进行切分
  // inner_tiles = [16, 16] 对应 Cube Unit 的硬件限制
  %packed = tensor.pack %input
      inner_dims_pos = [0, 1]
      inner_tiles = [16, 16]
      : tensor<1024x1024xf16> -> tensor<64x64x16x16xf16>

  return %packed
}

// 融合说明：
// 在实际编译中，编译器会尝试将这个 tensor.pack 操作向上融合 (Fuse Up)
// 到产生 %input 的算子（如 Previous Conv/MatMul）的 Epilogue 中，
// 使得前一个算子直接写出 16x16 的分形格式，避免单独的内存搬运。
```

#### 3.2.2 Swizzled Layout (For Shared Memory)

##### 背景
在现代加速器（比如 GPU 和 NPU）中，片上高速缓存（Shared Memory / L1 / UB）通常被划分为多个**存储体（Banks）**（例如 32 个 Bank）。
当一个 Warp/Wavefront 中的多个线程同时访问不同地址，但这些地址映射到同一个 **Bank** 时，就会发生 **Bank Conflict（存储体冲突）**。硬件必须将这些访问**串行化（Serialization）**，导致有效带宽成倍下降。

##### 技术原理

**Swizzling（地址重排/混洗）** 是一种通过数学变换打乱数据在 Shared Memory 中物理存放顺序的技术，目的是让逻辑上连续的访问在物理上分散到不同的 Bank 中。

1. **XOR Swizzling (异或混洗)**：
   这是目前最主流的方案（NVIDIA Ampere+ 硬件原生支持）。

   *   **冲突场景**：假设 stride=32，Bank 映射通常是 `Address % 32`。如果线程 $T_0$ 访问 `A[0][0]`，$T_1$ 访问 `A[1][0]` (地址 32)，它们都会命中 Bank 0。
   *   **解法**：引入 XOR 映射。
       $$
       Bank_{Index}=(Col_{Index})⊕(Row_{Index} >> Shift)
       $$
   *   通过让列索引与行索引的高位进行异或，原本每一列都映射到 Bank 0 的局面被打乱，变为对角线式分布，从而消除冲突。

2. **Vectorized Access Optimization**：
   Swizzled Layout 还是为了适配硬件特殊的加载指令（如 NVIDIA 的 `ldmatrix`）。这些指令要求数据以特定的“交错”格式存放，以便一个 Warp 能够用一条指令加载一个 $8 \times 8$ 或 $16 \times 16$ 的矩阵块到 Tensor Core 寄存器。

3. **Ascend NPU: ？？？**

##### MLIR 示例：GPU 显式 Swizzling (NVGPU)

MLIR 通过 `nvgpu` 方言支持显式的 GPU Swizzling，而对于 Ascend，通常在 Bufferization 后的 Copy 阶段隐式处理。

```cpp
// 场景：将数据从 Global 拷贝到 Shared，启用 Swizzle 以优化 Tensor Core 读取
func.func @gpu_swizzle_copy(%global: memref<?xf16>, %shared: memref<?xf16, #gpu.address_space<workgroup>>) {
  // swizzle 属性指示编译器生成 XOR 地址计算逻辑
  %token = nvgpu.device_async_copy %global[%i] to %shared[%j] 
           dst_elements = 8 
           {bypass_l1, src_elements = 8} 
           : memref<?xf16> to memref<?xf16, #gpu.address_space<workgroup>>
  
  // 后续使用 ldmatrix 读取时，硬件会自动解码 Swizzled 地址
}
```

### 3.3 填充与对齐（Padding & Alignment）

#### 3.3.1 Dimension Padding

##### 背景
真实世界的模型 Tensor 维度往往是不规则的（如 Channel=3 或 Vocab=30522）。然而，高性能硬件单元通常有严格的**对齐要求**（Alignment Requirements）。如果不满足这些要求，硬件效率会大幅下降甚至无法运行。

##### 技术原理

**Padding（填充）** 的核心价值在于以少量的空间浪费换取**控制流简化**和**硬件兼容性**。

1.  **消除 Loop Tail (通用/GPU)**：
    如果矩阵尺寸不能被 Tiling Size 整除，编译器需要生成主循环和低效的余数循环（Loop Tail）。通过 Padding 将尺寸补齐（如补齐到 128 的倍数），编译器可以生成单一的完美循环，消除分支跳转，利于流水线满载。

2.  **强制 C0 对齐 (Ascend NPU 硬约束)**：
    达芬奇架构的 Cube Unit 物理上是一个 $16 \times 16$ 的阵列。它无法处理非 16 倍数的维度（如 `Channel=3`）。
    *   **On-the-fly Padding**：Ascend 编译器利用 **MTE** 引擎支持“搬运时填充”。从 HBM 读取 3 个数，MTE 在写入片上缓存时自动补 13 个 0。这实现了**零带宽开销**（HBM 带宽只消耗了 3 个数的量，虽然片上占了 16 个数的空间）。

3.  **避免 Stride Conflict (GPU)**：
    如果矩阵的 Stride 恰好是 Shared Memory Bank 数量的整数倍，会导致列访问冲突。Padding Stride（增加一个哑元）可以错开 Bank 映射。

##### MLIR 实现

`tensor.pad` 操作用于显式填充，通常在 Tiling 之前或之后进行，以确保每个 Tile 都是完整的（Full Tile）。

```cpp
// 场景：处理输入 Channel=3 的卷积，适配 Ascend C0=16 要求
func.func @ascend_padding_c0(%img: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224xf16> {
  
  // 显式 Padding：逻辑上将 C=3 扩展为 C=16
  // 这告诉编译器：计算循环的边界是 16，不是 3
  // low/high 指定了在维度前后填充 0 的数量
  %padded = tensor.pad %img low[0,0,0,0] high[0,13,0,0] {
    ^bb0(...):
      %c0 = arith.constant 0.0 : f16
      tensor.yield %c0 : f16
  } : tensor<1x3x... > to tensor<1x16... >

  // 后续 Packing 会将这个 Padded Tensor 重排为 NC1HWC0
  // 后端优化：识别到 Padding 为全 0，生成 MTE 指令参数 src_stride=3, dst_stride=16
  return %padded
}
```

---

### 3.4 缓冲区化与原地更新（Bufferization & In-place）

#### 3.4.1 One-Shot Bufferization

##### 背景

在编译器的中高层 IR（如 MLIR Linalg/Tensor 级别），数据通常被表示为**不可变张量（Immutable Tensors）**，遵循 SSA（静态单赋值）原则。这便于数学分析，但与底层硬件的**冯·诺依曼架构**（可变内存、指针读写）不符。
**Bufferization** 的任务是将 Tensor 转换为 Memref（内存缓冲区）。

*   **Naive Bufferization**：为每个算子的输出都 `malloc` 新内存。结果：内存爆炸，拷贝频繁。
*   **One-Shot Bufferization**：对整个函数进行全局分析，寻找**原地更新（In-place Update）**的机会，即复用输入 Buffer 来存储输出结果。

##### 技术原理

1.  **Destination-Passing Style (DPS, 目标传递风格)**：
    这是现代 Bufferization 的核心范式。算子不再“返回”一个新的 Tensor，而是接收一个“输出 Buffer”作为参数，并将结果写进去。
    *   *Tensor IR*: `%result = op(%input)`
    *   *Buffer IR*: `op(%input, %output_buffer)`

2.  **读写冲突检测 (RaW Conflict Detection)**：
    编译器必须证明复用是安全的。
    *   *冲突场景*：如果 Tensor A 被 Op1 读取，同时被 Op2 写入（复用）。如果 Op2 先于 Op1 执行，Op1 就会读到错误的数据。
    *   *分析算法*：构建 **Interference Graph（干涉图）**。如果输入 Tensor 在写入点之后不再被读取（Last Use），则标记为可复用；否则，插入显式的 `memref.copy` 进行保护。

3.  **硬件视角**：
    *   **GPU**：Bufferization 通常映射为动态的 `malloc/free` 或显式的 Shared Memory 分配。
    *   **Ascend NPU**：由于 NPU 运行时（Runtime）通常不支持高效的动态内存分配，Bufferization 是后续 **Static Memory Planning** 的前置步骤。编译器倾向于极度激进的 In-place 复用，以减少 **Workspace** 的总需求量，确保模型能塞进有限的 Device Memory。

##### MLIR 实现
MLIR 通过 `bufferization` dialect 和 `one-shot-bufferize` pass 实现这一过程。

```cpp
// 转换前：Tensor Level (Value Semantics)
// 原始图：SSA 形式，%t1 是不可变的
func.func @tensor_calc(%A: tensor<1024xf32>, %B: tensor<1024xf32>) -> tensor<1024xf32> {
  
  // 1. Init Tensor (逻辑上的空张量)
  %init = tensor.empty() : tensor<1024xf32>
  
  // 2. Generic Op (DPS 风格: outs 指定了潜在的复用目标)
  %t1 = linalg.generic { ... } 
      ins(%A, %B : ...) 
      outs(%init : ...) { ... }
  
  return %t1
}

// 转换后：Memref Level (In-place Bufferization)
// 优化后：指针形式，In-place 更新
func.func @buffer_calc(%A: memref<1024xf32>, %B: memref<1024xf32>, %Out: memref<1024xf32>) {
  
  // 编译器分析发现 %init 只是占位符，且 %Out Buffer 在此处可写
  // 因此，直接将计算结果写入传入的 %Out 指针，完全消除了中间 malloc
  
  linalg.generic { ... }
      ins(%A, %B : memref<1024xf32>, memref<1024xf32>)
      outs(%Out : memref<1024xf32>) { ... }
  
  // 函数无返回值 (void)，结果副作用在 %Out 中
  return
}
```

##### 性能影响案例

*   **Case**: `Y = ReLU(X)`
*   **Copying**: `Alloc(Y); Load(X); Compute; Store(Y); Free(Y)`。带宽消耗 $2N$。
*   **In-place**: `Load(X); Compute; Store(X)`. 带宽消耗 $2N$（读写同一地址，Cache 命中率极高），且省去了 `Alloc/Free` 的系统调用开销（约 5-10us）。

---

## 4. 内存层次与多级分块（Memory Hierarchy & Tiling）

在解决了数据布局后，编译器必须面对现代处理器最严峻的挑战——**内存墙（Memory Wall）**问题。

本章关注如何将庞大的数据与算力映射到硬件陡峭的**存储金字塔（Memory Pyramid）**上。核心目标是掩盖不同层级存储器（HBM/DDR $\to$ L2 $\to$ L1/Shared $\to$ Register）之间的延迟差异。通过**多级分块（Tiling）**技术，编译器将计算限制在高速缓存内完成，并利用显式的异步搬运与生命周期管理技术，确保计算单元永远"有数可算"，实现带宽利用率的最大化。

### 4.1 多级分块（Multi-level Tiling）

#### 4.1.1 Register/L1/L2 Tiling

##### 背景

为了掩盖 DRAM 的高延迟，必须将大张量切分为适应各级 Cache 大小的 **Tile（图块）**。
**Tile-local Fusion** 的本质是将原本串行的"全图算子"，转换为在 Tile 粒度上紧密耦合的"子图融合"，确保生产者（Producer）生成的 Tile 在被驱逐出 Cache 前，立刻被消费者（Consumer）使用。

##### 技术原理

多级分块不仅仅是循环变换，它是对硬件**存储带宽金字塔**的数学适配。其核心原理包含以下三个层面：

1.  **带宽放大效应 (Bandwidth Amplification)**：
    *   硬件的存储带宽呈倒金字塔状：$BW_{Reg} \gg BW_{L1/Shared} \gg BW_{L2} \gg BW_{HBM}$。
    *   Tiling 的目标是确保：**越慢的内存，访问频率越低**。
    *   通过分块，数据从 HBM 加载一次到 L2，从 L2 加载一次到 L1，但在 L1 和寄存器之间进行成百上千次的高频读写。这使得计算单元（ALU）感受到的是寄存器的超高带宽，而非 HBM 的低带宽。

2.  **表面积-体积比优化 (Surface-to-Volume Ratio Optimization)**：
    *   以矩阵乘法为例，计算量是 $O(N^3)$（体积），数据量是 $O(N^2)$（表面积）。
    *   如果对全图计算，数据复用率低。
    *   如果切分为 $K \times K$ 的 Tile，加载 $2K^2$ 的数据可以进行 $K^3$ 次计算。
    *   **原理**：只要 Tile 大小 $K$ 足够大，计算密度（Compute/Memory Ratio）就会随 $K$ 线性增长，直到填满 Cache 容量。

3.  **硬件作用域映射 (Hardware Scope Mapping)**：
    编译器将不同层级的 Tile 映射到硬件不同的并行层级，这决定了数据的**共享范围**：
    *   **L2 Tile (Grid Level)**：映射到 GPU Grid 或 NPU Cluster。数据在不同核心间**不共享**（或通过 L2 弱共享）。
    *   **L1/Shared Tile (Block Level)**：映射到 GPU Thread Block 或 NPU AI Core。数据加载到 Shared Memory/UB，被这一组线程/Core **显式共享**。
    *   **Register Tile (Thread Level)**：映射到 GPU Thread 或 NPU Vector Unit。数据驻留在寄存器堆，仅被当前线程/指令**独占**，速度最快。

##### 策略

*   **L2 Tiling**：适应 L2 Cache 大小，优化网格（Grid）调度。
*   **L1/Shared Tiling**：适应 L1/SRAM 大小，优化线程块（Thread Block）调度。
*   **Register Tiling**：适应寄存器文件大小，优化指令级并行。

##### MLIR 实现

MLIR 通过 `scf` 循环嵌套表达分块逻辑，并利用 `linalg.tile` pass 自动生成。

```cpp
// 原始算子：全量计算
%0 = linalg.matmul ins(%A, %B) outs(%C)
  
  
// 优化后：三级分块结构 (L2 -> L1 -> Register)
// 外层：L2 Cache Tile, Size=64 (对应 Grid/WorkGroup)
scf.for %i0 = 0 to 1024 step 64 {
  scf.for %j0 = 0 to 1024 step 64 {
    // 中层：L1/Shared Memory Tiling, Size=8 (对应 ThreadBlock)
    scf.for %i1 = %i0 to min(%i0 + 64, 1024) step 8 {
      scf.for %j1 = %j0 to min(%j0 + 64, 1024) step 8 {
        // 内层：Register Tiling (对应 SIMD/Vector 指令)
        // 此时数据驻留在寄存器堆，执行极致的融合计算
        %tile = linalg.matmul ... // 8x8 micro-kernel矩阵乘法
      }
    }
  }
}
```

---

### 4.2 显式内存层级管理（Explicit Hierarchy Management）

#### 4.2.1 Memory Promotion & Copy-Compute Overlap

##### 背景
大多数 AI 处理器（GPU/TPU/NPU/DSP）都包含用户可控的**片上高速缓存（Scratchpad Memory）**（如 GPU 的 Shared Memory 或 NPU 的 Local Memory）。
**内存提升（Promotion）** 指将频繁访问的数据从慢速的主存（HBM/DDR）显式搬运到片上高速缓存。为了避免数据搬运阻塞计算单元，必须利用硬件的 **DMA 机制** 实现**拷贝与计算的时间重叠（Overlap）**。

##### 技术原理

* **Asynchronous Data Movement (异步数据搬运)**：
    利用硬件独立的 **DMA 引擎** 或 **异步拷贝指令**，在计算单元（ALU/Tensor Core）处理当前数据的同时，后台静默地预取下一块数据。
    *   *通用性说明*：在 NVIDIA GPU 上映射为 `cp.async`，在 Ascend NPU 上映射为 `DataCopy`，在 TPU 上映射为 DMA 指令。

*  **Double Buffering (双缓冲/乒乓机制)**：
    一种软件流水线（Software Pipelining）策略。分配两块片上缓存（Buffer A 和 Buffer B），当计算单元处理 Buffer A 时，DMA 引擎填充 Buffer B，交替进行。

* **Bank Conflict Avoidance (存储冲突避免)**：
    针对采用多体存储（Banked Memory）架构的片上缓存，通过 **Padding（填充）** 或 **Swizzling（地址重排）** 优化数据布局，防止并行访问冲突。

##### MLIR 示例

```cpp
// 这是一个通用的"异步拷贝 + 融合计算"模式
func.func @explicit_memory_hierarchy(%A_global, %B_global, %C_global) {
  // 1. 分配片上高速缓存 (Scratchpad/Shared Memory)
  // 这里的 memory_space 是一个通用属性，不同硬件对应不同层级
  %A_local = memref.alloc() : memref<128x32xf16, #gpu.address_space<workgroup>>
  %B_local = memref.alloc() : memref<128x32xf16, #gpu.address_space<workgroup>>

  // 2. 启动异步 DMA 搬运 (Async DMA Start)
  // 编译器后端会将其映射为特定硬件的 DMA 指令 (如 cp.async 或 dma_copy)
  %token = gpu.device_async_copy %A_global[...] to %A_local[...] 
  
  // ... (此处可以插入不依赖 A_local 的其他独立计算以掩盖延迟) ...

  // 3. 等待数据就绪 (DMA Wait)
  gpu.device_async_wait %token
  gpu.barrier

  // 4. 在高速缓存上执行融合算子 (Compute on Scratchpad)
  // 此时算子全速运行，无主存带宽瓶颈
  %acc = linalg.matmul ins(%A_local, %B_local) ... 
  
  // 5. 结果写回
  memref.store %acc, %C_global[...]
}
```

---

### 4.3 内存生命周期优化（Lifetime Optimization）

#### 4.3.1 Static Memory Planning & Reuse

##### 背景

深度学习模型中，中间 Tensor（Activation）的生命周期通常较短。**内存规划**旨在通过**活跃度分析（Liveness Analysis）**，让互斥的 Tensor 共享同一块物理内存（Memory Arena），从而降低**峰值内存（Peak Memory Usage）**。

这在**边缘设备**（如手机 NPU、MCU）上尤为重要，通常在编译期确定所有 Buffer 的偏移量（Offset）。

##### 技术原理

静态内存规划的核心是将**时间维度上的互斥**转化为**空间维度上的共享**。其技术实现通常包含三个步骤：

1.  **活跃度分析与干涉图构建 (Liveness Analysis & Interference Graph)**：
    *   编译器对计算图进行拓扑排序，确定每个张量的**定义点（Def）**和**最后使用点（Last Use）**，从而得到一个**活跃区间（Live Interval）**。
    *   构建**干涉图（Build Interference Graph）**：节点代表张量，边代表活跃区间重叠。如果两个节点之间**有边（Edge）**，表示它们**“相互干涉”**，不能共享同一块物理内存。

2.  **内存分配算法 (Offset Assignment / Tensor Allocation)**：
    *   这是一个经典的**图着色问题（Graph Coloring Problem）**。目标是用最少的“颜色”（内存偏移量）为干涉图的所有节点着色，使得相邻节点颜色不同。
    *   由于图着色是 NP-Hard 问题，编译器通常采用 **Greedy Best-Fit** 算法：按张量大小排序，依次为每个张量分配一个**偏移量（Offset）**。分配时，尝试寻找空闲区间（Free Block）中最小且满足大小的块。
    *   目标：最小化所有张量所需的**总内存池大小（Total Workspace Size）**。

3.  **Arena 机制 (Single Arena / Workspace)**：
    *   为了避免运行时频繁调用操作系统级的 `malloc/free`（会导致碎片和系统调用开销），编译器计算出整个图运行所需的**峰值内存（Peak Memory）**。
    *   在运行时，仅分配**一块**巨大的连续内存（Arena/Workspace）。所有算子的输入输出都通过 `Base_Ptr + Offset` 进行访问。

##### MLIR 策略
*   **Bufferization**：在将 Tensor 降级为 Memref 时，使用 `BufferDeallocation` pass 插入 `dealloc`，并合并 `alloc`。
*   **In-place Bufferization**：尽可能复用输入 buffer 作为输出 buffer（如果输入不再被使用）。

```cpp
// 场景：两个串行的算子 Op1 -> Op2
// Op1 的输出 %buf1 在 Op2 执行完后就不再需要
// Op3 需要一个新的 Buffer，复用 %buf1 的空间

func.func @static_memory_planning(%arena: memref<10MB>) {
  
  // 1. 静态规划结果：Buf1 分配在偏移 0
  %buf1 = memref.view %arena[0] ... : memref<1MB>
  call @op1_produce(%buf1)
  
  // 2. Op2 读取 Buf1
  call @op2_consume(%buf1)
  // --- 此时 %buf1 生命周期结束 ---

  // 3. 内存复用：Buf2 复用偏移 0 的空间
  // 编译器静态分析确认 %buf1 和 %buf2 无干涉 (Interference)
  %buf2 = memref.view %arena[0] ... : memref<1MB>
  call @op3_produce(%buf2)
  
  return
}
```

#### 4.3.2 Memory-Constrained Operator Splitting

> 虽然本节在形式上和 **4.1 多级分块** 很像（都是切分），但两者的**目标函数**截然不同：
>
> 4.1 是为了**快**（Cache 命中率），而 4.3.2 是为了**活下去**（避免 OOM，内存溢出）

##### 背景

当单个算子（即使是分块后）所需的临时空间超过硬件限制（如 SRAM 大小或 TPU 内存），或者为了适配特定的内存 Bank 限制，编译器必须将算子**拆分（Split）**为多次执行。

此技术常用于**超大模型训练**（Activation Checkpointing 也是一种变体）或**受限内存嵌入式推理**。

##### 技术原理

内存受限拆分的核心逻辑是**时空置换（Space-Time Trade-off）**：通过增加少量的控制流开销（时间），换取峰值内存占用的显著降低（空间）。其技术实现基于以下机制：

1.  **工作集缩减 (Working Set Reduction)**：
    *   对于一个大算子（如 $4096 \times 4096$ 的 Element-wise），其**峰值内存**需求是输入+输出的总和。如果这个总和超过了硬件的**片上内存（SRAM/UB）**容量，算子将无法执行。
    *   **Strip Mining** 将空间的维度（Spatial Dimension）转化为时间的维度（Temporal Dimension）。通过将大循环切分为小循环，系统只需要分配能容纳**单个切片（Tile）**的内存。
    *   *数学效果*：峰值内存从 $O(N)$ 降低到 $O(Tile)$。

2.  **流式执行与流水线 (Streaming Execution & Pipelining)**：
    *   拆分后的算子变成了“流式处理”模式：`Load Tile -> Compute -> Store Tile`。
    *   为了掩盖拆分带来的频繁 I/O 开销，编译器通常结合 **Double Buffering (双缓冲)** 技术。在计算第 $i$ 个 Tile 时，DMA 引擎并行搬运第 $i+1$ 个 Tile。

3.  **算子裂变 (Operator Fission)**：
    *   与通常追求的“融合（Fusion）”相反，这里有时需要**反向操作**。如果一个融合算子（OpA + OpB）所需的中间 Buffer 太大而无法驻留片上，编译器会选择将其**裂变**为两个循环，中间结果写回主存（DRAM），以牺牲带宽为代价换取内存可行性。

##### 硬件视角：Ascend NPU 的 UB 适配

在 **Ascend NPU** 上，这一技术是刚需。

*   **UB 强约束**：Ascend 的 Unified Buffer (UB) 通常只有几百 KB。对于大分辨率图像（如 4K 图）或大语言模型 Tensor，绝对无法一次性塞入 UB。
*   **自动切分**：Ascend 编译器需要根据 UB 的实际大小计算最大可能的 Tile Size，自动插入循环结构。如果计算涉及多个 Tensor，必须保证它们切片后的总和 $\le UB_{Capacity}$。

##### MLIR 示例：Strip Mining (Loop Fission)

假设我们需要处理一个巨大的张量（如 1GB），但硬件的片上内存（SRAM）仅有 1MB。直接对整个张量执行 `linalg.generic` 会导致内存溢出（OOM）。
编译器通过 **Strip Mining** 将其重写为循环形式，每次只申请一小块缓冲区。

```cpp
// 原始：大内存需求 (High Peak Memory)
// 需要分配整个 %large_result，可能导致 OOM
func.func @naive_large_op(%in: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %large_result = linalg.generic ... ins(%in) ... 
  return %large_result
}

// 优化后：受限内存分割 (Low Peak Memory)，峰值内存 (Peak Memory) 降低到 1MB
func.func @split_large_op(%in: tensor<1024x1024xf32>, %out_buf: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %step = arith.constant 32 : index // 拆分步长，由 SRAM 大小决定

  // 将大算子拆分为循环
  scf.for %i = %c0 to %c1024 step %step {
    // 1. Slice: 逻辑切片，不发生物理拷贝
    %sub_in = tensor.extract_slice %in[%i, 0] [32, 1024] [1, 1] 
              : tensor<1024x1024xf32> to tensor<32x1024xf32>

    // 2. Alloc: 分配微小的临时 Buffer (驻留在 SRAM)
    %small_init = tensor.empty() : tensor<32x1024xf32>
    
    // 3. Fused Compute: 在小 Buffer 上执行融合计算
    // 即使是大算子，在这一步也被局部化了，结果存在 SRAM 中
    %sub_res = linalg.generic ... ins(%sub_in) outs(%small_init) ...

    // 4. Copy Back: 将结果从 SRAM 刷回主存 (DRAM)
    // 这里的 bufferization.to_memref 示意将 Tensor 结果写入 %out_buf
    memref.subview %out_buf[%i, 0] ... 
    // ... copy %sub_res content to %out_buf subview ...
  }
}
```

---

## 5. 并行性与分布式融合（Parallelism & Distributed Fusion）

随着摩尔定律的演进，算力的增长主要源于**并行度（Parallelism）**的提升。

本章关注如何将融合后的内核映射到硬件海量的**执行单元**上。优化的跨度极其宽广：从微观的**指令级**（SIMD/Vector）和**线程级**（Thread/Warp），扩展到宏观的**任务级**（Streams）以及跨芯片的**分布式集群**（SPMD）。核心目标是解决“如何切分数据”与“如何分配任务”的问题，通过融合通信与计算，确保成千上万个核心能够协同工作且互不阻塞。

### 5.1 指令与向量级并行（Instruction/Vector Parallelism）

#### 5.1.1 SIMD Vectorization Fusion

##### 背景

**SIMD**（Single Instruction Multiple Data）允许单条指令处理多个数据元素。这是 CPU (AVX/SVE)、DSP (Hexagon) 以及部分 NPU 的基础并行方式。
编译器通过 **Loop Vectorizer** 将多个标量操作融合为向量指令，并利用 **Predication/Masking** 技术处理非对齐的边界条件。

##### 技术原理

1.  **Loop Strip-mining (循环条带化)**：
    向量寄存器有固定宽度（如 256-bit 或 2048-bit）。编译器必须将逻辑循环切分为步长等于向量宽度的“条带”。
2.  **Masked Execution (掩码执行)**：
    处理循环尾部（Loop Tail）或条件分支时，为了避免标量回退，编译器生成**带掩码（Predicate Mask）**的向量指令，仅对 Mask 为 1 的 Lane 进行计算和写回。

##### 硬件视角：Ascend NPU 的宽向量掩码

Ascend 的 Vector Unit 非常宽（通常 256 Bytes，即一次处理 128 个 FP16）。
*   **全掩码模式**：Ascend 指令集强依赖 `mask` 参数（如 `vector_add(..., mask, ...)`）。编译器通过设置特殊的 Mask 寄存器（64-bit），精确控制哪部分数据参与计算。
*   **指令级 Repeat**：如前所述，Ascend 编译器倾向于将 SIMD 进一步融合为 Repeat 指令，一条指令执行多达 255 个 Vector 宽度的计算，极大降低取指开销。

##### MLIR Vector Dialect

```cpp
// 融合后的向量化操作
// vector<256xf32> 对应硬件的向量寄存器宽度 (如 AVX-512 或 SVE)
%v_a = vector.load %A[%i] : memref<?xf32>, vector<256xf32>
%v_b = vector.load %B[%i] : memref<?xf32>, vector<256xf32>

// 融合 FMA (Fused Multiply-Add) 指令
%v_sum = vector.fma %v_a, %v_b, %v_c : vector<256xf32>

// 融合 Masked Store (处理 Loop Tail 边界)
// %mask 是通过 vector.create_mask 生成的谓词掩码
vector.transfer_write %v_sum, %C[%i], %mask : vector<256xf32>, memref<?xf32>
```

---

### 5.2 线程级并行融合（Thread-level Parallelism）

#### 5.2.1 Workgroup & Subgroup Fusion

##### 背景

在 SIMT 架构（如 GPU）中，线程被组织为层级结构：**Workgroup**（对应 CUDA Block）和 **Subgroup**（对应 CUDA Warp 或 AMD Wavefront）。
**Subgroup-level Fusion** 利用硬件提供的 **Shuffle/Permute** 指令，使同一 Subgroup 内的线程能直接交换寄存器数据，无需经过 Shared Memory，从而实现极低延迟的归约（Reduction）融合。

##### 技术原理

1.  **资源划分 (Resource Partitioning)**：
    编译器根据计算图的并行度，将任务划分为多个 **Workgroup (Grid)**。每个 Workgroup 内部共享 L1/Shared Memory。
2.  **Barrier Optimization (屏障优化)**：
    在 Subgroup（如 Warp/Wavefront）内部，编译器利用硬件隐式同步特性消除显式 Barrier；在 Workgroup 内部，编译器尝试移动 Barrier 位置以最大化指令流水线重叠。

##### 硬件视角：Ascend NPU 的 Block 映射与流水线同步

Ascend 没有 CUDA 那样显式的 "Warp" 概念。
*   **Block = AI Core**：MLIR 的 `Workgroup` 通常直接映射为一个 **AI Core (Block)** 任务。
*   **Intra-Core Parallelism (核内并行)**：Ascend 的“Subgroup 融合”实际上体现为 **Cube、Vector、MTE 三条流水线的并行**。
    *   *融合策略*：编译器通过**指令重排（Instruction Reordering）**，让 MTE 搬运数据、Vector 处理数据、Cube 计算矩阵在同一个时间窗口内并发运行。同步通过 **Set/Wait Event** 指令（而非简单的 `__syncthreads`）实现。

##### MLIR GPU/Vector Dialect 实现

```cpp
// Subgroup-level Reduction Fusion
// 场景：融合 加法 和 线程间通信 (AllReduce within a subgroup)
%val = ... : f32

// 使用通用的 shuffle 指令在 lanes 之间交换数据
// 下层会映射为 NVVM shfl (NVIDIA) 或 ROCDL ds_bpermute (AMD)
%shuffled = gpu.shuffle xor %val, %offset, %width : f32

// 融合计算
%reduced = arith.addf %val, %shuffled : f32
```

#### 5.2.2 Cooperative Matrix Fusion (MMA)

##### 背景

现代 AI 处理器通常配备专用的矩阵加速单元（如 NVIDIA Tensor Core, Google TPU MXU, Intel AMX, ARM SME, Ascend NPU AIC）。
为了利用这些单元，编译器必须执行 **MMA Fusion**，将标准的矩阵乘法循环重构为**协作式（Cooperative）**指令，让一组线程（Subgroup）协同完成一个小块矩阵的加载与计算。

##### 技术原理

1.  **Intrinsic Mapping (内建指令映射)**：
    编译器将高层的 `linalg.matmul` 识别为特定硬件的**协作矩阵指令**（如 `mma.sync` 或 `wmma`）。
2.  **Layout Conformation (布局适配)**：
    协作指令通常要求输入数据满足特定的布局（如 Fragment Layout）。编译器在 Lowering 阶段插入隐式的 `Pack/Unpack` 或利用寄存器重用策略。

##### 硬件视角：Ascend NPU 的 Cube 融合

Ascend 的核心就是 **Cube Unit (矩阵加速器)**。
*   **Fractal Computing (分形计算)**：Cube 单元只能计算 $16 \times 16$ 的分形块。
*   **L0 Buffer Fusion**：融合的关键在于**L0A/L0B/L0C 缓存的管理**。
    *   编译器生成代码，指示 MTE 将数据直接搬运到 L0 缓存（并自动完成分形转换）。
    *   Cube 指令连续发射，直接复用 L0C 中的累加结果（Accumulator），避免写回 UB 或 GM。这就是 Ascend 上最高效的 MMA 融合。

##### MLIR Vector Dialect 实现

```cpp
// 使用 vector.contract 或 nvgpu/amx dialect 描述协作计算
// 这里的语义是：一组线程共同持有一个 vector 片段 (Fragment)
%res = vector.contract {
    indexing_maps = [#map_a, #map_b, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
} %a_vec, %b_vec, %acc_vec : vector<4x8xf16>, vector<8x4xf16> into vector<4x4xf32>
```

---

### 5.3 分布式与张量并行（Distributed & Tensor Parallelism）

#### 5.3.1 SPMD Sharding Propagation（SPMD 切分传播）

##### 背景

在分布式训练（如 Megatron-LM/GSPMD）中，大张量被**切分（Sharded）**分布在设备集群上。
**切分感知融合（Sharding-aware Fusion）**的核心逻辑是：如果相邻算子的**分布式布局（Mesh Layout）**一致，则它们可以在本地融合执行，完全消除中间的通信（Resharding）。

##### 技术原理

1.  **Slice Propagation (切片传播)**：
    编译器根据设备网格（Mesh）和用户指定的 Sharding 策略（如 `Shard([0], [1])`），推导图中每个 Tensor 的**分布式布局**。如果生产者和消费者的切分方式一致（Compatible），则它们之间的边被标记为 **Local**，无需通信。
2.  **Communication Insertion (通信插入)**：
    当布局不一致时（如 Row-parallel 转 Col-parallel），编译器插入 `AllGather` 或 `AllToAll` 算子。

##### 硬件视角：Ascend NPU 的 HCCS 拓扑感知

*   **HCCS (Huawei Cache Coherent System)**：Ascend 芯片间的高速互连。
*   **Alignment Constraints (对齐约束)**：在切分 Tensor 时，Ascend 编译器必须保证切分后的 Local Shape 依然满足 **32-byte 对齐** 或 **C0=16 对齐**。
    *   *融合策略*：如果切分导致数据不对齐（例如切分 `Channel=32` 为两份 `16`），编译器会接受；但如果切分 `Channel=16` 为 `8`，编译器会拒绝该 Sharding 策略，或者强制插入 Padding，因为不对齐会导致 Cube 单元无法计算 Local Slice。

##### MLIR Mesh Dialect 示例

```cpp
// 定义通用的设备拓扑：2x4 的处理器集群
%mesh = mesh.cluster @device_mesh(rank = 2, dim_sizes = [2, 4])

// 切分感知融合：
// MatMul 和 Elementwise 算子共享相同的 Sharding [[0], [1]]
// 编译器不仅融合了计算，还消除了它们之间的网络通信
func.func @spmd_fused(%A: tensor<1024x1024xf32, #mesh.shard<@device_mesh, [[0], [1]]>>,
                      %B: tensor<1024x1024xf32>) {
  // 1. 本地计算 MatMul (Local Compute)
  %local_matmul = linalg.matmul ... 

  // 2. 本地融合 Elementwise (Zero-cost fusion)
  // 因为布局一致，无需 AllGather，直接在切片上计算
  %local_result = arith.maxf %local_matmul, %c0 ...
  
  // 3. 仅在必要时插入集合通信
  mesh.all_reduce %local_result on @device_mesh ...
}
```

#### 5.3.2 Collective Operation Fusion (通信算子融合) 

##### 背景

在分布式数据并行训练中，模型往往包含数千个参数张量。如果对每个张量单独执行 `AllReduce`，**通信握手（Handshake）**和**内核启动（Kernel Launch）**的延迟将远超实际的数据传输时间。
**分桶（Bucketing）融合**技术：编译器将多个逻辑上独立的小张量通信，通过**内存拷贝（Memcpy）**打包到一个大的连续缓冲区（Buffer）中，执行单次大通信，然后再解包。这能显著提升带宽利用率。

##### 技术原理

1.  **Buffer Coalescing (缓冲区合并)**：
    小包通信受限于**延迟（Latency）**而非带宽。编译器或运行时将多个小的通信 Tensor（如不同层的梯度）通过 `Memcpy` 拼接到一个大的连续 **Bucket** 中。
2.  **Deterministic Scheduling (确定性调度)**：
    为了防止死锁，编译器必须保证所有设备上执行 `AllReduce(Bucket)` 的顺序严格一致。

##### 硬件视角：Ascend NPU 的任务调度优化

*   **Task Launch Overhead**：在 Ascend 上，启动一个 HCCL 任务（通信任务）需要 CPU 下发描述符，开销不可忽视。
*   **HCCL Fusion**：
    *   Ascend 编译器计算最优的 **Bucket Size**（例如 32MB）。
    *   利用专门的 **DMA 引擎** 在后台进行 Bucket 的打包（Packing）。
    *   由于 HCCS 带宽极高（通常 30GB/s+），**打包（Memory Copy）** 往往成为瓶颈。因此融合策略会尝试使用 Vector Unit 协助打包，或者在计算结束时直接写入 Bucket 预留的地址（Zero-copy Bucketing）。

##### 逻辑示意

```cpp
// 原始：
AllReduce(Grad_Layer1) -> Wait -> AllReduce(Grad_Layer2) -> Wait

// 融合后 (Bucketing)：
Buffer = Concat(Grad_Layer1, Grad_Layer2)
AllReduce(Buffer)  <-- 融合为单次通信内核调用
Grad_Layer1, Grad_Layer2 = Split(Buffer)
```

##### MLIR 伪代码示例

这段代码展示了编译器如何将两个独立梯度的 AllReduce 融合为一次通信：

```cpp
func.func @bucket_allreduce(%grad_a: tensor<1024xf32>, %grad_b: tensor<2048xf32>) 
    -> (tensor<1024xf32>, tensor<2048xf32>) {
  
  // 1. [Packing] 将分散的小张量平铺并拼接 (Concat) 到一个大 Buffer
  // 这一步通常在本地内存通过 memref.copy 完成
  %packed_buffer = tensor.concat %grad_a, %grad_b dim(0) 
                 : tensor<1024xf32>, tensor<2048xf32> -> tensor<3072xf32>

  // 2. [Fused Communication] 执行单次集合通信
  // 大消息体能更好地利用互连网络带宽 (NVLink/Infiniband/HCCS)
  %reduced_buffer = mesh.all_reduce %packed_buffer on @device_mesh reduction("sum")
                  : tensor<3072xf32>

  // 3. [Unpacking] 将结果切片 (Slice) 回原始形状
  %res_a = tensor.extract_slice %reduced_buffer[0] [1024] [1] 
         : tensor<3072xf32> to tensor<1024xf32>
  %res_b = tensor.extract_slice %reduced_buffer[1024] [2048] [1] 
         : tensor<3072xf32> to tensor<2048xf32>

  return %res_a, %res_b
}
```

#### 5.3.3 Communication-Computation Overlap（通信-计算重叠融合）

##### 背景

将**通信算子**与**无依赖的计算算子**进行流水线融合。通过将大算子切分为微操作（Micro-ops），在执行计算流（Compute Stream）的同时，并行触发通信流（Comm Stream）。

##### 技术原理

通信-计算重叠的核心在于打破“计算完成才能传输”的串行依赖，利用硬件的独立部件并行工作。其实现依赖两个关键机制：

1.  **多流流水线 (Multi-Stream Pipelining)**：
    *   将硬件指令队列划分为 **Compute Stream**（负责矩阵乘、向量计算）和 **Communication Stream**（负责 Send/Recv/AllReduce）。
    *   两个 Stream 只有在显式的**同步点（Synchronization Point）**才会互相等待，其余时间并行执行。

2.  **微批次与双缓冲 (Micro-batching & Double Buffering)**：
    *   为了实现重叠，数据必须切分。当计算单元正在处理第 $i+1$ 个数据块（Micro-batch）时，通信单元正在传输第 $i$ 个数据块的结果。
    *   这要求内存分配上采用 **Ping-Pong Buffer**，防止读写冲突（Race Condition）。

##### 硬件视角：Ascend NPU 的 TS 调度与 HCCS

在 Ascend 达芬奇架构中，重叠的实现高度依赖于硬件调度器 **Task Scheduler (TS)** 和互连架构 **HCCS**：

1.  **物理独立的执行引擎**：
    *   **AICore**：执行 Cube 和 Vector 指令。
    *   **HCCL Engine**：独立的通信控制器，直接通过 HCCS 链路搬运 HBM 中的数据，不占用 AICore 的算力。
    *   *原理*：编译器将计算算子下发到 Stream 0，将 HCCL 算子下发到 Stream 1。**TS (Task Scheduler)** 会同时从两个队列中提取任务并分发给对应的硬件引擎，实现真正的物理并行。

2.  **硬件级事件同步 (Hardware Event Synchronization)**：
    *   为了保证数据一致性（例如：必须等计算写完 HBM，通信才能读），编译器插入 **Event Record** 和 **Stream Wait** 指令。
    *   *Ascend 特性*：这种等待是**硬件级**的。TS 会挂起依赖流的执行，直到 Event 信号到达，而不会阻塞 Host CPU 的下发线程，也不会阻塞其他无依赖的 Stream。

##### 伪代码逻辑

```cpp
// 融合优化（Pipelined Schedule）：
// 编译器自动插入 Event Record/Wait 实现流同步
Stream_Compute:  [Comp A0]  [Comp A1]  [Comp A2]
Stream_Comm:                [Send A0]  [Send A1]  [Send A2]
// 此时 Comp A1 与 Send A0 实现时间轴上的融合（重叠）
```

##### MLIR 实现：Async Dialect 映射 Stream

在 MLIR 中，通过 `async.execute` 创建的 Token 依赖关系，在 Ascend 后端需要 Lowering 为 Stream 和 Event 操作。

```cpp
// 场景：流水线并行中的 1F1B 模式
// 目标：在计算 MB[i] 的同时，发送 MB[i-1]

func.func @ascend_overlap(%input_i: tensor<...>, %result_i_minus_1: tensor<...>) {
  
  // Stream 0: Compute (映射为 AICore 任务)
  %compute_token = async.execute {
    // 这是一个计算密集型算子
    %res_i = linalg.matmul ins(%input_i, ...) ...
    
    // 记录 Event，表示计算完成，数据已准备好
    // (隐式: gpu.record_event %evt_compute)
    async.yield %res_i : tensor<...>
  }

  // Stream 1: Communication (映射为 HCCL 任务)
  // 必须等待上一轮的 buffer 可用，但不阻塞当前轮的计算
  %comm_token = async.execute {
    // 这是一个通信密集型算子
    // 它在 HCCS 链路上运行，与 AICore 并行
    mesh.send %result_i_minus_1 to @next_stage ...
    
    async.yield
  }

  // 屏障：确保本轮所有操作提交完成
  async.await %compute_token, %comm_token
}
```

#### 5.3.4 Pipeline P2P Fusion (流水线点对点融合)

##### 背景
在超大模型训练（如 GPT-4, DeepSeek）中，通常采用 **Pipeline Parallelism (PP)** 将模型的不同层分布在不同设备上。PP 的核心挑战是 **Bubble（流水线气泡）**。
为了掩盖跨设备的 **P2P 通信（Send/Recv）** 延迟，编译器需要实现 **1F1B (One-Forward-One-Backward)** 调度的自动化融合：即在执行当前 Micro-Batch 计算的同时，异步发送/接收上一个 Micro-Batch 的边界数据。

##### 技术原理
**计算-通信交错（Inter-op Overlap）**：
不同于 Tensor Parallel 的层内通信，PP 的通信发生在层间。编译器将 `Send/Recv` 指令下沉到计算图中，并使用异步原语将其与**无依赖的计算任务**（通常是不同 Micro-Batch 的计算）并行化。

##### MLIR 示例：Async P2P Communication

```cpp
// 模拟 Pipeline Stage N 的 1F1B 调度融合
// 同时处理：计算 MicroBatch[i] 和 发送 MicroBatch[i-1] 的结果

func.func @pipeline_stage_fused(%mb_curr: tensor<...>, %mb_prev_result: tensor<...>) {
  
  // Stream 1: 计算当前 Micro-Batch (计算密集)
  %compute_token = async.execute {
    // 前向传播计算
    %res = call @forward_layers(%mb_curr)
    async.yield %res : tensor<...>
  }

  // Stream 2: 发送上一个 Micro-Batch 的结果给 Stage N+1 (网络密集)
  // 这个操作与 Stream 1 并行执行，掩盖了 P2P 延迟
  %comm_token = async.execute {
    // P2P 发送指令
    mesh.send %mb_prev_result to @stage_next_mesh ...
    async.yield
  }

  // 同步点：等待两者完成，准备进入下一个 Step
  async.await %compute_token, %comm_token
  return
}
```

---

### 5.4 任务级与异构并行（Task & Heterogeneous Parallelism）

#### 5.4.1 Async & Multi-stream Fusion

##### 背景

在异构系统（Host CPU + Device Accelerator）中，利用**事件（Event）**和**流（Stream/Queue）**机制，将 CPU 逻辑、DMA 数据搬运和 Device 计算融合在同一个时间窗口内并发执行。

##### 技术原理

多流融合的核心是将无数据依赖的子图（Subgraphs）调度到独立的执行队列中，利用硬件的资源冗余来实现任务级并行。

1.  **DAG 分区与流映射 (DAG Partitioning & Stream Mapping)**：
    *   编译器对计算图进行**依赖分析（Reachability Analysis）**。如果图在某点分叉为两个互不依赖的分支（Branch A 和 Branch B），编译器将它们标记为可并行。
    *   **流分配**：Branch A 分配给 `Stream 0`，Branch B 分配给 `Stream 1`。
    *   **同步插入**：在两个分支汇合（Join）的地方插入 `StreamWaitEvent`，确保结果正确性。

2.  **异构引擎并行 (Heterogeneous Engine Parallelism)**：
    *   这是 AI 加速器区别于 CPU 的关键。硬件内部包含多种专用引擎。
    *   **DMA 引擎**：负责搬运。
    *   **Compute 引擎**：负责计算。
    *   **Scalar/Host 引擎**：负责复杂逻辑控制。
    *   *原理*：编译器将不同类型的算子（Copy vs MatMul vs Shape Inference）下发到对应的流，使得 DMA 在搬运数据的同时，Compute 引擎在计算，Scalar 引擎在处理动态 Shape 逻辑。

##### 硬件视角：Ascend NPU 的异构多核调度

在 Ascend 达芬奇架构中，多流融合的实现高度依赖于 **Task Scheduler (TS)** 和 **AICPU** 的协同：

1.  **TS 硬件分发 (Task Scheduler Dispatch)**：
    *   Ascend 芯片内置了一个硬件级的 **TS (Task Scheduler)** 模块。
    *   Host CPU 只需将任务描述符（Task Descriptor）推送到内存中的 SQ (Submission Queue)。
    *   **并行分发**：TS 会自动从不同的 Queue 中抓取任务，如果任务依赖满足（Event 信号到达），TS 会将 AICore 任务发给 Cube/Vector，将系统任务发给 AICPU，将拷贝任务发给 DMA。**这种分发完全由芯片硬件完成，无 Host CPU 负载。**

2.  **AICore 与 AICPU 的异步并行**：
    *   **AICore**：擅长张量计算。
    *   **AICPU**：擅长标量逻辑、OS 交互、打印日志等。
    *   *融合策略*：对于无法被 AICore 融合的复杂算子（如 `Unique`, `Where`, `TopK` 的某些变体），编译器将其调度到 AICPU 上异步执行。与此同时，AICore 继续执行后续无依赖的矩阵计算。两者通过 Stream/Event 机制同步。

##### MLIR 实现：Async Region

MLIR 的 `async` dialect 提供了完美的抽象。在 Ascend 后端，`async.execute` 需要映射为不同的 Stream ID。

```cpp
// 场景：Inception 模块的多分支并行 + 异构计算
func.func @ascend_multi_stream(%input: tensor<...>) {
  
  // Stream 0: 密集计算分支 (映射为 AICore Task)
  %token0 = async.execute {
    %conv = linalg.conv_2d ... ins(%input) ...
    async.yield %conv : tensor<...>
  }

  // Stream 1: 复杂逻辑/预处理分支 (映射为 AICPU Task)
  %token1 = async.execute {
    // 假设这是一个 AICore 不支持的复杂 CPU 算子
    %processed = "tf.Unique"(%input) ... 
    async.yield %processed : tensor<...>
  }

  // Stream 2: 独立的数据搬运 (映射为 DMA Task)
  %token2 = async.execute {
    %copy = memref.copy %input, %host_buffer ...
    async.yield
  }

  // 融合点：TS 硬件调度器会等待所有 Stream 完成
  async.await %token0, %token1, %token2
  
  // 后续处理...
}
```

#### 5.4.2 Host-Device Prefetch Fusion (预取融合)

##### 背景

在推荐系统（Embedding Lookup）或图神经网络等场景中，Host（CPU）到 Device（处理器）的数据搬运往往是瓶颈。
如果采用简单的串行模式（Copy -> Compute），处理器会频繁处于**饥饿（Starvation）**状态。
编译器通过**软件流水线（Software Pipelining）**技术，将"当前批次的计算"与"下一批次的搬运"融合在同一个时间窗口内，实现 Host 与 Device 的全并行。

##### 技术原理

编译器进行**软件流水线化（Software Pipelining）**：

1.  **Prologue**: 启动 Batch 0 的拷贝。
2.  **Steady State (Kernel)**: 
    *   并行执行：`Compute(Batch i)` **&&** `Copy(Batch i+1)`
3.  **Epilogue**: 计算最后的 Batch。

##### MLIR 伪代码示例

这里展示了一个典型的 **Double Buffering（双缓冲）** 预取流水线：

```cpp
// 预先分配两个 Device Buffer 用于乒乓操作
%buf0 = memref.alloc() : memref<...>
%buf1 = memref.alloc() : memref<...>

// [Prologue] 预取第 0 个 Batch 的数据
%token_init = gpu.memcpy async %buf0, %host_data[0] : memref<...>, memref<...>

// 主循环：处理 Batch i，同时预取 Batch i+1
// iter_args 携带当前的 token 和 buffer 索引
scf.for %i = 0 to %num_batches step 1 
    iter_args(%prev_token = %token_init, %curr_buf = %buf0, %next_buf = %buf1) {
  
  // 1. [Wait] 等待当前 Buffer 的数据搬运完成
  // 这里的等待时间通常被上一轮的计算掩盖了
  async.await %prev_token

  // 2. [Prefetch] 立即启动下一 Batch 的搬运 (i+1) 到 spare buffer
  // 这是一个异步操作，CPU 立即返回，DMA 开始工作
  %next_token = gpu.memcpy async %next_buf, %host_data[%i + 1] 
  
  // 3. [Compute] 在当前 Buffer 上全速计算
  // 此时 Device 在计算 %curr_buf，DMA 在搬运 %next_buf
  // 实现了 Host-Device Overlap
  linalg.generic ... ins(%curr_buf) ...

  // 交换 Buffer 和 Token 进入下一次迭代
  scf.yield %next_token, %next_buf, %curr_buf
}
```

---

## 6. 硬件适配与计算-内存权衡（Hardware Adaptation & Compute-Memory Trade-off）

理论上的完美融合，在真实硬件上往往会遭遇严酷的**资源约束（Resource Constraints）**。

在现代 AI 处理器上，性能并非只由“算了多少 FLOPs”决定，而是由**寄存器、片上缓冲区、并行度与专用计算单元**之间的微妙平衡所主导。

本章关注**理想算法**与**物理现实之间的博弈**：

- 是通过更激进的融合来减少访存，
- 还是主动拆分内核，以换取更高的 Occupancy 或更稳定的流水线？

大量工程实践表明，

> “融合反而变慢”的 90% 原因，
>  并非出现在算法层或循环层，
>  而是发生在 **硬件资源失衡** 这一层面。

本章通过**重计算**、**专用指令映射**及**混合精度技术**，叙述对特定硬件特性的极致适配。

### 6.1 资源感知内核融合（Resource-Aware Kernel Fusion）

#### 6.1.1 Register Pressure Controlled Fusion（寄存器压力控制融合）

##### 背景

在 AI 处理器上，**寄存器既是最快的存储层级，也是最稀缺的资源**。

算子融合虽然减少了全局内存访问，但也会显著拉长变量的**活跃区间（Live Interval）**。
 当融合后的内核需要同时持有过多中间值时：

- 在 GPU 上，会触发 **Register Spill** 到 Local Memory，
- 在 Ascend NPU 上，则可能导致 **L0 Buffer 溢出或流水线气泡（Pipeline Bubble）**，
   性能都会出现断崖式下降。

##### 技术原理

编译器构建**代价模型（Cost Model）**：

1. **估算活跃变量数**：$Regs_{fused} \approx \sum Regs_{ops} - Regs_{Shared}$

2. **Occupancy 阈值检查**：

   * **GPU**：单线程寄存器使用量 ↑ → 可同时驻留的 Warp ↓ → Occupancy Collapse

   * **Ascend NPU**：？？？

3. **Cut Strategy**：当预测寄存器不足时，编译器主动"切断"融合，插入显式的 Store/Load。

##### MLIR 实现逻辑

MLIR 通常通过 `transform` dialect 或后端 pass 来控制这种切分。

```cpp
// 场景：一个巨大的 Element-wise 链，可能耗尽寄存器
// 编译器决策：将其切分为两个 Kernel，而不是融合为一个

// Kernel 1: 生产物化 %temp
func.func @part1(%in: tensor<...>) -> tensor<...> {
  %1 = arith.addf ...
  %2 = arith.mulf ...
  %temp = math.exp %2  // 这里的活跃变量达到峰值
  return %temp
}

// Kernel 2: 消费 %temp
func.func @part2(%temp: tensor<...>) -> tensor<...> {
  %3 = arith.divf %temp, ...
  return %3
}

// 注意：如果强行融合，%1, %2, %3 以及后续变量可能需要同时存活在寄存器中
```

---

### 6.2 推测性融合与重计算（Speculative Fusion & Rematerialization）

#### 6.2.1 Activation Checkpointing (Rematerialization)

##### 背景

在训练超大模型或显存受限的推理场景中，**内存容量**是硬约束。
**重计算（Rematerialization）** 是一种以"时间换空间"的策略：为了避免存储某个中间张量（Activation），编译器选择在消费者算子中**重新计算**它，而不是从内存读取。

##### 技术原理

重计算不仅仅是简单的“删掉再算”，它是一个复杂的**图优化与资源调度**问题。其核心技术原理包含三个层面：

1.  **时空置换与带宽红利 (Space-Time Trade-off & Bandwidth Bonus)**：
    *   **基本逻辑**：牺牲计算资源（Time）来节省显存占用（Space）。
    *   **隐形红利**：在 Memory-bound（访存受限）的算子（如 ReLU, Dropout, Add）中，从 HBM 读取数据的延迟往往高于 ALU 重新计算数据的延迟。因此，**重计算有时反而比“保存-读取”更快**，因为它避免了 HBM 的往返访问（Round-trip），减少了对 Memory Wall 的撞击。

2.  **基于代价的检查点选择 (Cost-driven Checkpoint Selection)**：
    *   编译器不会重算所有节点，而是通过**贪心算法**或**动态规划**选择最佳的“检查点（Checkpoints）”。
    *   *保留策略*：保留**计算密集型**（High Arithmetic Intensity）算子的输出（如 MatMul, Conv），因为重算它们的代价太高。
    *   *重算策略*：丢弃**访存密集型**（Low Arithmetic Intensity）算子的输出（如 Activation, Element-wise），仅在反向传播需要时，依据保留的检查点重新推导。

3.  **子图克隆与重连 (Subgraph Cloning & Rewiring)**：
    *   在编译器 IR 层面，这表现为**子图复制**。
    *   编译器识别出反向传播（Backward）中依赖前向（Forward）结果的边。如果该结果被标记为“重计算”，编译器会将生成该结果的前向子图（Op Sequence）**克隆一份**并插入到反向图的对应位置，切断与原前向结果的数据依赖。

##### 硬件视角：Ascend NPU 的 UB 驻留优化

在 **Ascend NPU** 上，重计算的意义被进一步放大，它与 **L1/UB 融合** 紧密相关：

1.  **UB 溢出规避 (UB Spilling Avoidance)**：
    *   Ascend 的 Unified Buffer (UB) 容量很小。在进行 LayerNorm 或 Softmax 融合时，如果中间变量太多导致 UB 放不下，传统做法是发生 **Spill**（写回 HBM 再读回）。
    *   *优化*：编译器倾向于在 UB 内部直接重算中间变量。因为 UB 的带宽极高（TB/s 级），在 UB 内多算一次的开销几乎可以忽略不计，远优于 Spill 到 HBM 的巨大延迟。

2.  **Tiling 维度的权衡**：
    *   为了塞进 UB，编译器可能被迫将 Tiling 切得很小，导致 Cube 单元利用率低。
    *   通过重计算减少活跃 Tensor 的数量，可以腾出 UB 空间，允许编译器选择**更大的 Tile Size**，从而提升整体计算效率。

##### 适用场景

*   **计算廉价但传输昂贵**的操作：如 ReLU, Cast, Element-wise Add。
*   **长距离依赖**：生产和消费之间间隔了大量其他算子，导致 Activation 长期占用显存。

##### MLIR 伪代码示例

```cpp
// 原始图：A -> B -> C，其中 B 的结果需要被保存以供 C 使用
%b = linalg.generic ... ins(%a) ... // Op B
%c = linalg.generic ... ins(%b) ... // Op C

// 优化后 (Rematerialization)：
// 编译器发现保存 %b 的显存代价 > 重算 %b 的计算代价
// 因此删除了全局的 %b，将 Op B 的逻辑"克隆"并内联到 Op C 之前

func.func @fused_recompute(%a: tensor<...>) {
  // 在消费者内部重新生成数据，通常是在寄存器层面
  %b_recalc = linalg.generic ... ins(%a) ... 
  
  // 直接使用重算的值
  %c = linalg.generic ... ins(%b_recalc) ...
}
```

---

### 6.3 专用硬件指令映射（Accelerator Intrinsic Mapping）

#### 6.3.1 Matrix/Vector Intrinsics Fusion

##### 背景

AI 处理器（GPU/TPU/NPU）通常包含专用的矩阵加速单元（如 Tensor Core, AMX, Matrix Core, Ascend NPU AIC）。
这些单元通常只支持特定的**形状（Shape）**（如 16x16）和**布局（Layout）**。编译器必须将高层的 `MatMul` 算子进行 **Tiling + Packing**，并融合为一个能直接映射到硬件指令（Intrinsic）的形态。

> 对 Ascend NPU 而言，是否能够成功映射到 Cube 指令，往往是 MatMul 性能的分水岭：一旦退化为 Vector 路径，性能差距可能达到一个数量级。

##### 技术原理

编译器通过 **渐进式降级（Progressive Lowering）** 和 **布局适配（Layout Adaptation）** 将高层算子映射为硬件 Intrinsic，其核心机制包含：

1.  **中间抽象层 (The Vector Contract Abstraction)**：
    *   为了避免 `Linalg` 直接跳跃到汇编（LLVM IR），MLIR 引入了 `vector.contract` 或 `vector.outerproduct` 作为中间层。
    *   它保留了多维结构，但语义上已经接近硬件的 FMA（Fused Multiply-Add）指令。编译器在此层进行**形状推断**，确认 `16x16x16` 的 `vector.contract` 可以被一对一替换为硬件指令。

2.  **碎片化与寄存器重用 (Fragmentation & Register Reuse)**：
    *   **Fragment 抽象**：GPU Tensor Core 不直接操作内存，而是操作 **Fragment**（分布在多个线程寄存器中的数据片段）。
    *   **Fusion 逻辑**：编译器将数据的 `Load` 转化为 `LoadMatrix`（生成 Fragment），将 `MatMul` 转化为 `MMA`（消耗 Fragment），将 `Store` 转化为 `StoreMatrix`。
    *   **Accumulator Reuse**：最关键的融合发生在累加器上。编译器分析循环，保持 `Accumulator Fragment` 在寄存器中不动，只更新 `A` 和 `B` 的 Fragment，从而将数百次乘加指令融合为一条流水线。

3.  **布局感知的向量化 (Layout-aware Vectorization)**：
    *   如果硬件指令要求输入是 Blocked Layout（如 Tensor Core 需要数据在 Shared Memory 中按特定 Swizzle 排列），编译器会在 Intrinsic 调用前插入隐式的 **Shuffle** 或 **Pack** 操作，并将其融合到数据加载阶段。

##### MLIR Vector Dialect 实现 (通用抽象)

`vector.contract` 是 MLIR 中连接上层算法与底层硬件指令的桥梁。

```cpp
// 高层：linalg.matmul
// ↓ Lowering
// 中层：vector.contract (抽象的向量收缩)
// 这一步融合了 FMA (乘加) 操作
%result = vector.contract {
    indexing_maps = [#map_a, #map_b, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
} %lhs_vec, %rhs_vec, %acc_vec 
  : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>

// ↓ CodeGen (后端映射)
// NVIDIA GPU -> nvgpu.mma.sync
// Intel CPU  -> x86vector.avx512.vpdpbusd
// ARM CPU    -> arm_neon.sdot
// Ascend NPU -> matmul
```

---

### 6.4 混合精度与量化融合（Mixed Precision & Quantization Fusion）

#### 6.4.1 Quantization-Aware Fusion

##### 背景

在**推理**端，量化（**INT8/FP8**）是主流。量化通常包含 `Dequantize` (反量化) -> `Compute` (计算) -> `Quantize` (量化) 的流程。
如果这些转换操作独立执行，带宽开销巨大。**量化感知融合**将 `Dequant` 和 `Quant` 算子分别融合到主计算算子的**输入端（Prologue）**和**输出端（Epilogue）**。

##### 技术原理

量化融合不仅仅是算术运算的合并，更是**数据位宽（Bit-width）的管理艺术**。其核心技术原理包含以下三个层面：

1.  **Requantization Epilogue Fusion (重量化尾部融合)**：
    *   *问题*：在 INT8 矩阵乘法中，为了防止溢出，累加器（Accumulator）通常使用 **INT32** 类型。如果将 INT32 结果直接写回 HBM，数据量会膨胀 4 倍，抵消了量化的带宽优势。
    *   *原理*：编译器将 **Requantize** 逻辑（`Int32 -> Float(Scale) -> Int8`）直接融合到 MatMul 的 **Epilogue** 阶段。
    *   *效果*：INT32 数据只存在于寄存器或片上缓存中，写回主存的永远是压缩后的 INT8 数据。

2.  **Register-level Type Promotion (寄存器级类型提升)**：
    *   *场景*：算子输入是 INT8，但后续计算需要高精度。
    *   *原理*：编译器在从内存加载 INT8 数据到寄存器后，立即执行 **Cast/Extend** 指令将其提升为 FP16/FP32。
    *   *收益*：内存带宽占用保持在低位（INT8），而计算精度保持在高位（FP/INT32）。

3.  **Correctness-driven Fusion (正确性驱动的融合)**：
    *   不同于为了“快”而做的融合，量化融合往往是为了“对”。
    *   许多专用加速单元（如 Ascend Cube）的 INT8 指令有严格的 **Input/Output Layout** 和 **Clipping** 要求。编译器必须融合特定的 `Clamp` 或 `Pack` 算子，否则生成的指令流无法通过硬件的合法性检查。

##### 硬件视角：Ascend NPU 的 Fixpipe 流水线

在 **Ascend NPU** 上，量化融合有专门的硬件路径支持，不仅仅是软件指令的组合：

1.  **Cube-to-Vector 桥接**：
    *   Cube 单元计算出的结果是 **INT32** 格式，存储在 **L0C Buffer** 中。
    *   为了进行下一层计算，必须将其转回 **INT8**。
    *   Ascend 提供了专门的 **量化后处理指令**（在旧架构称为 `fixpipe`，新架构融合在 Vector 指令中）。

2.  **融合策略**：
    *   编译器生成代码，指示 Vector 单元直接从 L0C 读取 INT32 数据，执行 `Reqant`（乘 Scale、加 Offset）、`Clip`（截断）和 `Cast`（转 INT8）。
    *   整个过程数据流为 `L0C(Int32) -> UB(Int32->Int8) -> GM(Int8)`，避免了中间宽数据的无效搬运。

##### 典型数据流

*   **Prologue Fusion**: Load INT8 -> Convert to FP32 -> Compute。
*   **Epilogue Fusion**: Compute result (FP32) -> Scale/Shift -> Convert to INT8 -> Store。

##### MLIR Linalg 实现

```cpp
// 融合后的 INT8 MatMul 核心
// 注意：输入是 i8，累加器是 i32，计算逻辑融合了类型转换
%res = linalg.generic {
  indexing_maps = ...,
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%a_i8, %b_i8 : tensor<MxKxi8>, tensor<KxNxi8>)
  outs(%acc_i32 : tensor<MxNxi32>) {
  
  ^bb0(%a: i8, %b: i8, %acc: i32):
    // 1. [Prologue] 在寄存器中即时扩展类型 (i8 -> i32)
    %a_ext = arith.extsi %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    
    // 2. [Compute] 计算乘加
    %prod = arith.muli %a_ext, %b_ext : i32
    %new_acc = arith.addi %prod, %acc : i32
    
    // 3. [Yield] 返回累加结果 (Epilogue 量化将在循环外处理或继续融合)
    linalg.yield %new_acc : i32
}
```

#### 6.4.2 Redundant Cast Elimination (冗余类型消除)

##### 背景
在混合精度（AMP）场景中，图层中可能充斥着 `FP16 -> FP32 -> FP16` 的转换链。
编译器在融合过程中，会执行**类型传播（Type Propagation）**，消除那些**"来回转换"**的冗余 Cast 操作，确保数据尽可能保持在低精度格式下流动，仅在累加器（Accumulator）中临时提升精度。

##### MLIR Linalg 实现 (INT8 MatMul 示例)

```cpp
// 融合后的 INT8 MatMul 核心
// 展示了如何在计算内核中融合类型转换 (Cast Fusion)
%res = linalg.generic {
  indexing_maps = ...,
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%a_i8, %b_i8 : tensor<MxKxi8>, tensor<KxNxi8>)
  outs(%acc_i32 : tensor<MxNxi32>) {
  
  ^bb0(%a: i8, %b: i8, %acc: i32):
    // 1. [Cast Fusion] 在寄存器中即时扩展类型 (i8 -> i32)
    // 这一步消除了显式的 Input Cast Kernel
    %a_ext = arith.extsi %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    
    // 2. [Compute] 计算乘加
    %prod = arith.muli %a_ext, %b_ext : i32
    %new_acc = arith.addi %prod, %acc : i32
    
    // 3. [Yield] 返回累加结果
    linalg.yield %new_acc : i32
}
```

#### 6.4.3 Loss Scaling Fusion (混合精度 Loss 缩放融合)

##### 背景
在 **FP16/BF16 混合精度训练** 中，梯度的数值范围可能极小，容易下溢（Underflow）变为 0。解决方案是 **Loss Scaling**：在前向结束后将 Loss 乘以一个大因子（Scale），在反向结束后将梯度除以该因子（Unscale）。
如果作为独立算子执行，这会引入两次额外的全图内存读写（Memory-bound）。

##### 技术原理

Loss Scaling 的融合本质上是**内存带宽优化**问题。其技术实现基于以下两个核心机制：

1.  **访存“搭载” (Memory Access Piggybacking)**：
    *   *问题*：`Unscale` 操作（$G = G / S$）是一个典型的 **Memory-bound** 操作。它需要遍历所有梯度张量，计算密度（Arithmetic Intensity）极低。如果单独执行，时间全花在读写 HBM 上，ALU 几乎空转。
    *   *原理*：编译器将乘除法操作“搭载”到邻近的**计算密集型**或**必须读写内存**的算子中。
    *   *Prologue*：`Loss * Scale` 融合进 Loss Function（如 CrossEntropy）。由于 Loss Function 本身就要读取 Logits 和 Labels，多做一次乘法是“免费”的。
    *   *Epilogue*：`Grad / Scale` 融合进 Optimizer Update 或 Gradient Clipping。这些算子本就要读取梯度，此时顺便执行除法，**完全消除了 Unscale 算子的内存访问开销**。

2.  **状态检查融合 (Finite Check Fusion)**：
    *   混合精度训练要求检查梯度是否包含 `NaN` 或 `Inf`。
    *   *未融合*：`Unscale Kernel` -> `Check Finite Kernel` -> `Update Kernel`。需要 3 次全量读写。
    *   *融合*：在寄存器中计算 `val = raw_val * scale_inv` 后，立即执行 `is_finite(val)` 检查，并累加状态标志。整个过程在一次内存扫描中完成。

##### 硬件视角：Ascend NPU 的 Vector 流水线优化

在 **Ascend NPU** 上，Loss Scaling 的融合策略主要围绕 **Vector Unit** 和 **Unified Buffer (UB)** 展开：

1.  **UB 驻留与减少搬运**：
    *   在未融合的情况下，梯度数据流为 `GM -> UB -> Vector(Mul) -> UB -> GM`。这仅仅为了改一个数值就消耗了宝贵的搬运带宽。
    *   通过融合，数据流变为 `GM -> UB -> Vector(Unscale + Clip + Update) -> UB -> GM`。`Unscale` 操作利用了 Vector Unit 强大的流水线能力，在数据驻留 UB 期间完成变换，不占用额外的搬运时间。

2.  **全归约指令利用 (Global Reduction)**：
    *   针对 `IsFinite` 检查，Ascend 的 Vector Unit 支持高效的 **ReduceMax/Min** 指令。
    *   融合 Kernel 可以在 UB 内并行检查当前切片（Tile）的浮点状态，生成一个标量结果，最后只将这个布尔值（HasInf/NaN）写回 Global Memory，极大地减少了同步开销。

##### MLIR 实现：Fused Unscale & Clip

```cpp
// 融合 Unscale 和 Gradient Clipping
// 避免单独启动一个 div kernel
func.func @fused_unscale_clip(%raw_grad: tensor<?xf32>, %scale: f32, %clip_norm: f32) {
  
  // 在一次遍历中完成除法和裁剪
  %final_grad = linalg.generic { ... } 
    ins(%raw_grad : tensor<?xf32>) 
    outs(%out : tensor<?xf32>) {
    
    ^bb0(%g: f32, %o: f32):
      // 1. Unscale: g_real = g_raw / scale
      // 乘法比除法快，通常实现为 * (1.0/scale)
      %scale_inv = arith.divf %c1, %scale : f32
      %g_real = arith.mulf %g, %scale_inv : f32
      
      // 2. Clip: clamp(g_real, -clip, +clip)
      %neg_clip = arith.negf %clip_norm : f32
      %t1 = arith.maxf %g_real, %neg_clip : f32
      %clipped = arith.minf %t1, %clip_norm : f32
      
      linalg.yield %clipped : f32
  }
}
```

---

### 6.5 **指令级数据打包（Instruction-Specific Packing）**

*注意：与第 3 章的全局布局优化不同，本节关注为了适配特定硬件指令（如 Tensor Core 或 VNNI）的输入格式要求，而在寄存器或 L1 传输层级进行的微观数据重排。*

在 GPU 上，Instruction-specific Packing 往往发生在寄存器加载阶段；

而在 Ascend NPU 上，这一步**直接决定数据能否进入 Cube Buffer（L0A/L0B）**，是从“能跑”到“跑满”的关键门槛。

#### 6.5.1 Intrinsic-Compatible Packing (Tensor Packing)

##### 背景

现代 AI 处理器的专用单元（如 NVIDIA Tensor Core 或 Intel AMX）通常要求输入数据遵循特定的**块状布局（Blocked Layout）**（例如：将矩阵切分为 $32 \times 32$ 的小块，或者在通道维度进行 $4$ 元素交错）。
如果全局内存布局是标准的 Row-major，编译器必须在数据加载到寄存器之前，通过 `tensor.pack` 将其转换为硬件指令兼容的**物理布局**。

##### 技术原理

指令级打包是为了解决**逻辑数据视图**与**硬件物理视图**之间的“阻抗失配”。其核心技术原理包含：

1.  **寄存器碎片映射 (Register Fragment Mapping)**：
    *   *问题*：在高层 IR 中，矩阵是一个连续的二维数组。但在硬件底层（如 NVIDIA Tensor Core），一个 $16 \times 16$ 的矩阵片段（Fragment）实际上是**被打散**分布在 Warp 内 32 个线程的私有寄存器中的。
    *   *原理*：编译器必须执行一种复杂的**线程到寄存器的索引变换**。例如，线程 0 可能持有矩阵坐标 $(0,0), (0,1), (8,0), (8,1)$ 的值。Packing 的过程就是生成这种特定的 `LaneID` 和 `RegisterID` 的映射逻辑，以便直接喂给 `mma.sync` 指令。

2.  **随路重排 (In-flight Data Shuffling)**：
    *   *原理*：为了避免显式的 Packing 开销，编译器利用硬件的 **Vector Load/Store** 指令特性，在从 L1/Shared Memory 加载数据到寄存器的**途中**完成重排。
    *   *实现*：通过生成特定的 **Permutation Mask** 或利用硬件支持的 **Block Load** 指令，使得数据进入寄存器时就已经“各就各位”，无需额外的 `Shuffle` 指令。

##### MLIR 实现：适配 Tensor Core 的 Block Layout

```cpp
// 原始：标准的 Row-major 矩阵乘法
// 问题：硬件 MMA 指令可能要求输入是 Blocked 格式
func.func @matmul_logical(%A: tensor<1024x1024xf32>, %B: tensor<1024x1024xf32>) {
  
  // 1. [Packing] 逻辑变换：将大矩阵视为小块的集合
  // 变换为：tensor<32x32x32x32xf32> (Outer_M, Outer_N, Inner_tile_m, Inner_tile_n)
  // 这一步通常融合在 Local Memory Tiling 的 Load 阶段
  %B_packed = tensor.pack %B
      inner_dims_pos = [0, 1]
      inner_tiles = [32, 32] 
      : tensor<1024x1024xf32> -> tensor<32x32x32x32xf32>

  // 2. [Intrinsic Compute] 使用打包后的数据执行计算
  // 此时数据布局完美匹配 nvgpu.mma 或 vector.contract 的输入要求
  %res = linalg.generic ... ins(%A_packed, %B_packed) ...
}
```

#### 6.5.2 VNNI/Dot-Product Packing (CPU Vectorization)

##### 背景
在通用 CPU（x86 AVX-512 或 ARM NEON）上执行 INT8 矩阵乘法时，硬件通常提供**点积指令**（如 `vpdpbusd` 或 `sdot`）。
这些指令要求输入数据在内存中具有特定的微观结构：例如，为了一次性加载 4 个 `int8` 元素并与另一个向量进行点积，数据必须在**归约维度（K轴）**上连续存放。
如果原始数据是普通的 Row-major，编译器必须执行 **Packing**，将逻辑上的 `[M, K]` 转换为物理上的 `[M, K/4, 4]`。

##### 技术原理

CPU 上的深度学习加速主要依赖 SIMD 指令集（如 AVX-512 VNNI 或 ARM NEON-DotProd）。这些指令引入了 **“垂直归约（Vertical Reduction）”** 的计算模式，要求数据在内存中进行微观重排。

1.  **归约维连续性 (Reduction-Dimension Contiguity)**：
    *   *问题*：标准的向量乘法是 `v1 * v2`（逐元素相乘）。但 INT8 点积指令（如 `vpdpbusd`）执行的是 $\sum_{i=0}^3 a_i \times b_i$。它要求参与归约的 4 个元素（INT8）必须打包在一个 32-bit 的**通道（Lane）**内。
    *   *原理*：Packing 将逻辑上的 **K 维度**（归约维）折叠到最内层。例如，将 `[K]` 变为 `[K/4, 4]`。
    *   *效果*：当 CPU 加载一个 32-bit 整数时，它实际上加载了逻辑上的 `A[k], A[k+1], A[k+2], A[k+3]`。这使得硬件能在一个时钟周期内完成这 4 个数的乘加运算。

2.  **权重预打包与常量折叠 (Weight Pre-packing & Constant Folding)**：
    *   *问题*：如果在推理运行时现场进行这种 Packing，重排数据的开销可能会抵消 VNNI 指令带来的加速。
    *   *原理*：由于模型推理中权重（Weights）是固定的，编译器将 `tensor.pack` 操作**上提（Hoist）** 到常量初始化阶段。
    *   *效果*：运行时加载的权重已经是物理上 `[K/4, N, 4]` 排布的数据，完全消除了 Packing 开销。

##### 硬件视角：Ascend NPU 的 AICPU (ARM) 协同

虽然 Ascend NPU 的主力是 Cube Unit（使用分形格式），但其 SoC 上还集成了强大的 **AICPU (基于 ARM 架构)**，用于处理 Cube 不支持的算子或复杂控制流。

1.  **AICPU 的 NEON DotProd 利用**：
    *   当编译器将某些小算子或不支持的算子回退（Fallback）到 AICPU 执行时，底层使用的是 ARM NEON 指令集。
    *   ARM v8.2+ 引入了 **SDOT (Signed Dot Product)** 指令。
    *   *融合策略*：Ascend 编译器在生成 AICPU 二进制代码时，同样会应用 VNNI 风格的 Packing 策略，将数据在 K 维度每 4 个一组打包，以触发 `SDOT` 加速，避免标量计算的低效。

##### MLIR Tensor Pack 实现

下面的示例展示了如何使用 `tensor.pack` 为 VNNI 指令准备数据。这一步通常作为**权重预处理（Weight Pre-packing）**在编译期完成，或者是作为 Constant Folding 的一部分。

```cpp
// 原始：INT8 权重矩阵 [K=1024, N=1024]
// 目标：适配 VNNI 指令，需要在 K 维度上每 4 个元素打一个包
// 物理布局变为：[K/4, N, 4] -> [256, 1024, 4]

func.func @vnni_packing(%weight: tensor<1024x1024xi8>) -> tensor<256x1024x4xi8> {
  
  // padding_value: 处理 K 维度无法被 4 整除的边界情况
  %pad = arith.constant 0 : i8

  // tensor.pack: 执行布局变换
  // inner_dims_pos = [0] 表示对第 0 维 (K) 进行切分
  // inner_tiles = [4] 表示切分大小为 4 (适配 32-bit 累加器: 4 * 8-bit)
  %packed_weight = tensor.pack %weight
      padding_value(%pad : i8)
      inner_dims_pos = [0]
      inner_tiles = [4]
      : tensor<1024x1024xi8> -> tensor<256x1024x4xi8>

  return %packed_weight
}

// 后续计算说明：
// 这个 %packed_weight 会被喂给 linalg.generic 或 vector.contract
// 后端编译器（LLVM）识别到 [..., 4] 的连续维度后，
// 会自动生成 vpdpbusd (x86) 或 sdot (ARM) 指令。
```

---

## 7. 控制流与动态性（Control-flow & Dynamism）

现实世界的 AI 模型往往包含分支跳转、变长序列或稀疏数据，这构成了对静态编译优化的最大挑战。

本章关注如何在**运行时具有不确定性**的环境下，依然维持高效的融合执行。核心目标是将动态行为**静态化**或**规范化**。编译器通过控制流的**扁平化（Flattening）**与**谓词化（Predication）**来规避分支发散，通过**符号化分析（Symbolic Analysis）**来处理未知的张量形状，并通过运行时的**即时特化（JIT Specialization）**，确保动态模型也能享受到静态融合带来的性能红利。

### 7.1 控制流扁平化与谓词融合（Control-Flow Flattening & Predication）

#### 7.1.1 Predication (Select-based Fusion)

##### 背景

AI 处理器（尤其是 GPU/NPU）偏好 SIMD/SIMT 并行，极其忌讳分支跳转（Branch Divergence）。
当算子内部存在条件逻辑（如 `ReLU`, `Dropout` 或分段函数）时，编译器不生成 `if-else` 跳转指令，而是采用**谓词化（Predication）**：同时计算两个分支的结果，然后使用 `Select` 指令根据条件掩码选择最终值。这使得带有控制流的算子依然可以被融合到由 `vector` 或 `linalg` 构成的密集计算循环中。

融合通常发生在**基本块（Basic Block）**内部。如果存在 `if` 跳转，基本块会被打断，阻碍指令调度和流水线优化。谓词化消除了跳转，使整个逻辑变成一个大的基本块，不仅方便融合，还让 Loop Vectorizer 能够轻松对循环进行 SIMD 化。

##### 技术原理

1.  **条件物化（Condition Materialization）**：
    首先计算条件表达式，生成一个**布尔掩码（Boolean Mask）**或谓词寄存器。对于向量处理器，这通常是一个与数据宽度一致的掩码向量。
    *   *Example*: `mask = (input > 0)`
2.  **推测性执行（Speculative Execution / Compute Both）**：
    无论条件如何，编译器会让硬件**同时计算两个分支**的结果。
    *   *True Path*: `res_true = compute_true_block(input)`
    *   *False Path*: `res_false = compute_false_block(input)`
    *   *注意*：这要求分支内的操作是**无副作用（Side-effect free）**的（例如纯算术运算）。如果包含内存写操作或异常抛出，则不能简单推测执行。
3.  **指令选择（Select / Blend）**：
    使用硬件提供的 `Select`、`CMOV`（Conditional Move）或 `Blend` 指令，根据掩码选择最终结果。
    *   *Logic*: `result = (mask & res_true) | (~mask & res_false)`
    *   *Semantics*: `result = select(mask, res_true, res_false)`

##### 代价模型与权衡 (Trade-off)

编译器并非对所有分支都进行谓词化融合，通常基于以下权衡：

*   **收益**：消除了分支预测失败的惩罚（CPU）或分支发散的串行化（GPU）；增加了指令级并行度（ILP）。
*   **成本**：执行了无用的计算（被丢弃的那个分支）。
*   **决策阈值**：仅当分支内的**指令数量较少**（如 ReLU, Clip, Thresholding）时，谓词化才是划算的。如果分支内包含矩阵乘法等重负载操作，推测执行的代价过大，此时应保留控制流。

##### 硬件指令映射

*   **x86 AVX-512**: `vpblendvb` (根据掩码混合两个向量)。
*   **ARM NEON/SVE**: `bsl` (Bitwise Select) 或 `sel` 指令。
*   **NVIDIA PTX**: `selp` (Select based on predicate register) 或利用 Predicate Register (`@p add.f32 ...`) 控制单条指令执行。
*   **Ascend NPU**：使用`Mask`机制，利用硬件的谓词寄存器支持在运行时控制数据流动。

##### MLIR 实现：从 SCF 到 Arith

```cpp
// 原始逻辑：显式的控制流分支 (难以向量化，难以融合)
// scf.if %cond { ... } else { ... }

// 优化后：谓词化融合 (Predicated Fusion)
// 所有的计算路径都被平铺，适合 SIMD 执行
func.func @relu_fused(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0.0 : f32
  
  %0 = linalg.generic ... ins(%arg0) ... {
    ^bb0(%in: f32):
      // 1. 生成谓词掩码 (Predicate Mask)
      %cond = arith.cmpf ogt, %in, %c0 : f32
      
      // 2. 同时"计算"两个分支 (一个是 %in，一个是 %c0)
      // 3. 使用 Select 指令融合
      %res = arith.select %cond, %in, %c0 : f32
      
      linalg.yield %res : f32
  }
  return %0
}
```

#### 7.1.2 隐式掩码融合 (即时计算掩码）

##### 背景
在长序列 LLM（如 Context Length = 128k）中，**Attention Mask** 是一个 $N \times N$ 的下三角矩阵。如果显式创建该 Tensor（Materialization），将消耗 $128k^2$ 个 bool/float 空间，导致瞬间显存溢出（OOM）。
**隐式掩码融合（Implicit Mask Fusion）**指编译器不分配物理内存存储 Mask，而是在 Attention Score 计算的 Kernel 内部，利用当前线程的坐标 `(row, col)` **即时计算**掩码值（On-the-fly Mask Generation）。

##### 技术原理
**Index-to-Value Fusion（索引即数值）**：
将“读取内存”操作替换为“逻辑比较”操作。
$$
Mask[i][j] = (i \ge j) ? 0 : -\infty $$
$$

这本质上是将**静态数据结构**转化为**动态控制流（谓词逻辑）**，彻底消除了 Mask 的内存占用。

##### MLIR 实现：Linalg Indexing

```cpp
// 融合 Causal Mask 到 Attention Score 计算
// 场景：Score = Softmax(Q * K^T + Mask)
// 优化：不读取 Mask Tensor，直接利用 linalg.index 动态生成

%scores = linalg.generic {
  indexing_maps = [ ... ], 
  iterator_types = ["parallel", "parallel"]
} ins(%Q, %K : ...) outs(%Out : ...) {

  ^bb0(%q: f32, %k: f32, %out: f32):
    // 1. 计算点积 Q * K^T
    %dot = arith.mulf %q, %k : f32
    
    // 2. [Implicit Fusion] 获取当前计算元素的几何坐标
    %row_idx = linalg.index 0 : index
    %col_idx = linalg.index 1 : index
    
    // 3. 生成 Causal Mask 逻辑
    // 判定条件：row >= col ?
    %is_causal = arith.cmpi sge, %row_idx, %col_idx : index
    
    // 4. 选择掩码值 (Predication)
    // 这是一个纯寄存器操作，无内存读取
    %c0 = arith.constant 0.0 : f32
    %neg_inf = arith.constant -1.0e+4 : f32
    %mask_val = arith.select %is_causal, %c0, %neg_inf : f32
    
    // 5. 融合到结果
    %masked_score = arith.addf %dot, %mask_val : f32
    linalg.yield %masked_score : f32
}
```

---

### 7.2 动态形状融合（Dynamic Shape Fusion）

#### 7.2.1 Symbolic Shape Analysis (符号化形状分析)

##### 背景

在动态批处理（Dynamic Batching）或 NLP 变长序列场景中，Tensor 的维度在编译期是未知的（`?`）。
传统的静态编译器遇到 `?` 通常会放弃融合，退化为解释执行或生成大量胶水代码。现代 AI 编译器通过**符号化分析**，在不知道具体数值的情况下，依然能够生成高效的融合算子。

##### 技术原理

1.  **符号化建模（Symbolic Modeling）**：
    编译器将所有的动态维度视为代数符号（Symbols，如 $s_0, s_1$）。所有的索引计算不再基于常量，而是基于**仿射表达式（Affine Expressions）**，例如 `(i, j) -> (i * s_1 + j)`。这使得编译器可以推理出内存访问的线性关系，即使步长 $s_1$ 是未知的。

2.  **约束满足性检查（Constraint Solving）**：
    为了安全地融合两个算子（如 `Add(A, B)`），编译器必须证明 $Shape(A) \equiv Shape(B)$。对于动态形状，编译器执行**SSA 值等价分析**：如果两个维度的定义来源（Definition Source）相同（例如都来自同一个 Input Argument 的第 0 维），则认为它们在运行时必然相等，允许融合。

3.  **形状实体化（Shape Reification）**：
    这是生成可执行代码的关键。编译器将形状计算逻辑从数据计算中剥离，生成一组独立的**标量运算指令**（通常是 index 类型）。这些指令在运行时率先执行，计算出具体的循环边界，然后喂给融合后的 Kernel。

##### MLIR 实现：基于 Dim 的动态融合

```cpp
// 动态形状融合示例
// 编译器不需要知道具体大小，只需要生成依赖运行时 Dim 的代码
func.func @dynamic_fusion(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // 1. [Reify] 获取运行时形状 (Reify Shapes)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %A, %c1 : tensor<?x?xf32>

  // 2. [Init] 使用动态尺寸创建输出 Tensor
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>

  // 3. [Generic] 融合算子体
  // linalg.generic 天然支持动态形状，因为它的循环边界由输入 (%A) 决定
  %res = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  }
  return %res
}
```

---

### 7.3 稀疏性与不规则融合（Sparsity & Irregular Fusion）

#### 7.3.1 Sparse-Dense Fusion (稀疏-稠密融合)

##### 背景

虽然"动态性"通常指形状变化，但**数据分布的动态性（稀疏性）**也是关键挑战。
当稀疏张量（Sparse Tensor）与稠密张量（Dense Tensor）进行运算时（如 GNN 或推荐系统），融合策略不能遍历整个空间，而必须**跟随稀疏索引（Sparse Indices）**进行迭代。
编译器利用 **Co-iteration（协同迭代）** 技术，只对非零元素的位置执行融合算子链，从而获得 $O(NNZ)$ 而非 $O(N^2)$ 的性能。

##### 技术原理

为了让编译器能够像处理稠密张量一样处理稀疏张量，并实现自动化融合，现代编译器（如 MLIR Sparse Compiler, TACO）采用了以下核心技术：

1.  **基于维度的层级抽象（Per-dimension Level Abstraction）**：
    编译器不硬编码 "CSR" 或 "COO" 格式，而是将稀疏格式解构为**维度属性**的组合。
    *   **Dense**: 该维度的所有坐标都存在（$O(1)$ 访问）。
    *   **Compressed**: 只存储非零元素的坐标（需要查表访问，如 `indices` 数组）。
    *   **Singleton**: 该维度只有一个元素（如 COO 格式中的坐标）。
        通过组合这些属性，编译器可以描述任意稀疏格式，并生成对应的遍历代码。

2.  **协同迭代与集合运算（Co-iteration & Set Operations）**：
    当融合两个稀疏张量（或稀疏+稠密）时，编译器必须生成能够**同步遍历**它们的循环逻辑。这本质上是**集合论**问题：
    *   **Sparse $\times$ Sparse (Intersection)**: 只有两个张量在位置 $i$ 都有值时才计算。编译器生成类似"双指针归并"的逻辑，跳过只要有一方为零的位置。
    *   **Sparse $+$ Sparse (Union)**: 只要有一方有值就计算。编译器生成复杂的 `while` 循环来处理指针的对齐和推进。
    *   **Sparse $\times$ Dense**: 编译器利用稀疏张量的索引去驱动对稠密张量的随机访问（Gather）。

3.  **循环重构（Loop Reconstruction）**：
    编译器将高层的 `linalg.generic` 降级为底层的 `while` 循环和间接寻址（Indirect Addressing）。
    *   *Dense Loop*: `for (i=0; i<N; i++)`
    *   *Sparse Loop*: `pos = ptr[i]; while (pos < ptr[i+1]) { idx = indices[pos]; ... }`

##### MLIR SparseTensor Dialect 实现

```cpp
// 稀疏融合：Sparse Matrix * Dense Vector + Bias
// 关键点：循环结构不是 dense 的 for-loop，而是由稀疏张量的压缩格式驱动
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

func.func @sparse_fusion(%sp_mat: tensor<?x?xf32, #CSR>, 
                         %vec: tensor<?xf32>, 
                         %bias: tensor<?xf32>) -> tensor<?xf32> {
  
  // 融合后的操作：linalg.generic 看起来和稠密一样
  // 但编译器后端会将其 Lowering 为"遍历非零元素索引"的复杂循环
  %res = linalg.generic {
    indexing_maps = ...
    iterator_types = ...
  } ins(%sp_mat, %vec, %bias) outs(...) {
    ^bb0(%a: f32, %b: f32, %c: f32, %out: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %prod, %c : f32
      linalg.yield %sum : f32
  }
  return %res
}
```

*注：扩展阅读 [MLIR的SparseTensor方言是如何分析矩阵的稀疏性的？](https://www.cnblogs.com/notlate-cn/p/19525701)*

---

### 7.4 运行时特化（Runtime Specialization）

#### 7.4.1 Just-In-Time (JIT) Specialization

##### 背景

在处理动态形状时，生成一个通用的 Kernel（处理任意 `?`）通常比针对特定形状（如 `1024`）生成的 Kernel 性能差（因为无法做常量折叠、向量化对齐或循环展开）。
**运行时特化**策略指：编译器保留一份通用的 IR 模板，在运行时检测到具体的形状参数（如 `Batch=1` 或 `Batch=32`）时，即时触发编译，生成**完全静态化（Static）**的高性能融合 Kernel。

##### 技术原理

编译器通过运行时特化提升性能，主要依赖以下三个底层机制：

1.  **常量提升与传播 (Constant Promotion & Propagation)**：
    在特化路径（Specialized Path）中，原本在运行时才能确定的变量（如 Batch Size = 1），被强制视为**编译期常量**。
    这使得编译器可以执行**常量折叠（Constant Folding）**，预先计算出所有与形状相关的偏移量（Offsets）和步长（Strides），将复杂的地址计算简化为立即数加法。

2.  **循环完全展开与向量化 (Loop Unrolling & Vectorization)**：
    这是特化带来的最大收益。
    *   **动态循环**：编译器必须生成循环头、循环体、循环尾（处理余数）以及边界检查代码，且不敢激进使用寄存器（因为不知道循环次数是否足以填满流水线）。
    *   **静态特化循环**：当 $N$ 已知且较小（如 $N=1$ 或 $N=4$）时，编译器可以**完全消除循环结构**，直接生成 $N$ 条线性的 FMA 指令。这不仅消除了跳转开销，还允许编译器进行精确的**寄存器分配（Register Allocation）**，实现 100% 的 ALU 利用率。

3.  **代码版本管理与分发 (Versioning & Dispatch)**：
    编译器采用 **Guard-based Dispatch** 策略。它不试图生成一个"万能优化的内核"，而是生成多个版本：
    *   **Fast Path**: 针对热点形状（如 `B=1`, `Seq=128`）的极致优化无分支代码。
    *   **Generic Path**: 带有完整循环开销的兜底代码。
        运行时的开销仅仅是几个整数比较指令（Guard Check），换来的是核心计算路径的数倍性能提升。

##### MLIR 实现：Shape Guarding & Versioning

下面的示例展示了如何处理动态 Batch 的矩阵乘法。编译器为最常见的 `Batch=1` 生成了特化路径。

```cpp
func.func @matmul_dispatch(%A: tensor<?x1024xf32>, %B: tensor<1024x1024xf32>) -> tensor<?x1024xf32> {
  
  // 1. [Guard] 获取运行时维度并进行检查
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  %batch_size = tensor.dim %A, %c0 : tensor<?x1024xf32>
  %is_batch_1 = arith.cmpi eq, %batch_size, %c1 : index

  // 2. [Dispatch] 使用 scf.if 进行分发
  %result = scf.if %is_batch_1 -> (tensor<?x1024xf32>) {
    
    // === 特化路径 (Specialized Path: Batch=1) ===
    // 关键点：使用 tensor.cast 将动态 ? 转换为静态 1
    // 这告诉编译器：在这个块内，A 的形状是确定的 1x1024
    %A_static = tensor.cast %A : tensor<?x1024xf32> to tensor<1x1024xf32>
    %init_static = tensor.empty() : tensor<1x1024xf32>

    // 此时 linalg.matmul 看到的是静态形状
    // 后端可以将其从循环 Lowering 为一串扁平的 FMA 指令 (全展开)
    %res_static = linalg.matmul ins(%A_static, %B : ...) outs(%init_static : ...)
    
    // 转换回动态类型以统一返回接口
    %res_cast = tensor.cast %res_static : tensor<1x1024xf32> to tensor<?x1024xf32>
    scf.yield %res_cast : tensor<?x1024xf32>

  } else {
    
    // === 通用/回退路径 (Fallback Path: Dynamic Batch) ===
    // 这里的 ? 仍然是动态的，生成带有循环开销的通用代码
    %init_dynamic = tensor.empty(%batch_size) : tensor<?x1024xf32>
    %res_dynamic = linalg.matmul ins(%A, %B : ...) outs(%init_dynamic : ...)
    
    scf.yield %res_dynamic : tensor<?x1024xf32>
  }

  return %result : tensor<?x1024xf32>
}
```

##### 性能收益解析

*   **Static Path (`Batch=1`)**：由于形状已知，后端编译器可以消除对 `Batch` 维度的循环，直接生成向量化的点积指令（如 `tensor<1x1024>` $\times$ `tensor<1024x1024>` 变为 1024 次向量乘加）。
*   **Dynamic Path**：必须保留外层循环 `for (i=0; i<N; ++i)`，并处理余数循环（Loop Tail），开销显著。

---

## 8. 跨层次全局优化（Cross-layer Global Optimization）

在前述章节中，我们介绍了单一维度的优化技术。然而，局部最优往往不等于全局最优。

本章跳出具体的算子变换，站在**编译器架构师**的视角，关注全图范围内的**决策（Decision Making）**与**搜索（Search）**。核心目标是解决不同优化策略之间的冲突。通过全图的布局传播算法、基于量化分析的**代价模型（Cost Model）**以及**调度与计算分离**的自动化搜索架构，编译器能够在庞大的优化空间中自动寻找出全局性能最佳的融合方案。

### 8.1 全局布局与Buffer传播（Global Layout & Buffer Propagation）

#### 背景

前面章节讨论了如何将 NCHW 转换为 NCHWc 以适配硬件（第3章、第6章）。但如果在每个算子前后都插入 `Pack`/`Unpack`，开销会抵消收益。
**全局布局优化**将布局选择视为一个**约束满足问题（CSP）**：在整个计算图中传播布局约束，只在必要的边界（如网络输入输出或 CPU/NPU 交互点）插入格式转换，确保 **Format Conversion 开销最小化**。

#### 技术原理

1.  **布局传播（Layout Propagation）**：类似于常量传播。如果 Op A 要求输出为 `Blocked_32`，则该约束会传播给 Op B 的输入。
2.  **冲突解决（Conflict Resolution）**：当 Op A 产出 `NCHW` 但 Op B 强制要求 `NHWC` 时，编译器需要在该边上插入 `Transpose` 或 `LayoutTransform`。全局求解器（如 Union-Find 算法）会寻找最小化插入代价的方案。

#### MLIR 实现：Bufferization 期间的布局传播

MLIR 在 `one-shot-bufferize` 阶段或专门的 Layout Pass 中执行此逻辑。

```cpp
// 原始图：Op1 (Unknown Layout) -> Op2 (Requires Blocked)
// 优化器决定将 Op1 的输出直接产生为 Blocked 布局，从而消除中间转换

func.func @layout_propagation(%input: tensor<?x?xf32>) {
  // Op2 要求 Blocked Layout (e.g., for Tensor Core)
  // 编译器反向传播，强制 Op1 直接输出 Blocked 格式
  
  // 1. [Global Decision] Op1 被重写为 Pack-Fused 版本
  %0 = linalg.generic ... outs(%blocked_buffer) ... // 直接产出 Blocked

  // 2. [Zero-Copy] Op2 直接读取 %0，无需 layout.transform
  %1 = linalg.matmul ins(%0, ...) ...
  
  return %1
}
```

---

### 8.2 代价模型驱动的融合决策 (Cost-Model Driven Fusion)

#### 背景
**"能融合"不代表"应该融合"**。

*   **过度融合（Over-fusion）**：将过多算子融合到一个 Kernel，会导致寄存器溢出（Spilling），降低 GPU Occupancy，或者破坏并行度（例如将 Reduce 与 Element-wise 强行融合可能导致并行度受限于 Reduce 的维度）。
*   **融合不足（Under-fusion）**：增加了不必要的内存读写。

**代价模型**通过量化计算密集度（Arithmetic Intensity）和硬件资源限制，做出"切断融合"的决策。

#### 技术原理
1.  **Roofline Model 分析**：计算算子子图的 FLOPs/Byte 比率。如果融合后不仅没提升，反而因为资源争抢导致性能下降，则拒绝融合。
2.  **饱和度分析（Saturation Analysis）**：分析融合后的 Loop Body 大小是否超过指令缓存（I-Cache）或寄存器堆限制。
3.  **贪心与动态规划**：在图上通过聚类算法（Clustering）寻找最优的融合子图切割点。

#### MLIR 实现：Transform Dialect 中的决策逻辑

MLIR 使用 `transform` dialect 来编写这种可编程的决策逻辑，而不是硬编码在 C++ Pass 中。

```cpp
// 这是一个描述"如何决策"的 Meta-Program
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  // 1. 找到所有 MatMul 算子
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
  
  // 2. [Cost Model Check] 检查是否值得 Tile 和 Fuse
  // 假设有一个自定义扩展 transform.check_profitability
  %profitable = transform.check_profitability %matmul { min_flops = 1000 }
  
  // 3. 只有收益高时才执行融合
  transform.if %profitable {
    %tiled, %loops = transform.structured.tile %matmul [32, 32]
    transform.structured.fuse_into_containing_op %tiled
  }
}
```

*注：扩展阅读 [如何基于 MLIR 实现代价模型驱动的融合决策机制](https://www.cnblogs.com/notlate-cn/p/19526452)*

---

### 8.3 自动调优与调度分离 (Auto-tuning & Schedule Separation)

#### 背景

对于通用 AI 处理器，最佳的 Tiling Size、Vector Length 和 Unroll Factor 随硬件架构剧烈变化。手动编写 Heuristics（启发式规则）很难覆盖所有场景。
**计算与调度分离**的思想（源于 Halide/TVM，在 MLIR 中通过 Transform Dialect 实现）允许编译器将"算什么（Compute）"保持不变，而通过搜索算法自动生成"怎么算（Schedule）"。

#### 技术原理
1.  **搜索空间构建（Search Space Construction）**：定义可调参数（Tile Size $\{16, 32, 64\}$, Vectorize $\{True, False\}$, Unroll $\{4, 8\}$）。
2.  **代码生成与评估（Codegen & Eval）**：生成多个版本的 Kernel，在真实硬件或模拟器上运行/评估。
3.  **机器学习预测（ML Cost Model）**：使用 XGBoost 或 GNN 预测某组参数的性能，避免全空间搜索（如 TVM Ansor）。

#### MLIR Transform Dialect 示例

这就是 MLIR 这一代编译器最先进的地方：它将调优过程变成了**IR 变换脚本**。

```cpp
// 计算部分 (IR 保持纯净)
func.func @matmul(%A, %B, %C) {
  linalg.matmul ins(%A, %B) outs(%C)
  return
}

// === 调度脚本 (由 Auto-tuner 生成) ===
// 自动调优器会生成成百上千个这样的脚本，寻找最优解
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %target = transform.structured.match ops{["linalg.matmul"]} in %arg1
  
  // 调优参数：Tile Sizes [128, 128, 16]
  %tiled, %loops:3 = transform.structured.tile %target [128, 128, 16]
  
  // 调优参数：Pad = True
  %padded = transform.structured.pad %tiled ...

  // 调优参数：Vectorize = True
  %vec = transform.structured.vectorize %padded
  
  // 调优参数：Bufferize = True
  transform.bufferization.one_shot_bufferize %arg1
}
```

*注：扩展阅读 [如何基于 MLIR 实现自动调优](https://www.cnblogs.com/notlate-cn/p/19526556)*

---

## 9. 特殊应用场景的融合策略映射

本章不再孤立地介绍应用场景，而是基于前 8 章建立的理论体系，对典型负载进行**“融合诊断”**。我们将分析每个场景的核心性能瓶颈，并通过**引用具体的章节编号**，展示如何组合多种融合技术来破解难题。

---

### 9.1 大语言模型优化 (LLM / Transformer)

**【融合诊断】**
LLM 的核心瓶颈在于 **Memory Wall (显存带宽)** 和 **Sequence Length (序列长度)** 的平方级复杂度。
*   **诊断 1 (Attention)**：标准 Attention 产生巨大的 $N \times N$ 中间矩阵，导致 HBM 读写爆炸。
    *   $\to$ **解法**：**4.1 多级分块** + **4.2.1 显式内存提升**（在 SRAM 内完成 Softmax）+ **7.1.2 隐式 Mask**（不存 Mask Tensor）。
*   **诊断 2 (Deep Depth)**：层数极深，导致 Kernel Launch 开销大且显存占用高。
    *   $\to$ **解法**：**1.3.2 Block Fusion**（融合 RoPE/RMSNorm）+ **6.2.1 Activation Checkpointing**（以算换存）。

**【核心技术映射】**

*   **FlashAttention (v2/v3)**: 是 **[4.1] Tiling** + **[4.2] Copy-Compute Overlap** + **[2.2] Loop-carried State** 的极致结合。
*   **SwiGLU Fusion**: 利用 **[1.1] 垂直融合**，在寄存器中完成 Gate * Up 操作。

**【硬件差异】**
*   **GPU**: 强依赖 **Shared Memory Swizzling [3.2.2]** 避免 Bank Conflict。
*   **Ascend NPU**: 强依赖 **UB Tiling [4.3]** 和 **Fractal Layout Packing [3.2.1]** 适配 Cube 单元。

---

### 9.2 混合专家模型 (Mixture of Experts, MoE)

**【融合诊断】**
MoE 的核心挑战在于 **Dynamic Routing (动态路由)** 导致的计算碎片化和负载不均衡。

*   **诊断 1 (Branching)**：不同 Token 走不同专家，控制流极其复杂。
    *   $\to$ **解法**：**7.1 控制流扁平化**（通过 Mask 处理）或 **5.3.4 Pipeline Fusion**。
*   **诊断 2 (Fragmentation)**：专家计算是大量的小矩阵乘法。
    *   $\to$ **解法**：**1.2.2 Batch Fusion**（Grouped GEMM）。

**【核心技术映射】**
*   **Grouped GEMM**: 将多个小 GEMM 拼成一个大 Kernel，利用 **[5.2.1] Thread Block Fusion** 减少启动开销。
*   **Expert Parallelism**: 本质是 **[5.3.1] SPMD Sharding** 的一种特殊形式（在专家维度切分）。

**【硬件差异】**
*   **GPU**: 依赖 **Triton JIT [7.4]** 动态生成 Grouped GEMM Kernel。
*   **Ascend NPU**: 倾向于 **Padding [3.3.1]**，将动态负载补齐为静态 Shape 以喂饱 Cube。

---

### 9.3 稀疏计算 (Sparse Computing)

**【融合诊断】**
稀疏计算的核心在于 **Irregular Memory Access (不规则访存)**，这与 SIMD/SIMT 硬件背道而驰。
*   **诊断 1 (Indexing)**：无法使用简单的仿射映射（Affine Map）。
    *   $\to$ **解法**：**7.3.1 Sparse-Dense Fusion**（协同迭代）。
*   **诊断 2 (Bandwidth)**：元数据（Indices）占用带宽。
    *   $\to$ **解法**：**3.2.1 Tensor Packing**（压缩元数据，如 2:4 稀疏）。

**【核心技术映射】**
*   **Sparse Co-iteration**: 编译器后端生成的 **[7.3] 稀疏循环**，替代标准的 `scf.for`。
*   **Block Sparsity**: 利用 **[3.1.1] Layout Propagation** 强制要求输入为 Blocked 格式。

**【硬件差异】**
*   **GPU**: 利用寄存器做 **Indirect Gather**。
*   **Ascend NPU**: 必须将稀疏块 **Densify (致密化)** 为 $16 \times 16$ 小块，利用 Cube 算力掩盖零填充。

---

### 9.4 量化部署 (Quantization)

**【融合诊断】**
量化的核心挑战是 **Type Mismatch (类型不匹配)** 和 **Accuracy (精度保持)**。
*   **诊断 1 (Bandwidth)**：FP32/16 传输太慢。
    *   $\to$ **解法**：**6.4.1 Quantization-Aware Fusion**（Prologue/Epilogue 融合类型转换）。
*   **诊断 2 (Alignment)**：INT8 指令对数据排布要求极高。
    *   $\to$ **解法**：**6.5.2 VNNI Packing**（指令级重排）。

**【核心技术映射】**
*   **Requantize Fusion**: 将 `Int32->FP32->Int8` 的 Scaling 逻辑融合到 **[6.3.1] Intrinsic** 的尾部。
*   **Weight Pre-packing**: 在编译期利用 **[3.2.1] Packing** 将权重重排为 VNNI/DotProd 格式。

**【硬件差异】**
*   **GPU**: 利用 **Tensor Core INT8**，配合 Shared Memory Swizzle。
*   **Ascend NPU**: 利用 **MTE On-the-fly Quant** 和 **Vector Fixpipe** 指令。

---

### 9.5 推荐系统 (DLRM)

**【融合诊断】**
推荐系统是典型的 **Memory-Bound (Embedding)** + **Compute-Bound (MLP)** 混合体。
*   **诊断 1 (Latency)**：Embedding 查表主要受限于 DRAM 延迟而非带宽。
    *   $\to$ **解法**：**5.4.2 Prefetch Fusion**（软件流水线）。
*   **诊断 2 (Tiny Ops)**：Gather 后的 Concat/Reshape 开销巨大。
    *   $\to$ **解法**：**1.2.1 Multi-output Fusion**（融合查表与后续处理）。

**【核心技术映射】**
*   **Fused Embedding**: 将 `Gather` + `Reduce` + `Quant` 融合，本质是 **[7.3] 稀疏融合** 的一种特例。
*   **Software Pipelining**: 利用 **[5.4.1] Async Fusion** 掩盖 Host-to-Device 拷贝延迟。

**【硬件差异】**
*   **GPU**: 利用 **UVM** 和 **Massive Parallelism**。
*   **Ascend NPU**: 利用专用 **DMA 引擎** 和 **Embedding Cache** 硬件特性。

---

### 9.6 状态空间模型 (Mamba / SSM)

**【融合诊断】**
SSM 引入了 **Sequential Scan (串行扫描)**，打破了 Transformer 的并行优势。
*   **诊断 1 (Dependency)**：$h_t$ 强依赖 $h_{t-1}$，无法直接并行。
    *   $\to$ **解法**：**2.2.2 Temporal Fusion**（状态寄存器传递）或 **Parallel Scan** 算法变换。
*   **诊断 2 (IO)**：中间状态 $h$ 极大，写回 HBM 不可接受。
    *   $\to$ **解法**：**4.2.1 Explicit Hierarchy**（SRAM 驻留）。

**【核心技术映射】**
*   **Hardware-aware Scan**: 利用 **[5.2.1] Warp Shuffle** 或 **Subgroup Fusion** 实现寄存器级扫描。
*   **Kernel Fusion**: 将 `Discretization` + `Scan` + `Proj` 融合，通过 **[1.3] 模式融合** 生成单 Kernel。

---

### 9.7 边缘设备 (Edge / Mobile)

**【融合诊断】**
边缘设备的核心约束是 **Strict Memory Limit (极小的 SRAM/RAM)**。
*   **诊断 1 (Peak Mem)**：无法容纳全图中间结果。
    *   $\to$ **解法**：**4.3.2 Operator Splitting**（Strip Mining）。
*   **诊断 2 (Reuse)**：内存分配开销大。
    *   $\to$ **解法**：**4.3.1 Static Planning**（编译期绝对地址分配）。

**【核心技术映射】**
*   **Liveness Analysis**: 构建 **[4.3] 干涉图**，实现极致的内存复用。
*   **Zero-copy**: 利用 **[3.4] Bufferization** 消除一切不必要的 Tensor Copy。

---

### 9.8 动态 Batch (Dynamic Batching)

**【融合诊断】**
Serving 系统中 Batch Size 随请求流量剧烈波动。
*   **诊断 1 (Unknown Shape)**：静态 Loop Unrolling 失效。
    *   $\to$ **解法**：**7.2.1 Symbolic Analysis**（生成通用 Kernel）或 **7.4.1 JIT Specialization**（运行时特化）。

**【核心技术映射】**
*   **Shape Guard**: 在运行时插入 **[7.1] 控制流**，根据 Batch 大小分发到不同的 **[7.4] 特化 Kernel**。
*   **Bucketing**: NPU 上通过 **[3.3.1] Padding** 将动态形状规整化。

---

## 附：主流编译器

### XLA (XLA: Accelerated Linear Algebra)

- **主要维度**：依赖拓扑 + 数据布局 + 跨层优化
- **核心融合技术**：
  - HLO Fusion (Vertical/Horizontal)
  - Layout Assignment
  - Buffer Assignment
- **适用场景**：TensorFlow, JAX 训练/推理
- **链接**：[XLA Documentation](https://www.tensorflow.org/xla)

---

### TVM (Tensor Virtual Machine)

- **主要维度**：全维度覆盖
- **核心融合技术**：
  - Tensor Expression Fusion
  - Auto-scheduling (AutoTVM/Ansor)
  - Multi-level Tiling
- **适用场景**：跨硬件部署
- **链接**：[TVM GitHub](https://github.com/apache/tvm)

---

### TensorRT

- **主要维度**：硬件适配 + 模式融合
- **核心融合技术**：
  - Layer Fusion
  - INT8 Calibration
  - Kernel Auto-tuning
- **适用场景**：NVIDIA GPU 推理
- **链接**：[TensorRT Documentation](https://developer.nvidia.com/tensorrt)

---

### Triton

- **主要维度**：内存层次 + 并行性
- **核心融合技术**：
  - Block-level Programming
  - Tile-based Fusion
  - Auto-tuning
- **适用场景**：GPU 自定义 kernel 开发
- **链接**：[Triton GitHub](https://github.com/openai/triton)

---

### MLIR (Multi-Level Intermediate Representation)

- **主要维度**：多级抽象 + 跨层优化
- **核心融合技术**：
  - Linalg Fusion
  - Affine Loop Fusion
  - Progressive Lowering
- **适用场景**：编译器基础设施
- **链接**：[MLIR GitHub](https://github.com/llvm/llvm-project/tree/main/mlir)

---

### TorchInductor

- **主要维度**：依赖拓扑 + 动态性
- **核心融合技术**：
  - Graph Pattern Matching
  - Triton Codegen
  - Dynamic Shape Support
- **适用场景**：PyTorch 2.0 推理/训练
- **链接**：[PyTorch 2.0 Inductor](https://pytorch.org/get-started/pytorch-2.0/)

---

## 参考资料

1. [MLIR Documentation](https://mlir.llvm.org/)
2. [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
3. [Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
4. [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
5. [IREE Compiler](https://iree.dev/)
6. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
