# AI编译器融合技术系统化分类总结

## 1. 依赖拓扑（Dependency Topology）

依赖拓扑优化关注算子之间的数据依赖关系，通过分析和重组计算图来减少内存开销和提升执行效率。这是编译器前端优化中最核心的环节。

### 1.1 垂直融合（Vertical Fusion）

垂直融合沿着数据流的生产者-消费者链条进行优化，是最经典的融合模式。

#### 1.1.1 Producer-Consumer Fusion（生产者-消费者融合）

##### 背景

在深度学习模型中，算子之间往往存在严格的数据依赖关系：后一个算子需要前一个算子的输出作为输入。这种依赖链形成了 **"生产者-消费者"**（Producer-Consumer）关系。

传统执行模式下，每个算子独立编译成 kernel，中间结果需要写入全局内存（Global Memory），然后下一个 kernel 再重新读取。这种 **"物化"**（Materialization）过程带来了显著的开销：

- **内存带宽压力**：每个中间张量都需要写回主存再读出
- **Cache利用率低**：中间结果可能驱逐有用的Cache内容
- **Kernel启动开销**：每个算子单独启动一个Kernel

**核心思想**

**消除中间物化（Intermediate Materialization Elimination）**：将生产者的计算内联到消费者中，使中间结果仅存活于寄存器/L1 Cache，避免写回全局内存。

```cpp
// 融合前
C = Add(A, B)        // 写入全局内存
D = ReLU(C)          // 从全局内存读取C 

// 融合后
C = ReLU(Add(A, B))  // 中间结果仅存活于寄存器
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
    // 原始 ReLU 操作（中间结果 %sum 无需物化）
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

**技术原理**

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

**背景**

标准的垂直融合通常在 **归约（Reduction）** 操作处断开。例如 Softmax 或 LayerNorm，传统实现需要多次遍历内存（Pass 1: 求和/均值 $\to$ Pass 2: 归一化）。

**归约-逐元素融合** 旨在打破归约操作的同步屏障，通过**数学算法的重写**（如 Welford 或 Online Softmax），将多次内存扫描合并为一次扫描（One-pass）。

**核心思想**

1. **Welford 算法 (For LayerNorm)**：
   - **传统**：先遍历一次求 Mean，再遍历一次求 Variance，最后遍历求 Output。
   - **融合**：在单次循环中同时维护 Mean 和 Variance 的迭代更新公式，一次遍历即可得到最终统计量并应用 Normalization。
2. **Online Softmax / Safe Softmax**：
   * **传统**：$max = reduce_{max}(x)$ -> $sum = reduce_{sum}(e^{x - max})$-> $out = \frac{e^{x-max}}{sum}$ (3 Pass)。
   * **融合**：利用数学性质 $\frac{e^{x_i - \max}}{\sum e^{x_j - \max}}$，在一次遍历中动态更新全局 Max 和 Sum，无需预先扫描最大值。

 **应用场景**

| 算子                    | 涉及算法          | 收益                    |
| ----------------------- | ----------------- | ----------------------- |
| **LayerNorm / RMSNorm** | Welford Algorithm | 减少 1-2 次全局内存读写 |
| **Softmax**             | Online Softmax    | 减少 2 次全局内存读写   |
| **Cross Entropy Loss**  | Log-Sum-Exp Trick | 提升数值稳定性与性能    |

这种融合无法通过简单的 `linalg.fuse` 实现，通常需要使用 `scf.for` 携带状态（`iter_args`）来表达复杂的更新逻辑。

**MLIR 示例**

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

**Multi-output Fusion**（也称 **Sibling Fusion**）针对**多个算子共享同一输入**的场景。如果每个消费者独立执行，共享输入会被多次加载。水平融合将这些计算合并，实现“一次读取，多次计算”。

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
   融合后：同时存储所有中间结果需要 R × N 寄存器
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

#### 1.3.2 Multi-Head Attention Fusion (FlashAttention Style)

##### 背景

Transformer 的 **Multi-Head Attention (MHA)** 是模型性能的决定性瓶颈。

标准实现（`MatMul(Q,K) -> Softmax -> MatMul(S,V)`）存在两大痛点：

1.  **$O(N^2)$ 内存复杂度**：中间生成的 Attention Score 矩阵形状为 $[Batch, Heads, Seq, Seq]$。当序列长度 $Seq$ 增长时，该矩阵占用的显存呈平方级增长。
2.  **Memory Wall（内存墙）**：在传统执行模式下，这个巨大的 $N \times N$ 矩阵需要被完整写入全局内存（HBM），再重新读回以进行 Softmax 和下一次 MatMul。这种频繁的 HBM 读写导致的延迟远超计算本身的耗时（IO-bound）。

因此，MHA 融合不仅仅是简单的算子合并，而是一种**IO 感知的算法级融合（IO-aware Algorithmic Fusion）**，旨在利用 Tiling 技术完全消除 $N \times N$ 矩阵对 HBM 的访问。

##### 核心原理

通过 **Tiling（分块）** 和 **Recomputation（重计算）**，将所有计算限制在片上 SRAM 中进行，完全消除 $O(N^2)$ 的中间结果物化。

```cpp
传统实现 (Standard Attention):
1. S = Q @ K^T          (Write S to HBM, size N^2)
2. P = Softmax(S)       (Read S, Write P to HBM, size N^2)
3. O = P @ V            (Read P, Write O to HBM)

融合后 (FlashAttention / Memory-Efficient Attention):
Block-wise loop:
  Load block of Q, K, V into SRAM
  Compute block of S = Q_i @ K_j^T (on SRAM)
  Compute block of P = Softmax(S)  (on SRAM, using Online Softmax)
  Compute block of O += P @ V_j    (on SRAM, accumulate to Output)
  (中间矩阵 S 和 P 从未离开过片上 SRAM)
```

##### MLIR 实现：Tiled Attention Logic

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

*注：扩展阅读 [MLIR如何高效实现Attention？](https://www.cnblogs.com/notlate-cn/p/19522984)*

---

## 2. **循环与迭代空间优化（Loop & Iteration Space Optimization）**

### 2.1 循环融合（Loop Fusion）

#### 2.1.1 Loop Fusion & Tiling

##### 背景

在底层 IR（如 Affine 或 LLVM IR）中，**循环融合**是指将两个具有相同或兼容**迭代空间**的相邻循环合并为一个循环体的代码变换技术。

与图层面的算子融合不同，循环融合更关注**指令执行流**的优化。其核心收益不仅仅是减少全局内存访问，还包括：
1.  **提升时间局部性（Temporal Locality）**：将数据的**"定义"**与**"使用"**拉近，使其在**寄存器文件**或 **L1/L2 Cache** 中保持活跃（Hot），避免被驱逐。
2.  **减少循环控制开销**：减少了循环计数器的增量指令、条件分支跳转指令（Branch）以及未命中的分支预测惩罚。
3.  **隐式同步消除**：在并行编程（如 OpenMP 或 CUDA）中，两个独立的循环之间通常隐含一个**同步信号（Barrier）**。融合后，这个屏障被消除，减少了线程空转等待。

##### 约束

循环融合并非总是合法的。编译器必须进行严格的**依赖分析（Data Dependency Analysis）**，确保融合后的执行顺序不会破坏原有的读写依赖（例如：不能将“先读后写”的依赖变成“先写后读”）。

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
    // %C 的 store 操作被消除，中间结果 %sum 直接用于下一步计算
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

在执行**归约（Reduction）**（如求和、点积）或**自回归（Autoregressive）**计算时，当前迭代的输入依赖于上一迭代的输出。
*   **传统低效模式**：每次迭代都从内存（或栈）中读取状态变量，更新后再写回内存。
*   **优化目标**：**寄存器提升（Register Promotion）**。将循环携带的状态变量（Accumulator/State）强行驻留在 CPU/GPU 的**寄存器**中，在整个循环执行期间不发生溢出（Spill）到内存的操作。

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
在处理循环神经网络（RNN/LSTM/GRU）时，存在严格的时间序列依赖：$State_t = f(Input_t, State_{t-1})$。
**时间融合（Temporal Fusion）** 指将一个时间步（Time Step）内的所有操作（MatMul, Activation, Gate Operations）融合为一个 Kernel，同时将**Hidden State** 保持在高速缓存或寄存器中传递给下一个时间步，而不是写回全局显存。

##### 挑战

*   **串行依赖**：时间步之间无法并行，因此融合重点在于减少每个时间步的**启动开销（Launch Overhead）**和**状态读写开销**。

##### 技术实现
在 MLIR 中，这通常表现为 `scf.for` 内部嵌套复杂的 `linalg` 操作，随后通过 **Loop Unrolling**（循环展开）和 **Invariant Code Motion**（不变量外提）来优化权重矩阵的加载。

---

### 2.3 **循环展开与流水线（Loop Unrolling & Pipelining）**

此技术通过重组循环内的指令调度，最大化硬件单元的利用率，主要解决指令流水线气泡和内存延迟问题。

#### 2.3.1 Loop Unrolling (Instruction Level Parallelism)

##### 背景

现代 CPU/GPU 拥有超标量架构（Superscalar），每个时钟周期可发射多条指令。紧凑的循环（Tight Loop）由于频繁的分支跳转检测（Branch Compare & Jump），会打断指令流水线。
**循环展开**通过复制循环体代码，减少跳转次数，并暴露更多的独立指令供硬件调度器进行**指令级并行（ILP）**优化。

##### 收益
1.  **减少分支开销**：$N$ 次迭代变成 $N/K$ 次跳转。
2.  **向量化机会**：展开后的连续访存指令更容易被合并为向量加载（Vector Load）。

##### MLIR 实现

在 `affine` 或 `scf` 方言中，可以通过属性标记或变换 Pass 显式控制展开因子。

```cpp
// 原始循环
affine.for %i = 0 to 1024 {
  %x = affine.load %A[%i]
  %y = arith.mulf %x, %c2
  affine.store %y, %A[%i]
}

// 展开因子 = 4 (Unroll Factor = 4)
// -affine-loop-unroll="unroll-factor=4"
affine.for %i = 0 to 1024 step 4 {
  // 编译器生成 4 个独立的计算链，允许 CPU 并行发射指令
  %x0 = affine.load %A[%i]
  %x1 = affine.load %A[%i+1]
  %x2 = affine.load %A[%i+2]
  %x3 = affine.load %A[%i+3]
  // ... 计算 x0...x3 ...
  // ... 存储 x0...x3 ...
}
```

#### 2.3.2 Software Pipelining (Latency Hiding)

##### 背景
在深度学习算子（如 GEMM, Attention）中，从全局内存（HBM）加载数据到片上缓存（SRAM/Register）的延迟极高（数百个时钟周期）。

如果采用 **Load $\to$ Compute** 的串行模式，计算单元在等待数据时会空转。
**软件流水线**（配合双缓冲/多缓冲 Double Buffering）将不同迭代的阶段重叠执行：在计算当前块（Tile $i$）的同时，预取下一块（Tile $i+1$）的数据。

##### 机制

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

本章关注数据在内存中的物理组织形式。优秀的布局能最大化内存带宽利用率，并满足特定硬件指令（如 Tensor Core, AVX-512）的数据格式要求。

### 3.1 全局布局优化（Global Layout Optimization）

#### 3.1.1 Layout Propagation & Assignment

##### 背景

不同算子对内存布局有不同偏好（例如：Conv2d 在 GPU 上偏好 `NHWC`，在 NPU 上可能偏好 `NC1HWC0`）。
如果在图中频繁插入 `Transpose` 或 `Reshape`，数据搬运开销将抵消计算收益。**布局传播**旨在为整个子图选择统一的最佳布局。

##### 策略

*   **Source-Sink Propagate**：从对布局敏感的“锚点算子”（如 Conv）开始，向上下游传播布局约束。
*   **Layout Transform Elimination**：利用数学性质消除冗余转换（例如：`Transpose(Transpose(x)) == x`）。

##### MLIR 实现

在 `linalg` 层级，通过**索引映射（Indexing Maps）**的置换（Permutation）来隐式表达布局，而不是物理 Transpose。

```cpp
// 逻辑上的 Transpose (不产生数据搬运，仅改变访问模式)
%output = linalg.matmul {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,  // A: Row-major
    affine_map<(m, n, k) -> (k, n)>,  // B: Column-major
    affine_map<(m, n, k) -> (m, n)>   // C: Output
  ]
} ins(%A, %B : ...) outs(%C : ...)
```

---

### 3.2 数据打包与微布局（Data Packing & Micro-layout）

#### 3.2.1 Tensor Packing (Block Layout)

##### 背景

现代加速器（GPU Tensor Cores, CPU AMX/AVX-512）通常只能高效处理**特定形状的微块**。
例如，AVX-512 偏好 `nchw16c`（通道维每16个元素打包在一起），NVIDIA Tensor Core 偏好特定的 Swizzle 布局。
**Packing（打包）** 指将逻辑上连续的张量重组为**分块物理布局**，以保证内存访问的连续性（Coalesced Access）。

##### 机制

将逻辑维度拆分并重排：`[N, C, H, W] -> [N, H, W, C/k, k]`。

##### MLIR 实现：`tensor.pack` / `tensor.unpack`
MLIR 引入了专门的 Op 来表达这种变换，这对 CPU/GPU 推理优化至关重要。

```cpp
// 将 NCHW 转换为 NCHW8c (Inner tile size = 8)
// 这对于向量化 (Vectorization) 至关重要
%packed = tensor.pack %input
  inner_dims_pos = [1]       // 在 Channel 维进行打包
  inner_tile_sizes = [8]     // 块大小为 8
  : tensor<1x32x224x224xf32> -> tensor<1x4x224x224x8xf32>

// 融合计算：在 Packed 布局上直接执行 Conv
%result = linalg.conv_2d ... ins(%packed ...)
```

#### 3.2.2 Swizzled Layout (For Shared Memory)

##### 背景
在 GPU Shared Memory 中，为了避免 **Bank Conflict**（存储体冲突），通常需要对数据进行 **Swizzling**（交错排列）。Triton 等编译器将 Layout 视为类型系统的一部分。

##### MLIR/Triton 示例
```cpp
// Triton-MLIR 中的 Layout Encoding
// #blocked 表示数据在线程间的分布方式
// #dot_op 表示针对 MMA (Matrix Multiply Accumulate) 优化的布局
%a = triton_gpu.convert_layout %src 
     : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dot_op>
```

### 3.3 填充与对齐（Padding & Alignment）

#### 3.3.1 Dimension Padding

##### 背景
硬件指令通常要求张量维度是特定数值（如 16, 32, 128）的倍数。如果实际 Shape 不满足（如 `batch=3`），需要**Padding** 到最近的倍数（如 `batch=4`），否则需要生成带有大量 `if` 判断的低效代码（Scalar loop）。

##### MLIR 实现
`tensor.pad` 操作用于显式填充，通常在 Tiling 之前或之后进行，以确保每个 Tile 都是完整的（Full Tile）。

```cpp
// 填充到 4 的倍数
%padded = tensor.pad %input low[0, 0] high[0, 1] {
  ^bb0(%arg0: index, %arg1: index):
    %c0 = arith.constant 0.0 : f32
    tensor.yield %c0 : f32
} : tensor<3x3xf32> to tensor<3x4xf32>
```

---

### 3.4 缓冲区化与原地更新（Bufferization & In-place）

#### 3.4.1 One-Shot Bufferization

##### 背景

在 MLIR 的上层（Tensor Level），数据是**不可变（Immutable）**的（SSA 语义），类似函数式编程。
但在底层（Memref Level），数据必须映射到具体的内存地址，并进行**可变（Mutable）**的读写。
**Bufferization** 是编译器将 Tensor 转换为 Memref 的过程。如果处理不好，会产生大量的 `alloc` 和 `memcpy`。

##### 核心技术：In-place Update
分析 Tensor 的 `Use-Def` 链，判断何时可以直接复用输入 Buffer 来存储输出（In-place），而非分配新内存。

##### MLIR 实现
MLIR 采用 **One-Shot Bufferize** pass，这是一种基于全局分析的算法。

```cpp
// Tensor Level (Value Semantics)
%0 = tensor.insert_slice %sub into %dest[...]

// ↓ Bufferized (Side-effect Semantics)
// 编译器分析出 %dest 在此处是最后一次使用，因此可以直接写入
memref.subview %view = %dest_mem[...]
memref.copy %sub_mem, %view : memref<...> to memref<...>
// 无需 alloc 新内存
```

---

## 4. 内存层次与多级分块（Memory Hierarchy & Tiling）

在深度学习加速器（GPU/TPU/NPU）中，计算单元的算力往往远超内存带宽（即 **Memory Wall** 问题）。
**内存层次优化**的核心目标，是将算子融合与硬件的存储层级（**HBM -> L2 -> L1/Shared -> Register**）对齐。通过**多级分块（Tiling）**，编译器将计算限制在高速缓存（Cache）内完成，使中间结果**在寄存器或 Shared Memory 中直接流转**，彻底消除对全局内存的冗余读写。

### 4.1 多级分块（Multi-level Tiling）

#### 4.1.1 Register/L1/L2 Tiling

##### 背景

为了掩盖 DRAM 的高延迟，必须将大张量切分为适应各级 Cache 大小的 **Tile（图块）**。
**Tile-local Fusion** 的本质是将原本串行的“全图算子”，转换为在 Tile 粒度上紧密耦合的“子图融合”，确保生产者（Producer）生成的 Tile 在被驱逐出 Cache 前，立刻被消费者（Consumer）使用。

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
大多数 AI 加速器（GPU/TPU/NPU/DSP）都包含用户可控的**片上高速缓存（Scratchpad Memory）**（如 GPU 的 Shared Memory 或 TPU 的 Local Memory）。
**内存提升（Promotion）** 指将频繁访问的数据从慢速的主存（HBM/DDR）显式搬运到片上高速缓存。为了避免数据搬运阻塞计算单元，必须利用硬件的 **DMA 机制** 实现**拷贝与计算的时间重叠（Overlap）**。

##### 关键技术

* **Asynchronous Data Movement (异步数据搬运)**：
    利用硬件独立的 **DMA 引擎** 或 **异步拷贝指令**，在计算单元（ALU/Tensor Core）处理当前数据的同时，后台静默地预取下一块数据。
    *   *通用性说明*：在 NVIDIA GPU 上映射为 `cp.async`，在 Ascend NPU 上映射为 `DataCopy`，在 TPU 上映射为 DMA 指令。

*  **Double Buffering (双缓冲/乒乓机制)**：
    一种软件流水线（Software Pipelining）策略。分配两块片上缓存（Buffer A 和 Buffer B），当计算单元处理 Buffer A 时，DMA 引擎填充 Buffer B，交替进行。

* **Bank Conflict Avoidance (存储冲突避免)**：
    针对采用多体存储（Banked Memory）架构的片上缓存，通过 **Padding（填充）** 或 **Swizzling（地址重排）** 优化数据布局，防止并行访问冲突。

##### MLIR 示例

```cpp
// 这是一个通用的“异步拷贝 + 融合计算”模式
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

1.  **Liveness Analysis**：构建张量的 Define-Use 链，确定其存活的时间区间。
2.  **Graph Coloring / Greedy Allocation**：将内存分配问题转化为区间着色问题。

##### MLIR 策略
*   **Bufferization**：在将 Tensor 降级为 Memref 时，使用 `BufferDeallocation` pass 插入 `dealloc`，并合并 `alloc`。
*   **In-place Bufferization**：尽可能复用输入 buffer 作为输出 buffer（如果输入不再被使用）。

```mlir
// 内存复用示例
func.func @reuse_example() {
  // Buf1 分配
  %buf1 = memref.alloc() : memref<1MB>
  call @op1(%buf1)
  // op1 结束后 buf1 不再活跃
  
  // Buf2 复用 Buf1 的物理地址，不进行新的 malloc
  %buf2 = memref.view %buf1 ... : memref<1MB>
  call @op2(%buf2)
}
```

#### 4.3.2 Memory-Constrained Operator Splitting

##### 背景

当单个算子（即使是分块后）所需的临时空间超过硬件限制（如 SRAM 大小或 TPU 内存），或者为了适配特定的内存 Bank 限制，编译器必须将算子**拆分（Split）**为多次执行。

此技术常用于**超大模型训练**（Activation Checkpointing 也是一种变体）或**受限内存嵌入式推理**。

##### 策略

*   **Strip Mining**：将大循环拆分为多个小循环，每次只处理一部分数据，降低瞬时内存占用。

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

本章节关注如何通过融合技术最大化利用硬件的并行能力，涵盖了从**指令 -> 线程 -> 核心 -> 芯片 -> 集群**的全层级优化。

### 5.1 指令与向量级并行（Instruction/Vector Parallelism）

#### 5.1.1 SIMD Vectorization Fusion

##### 背景

**SIMD**（Single Instruction Multiple Data）允许单条指令处理多个数据元素。这是 CPU (AVX/SVE)、DSP (Hexagon) 以及部分 NPU 的基础并行方式。
编译器通过 **Loop Vectorizer** 将多个标量操作融合为向量指令，并利用 **Predication/Masking** 技术处理非对齐的边界条件。

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

##### MLIR Mesh Dialect 示例

```cpp
// 定义通用的设备拓扑：2x4 的加速器集群
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

##### 伪代码逻辑

```cpp
// 融合优化（Pipelined Schedule）：
// 编译器自动插入 Event Record/Wait 实现流同步
Stream_Compute:  [Comp A0]  [Comp A1]  [Comp A2]
Stream_Comm:                [Send A0]  [Send A1]  [Send A2]
// 此时 Comp A1 与 Send A0 实现时间轴上的融合（重叠）
```

##### MLIR 伪代码示例

下面的示例展示了反向传播中的**流水线优化**：当计算资源（Compute）正在计算当前层的梯度时，网络资源（Network）正在后台异步发送上一层计算好的梯度。

```cpp
// 初始化一个空的 token，表示没有待处理的通信
%init_token = async.create_group

// 循环遍历层 (Backward Pass): i = Layers down to 0
%final_token = scf.for %i = %n_layers to 0 step -1 iter_args(%prev_comm_token = %init_token) {
  
  // 1. [Compute] 计算当前层的梯度 (计算密集型)
  // 这个操作会阻塞主线程，直到计算完成
  %grads = linalg.matmul ... : tensor<...>

  // 2. [Overlap] 启动异步通信任务 (网络密集型)
  // async.execute 创建一个新的流/线程，不阻塞主线程继续执行下一次循环
  // [%prev_comm_token] 表示依赖关系：可选，确保通信顺序
  %curr_comm_token = async.execute [%prev_comm_token] {
    
    // 执行集合通信 (AllReduce)
    // 此时主线程已经进入下一次循环计算下一层的 %grads 了
    mesh.all_reduce %grads on @device_mesh ... 
    
    async.yield // 任务结束
  }

  // 将当前的通信 token 传递给下一次迭代
  // 注意：这里没有调用 async.await，从而实现了计算与通信的并行
  scf.yield %curr_comm_token
}

// 在整个图的末尾等待所有通信完成
async.await %final_token
```

---

### 5.4 任务级与异构并行（Task & Heterogeneous Parallelism）

#### 5.4.1 Async & Multi-stream Fusion

##### 背景

在异构系统（Host CPU + Device Accelerator）中，利用**事件（Event）**和**流（Stream/Queue）**机制，将 CPU 逻辑、DMA 数据搬运和 Device 计算融合在同一个时间窗口内并发执行。

##### MLIR Async Dialect

```cpp
// 硬件无关的异步任务图融合
%t1, %f1 = async.execute { 
  // Task 1: 独立分支计算 (如 Query Projection)
  linalg.matmul ... 
  async.yield %res : tensor<?xf32>
}
%t2, %f2 = async.execute { 
  // Task 2: 另一分支 (如 Key Projection)
  linalg.conv2d ...
  async.yield %res : tensor<?xf32>
}
// 融合点：等待两个逻辑流完成
// 运行时会自动映射到 GPU Streams 或多线程池
async.await %t1, %t2
%final = linalg.add %f1, %f2
```

#### 5.4.2 Host-Device Prefetch Fusion (预取融合)

##### 背景

在推荐系统（Embedding Lookup）或图神经网络等场景中，Host（CPU）到 Device（加速器）的数据搬运往往是瓶颈。
如果采用简单的串行模式（Copy -> Compute），加速器会频繁处于**饥饿（Starvation）**状态。
编译器通过**软件流水线（Software Pipelining）**技术，将“当前批次的计算”与“下一批次的搬运”融合在同一个时间窗口内，实现 Host 与 Device 的全并行。

##### 优化逻辑

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

本章节探讨编译器如何根据具体硬件特性（寄存器堆大小、专用加速单元、内存对齐要求）来制定融合策略，在“算力”与“带宽”之间寻找最优解，涵盖了从**资源限制**、**重计算**、**专用指令映射**到**量化融合**的全方位内容。

### 6.1 资源感知内核融合（Resource-Aware Kernel Fusion）

#### 6.1.1 Register Pressure Controlled Fusion（寄存器压力控制融合）

##### 背景

虽然融合能减少内存访问，但它会显著增加**寄存器压力（Register Pressure）**。
如果融合的算子过多，活跃变量（Live Intervals）的总量超过硬件寄存器堆（Register File）容量，就会导致**寄存器溢出（Register Spilling）**到慢速内存（Local Memory/Stack），性能反而暴跌。

##### 技术原理

编译器构建**代价模型（Cost Model）**：

1.  **估算活跃变量数**：$Regs_{fused} \approx \sum Regs_{ops} - Shared_{inputs}$
2.  **Occupancy 阈值检查**：在 GPU 上，寄存器使用过多会限制同时运行的 Wavefront/Warp 数量（Occupancy），降低并行度。
3.  **Cut Strategy**：当预测寄存器不足时，编译器主动“切断”融合，插入显式的 Store/Load。

##### MLIR 实现逻辑

MLIR 通常通过 `transform` dialect 或后端 pass 来控制这种切分。

```cpp
// 场景：一个巨大的 Element-wise 链，可能耗尽寄存器
// 编译器决策：将其切分为两个 Kernel，而不是融合为一个

// Kernel 1: 生产中间结果 %temp
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
**重计算（Rematerialization）** 是一种以“时间换空间”的策略：为了避免存储某个中间张量（Activation），编译器选择在消费者算子中**重新计算**它，而不是从内存读取。

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
// 因此删除了全局的 %b，将 Op B 的逻辑“克隆”并内联到 Op C 之前

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

通用 AI 处理器（GPU/TPU/NPU）通常包含专用的矩阵加速单元（如 Tensor Core, AMX, Matrix Core, Ascend NPU AIC）。
这些单元通常只支持特定的**形状（Shape）**（如 16x16）和**布局（Layout）**。编译器必须将高层的 `MatMul` 算子进行 **Tiling + Packing**，并融合为一个能直接映射到硬件指令（Intrinsic）的形态。

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
```

---

### 6.4 混合精度与量化融合（Mixed Precision & Quantization Fusion）

#### 6.4.1 Quantization-Aware Fusion

##### 背景

在推理端，量化（INT8/FP8）是主流。量化通常包含 `Dequantize` (反量化) -> `Compute` (计算) -> `Quantize` (量化) 的流程。
如果这些转换操作独立执行，带宽开销巨大。**量化感知融合**将 `Dequant` 和 `Quant` 算子分别融合到主计算算子的**输入端（Prologue）**和**输出端（Epilogue）**。

##### 技术原理

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
编译器在融合过程中，会执行**类型传播（Type Propagation）**，消除那些**“来回转换”**的冗余 Cast 操作，确保数据尽可能保持在低精度格式下流动，仅在累加器（Accumulator）中临时提升精度。

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

---

### 6.5 **指令级数据打包（Instruction-Specific Packing）**

*注意：与第 3 章的全局布局优化不同，本节关注为了适配特定硬件指令（如 Tensor Core 或 VNNI）的输入格式要求，而在寄存器或 L1 传输层级进行的微观数据重排。*

#### 6.5.1 Intrinsic-Compatible Packing (Tensor Packing)

##### 背景

现代加速器的专用单元（如 NVIDIA Tensor Core 或 Intel AMX）通常要求输入数据遵循特定的**块状布局（Blocked Layout）**（例如：将矩阵切分为 $32 \times 32$ 的小块，或者在通道维度进行 $4$ 元素交错）。
如果全局内存布局是标准的 Row-major，编译器必须在数据加载到寄存器之前，通过 `tensor.pack` 将其转换为硬件指令兼容的**物理布局**。

##### 技术原理

**Pack/Unpack Fusion**：编译器不生成单独的 Packing Kernel，而是利用 `vector.transfer_read` 的 permutation map 或硬件提供的 Block Load 指令，在**从 Cache 加载数据到寄存器的过程中**完成重排。

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

本章节探讨当计算图包含条件分支、循环不变量或运行时可变形状时，编译器如何通过**扁平化（Flattening）**、**符号化（Symbolization）**和**特化（Specialization）**技术，在维持语义正确性的前提下进行算子融合。

### 7.1 控制流扁平化与谓词融合（Control-Flow Flattening & Predication）

#### 7.1.1 Predication (Select-based Fusion)

##### 背景

通用 AI 处理器（特别是 GPU/TPU）喜欢 SIMD/SIMT 并行，极其厌恶分支跳转（Branch Divergence）。
当算子内部存在条件逻辑（如 `ReLU`, `Dropout` 或分段函数）时，编译器不生成 `if-else` 跳转指令，而是采用**谓词化（Predication）**：同时计算两个分支的结果，然后使用 `Select` 指令根据条件掩码选择最终值。这使得带有控制流的算子依然可以被融合到由 `vector` 或 `linalg` 构成的密集计算循环中。

融合通常发生在**基本块（Basic Block）**内部。如果存在 `if` 跳转，基本块会被打断，阻碍指令调度和流水线优化。谓词化消除了跳转，使整个逻辑变成一个大的基本块，不仅方便融合，还让 Loop Vectorizer 能够轻易地对循环进行 SIMD 化。

##### 技术原理

1.  **条件物化（Condition Materialization）**：
    首先计算条件表达式，生成一个**布尔掩码（Boolean Mask）**或谓词寄存器。对于向量处理器，这是一个与数据宽度一致的掩码向量。
    *   *Example*: `mask = (input > 0)`
2.  **推测性执行（Speculative Execution / Compute Both）**：
    不管条件是真还是假，编译器生成代码让硬件**同时计算两个分支**的结果。
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
*   **Ascend NPU**：`Mask`机制

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
      
      // 2. 同时“计算”两个分支 (一个是 %in，一个是 %c0)
      // 3. 使用 Select 指令融合
      %res = arith.select %cond, %in, %c0 : f32
      
      linalg.yield %res : f32
  }
  return %0
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

虽然“动态性”通常指形状变化，但**数据分布的动态性（稀疏性）**也是关键挑战。
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
    *   **Sparse $\times$ Sparse (Intersection)**: 只有两个张量在位置 $i$ 都有值时才计算。编译器生成类似“双指针归并”的逻辑，跳过只要有一方为零的位置。
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
  // 但编译器后端会将其 Lowering 为“遍历非零元素索引”的复杂循环
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
    编译器采用 **Guard-based Dispatch** 策略。它不试图生成一个“万能优化的内核”，而是生成多个版本：
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

本章节不再关注单个算子的融合技术，而是探讨编译器如何在全图范围内解决**局部最优与全局最优的冲突**。它涵盖了全局布局流转、基于代价模型的融合决策以及自动化搜索架构。

### 8.1 全局布局与Buffer传播（Global Layout & Buffer Propagation）

#### 背景
在前几章中，我们讨论了如何将 NCHW 转换为 NCHWc 以适配硬件（第3章、第6章）。但如果在每个算子前后都插入 `Pack`/`Unpack`，开销会抵消收益。
**全局布局优化**将布局选择视为一个**约束满足问题（CSP）**：在整个计算图中传播布局约束，只在必要的边界（如网络输入输出或 CPU/NPU 交互点）插入格式转换，使整个图的 **Format Conversion 开销最小化**。

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

**代价模型**负责量化计算密集度（Arithmetic Intensity）和硬件资源限制，做出“切断融合”的决策。

#### 技术原理
1.  **Roofline Model 分析**：计算算子子图的 FLOPs/Byte 比率。如果融合后不仅没提升，反而因为资源争抢导致性能下降，则拒绝融合。
2.  **饱和度分析（Saturation Analysis）**：分析融合后的 Loop Body 大小是否超过指令缓存（I-Cache）或寄存器堆限制。
3.  **贪心与动态规划**：在图上通过聚类算法（Clustering）寻找最优的融合子图切割点。

#### MLIR 实现：Transform Dialect 中的决策逻辑

MLIR 使用 `transform` dialect 来编写这种可编程的决策逻辑，而不是硬编码在 C++ Pass 中。

```cpp
// 这是一个描述“如何决策”的 Meta-Program
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
**计算与调度分离**的思想（源于 Halide/TVM，在 MLIR 中通过 Transform Dialect 实现）允许编译器将"算什么（Compute）"保持不变，而通过搜索算法自动生成“怎么算（Schedule）”。

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

以下场景是多个理论维度的**组合应用**：

### 1. 大语言模型优化 (LLM Optimization)

**涉及维度**：依赖拓扑 + 内存层次 + 并行性

**核心融合技术**：

- **FlashAttention**：IO 感知的 Tile-local Fusion，融合 Softmax 与 Score 计算。
- **Operator Fusion**：`MatMul` + `Bias` + `GELU` / `RMSNorm`。
- **KVCache Fusion**：将 Paged Attention 的索引计算与搬运融合。

**硬件视角差异**：

- **GPU (NVIDIA)**：
  - **策略**：利用 **Shared Memory** 和 **Tensor Core** 异步拷贝（Cp.Async）。高度依赖 JIT（如 Triton）生成的动态 Kernel 来处理变长序列（VarLen）。
  - **瓶颈**：HBM 带宽与 SRAM 容量。
- **NPU (Ascend/TPU)**：
  - **策略**：倾向于 **Static Shape**。对于变长序列，通常采用分桶（Bucketing）+ Padding 策略，以适配 Cube 单元的固定分形格式。高度依赖 **L1/UB 融合**（将 RMSNorm 等向量操作完全留在片上）。
  - **瓶颈**：Cube 与 Vector 单元之间的数据格式转换。

**代表系统**：

- **GPU**: [FlashAttention](https://github.com/Dao-AILab/flash-attention), [vLLM](https://github.com/vllm-project/vllm), [Triton](https://github.com/openai/triton)
- **NPU**: [MindSpore Lite](https://www.mindspore.cn/), [CANN (Ascend)](https://www.hiascend.com/software/cann)

---

### 2. 混合专家模型 (Mixture of Experts, MoE)

**涉及维度**：并行性 + 控制流 + 硬件适配

**核心融合技术**：

- **Grouped GEMM Fusion**：将路由到不同专家的多个小矩阵乘法，融合为单次 Kernel 启动（解决 GPU Launch Overhead）。
- **Token Permutation Fusion**：将 Token 的路由排序、Gather/Scatter 操作与计算流水线融合。

**硬件视角差异**：

- **GPU**：
  - **策略**：利用 **Triton/CUTLASS** 实现高效的 Grouped GEMM，支持不同大小的矩阵混合计算。利用原子操作（Atomics）进行梯度的重排聚合。
- **NPU**：
  - **策略**：倾向于 **Padding Fusion**。将分配给不同专家的 Token 数填充到相等，以便利用 Cube 单元进行规则的 Batch MatMul，牺牲部分计算量换取流水线满载。

**代表系统**：

- [MegaBlocks](https://github.com/stanford-futuredata/megablocks) (Sparse Kernels)
- [Tutel](https://github.com/microsoft/tutel) (Microsoft MoE Optimization)
- [DeepSpeed-MoE](https://www.deepspeed.ai/)

---

### 3. 推荐系统 (Recommendation Systems / DLRM)

**涉及维度**：内存层次 + 异构并行 + 数据布局

**核心融合技术**：

- **Fused Embedding Lookup**：`Gather` (查表) + `Pooling` (求和) + `Quant` (量化) 三合一，通常是 Memory-bound 的瓶颈。
- **Prefetch Fusion (Pipelining)**：将 Host-to-Device 的 Embedding 搬运与 MLP 计算进行软件流水线融合（Double Buffering）。

**硬件视角差异**：

- **GPU**：
  - **策略**：**Massive Parallelism**。利用海量线程处理稀疏查表请求。对于超大表，使用 **UVM (Unified Virtual Memory)** 技术融合 CPU/GPU 内存空间。
- **NPU**：
  - **策略**：**DMA Assist**。利用专用的 DMA 引擎或 Embedding Cache 硬件（如 Ascend 的 HBM-Cache 机制）在后台自动预取数据，计算单元（Vector Core）只负责归约。

**代表系统**：

- [TorchRec](https://github.com/pytorch/torchrec) (Meta PyTorch RecSys)
- [HugeCTR](https://github.com/NVIDIA/HugeCTR) (NVIDIA Optimized DLRM)
- [AITemplate](https://github.com/facebookincubator/AITemplate) (Meta Codegen)

---

### 4. 量化部署 (Quantization Deployment)

**涉及维度**：硬件适配 + 数据布局 + 混合精度

**核心融合技术**：

- **Quant-Dequant Fusion**：将量化/反量化节点融合进主计算节点的 Prologue/Epilogue。
- **Weight Packing**：将 INT4/INT8 权重重排以适配 SIMD/MMA 指令。

**硬件视角差异**：

- **GPU**：
  - **策略**：**Mixed Precision (FP16/INT8)** 动态切换。利用 Tensor Core (INT8/FP8) 加速，同时保持 Accumulator 为 FP32。
  - **技术**：Layout Transform (NCHW $\to$ NC/32HW32) 在 Shared Memory 加载时融合完成。
- **NPU**：
  - **策略**：**Full Integer (全整型)** 推理。要求算子间传递的都是 INT8/INT32，极力避免回退到 FP32。
  - **技术**：离线校准（Offline Calibration）融合，生成固定的 Scale/Shift 参数，直接烧录进指令中。

**代表系统**：

- **GPU**: [TensorRT](https://developer.nvidia.com/tensorrt), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **NPU**: [SNPE (Qualcomm)](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk), [Ascend AMCT](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha001/developmenttools/devtool/atlasamct_16_0004.html)

---

### 5. 动态 Batch (Dynamic Batching)

**涉及维度**：控制流 + 内存层次

**核心融合技术**：

- **Continuous Batching**：在 Serving 系统中动态拼凑不同长度的请求。
- **Shape Polyhedral Analysis**：符号化形状分析。

**硬件视角差异**：

- **GPU**：
  - **策略**：**JIT Compilation**。运行时根据 Batch Size 动态生成特化 Kernel（如 Batch=1 路径 vs Batch=32 路径）。
- **NPU**：
  - **策略**：**Bucketing (分桶) + Padding**。NPU 编译器（AOT）通常需要预置若干固定档位（如 Batch 1, 4, 8, 16）。运行时将实际 Batch Pad 到最近的档位，因为 NPU 的指令调度器通常不支持完全动态的循环边界。

**代表系统**：

- **GPU**: [TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html), [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **NPU**: [MindSpore (Dynamic Shape)](https://www.mindspore.cn/docs/en/master/design/dynamic_graph_and_static_graph.html)

---

### 6. 稀疏计算 (Sparse Computing)

**涉及维度**：数据布局 + 硬件适配

**核心融合技术**：

- **Sparse-Dense Fusion**：稀疏矩阵（A）与稠密矩阵（B）乘法的融合。
- **Structured Sparsity**：利用硬件支持的结构化稀疏（如 2:4）进行压缩存储与计算。

**硬件视角差异**：

- **GPU**：
  - **策略**：支持 **Fine-grained (细粒度)** 或 **2:4 结构化稀疏**。利用寄存器重用（Register Reuse）处理非零元素，依赖 `cuSPARSE` 或 `Magic Cube` 库。
  - **实现**：Indirect Memory Access (Gather/Scatter)。
- **NPU**：
  - **策略**：通常不支持非结构化稀疏。强依赖 **Block Sparsity (块稀疏)**，因为 DMA 搬运和 Cube 计算都需要对齐（如 16x16）。融合策略通常是将稀疏块“致密化”为小块后计算。
  - **实现**：Masked Computation 或 Block Indexing。

**代表系统**：

- **GPU**: [NVIDIA APEX (2:4)](https://github.com/NVIDIA/apex), [SparTA](https://github.com/microsoft/SparTA)
- **NPU**: [TVM Sparse](https://tvm.apache.org/), [TACO](http://tensor-compiler.org/)

---

### 7. 图神经网络 (GNN)

**涉及维度**：数据布局 + 依赖拓扑 + 稀疏性

**核心融合技术**：

- **Fused Message Passing**：`Gather` (邻居信息) + `Compute` (MLP) + `Scatter` (聚合) 融合。避免中间巨大的边列表（Edge List）物化。
- **Reordering Fusion**：在运行时对节点 ID 进行重排（Reorder）以提升 Cache 命中率。

**硬件视角差异**：

- **GPU**：
  - **策略**：**Load Balancing**。由于图的幂律分布（邻居数差异大），需要动态调度线程块以平衡负载。利用 Shared Memory 缓存热点节点的特征。
- **NPU**：
  - **策略**：**Sparse-to-Dense + Padding**。将邻接矩阵切分为规则小块，对于稀疏块也作为稠密块计算（乘零），或者使用特定的 Masked MatMul 指令。

**代表系统**：

- [PyG (PyTorch Geometric)](https://github.com/pyg-team/pytorch_geometric) (High-level Library)
- [DGL (Deep Graph Library)](https://github.com/dmlc/dgl) (High-level Library)
- [GraphIt](https://graphit-lang.org/) (Graph Compiler)
- [FeatGraph](https://amazon-science.github.io/FeatGraph/)

---

### 8. 状态空间模型 (State Space Models, SSM / Mamba)

**涉及维度**：循环依赖 + 内存层次 + 算法简化

**核心融合技术**：

- **Parallel Scan Fusion**：将原本串行的状态更新（Scan）转化为**分块并行（Chunk-wise Parallel）**的前缀和扫描（Prefix Sum）。
- **IO-Aware Kernel Fusion**：将 SSM 的 `Discretization`（离散化） + `Scan` + `Output Projection` 融合为单个 Kernel，完全避免 $H$（Hidden State）写回 HBM。

**硬件视角差异**：

- **GPU**：
  - **策略**：**Warp Shuffle / Register Scan**。利用 GPU 寄存器之间的高速通信能力，在一个 Warp 或 Thread Block 内完成扫描。
- **NPU**：
  - **策略**：**SRAM-resident Scan**。依赖 Vector Unit 的高带宽，将 Chunk 数据加载到 UB 后，在片上完成循环扫描，极力避免与 Global Memory 交互。

**代表系统**：

- [Mamba (Official Implementation)](https://github.com/state-spaces/mamba)
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) (Tile-based CUDA Kernels)
- [Triton](https://github.com/openai/triton) (Used for writing custom scan kernels)

---

### 9. 边缘设备 (Edge Devices)

**涉及维度**：硬件适配 + 内存层次 + 算子拆分

**核心融合技术**：

- **Operator Splitting (Strip Mining)**：受限于极小的 SRAM，将大算子拆分为小时间片执行。
- **Memory Planning Fusion**：严格的生命周期管理，实现内存复用。

**硬件视角差异**：

- **Mobile GPU (Adreno/Mali)**：
  - **策略**：**Texture Memory Optimization**（利用纹理缓存优化访存）。通常使用 OpenCL/Vulkan，融合策略主要针对 Image Layout。
- **Mobile NPU/DSP (Hexagon/Apple NE)**：
  - **策略**：**Zero-copy Fusion**。通过共享系统内存（System RAM）和特定的硬件 DMA 通道，实现 CPU 与 NPU 之间的无拷贝数据流转。

**代表系统**：

- **通用/GPU**: [TensorFlow Lite](https://www.tensorflow.org/lite), [MNN (Alibaba)](https://github.com/alibaba/MNN), [TNN (Tencent)](https://github.com/Tencent/TNN)
- **NPU/DSP**: [QNN (Qualcomm)](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk), [CoreML](https://developer.apple.com/documentation/coreml)

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
