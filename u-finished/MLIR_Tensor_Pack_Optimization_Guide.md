# MLIR Tensor Pack 性能优化实战指南

本文深入解析 MLIR 中 `tensor.pack` 操作的核心原理，并通过一个完整的矩阵乘法（GEMM）优化示例，讲解如何利用它解决 **Cache Line 利用率**、**向量化友好性（Vectorization Friendliness）** 和 **Bank Conflict** 三大性能痛点。

## 1. 核心概念：什么是 `tensor.pack`？

`tensor.pack` 是 MLIR `tensor` dialect 中的一个高层操作，用于改变张量的数据布局（Layout）。它将逻辑上连续的张量重新组织为**分块（Blocked/Tiled）** 的物理布局。

*   **输入**：标准的行优先（Row-Major）或列优先张量。
*   **输出**：嵌套的 Tile 结构。例如，将 `[M, N]` 变为 `[M/tile_m, N/tile_n, tile_m, tile_n]`。

**为什么需要它？**
现代硬件（CPU SIMD, GPU Tensor Core, TPU）偏好处理小块的连续数据。如果数据在内存中是离散的（Stride访问），性能会急剧下降。`tensor.pack` 负责将"逻辑相邻"的数据搬运到"物理相邻"的位置。

---

## 2. 三大性能杀手与 `tensor.pack` 的解法

### 2.1 Cache Line Utilization (缓存行利用率)
*   **问题**：在计算 $C_{ij} += 
s
k A_{ik} 	imes B_{kj}$ 时，如果 $B$ 是行优先存储，随着 $k$ 的增加，访问 $B_{kj}$ 需要跳跃 $N$ 个元素。这导致每次加载一个 Cache Line (通常 64字节) 只能用到其中一个 float，浪费了带宽。
*   **解法**：使用 `tensor.pack` 将 $B$ 的 $K$ 维打包到最内层。使得内层循环访问 $B$ 时是连续的内存地址，充分利用 Cache Line 加载的每一个字节。

### 2.2 Vectorization Friendliness (向量化友好性)
*   **问题**：SIMD 指令（如 AVX-512）一次需要加载 16 个 float。如果数据在内存中不连续，编译器需要生成 gather/scatter 指令，或者多次 load + shuffle，效率极低。
*   **解法**：通过 Packing，将最内层维度的大小（tile size）设置为向量寄存器的长度（或其倍数）。这样可以直接生成 `vmovups` 等高效向量加载指令。

### 2.3 Bank Conflict & Padding (存储体冲突与边界)
*   **问题**：
    1.  **Bank Conflict**: 在 GPU Shared Memory 或某些 L1 Cache 设计中，如果访问步长（Stride）恰好是 Bank 数量的倍数，会导致所有线程争抢同一个 Bank，串行化执行。
    2.  **边界检查**: 当矩阵尺寸不能被 Tile Size 整除时，循环内部需要大量的 `if (i < M)` 检查，破坏流水线。
*   **解法**：`tensor.pack` 支持 `padding_value`。
    1.  通过 Padding 补齐到 Tile Size 的整数倍，消除 Loop 内的边界检查分支。
    2.  Padding 后的数据在物理上是连续且对齐的，通常能天然避免部分 Bank Conflict（取决于具体架构的 Bank 映射函数的互质特性）。

---

## 3. 实战场景：GEMM 优化

我们以矩阵乘法 $Matmul(A, B) o C$ 为例。
假设场景：$M=128, N=128, K=128$。目标硬件支持 SIMD 宽度为 16 (例如 f32 on AVX-512)。

### 3.1 Baseline: 朴素的 Linalg Matmul

最原始的 IR，数据都是默认的 Row-Major。

```cpp
func.func @matmul_naive(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %res = linalg.matmul 
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
```

**性能痛点分析**：
对于 `linalg.matmul` 的内层 Reduction Loop (沿着 K 轴)：
*   读取 `A` 是连续的（Row-Major, Iterate over columns）。
*   读取 `B` 是**不连续**的（Row-Major, Iterate over rows）。**这是性能杀手。**

### 3.2 优化策略：Packing B 矩阵

我们希望在计算核心（Micro Kernel）中，沿着 K 维规约时，能连续读取 B 的数据。
因此，我们将 B 从 `[K, N]` 布局转换为 `[N/n_tile, K/k_tile, k_tile, n_tile]`。
这里我们选择 `n_tile = 16` (匹配 SIMD 宽度), `k_tile = 4` (Micro Kernel 深度)。

#### 变换后的 MLIR 代码 (手工模拟 `tensor.pack` 效果)

```cpp
func.func @matmul_packed(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // 1. Pack B 矩阵
  // inner_dims_pos = [0, 1] 表示我们将 K(0) 和 N(1) 维度都进行 Tiling 放入内层
  // outer_dims_perm = [1, 0] 表示外层循环先遍历 N，再遍历 K (Column-Major block layout)
  // 结果 layout: [N/16, K/4, 4(k_tile), 16(n_tile)]
  %B_packed = tensor.pack %B
    padding_value(0.0 : f32) // 可选：如果尺寸不匹配自动补0，消除后续边界检查
    outer_dims_perm = [1, 0] 
    inner_dims_pos = [0, 1] 
    inner_tiles = [4, 16] 
    into %empty_packed_B : tensor<128x128xf32> -> tensor<8x32x4x16xf32>

  // 2. 也是通常需要的：Pack C 矩阵 (为了累加时的写对齐)
  // layout: [M/m_tile, N/n_tile, m_tile, n_tile]
  %C_packed = tensor.pack %C
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 16]
    into %empty_packed_C : tensor<128x128xf32> -> tensor<16x8x8x16xf32>

  // 3. 执行计算 (此时是在 Packed Layout 上进行的 Generic 操作)
  // 注意：linalg.matmul 会被 lower 或者是替换为 linalg.generic
  // 这里展示概念性的 Packed Matmul
  %res_packed = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>, // A (未Pack, 假设这里只Pack了B和C演示重点)
      affine_map<(m, n, k) -> (n floordiv 16, k floordiv 4, k mod 4, n mod 16)>, // B (Packed)
      affine_map<(m, n, k) -> (m floordiv 8, n floordiv 16, m mod 8, n mod 16)>  // C (Packed)
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B_packed : tensor<128x128xf32>, tensor<8x32x4x16xf32>)
    outs(%C_packed : tensor<16x8x8x16xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32): 
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x8x8x16xf32>

  // 4. Unpack 结果回原始布局
  %res = tensor.unpack %res_packed
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 16]
    into %C : tensor<16x8x8x16xf32> -> tensor<128x128xf32>
    
  return %res : tensor<128x128xf32>
}
```

### 3.3 深入分析：为什么这样就变快了？

让我们聚焦到最内层的计算核心。当编译器将上述 `linalg.generic` Lowering 到 Loops 后，针对 `B_packed` 的访问模式会发生质变。

#### 3.3.1 解决 Cache Line 和 Vectorization
**Before (Row-Major B):**
Loop K:
  Load `B[k][n]`
  Load `B[k+1][n]` -> **Stride = N (128 floats)**。Cache Miss 高，无法向量化加载。

**After (Packed B):**
`B_packed` 的最内层维度是 `n_tile = 16`。
Loop Inner (Micro Kernel):
  Load Vector `B_tile[0...15]` -> **连续内存访问**。
  *   **Cache**: 一次 Cache Line Fill (64 bytes = 16 floats) 刚好填满需要的数据。利用率 100%。
  *   **Vectorization**: 直接映射为一条 `vmovups` (AVX) 或 `ld1` (NEON)。

#### 3.3.2 解决 Padding 和 Branching
如果原始 N = 130，而我们的 `n_tile = 16`。
*   **Before**: 内层循环必须写 `if (n < 130)`.
*   **After**: `tensor.pack` 指定了 `padding_value=0.0`。
    *   Packing 阶段：会将边界外的 14 个元素填 0。
    *   Compute 阶段：循环按照 `ceil(130/16) * 16 = 144` 次执行。完全没有 `if` 分支。
    *   结果：虽然多算了几个 0 的乘加，但消除了流水线停顿（Branch Misprediction），通常净收益是正的。

#### 3.3.3 Bank Conflict (Implicit)
通过将数据重排为 `[TileK, TileN]` 的小块，数据在局部存储（如 L1 D-Cache）中的映射变得更加规整。
例如，如果 L1 Cache 是 N-way Set Associative，行优先的大 stride 访问容易导致 Cache Thrashing（频繁替换同一 Set 的不同 Line）。
Packed Layout 保证了短时间内访问的数据集中在一个连续的内存块中，极大降低了 Conflict Miss 的概率。

## 4. 总结与最佳实践

1.  **Always Pack the Reduction Dimension**: 对于 Matmul，总是尝试 Pack "K" 维和 "N" 维的组合，使得对 B 的读取变成连续的 Block 读取。
2.  **Match Tile Size to Hardware**: `inner_tiles` 的大小应该直接对应硬件的 SIMD 宽度（CPU）或 Tensor Core 形状（GPU, e.g., 16x16）。
3.  **Hoist Packing**: `tensor.pack` 本身有数据搬运开销。最佳实践是将 `tensor.pack` **提升（Hoist）** 到主循环之外，或者如果 B 是常量权重（Inference 场景），在编译期这就变成了常量折叠（Constant Folding），运行时零开销。
4.  **Use Padding**: 勇敢使用 `padding_value`，用少量的无效计算换取控制流的极度简化。

通过 `tensor.pack`，MLIR 提供了一种优雅的方式，在不改变算子语义的前提下，显式地控制数据的物理布局，从而精确打击现代处理器的性能痛点。
