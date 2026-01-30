# MLIR Tile-based 编程深度指南

本文档全面介绍 MLIR 中的 Tiling（分块）技术。从基础的循环分块到高级的算子融合（Tile and Fuse），结合详细的代码示例和原理分析，帮助开发者掌握这一高性能计算优化的核心技术。

---

## 1. 核心概念

**Tiling** 是将大循环或大矩阵分解为小块（tiles）的技术，旨在：
*   **提升缓存局部性**：让当前计算的数据块能完全放入 L1/L2 缓存。
*   **利用并行性**：不同的小块可以分配给不同的线程或计算单元。

MLIR 通过 **`TilingInterface`** 统一了不同 Dialect（如 Linalg, Tensor, Mesh）的 Tiling 行为，使得变换逻辑（Transform Dialect）可以复用于所有实现了该接口的算子。

---

## 2. 基础分块示例

### 2.1 示例 1：Affine Loop Tiling

Affine Dialect 提供了最直观的循环分块。

**原始代码**：
```mlir
func.func @loop_tiling() {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}
```

**应用 `--affine-loop-tile=tile-size=32` 后**：
```mlir
func.func @loop_tiling() {
  // 外层 tile 循环，步长 32
  affine.for %ti = 0 to 256 step 32 {
    affine.for %tj = 0 to 512 step 32 {
      affine.for %tk = 0 to 1024 step 32 {
        // 内层点循环，处理单个 tile
        affine.for %i = %ti to min(%ti + 32, 256) {
          affine.for %j = %tj to min(%tj + 32, 512) {
            affine.for %k = %tk to min(%tk + 32, 1024) {
              "test.foo"(%i, %j, %k) : (index, index, index) -> ()
            }
          }
        }
      }
    }
  }
  return
}
```

### 2.2 示例 2：Linalg Tiling (tile_using_for)

针对结构化算子（如矩阵乘法），使用 Transform Dialect 进行分块。

**原始矩阵乘法**：
```mlir
func.func @matmul(%A: tensor<25x34xf32>, %B: tensor<34x25xf32>,
                  %C: tensor<25x25xf32>) -> tensor<25x25xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<25x34xf32>, tensor<34x25xf32>)
                     outs(%C: tensor<25x25xf32>)
    -> tensor<25x25xf32>
  return %0 : tensor<25x25xf32>
}
```

**Transform 脚本 (tile_size=9)**：
```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %tiled, %loop = transform.structured.tile_using_for %0
      tile_sizes [9] : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

**变换后结果**：
系统生成了 `scf.for` 循环，利用 `extract_slice` 提取子张量，计算后用 `insert_slice` 插入回结果。
```mlir
func.func @matmul(...) -> tensor<25x25xf32> {
  %c9 = arith.constant 9 : index
  %c0 = arith.constant 0 : index
  %c25 = arith.constant 25 : index

  // Tile 循环
  %result = scf.for %idx = %c0 to %c25 step %c9
    iter_args(%out = %C) -> (tensor<25x25xf32>) {

    // 提取 slice
    %A_tile = tensor.extract_slice %A[%idx, 0] [9, 34] [1, 1]
    %C_tile = tensor.extract_slice %out[%idx, 0] [9, 25] [1, 1]

    // 在 tile 上执行 matmul
    %tile_result = linalg.matmul ins(%A_tile, %B : tensor<9x34xf32>, tensor<34x25xf32>)
                                  outs(%C_tile : tensor<9x25xf32>)
                   -> tensor<9x25xf32>

    // 插入回结果
    %inserted = tensor.insert_slice %tile_result into %out[%idx, 0] [9, 25] [1, 1]
    scf.yield %inserted : tensor<25x25xf32>
  }
  return %result : tensor<25x25xf32>
}
```

---

## 3. 并行分块：`tile_using_for` vs `tile_using_forall`

随着多核 CPU 和 GPU 的普及，显式并行循环 `scf.forall` 变得至关重要。

### 3.1 核心对比

| 特性 | `tile_using_for` | `tile_using_forall` |
| :--- | :--- | :--- |
| **生成的 IR** | `scf.for` (顺序循环) | `scf.forall` (多维并行循环) |
| **并行性** | 隐式，需后续 pass 识别 | 显式并行 |
| **返回句柄** | 返回**每一个**生成的循环层句柄 | 返回**唯一一个** `scf.forall` 算子句柄 |
| **分块指定** | 仅支持 `tile_sizes` | 支持 `tile_sizes` 或 `num_threads` (线程数) |
| **硬件映射** | 不支持 | 支持 `mapping` (如映射到 GPU Thread/Block) |
| **适用场景** | 顺序嵌套优化、精细循环层控制 | 并行计算、GPU 核函数生成 |

### 3.2 示例：映射到 GPU
```mlir
// 将 M, N 维度分块并映射到 GPU 的 Block X 和 Y
%tiled, %forall = transform.structured.tile_using_forall %matmul
  tile_sizes [16, 16]
  (mapping = [#gpu.block<y>, #gpu.block<x>])
```

---

## 4. Tile and Fuse 模式（核心实战）

本章详细解析 **Tile and Fuse** 技术，这是优化复杂数据流图的关键。

### 4.1 背景：为什么需要 Tile and Fuse？

考虑一个典型的神经网络计算子图：**全连接层 (Matmul) + Bias (Add) + ReLU (Max)**。

**原始代码 (Payload IR)**：
```mlir
func.func @fc_relu(%lhs: tensor<512x512xf32>,
                   %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>,
                   %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // 操作1: 矩阵乘法
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // 操作2: 逐元素加法 (加上偏置)
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // 操作3: 逐元素最大值 (ReLU: max(x, 0))
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  func.return %relued : tensor<512x512xf32>
}
```

**未优化的内存问题**：
每次操作都将完整的大张量（512x512）写回 L3 缓存或主存。
*   `Matmul` 写回 262K 元素
*   `Add` 读取 262K，写回 262K
*   `Max` 读取 262K，写回 262K
总内存写入量巨大，带宽成为瓶颈。

### 4.1.1 Payload IR 与 Transform IR 的对应关系

理解 Transform 脚本如何“控制” Payload 代码是关键。

```
┌─────────────────────────────────────────────────────────┐
│  Transform IR (变换脚本)                                 │
│  module attributes {transform.with_named_sequence} {    │
│    transform.named_sequence @__transform_main(          │
│         %root: !transform.any_op,                       │
│         %matmul_handle: !transform.op<"linalg.matmul">, │
│         %elementwise_handle: !transform.op<...>) {      │
│      ...                                                │
│    }                                                    │
│  }                                                      │
└───────────────────────────┬─────────────────────────────┘
                            │ 作用于
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Payload IR (计算代码)                                   │
│  func.func @fc_relu(...) {                              │
│    %matmul = linalg.matmul ...    <── 绑定到 %matmul_handle
│    %biased = linalg.elementwise ... <── ┐ 绑定到
│    %relued = linalg.elementwise ... <── ┴ %elementwise_handle
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

**注意**：`%elementwise_handle` 在匹配时会同时包含 `add` 和 `max` 两个操作，这在处理时需要特别注意（见下文 Step 1）。

---

### 4.2 Transform 脚本逐步深度解析

这是实现 Tile and Fuse 的完整脚本，我们将逐行拆解其背后的逻辑。

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %root: !transform.any_op,
       %matmul_handle: !transform.op<"linalg.matmul">,
       %elementwise_handle: !transform.op<"linalg.elementwise">) {

    // ═════════════════════════════════════════════════════════
    // Step 1: 分离 elementwise handle
    // ═════════════════════════════════════════════════════════
    // %elementwise_handle 匹配了 [add, max] 两个操作。
    // 我们需要将它们分开，以便从最后一个操作开始逆序处理。
    %add_handle, %max_handle = transform.split_handle %elementwise_handle
        : (!transform.op<"linalg.elementwise">)
        -> (!transform.any_op, !transform.any_op)

    // ═════════════════════════════════════════════════════════
    // Step 2: Tile 最后一个操作 (max/ReLU)
    // ═════════════════════════════════════════════════════════
    // Tile and Fuse 的黄金法则：从数据流的终点开始 Tile。
    // 这里我们将 512x512 的输出分块为 [8, 32]。
    %tiled_max, %loop =
        transform.structured.tile_using_forall %max_handle
        tile_sizes [8, 32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 此时生成的 IR 结构预览：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %max_tile = linalg.elementwise kind=max on [ti*8, tj*32]
    // }

    // ═════════════════════════════════════════════════════════
    // Step 3: Fuse add 操作到循环中
    // ═════════════════════════════════════════════════════════
    // 将 add 操作融合进由 max 生成的循环 %loop 中。
    %add_fused, %loop_updated =
        transform.structured.fuse_into_containing_op %add_handle into %loop
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ═════════════════════════════════════════════════════════
    // Step 4: Fuse matmul 操作到循环中
    // ═════════════════════════════════════════════════════════
    // 将最上游的 matmul 融合进循环 %loop_updated 中。
    %matmul_fused, %loop_final =
        transform.structured.fuse_into_containing_op %matmul_handle into %loop_updated
          : (!transform.op<"linalg.matmul">, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

#### 💡 核心原理：`fuse_into_containing_op` 是如何工作的？

当我们在 Step 4 执行 `fuse_into_containing_op %matmul into %loop` 时，编译器在幕后完成了复杂的推理：

1.  **依赖分析**：编译器发现循环内的 `%add_tile` 需要用到 `%matmul` 的结果。
2.  **切片推导 (Slicing)**：
    *   为了计算输出位置 `C[ti*8 : ti*8+8, tj*32 : tj*32+32]` 的值，`matmul` 需要哪些输入数据？
    *   根据 `TilingInterface` 的定义：
        *   需要 **LHS (A)** 的整行：`A[ti*8 : ti*8+8, 0 : 512]`
        *   需要 **RHS (B)** 的整列：`B[0 : 512, tj*32 : tj*32+32]`
3.  **代码生成**：
    *   在循环**内部**插入 `extract_slice` 来获取上述的输入数据切片。
    *   在循环**内部**插入一个新的 `linalg.matmul`，其尺寸变为了 `8x512` * `512x32` -> `8x32`。
4.  **数据流重定向**：将 `%add_tile` 的输入从原来的大 `%matmul` 替换为这个新的 `%matmul_tile`。

---

### 4.3 融合后的完整代码结构

经过上述变换，我们得到了极度优化的 IR：

```mlir
func.func @fc_relu_tiled_fused(...) -> tensor<512x512xf32> {
  // 并行循环：以 8x32 为单位块遍历整个 512x512 输出
  %result = scf.forall (%ti, %tj) in (64, 16)
      shared_outs(%output_arg = %output) -> (tensor<512x512xf32>) {

    // 1. 数据加载 (只读)：从大张量中提取需要的 Slice
    //    LHS: 取 8 行 (8x512)
    %lhs_tile = tensor.extract_slice %lhs[%ti*8, 0] [8, 512] [1, 1]
    //    RHS: 取 32 列 (512x32)
    %rhs_tile = tensor.extract_slice %rhs[0, %tj*32] [512, 32] [1, 1]
    //    Bias: 取对应的 8x32 块
    %bias_tile = tensor.extract_slice %bias[%ti*8, %tj*32] [8, 32] [1, 1]
    //    Output Init
    %out_tile = tensor.extract_slice %output_arg[%ti*8, %tj*32] [8, 32] [1, 1]

    // 2. 核心计算 (全都在 Tile 上进行，中间结果驻留寄存器/L1)
    //    Matmul: 8x512 * 512x32 -> 8x32
    %matmul_tile = linalg.matmul
      ins(%lhs_tile, %rhs_tile) outs(%out_tile) -> tensor<8x32xf32>

    //    Add: 8x32 + 8x32 -> 8x32 (直接使用寄存器中的 matmul_tile)
    %add_tile = linalg.elementwise kind=add
      ins(%matmul_tile, %bias_tile) outs(%matmul_tile) -> tensor<8x32xf32>

    //    Max: 8x32 (直接使用寄存器中的 add_tile)
    %max_tile = linalg.elementwise kind=max
      ins(%add_tile, %c0f) outs(%add_tile) -> tensor<8x32xf32>

    // 3. 结果写回：只将最终的 Max 结果写回主存
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %max_tile
        into %output_arg[%ti*8, %tj*32] [8, 32] [1, 1]
    }
  }
  return %result : tensor<512x512xf32>
}
```

### 4.4 优化效果对比

| 指标 | 未优化版本 | Tile + Fuse 版本 |
| :--- | :--- | :--- |
| **内存写回 (Write-backs)** | Matmul(262K) + Add(262K) + Max(262K) <br> **合计 ~786K 元素** | 仅 Max 操作写回 <br> **合计 ~262K 元素** |
| **L3 缓存压力** | 极高，中间结果反复进出 | 低，中间结果在 L1/寄存器复用 |
| **性能瓶颈** | 内存带宽限制 (Memory Bound) | 计算限制 (Compute Bound) |

---

## 5. 复杂场景处理：大 K 维度与 Packing

### 5.1 K 维度过大问题
在上述融合中，Matmul 的 K 维度（512）保持未分块。如果 K=4096，单个 `8x4096` 的 tile 输入将远超 L1 缓存。

**解决方案：分级 Tiling**
1.  **Reduction Tiling**: 先对 K 维度进行分块（使用 `scf.for`）。
2.  **Parallel Tiling**: 再对 M, N 维度进行分块（使用 `scf.forall`）。

```mlir
// Transform 脚本
// 1. Tile K 维度 (reduction dim)
%tiled_k, %k_loop = transform.structured.tile_using_for %matmul tile_sizes [0, 0, 64]

// 2. Tile M, N 维度
%tiled_mn, %mn_loop = transform.structured.tile_using_forall %tiled_k tile_sizes [8, 32]
```

这会生成一个“外层并行循环 + 内层归约循环”的结构。

### 5.2 数据 Packing
对于不规则的访存，建议使用 `tensor.pack` 预先调整数据布局，或者使用 `transform.structured.pack_matrices` 进行布局优化，再进行 Tiling。

---

## 6. C++ API 与调试

对于编译器开发者，以下信息至关重要。

### 6.1 关键文件位置
*   **接口定义**: `mlir/include/mlir/Interfaces/TilingInterface.td`
*   **SCF 实现**: `mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`
*   **Linalg 实现**: `mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp`

### 6.2 C++ API 示例
```cpp
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

// 1. 基础 Tiling
SCFTilingOptions options;
options.tileSizes = {32, 32};
SCFTilingResult result = tileUsingSCF(op, options);

// 2. Tile and Fuse
SCFTileAndFuseOptions fuseOptions;
fuseOptions.tilingOptions = options;
// 自动寻找并融合 producer
SCFTileAndFuseResult fuseResult = tileConsumerAndFuseProducersUsingSCF(
    consumerOp, producersFilter, fuseOptions);
```

### 6.3 调试技巧
使用 `transform-interpreter` pass 进行快速验证：
```bash
mlir-opt input.mlir --transform-interpreter --debug-only=transform-dialect
```
