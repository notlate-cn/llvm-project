# MLIR Tile-based 编程指南

## 1. 核心概念

**Tiling** 是将大循环/大矩阵分解为小块（tiles）的技术，用于优化缓存局部性和并行性。

MLIR通过 **TilingInterface** 统一不同Dialect的Tiling操作：

```cpp
// mlir/include/mlir/Interfaces/TilingInterface.td
def TilingInterface : Interface {
  let methods = [
    InterfaceMethod<[{
      返回循环迭代器类型（并行/归约等）
    }], "getLoopIteratorTypes", ...],

    InterfaceMethod<[{
      返回循环边界和步长
    }], "getIterationDomain", ...],

    InterfaceMethod<[{
      生成Tiling后的实现
    }], "getTiledImplementation", ...],
  ];
}
```

---

## 2. 示例1：Affine Loop Tiling

### 原始代码

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

### 应用 `--affine-loop-tile=tile-size=32` 后

```mlir
func.func @loop_tiling() {
  // 外层tile循环，步长32
  affine.for %ti = 0 to 256 step 32 {
    affine.for %tj = 0 to 512 step 32 {
      affine.for %tk = 0 to 1024 step 32 {
        // 内层点循环，处理单个tile
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

---

## 3. 示例2：Linalg Tiling with Transform Dialect（单算子）

### 原始矩阵乘法

```mlir
func.func @matmul(%A: tensor<25x34xf32>, %B: tensor<34x25xf32>,
                  %C: tensor<25x25xf32>) -> tensor<25x25xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<25x34xf32>, tensor<34x25xf32>)
                     outs(%C: tensor<25x25xf32>)
    -> tensor<25x25xf32>
  return %0 : tensor<25x25xf32>
}
```

### Transform脚本（tile_size=9）

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %tiled, %loop = transform.structured.tile_using_for %0
      tile_sizes [9] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

### 变换后结果

```mlir
func.func @matmul(...) -> tensor<25x25xf32> {
  %c9 = arith.constant 9 : index
  %c0 = arith.constant 0 : index
  %c25 = arith.constant 25 : index

  // Tile循环
  %result = scf.for %idx = %c0 to %c25 step %c9
    iter_args(%out = %C) -> (tensor<25x25xf32>) {

    // 提取slice
    %A_tile = tensor.extract_slice %A[%idx, 0] [9, 34] [1, 1]
      : tensor<25x34xf32> to tensor<9x34xf32>
    %C_tile = tensor.extract_slice %out[%idx, 0] [9, 25] [1, 1]
      : tensor<25x25xf32> to tensor<9x25xf32>

    // 在tile上执行matmul
    %tile_result = linalg.matmul ins(%A_tile, %B : tensor<9x34xf32>, tensor<34x25xf32>)
                                  outs(%C_tile : tensor<9x25xf32>)
                   -> tensor<9x25xf32>

    // 插入回结果
    %inserted = tensor.insert_slice %tile_result into %out[%idx, 0] [9, 25] [1, 1]
      : tensor<9x25xf32> into tensor<25x25xf32>
    scf.yield %inserted : tensor<25x25xf32>
  }
  return %result : tensor<25x25xf32>
}
```

---

## 4. 示例：Tile and Fuse 模式（组合算子）

### 4.1 背景：什么是 Tile and Fuse？

Tile-and-fuse 是优化**数据流图**的核心技术，用于提升缓存局部性。主要逻辑是 **逆序遍历** 图中的算子，将可以融合的算子合并到一个 tile 中。

**问题场景**：考虑一个典型的神经网络全连接层 + ReLU 激活函数的计算：

```mlir
func.func @fc_relu(%lhs: tensor<512x512xf32>,
                   %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>,
                   %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // 操作1: 矩阵乘法
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>)
    -> tensor<512x512xf32>

  // 操作2: 逐元素加法 (加上偏置)
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>)
    -> tensor<512x512xf32>

  // 操作3: 逐元素最大值 (ReLU: max(x, 0))
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>)
    -> tensor<512x512xf32>

  func.return %relued : tensor<512x512xf32>
}
```

**内存访问问题**（未优化）：
```
每次操作都写回 L3 缓存/主存:

linalg.matmul  生成 %matmul  (写回 512x512 = 262K 元素)
linalg.elementwise(add) 生成 %biased  (写回 262K 元素)
linalg.elementwise(max) 生成 %relued  (写回 262K 元素)

总内存写入: ~786K 元素
中间结果在 L3 缓存和 CPU 之间频繁传输
```

---

### 4.1.1 Payload IR 与 Transform IR 的关系

**关键概念**：MLIR 的 Transform dialect 使用 **两层 IR** 的概念：

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLIR 模块结构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Transform IR (变换脚本)                                 │   │
│  │  - 描述"如何变换"                                         │   │
│  │  - 操作类型: transform.*                                 │   │
│  │  - 在本例中: @__transform_main 函数                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         │ applies to                            │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Payload IR (待优化的计算代码)                            │   │
│  │  - 描述"计算什么"                                         │   │
│  │  - 操作类型: linalg.*, scf.*, arith.* 等                  │   │
│  │  - 在本例中: @fc_relu 函数                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**对应关系表**：

| Transform IR (4.2节) | Payload IR (4.1节) | 说明 |
|---------------------|-------------------|------|
| `%root: !transform.any_op` | 整个 `module` | 变换的根节点 |
| `%matmul_handle: !transform.op<"linalg.matmul">` | `@fc_relu` 中的 `linalg.matmul` 操作 | 指向矩阵乘法算子 |
| `%elementwise_handle: !transform.op<"linalg.elementwise">` | `@fc_relu` 中的两个 `linalg.elementwise` 操作 | 指向 **add** 和 **max** 两个算子 |

**如何建立对应关系？**

```bash
# 方法1: 使用 transform-interpreter pass 的命令行参数
mlir-opt input.mlir --transform-interpreter \
  --debug-bind-trailing-args=linalg.matmul,linalg.elementwise

# 方法2: 在同一个文件中包含 payload IR 和 transform IR
module {
  // Payload IR (要优化的代码)
  func.func @fc_relu(...) { ... }

  // Transform IR (优化脚本)
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(...) { ... }
  }
}

# 方法3: 通过 C++ API
transform::applyTransforms(
    payloadOp,              // @fc_relu 函数
    transformRootOp,         // @__transform_main
    {...transformOps},        // 额外的 handles (matmul, elementwise)
    options
);
```

**完整的模块结构**：

```mlir
module {
  // ═════════════════════════════════════════════════════════
  // Payload IR: 实际的计算代码
  // ═════════════════════════════════════════════════════════
  func.func @fc_relu(%lhs: tensor<512x512xf32>,
                     %rhs: tensor<512x512xf32>,
                     %bias: tensor<512x512xf32>,
                     %output: tensor<512x512xf32>)
                     -> tensor<512x512xf32> {
    %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%output : tensor<512x512xf32>)
      -> tensor<512x512xf32>

    %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%output : tensor<512x512xf32>)
      -> tensor<512x512xf32>

    %c0f = arith.constant 0.0 : f32
    %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
      ins(%biased, %c0f : tensor<512x512xf32>, f32)
      outs(%output : tensor<512x512xf32>)
      -> tensor<512x512xf32>

    return %relued : tensor<512x512xf32>
  }

  // ═════════════════════════════════════════════════════════
  // Transform IR: 变换脚本
  // ═════════════════════════════════════════════════════════
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(
         %root: !transform.any_op,                          // ← 绑定到整个 module
         %matmul_handle: !transform.op<"linalg.matmul">,   // ← 绑定到 @fc_relu 中的 linalg.matmul
         %elementwise_handle: !transform.op<"linalg.elementwise">) {  // ← 绑定到 @fc_relu 中的 2 个 elementwise

      // 在这里对 payload IR 进行变换...
      %add, %max = transform.split_handle %elementwise_handle
      %tiled_max, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
      // ...

      transform.yield
    }
  }
}
```

**执行流程**：

```
1. mlir-opt 加载模块
   │
2. transform-interpreter pass 找到 @__transform_main
   │
3. 从命令行参数或 C++ API 获取绑定信息:
   │
   │  %matmul_handle → 匹配 module 中所有的 linalg.matmul 操作
   │                   → 在本例中找到 @fc_relu 内的 1 个 matmul
   │
   │  %elementwise_handle → 匹配 module 中所有的 linalg.elementwise 操作
   │                      → 在本例中找到 @fc_relu 内的 2 个 elementwise (add 和 max)
   │
4. 执行 @__transform_main 中的 transform 操作
   │
5. 变换直接修改 payload IR (@fc_relu 函数)
   │
6. 输出变换后的模块
```

---

### 4.2 Transform 脚本逐步解析

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %root: !transform.any_op,                          // 整个模块
       %matmul_handle: !transform.op<"linalg.matmul">,   // 指向 matmul 操作
       %elementwise_handle: !transform.op<"linalg.elementwise">) {  // ⚠️ 关键点

    // ═════════════════════════════════════════════════════════
    // Step 0: 理解 %elementwise_handle
    // ═════════════════════════════════════════════════════════
    // %elementwise_handle 关联了模块中 ALL 的 linalg.elementwise 操作！
    // 在我们的例子中，它关联了两个操作：
    //   1. %biased = linalg.elementwise kind=<add>    (加偏置)
    //   2. %relued = linalg.elementwise kind=<max>    (ReLU)
    //
    // 因为 transform dialect 的 match 操作会匹配所有符合条件的操作，
    // 所以 %elementwise_handle 是一个包含 2 个操作的列表。
    //
    // 数据流关系：
    //   %matmul → %biased (add) → %relued (max)
    //   我们需要从最后的 max 开始 tile，然后向上融合 add 和 matmul
    // ═════════════════════════════════════════════════════════

    // ═════════════════════════════════════════════════════════
    // Step 1: 分离 elementwise handle
    // ═════════════════════════════════════════════════════════
    // 将 [add, max] 分成两个独立的 handles
    %add_handle, %max_handle = transform.split_handle %elementwise_handle
        : (!transform.op<"linalg.elementwise">)
        -> (!transform.any_op, !transform.any_op)

    // 现在：
    //   %add_handle  → 指向 linalg.elementwise kind=<add>
    //   %max_handle  → 指向 linalg.elementwise kind=<max>

    // ═════════════════════════════════════════════════════════
    // Step 2: Tile 最后一个操作 (max/ReLU)
    // ═════════════════════════════════════════════════════════
    %tiled_max, %loop =
        transform.structured.tile_using_forall %max_handle
        tile_sizes [8, 32]          // 将 512x512 分成 [64,16] 个 [8, 32] 的 tiles
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 生成代码结构：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %max_tile = linalg.elementwise kind=max on [ti*8:ti*8+8, tj*32:tj*32+32]
    // }

    // ═════════════════════════════════════════════════════════
    // Step 3: Fuse add 操作到循环中
    // ═════════════════════════════════════════════════════════
    %add_fused, %loop_updated =
        transform.structured.fuse_into_containing_op %add_handle into %loop
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // Fusion 做什么：
    // 1. 找到 add 操作的输入（来自 matmul 的输出）
    // 2. 将 add 操作移动到循环内部
    // 3. 在循环内对输入做 extract_slice
    // 4. add 的输出直接作为 max 的输入，不写回内存

    // 生成代码结构：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %biased_tile = linalg.elementwise kind=add on tile
    //   %max_tile = linalg.elementwise kind=max on tile (使用 %biased_tile)
    // }

    // ═════════════════════════════════════════════════════════
    // Step 4: Fuse matmul 操作到循环中
    // ═════════════════════════════════════════════════════════
    %matmul_fused, %loop_final =
        transform.structured.fuse_into_containing_op %matmul_handle into %loop_updated
          : (!transform.op<"linalg.matmul">, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ═════════════════════════════════════════════════════════
    // 详细解释：fuse_into_containing_op 如何工作？
    // ═════════════════════════════════════════════════════════
    //
    // 原始 matmul 操作 (在循环外):
    //   %matmul = linalg.matmul
    //     ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
    //     outs(%output : tensor<512x512xf32>)
    //
    // 目标循环 (Step 3 之后):
    //   scf.forall (%ti, %tj) in (64, 16) {
    //     %add_tile = linalg.elementwise kind=add on tile
    //     %max_tile = linalg.elementwise kind=max on tile
    //   }
    //
    // fuse_into_containing_op 的执行流程：
    // ┌─────────────────────────────────────────────────────────┐
    // │ 1. 分析数据依赖                                          │
    // │    - 找到 matmul 的 consumer: %add 操作                 │
    // │    - 检测 %add 在循环内的 tiled 版本: %add_tile         │
    // │                                                          │
    // │ 2. 调用 TilingInterface (linalg.matmul 实现)            │
    // │                                                          │
    // │    getIterationDomain() 返回:                           │
    // │      - dim 0: [0, 512)  并行迭代                        │
    // │      - dim 1: [0, 512)  并行迭代                        │
    // │      - dim 2: [0, 512)  归约迭代 (K 维度)               │
    // │                                                          │
    // │    getLoopIteratorTypes() 返回:                         │
    // │      - [parallel, parallel, reduction]                  │
    // │                                                          │
    // │ 3. 匹配循环迭代空间                                      │
    // │    - 循环外层: scf.forall (%ti, %tj)                    │
    // │      %ti 对应 matmul 的 dim 0 (M 维度)                  │
    // │      %tj 对应 matmul 的 dim 1 (N 维度)                  │
    // │    - matmul 的 dim 2 (K 维度) 是归约，不需要外层循环    │
    // │                                                          │
    // │ 4. 计算需要提取的 tile 区域                              │
    // │    - 输入 %lhs: 从 [ti*8, 0] 提取 [8, 512]              │
    // │      (M 维度 tiling, K 维度完整)                         │
    // │    - 输入 %rhs: 从 [0, tj*32] 提取 [512, 32]            │
    // │      (K 维度完整, N 维度 tiling)                         │
    // │    - 输出: 从 add_tile 的输出位置继承                   │
    // │                                                          │
    // │ 5. 在循环内生成 tiled matmul                             │
    // │    %lhs_tile = tensor.extract_slice %lhs[ti*8, 0][8,512]│
    // │    %rhs_tile = tensor.extract_slice %rhs[0, tj*32][512,32]│
    // │    %matmul_tile = linalg.matmul                         │
    // │      ins(%lhs_tile, %rhs_tile)                          │
    // │      outs(%output_tile)                                 │
    // │                                                          │
    // │ 6. 更新数据流                                           │
    // │    原来使用 %add 的输入 %biased (来自循环外的 matmul)    │
    // │    现在使用 %matmul_tile (循环内的 tiled 版本)           │
    // └─────────────────────────────────────────────────────────┘
    //
    // 关键点：为什么 matmul 的输入提取方式不同？
    // ┌─────────────────────────────────────────────────────────┐
    // │ %lhs: tensor<512x512xf32>  (M=512, K=512)               │
    // │   提取 [ti*8:ti*8+8, 0:512]  → tensor<8x512xf32>        │
    // │   M 维度被 tiling (分成 8), K 维度保持完整             │
    // │                                                          │
    // │ %rhs: tensor<512x512xf32>  (K=512, N=512)               │
    // │   提取 [0:512, tj*32:tj*32+32]  → tensor<512x32xf32>    │
    // │   K 维度保持完整 (用于归约), N 维度被 tiling (分成 32)  │
    // │                                                          │
    // │ 这是由 matmul 的语义决定的：                             │
    // │   C[i,j] = sum_k(A[i,k] * B[k,j])                      │
    // │   每个输出 tile C[ti*8:ti*8+8, tj*32:tj*32+32] 需要：   │
    // │     - A[ti*8:ti*8+8, 0:512]  (完整的 K 列)              │
    // │     - B[0:512, tj*32:tj*32+32]  (完整的 K 行)           │
    // └─────────────────────────────────────────────────────────┘
    //
    // 生成代码结构：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %lhs_tile = extract_slice %lhs[ti*8, 0][8, 512]
    //   %rhs_tile = extract_slice %rhs[0, tj*32][512, 32]
    //   %matmul_tile = linalg.matmul on tiles
    //   %add_tile = linalg.elementwise kind=add (使用 %matmul_tile)
    //   %max_tile = linalg.elementwise kind=max (使用 %add_tile)
    // }
    // ═════════════════════════════════════════════════════════

    transform.yield
  }
}
```

---

### 4.3 融合后的代码结构

```mlir
func.func @fc_relu_tiled_fused(...) -> tensor<512x512xf32> {
  %result = scf.forall (%ti, %tj) in (64, 16)
      shared_outs(%output_arg = %output) -> (tensor<512x512xf32>) {

    // ═════════════════════════════════════════════════════════
    // 从原始大张量提取 tiles (只读，从 L3 缓存)
    // ═════════════════════════════════════════════════════════
    %lhs_tile = tensor.extract_slice %lhs[%ti*8, 0] [8, 512] [1, 1]
      : tensor<512x512xf32> to tensor<8x512xf32>

    %rhs_tile = tensor.extract_slice %rhs[0, %tj*32] [512, 32] [1, 1]
      : tensor<512x512xf32> to tensor<512x32xf32>

    %bias_tile = tensor.extract_slice %bias[%ti*8, %tj*32] [8, 32] [1, 1]
      : tensor<512x512xf32> to tensor<8x32xf32>

    %output_tile = tensor.extract_slice %output_arg[%ti*8, %tj*32] [8, 32] [1, 1]
      : tensor<512x512xf32> to tensor<8x32xf32>

    // ═════════════════════════════════════════════════════════
    // 三个操作在 tile 上完成，数据保持在寄存器/L1
    // ═════════════════════════════════════════════════════════
    %matmul_tile = linalg.matmul
      ins(%lhs_tile, %rhs_tile : tensor<8x512xf32>, tensor<512x32xf32>)
      outs(%output_tile : tensor<8x32xf32>)
      -> tensor<8x32xf32>

    %add_tile = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%matmul_tile, %bias_tile : tensor<8x32xf32>, tensor<8x32xf32>)
      outs(%matmul_tile : tensor<8x32xf32>)
      -> tensor<8x32xf32>

    %max_tile = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
      ins(%add_tile, %c0f : tensor<8x32xf32>, f32)
      outs(%add_tile : tensor<8x32xf32>)
      -> tensor<8x32xf32>

    // ═════════════════════════════════════════════════════════
    // 只写回最终结果到 L3 缓存
    // ═════════════════════════════════════════════════════════
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %max_tile
        into %output_arg[%ti*8, %tj*32] [8, 32] [1, 1]
        : tensor<8x32xf32> into tensor<512x512xf32>
    }
  }
  return %result : tensor<512x512xf32>
}
```

---

### 4.4 优化效果对比

#### 未优化版本
```
┌─────────────────────────────────────────────────────────┐
│ 内存访问 (每次操作写回 L3 缓存):                          │
├─────────────────────────────────────────────────────────┤
│ matmul:   读写 ~786K 元素 (512x512x3)                    │
│ add:      读写 ~524K 元素 (512x512x2)                    │
│ max:      读写 ~524K 元素 (512x512x2)                    │
│                                                        │
│ 总计: ~1.8M 元素访问                                     │
│ 中间结果在 L3 ↔ L1 之间传输 2 次                          │
└─────────────────────────────────────────────────────────┘
```

#### Tile + Fuse 版本
```
┌─────────────────────────────────────────────────────────┐
│ 每个 8x32 tile 的内存访问:                               │
├─────────────────────────────────────────────────────────┤
│ 读取: 8x512 + 512x32 + 8x32 = 9,216 元素 (从 L3)        │
│ 计算: 全部在 L1/寄存器中完成                             │
│ 写回: 8x32 = 256 元素 (到 L3)                            │
│                                                        │
│ 64 个 tiles 总写回: 64 x 256 = 16,384 元素               │
│ 相比未优化减少 ~98% 的内存写入量                          │
└─────────────────────────────────────────────────────────┘
```

---

### 4.5 关键概念总结

| 概念 | 说明 |
|------|------|
| **Handle** | Transform dialect 中指向 payload IR 操作的引用 |
| **Match** | `transform.structured.match` 返回匹配**所有**符合条件的操作 |
| **Split** | 将包含多个操作的 handle 分离成独立的 handles |
| **Tile** | 将大操作分解，引入外层循环结构 |
| **Fuse** | 将 producer 操作移动到 consumer 的循环内 |
| **数据局部性** | 中间结果保持在 L1/寄存器，不写回主存 |
| **操作顺序** | 必须从最后一个 consumer 开始，向上游融合 |
| **Handle 失效** | 被 consume 的 handle 会失效，不可再用 |

---

### 4.6 为什么需要 split_handle？

```mlir
// 假设原始代码有两个 linalg.elementwise 操作:
%1 = linalg.elementwise kind=add ...
%2 = linalg.elementwise kind=max ...

// transform.structured.match 会匹配两者:
%elementwise_handle = transform.structured.match ops{["linalg.elementwise"]}
// %elementwise_handle 现在包含 [%1, %2] 两个操作

// 如果直接 tile，会同时 tile 两个操作，这不是我们想要的
// 我们只想 tile 最后的 max 操作，然后向上融合

// 所以需要 split:
%add_handle, %max_handle = transform.split_handle %elementwise_handle
// %add_handle  → [%1]
// %max_handle  → [%2]
```

---

### 4.7 进阶：当 K 维度过大时如何处理？

#### 问题场景

```mlir
// 假设 K 维度非常大 (例如 4096 或更大)
func.func @large_k_matmul(%A: tensor<512x4096xf32>,
                          %B: tensor<4096x512xf32>,
                          %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %matmul = linalg.matmul
    ins(%A, %B : tensor<512x4096xf32>, tensor<4096x512xf32>)
    outs(%C : tensor<512x512xf32>)
    -> tensor<512x512xf32>
  return %matmul
}
```

**问题**：按照之前的融合方式，每个 tile 需要加载：
- `%lhs_tile`: 8 x 4096 = 32K 元素 (~128KB)
- `%rhs_tile`: 4096 x 32 = 131K 元素 (~512KB)

这会超出 L1 缓存（通常 32-64KB），导致性能下降。

---

#### 解决方案 1：对 K 维度进行 Tiling

使用 **Partial Reduction Tiling** 或 **Pack + Tiling** 策略：

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %root: !transform.any_op,
       %matmul: !transform.op<"linalg.matmul">) {

    // ═════════════════════════════════════════════════════════
    // Step 1: 先对归约维度 (K) 进行 tiling
    // ═════════════════════════════════════════════════════════
    // 使用 tile_reduction 进行 K 维度 tiling
    %tiled_k, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [64]              // 只 tile K 维度 (索引2)
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

    // 生成代码结构：
    // scf.for %tk = 0 to 4096 step 64 {
    //   // K 维度被分成 64 大小的块
    //   %partial = linalg.matmul on partial K
    //   // 注意：这是部分归约，需要累加
    // }

    // ═════════════════════════════════════════════════════════
    // Step 2: 再对 M, N 维度进行 tiling
    // ═════════════════════════════════════════════════════════
    %tiled_mn, %mn_loop = transform.structured.tile_using_forall %tiled_k
      tile_sizes [8, 32]           // Tile M 和 N 维度
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

**生成的代码结构**：

```mlir
func.func @large_k_matmul_tiled(...) -> tensor<512x512xf32> {
  // M, N 维度的外层循环 (并行)
  %result = scf.forall (%ti, %tj) in (64, 16)
      iter_args(%C_accum = %C) -> (tensor<512x512xf32>) {

    %C_tile = tensor.extract_slice %C_accum[%ti*8, %tj*32] [8, 32] [1, 1]

    // K 维度的内层循环 (归约)
    %final_tile = scf.for %tk = 0 to 4096 step 64
        iter_args(%accum = %C_tile) -> (tensor<8x32xf32>) {

      // 提取 K 维度的 tile
      %A_tile = tensor.extract_slice %A[%ti*8, %tk] [8, 64] [1, 1]
        : tensor<512x4096xf32> to tensor<8x64xf32>

      %B_tile = tensor.extract_slice %B[%tk, %tj*32] [64, 32] [1, 1]
        : tensor<4096x512xf32> to tensor<64x32xf32>

      // 在小 tile 上执行 matmul (8x64 * 64x32 = 8x32)
      %partial = linalg.matmul
        ins(%A_tile, %B_tile : tensor<8x64xf32>, tensor<64x32xf32>)
        outs(%accum : tensor<8x32xf32>)
        -> tensor<8x32xf32>

      scf.yield %partial : tensor<8x32xf32>
    }

    // 将最终 tile 写回
    tensor.parallel_insert_slice %final_tile
      into %C_accum[%ti*8, %tj*32] [8, 32] [1, 1]
  }
  return %result
}
```

**内存使用对比**：

| 方案 | 每个 tile 的内存使用 |
|------|---------------------|
| 不 tile K 维度 | 8x4096 + 4096x32 = 163K 元素 (~640KB) |
| Tile K=64 | 8x64 + 64x32 = 2.5K 元素 (~10KB) |

---

#### 解决方案 2：Pack + Tiling (数据重排)

对于非常不规则的数据访问模式，可以先 Pack 数据：

```mlir
// 先对 A 和 B 进行 pack 重排
%A_packed = tensor.pack %A
  inner_dims_pos = [0, 1]           // 保持 M, K 维度顺序
  inner_tiles = [8, 64]              //打包成 8x64 的小块
  into tensor<64x8x64xf32>           // [K/64, M/8, 8, 64]

%B_packed = tensor.pack %B
  inner_dims_pos = [0, 1]           // 保持 K, N 维度顺序
  inner_tiles = [64, 32]             // 打包成 64x32 的小块
  into tensor<64x16x64x32xf32>       // [K/64, N/32, 64, 32]

// 然后在 packed 数据上进行 matmul
// 这可以让缓存预取更有效
```

---

#### 解决方案 3：使用 Pack 算子

MLIR 提供专门的 `tensor.pack` 操作用于优化数据布局：

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%matmul: !transform.op<"linalg.matmul">) {

    // 在 tiling 之前先 pack
    %packed = transform.structured.pack_matrices %matmul
      packing_factors [8, 64, 32]     // [M_tile, K_tile, N_tile]
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op)

    // 对 packed matmul 进行 tiling
    %tiled, %loop = transform.structured.tile_using_forall %packed
      tile_sizes [1, 1, 1]           // 每个 tile 已经是 packed 大小
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

---

#### 关键点总结

| 场景 | 解决方案 | Tiling 策略 |
|------|----------|-------------|
| K 适度大小 (< L2) | 标准 tile | 只 tile M, N，保持 K 完整 |
| K 很大 (> L2) | K 维度 tiling | 先 tile K (归约)，再 tile M,N |
| 访问不规则 | Pack + Tile | 先 pack 数据布局，再 tile |
| 寄存器优化 | 微内核 | tile 到更小尺寸 (4x4, 8x8) |

---

## 5. C++ API使用方式

```cpp
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

// 配置tiling选项
SCFTilingOptions options;
options.tileSizes = {32, 32};           // tile大小
options.interchange = {1, 0};           // 循环交换

// 执行tiling
SCFTilingResult result = tileUsingSCF(op, options);

// 结果访问
// result.tiledOps - tiling后的操作
// result.loops - 生成的循环（scf.for）
// result.tileSizes - 实际使用的tile大小
```

### Tile and Fuse API

```cpp
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

SCFTileAndFuseOptions options;
options.tilingOptions = SCFTilingOptions{...};

SCFTileAndFuseResult result = tileConsumerAndFuseProducersUsingSCF(
    consumerOp,                        // 要tile的consumer
    producerOpsFilter,                 // 选择要融合的producers
    options
);

// result.fusedProducers - 融合后的producer操作
// result.loops - 生成的循环嵌套
```

---

## 6. Tiling类型对比

| 类型 | 用途 | 循环结构 | 适用场景 |
|------|------|----------|----------|
| `tile_using_for` | 通用tiling | `scf.for` | 顺序执行，通用优化 |
| `tile_using_forall` | 并行tiling | `scf.forall` | 多线程/GPU并行 |
| Continuous tiling | 不规则尺寸分割 | 多种循环组合 | 处理非2的幂次尺寸 |
| Partial reduction tiling | 归约优化 | `scf.for` + 归约处理 | 累加/点积优化 |

---

## 7. 高级特性

### 7.1 参数化Tiling

```mlir
// 运行时动态确定tile大小
#map0 = affine_map<()[s0] -> (s0 ceildiv 32)>
affine.for %ti = 0 to %N step #map0()[%N] {
  // ...
}
```

### 7.2 循环交换（Loop Interchange）

```cpp
options.interchange = {1, 0};  // 交换i,j循环顺序
```

### 7.3 Multi-way Split Tiling

```mlir
// 将25分割为 [18, 7], 再将7分割为 [4, 3]
%tile_sizes, %chunk_sizes = transform.structured.continuous_tile_sizes %0
  { dimension = 0, target_size = 9 }
%linalg_splits = transform.structured.split %0 after %chunk_sizes
  { dimension = 0, multiway }
```

---

## 8. 关键文件位置

### 接口定义
- `mlir/include/mlir/Interfaces/TilingInterface.td`
- `mlir/include/mlir/Interfaces/TilingInterface.h`

### 实现文件
- **SCF**: `mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`
- **Linalg**: `mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp`
- **Affine**: `mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp`
- **Tensor**: `mlir/lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp`

### 测试文件
- `mlir/test/Dialect/Linalg/continuous-tiling-full.mlir`
- `mlir/test/Dialect/Affine/loop-tiling.mlir`
- `mlir/test/Dialect/SCF/parallel-loop-tiling.mlir`

### 文档
- `mlir/docs/Tutorials/transform/Ch1.md` - Transform dialect教程
- `mlir/docs/Dialects/Transform.md` - Transform方言参考
- `mlir/docs/Dialects/Linalg/_index.md` - Linalg方言文档

---

## 9. 使用命令

```bash
# Affine loop tiling
mlir-opt input.mlir --affine-loop-tile="tile-size=32"

# Transform interpreter
mlir-opt input.mlir --transform-interpreter

# 完整pipeline示例
mlir-opt input.mlir --transform-interpreter --canonicalize \
  --convert-linalg-to-loops --lower-affine
```

---

## 10. 设计原理

Tiling在MLIR中的设计遵循以下原则：

1. **统一接口** - `TilingInterface` 允许不同方言共享tiling逻辑
2. **可组合性** - Transform dialect允许精确控制变换序列
3. **Handle机制** - 跟踪操作引用，防止悬空指针
4. **验证模式** - 支持expensive-checks模式检测未定义行为
5. **性能透明** - tile size可以是编译时常量或运行时参数
