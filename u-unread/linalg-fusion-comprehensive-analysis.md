# MLIR Linalg 方言 Fusion 机制全面分析

## 概述

**Fusion（融合）** 是一种重要的编译器优化技术，通过将多个操作合并为一个操作来减少内存访问开销，提高数据局部性，并为后续优化（如向量化、并行化）创造机会。

### Fusion 相关文件

```
mlir/lib/Dialect/Linalg/Transforms/
├── Fusion.cpp                      # 核心 Tensor 融合实现
├── ElementwiseOpFusion.cpp         # 逐元素操作融合
└── FusePadOpWithLinalgProducer.cpp # Pad 操作与 Linalg 生产者融合
```

### Fusion 的核心思想

```
融合前:
  %1 = linalg.matmul ins(%A, %B) outs(%C)
  %2 = linalg.generic ins(%1) outs(%D) { add(%arg0, %c) }

融合后:
  %2 = linalg.generic {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %tmp = arith.mulf %a, %b : f32
      %res = arith.addf %tmp, %c : f32
      linalg.yield %res : f32
  } ins(%A, %B, %c) outs(%D)
```

**好处**:
- 消除中间张量 `%1` 的内存读写
- 提高寄存器利用率
- 为向量化创造机会
- 减少内存带宽压力

---

## Fusion 核心概念

### 1. 生产者-消费者关系

```cpp
// 生产者: 产生数据
%1 = linalg.generic ins(%A) outs(%B) { ... }

// 消费者: 使用生产者的输出
%2 = linalg.generic ins(%1) outs(%C) { ... }
```

**融合条件**:
1. 生产者只有一个消费者（或消费者可以内联）
2. 操作的计算模式兼容（如索引映射兼容）
3. 融合后能提升性能（减少内存访问）

### 2. Buffer vs Tensor 语义

代码对比：

```llvm
// Buffer 语义 (使用 MemRef)
%subview = memref.subview %buffer[...]
%1 = linalg.generic ins(%subview) outs(%output)

// Tensor 语义 (使用 Tensor)
%slice = tensor.extract_slice %tensor[...]
%1 = linalg.generic ins(%slice) outs(%output)
```

#### 2.1 Buffer

**语义**：操作的是可变的 **内存缓冲区（mutable memory）**，直接映射到底层内存。

```llvm
// Buffer 语义 (使用 MemRef)
%subview = memref.subview %buffer[...]
%1 = linalg.generic ins(%subview) outs(%output)
```

* `%buffer` 是一个 **MemRef**（内存缓冲区），可能是 `[M, N]` 的二维数组。

* `memref.subview` 从 `%buffer` 上创建一个 **子视图（subview）**，指定起始索引 (offsets)、大小 (sizes) 和步长 (strides)。

* `%subview` 仍然是一个 **MemRef**，但它只指向原 buffer 的一部分，**不会**分配新内存复制数据。

* `linalg.generic` 操作 `%subview`，读取其数据，然后 **原地（in-place）**修改。

#### 2.2 Tensor

**语义**：操作的是不可变的 **张量值（immutable value）**，在 SSA 语义下每次操作会生成新的张量。

```llvm
// Tensor 语义 (使用 Tensor)
%slice = tensor.extract_slice %tensor[...]
%1 = linalg.generic ins(%slice) outs(%output)
```

* `%tensor` 是不可变的 **Tensor**（SSA 值）。

* `tensor.extract_slice` 创建 **张量的切片**，功能类似 `subview`，**但不是** 原地操作，而是返回一个 **新的张量值**（`%slice`）。

* `linalg.generic` 作用于 `%slice`，读取其数据，输出写入 `%output`，但不修改原 `%tensor`。

### 3. 索引映射 (Indexing Map)

```cpp
// 描述操作数如何映射到迭代空间
indexing_maps = [
  affine_map<(i, j) -> (i, j)>,  // 输入 A: 使用 i, j
  affine_map<(i, j) -> (j)>,     // 输入 B: 使用 j
  affine_map<(i, j) -> (i)>      // 输出 C: 使用 i
]
```

以`affine_map<(i, j) -> (i + j, j)>`为例介绍含义：

* 左侧的`(i, j)`是输入维度，表示**循环索引**；

* 右侧的`(i + j, j)`是输出表达式，表示 **Tensor 的索引**

等价的 for 循环伪代码：

> 假设输入 Tensor 为 A [M] [N]，映射到输出Tensor B上：
>
> ```cpp
> for (int i = 0; i < M; i++) {
>     for (int j = 0; j < N; j++) {
>         // affine_map<(i, j) -> (i + j, j)>
>         B[i + j][j] = A[i][j];
>     }
> }
> ```
>
> 所以 Tensor B的Shape=[M + N - 1, N]

---

## Fusion.cpp 核心功能解读

### 文件概述

`Fusion.cpp` 实现了基于 `tensor::ExtractSliceOp` 的生产者-消费者融合，是 `tileAndFuse` 转换流程的核心部分。

---

### 核心数据结构

#### ShapeDimension 结构

```cpp
struct ShapeDimension {
  Value shape;         // 形状值（操作数）
  unsigned dimension;  // 维度索引
};
```

**作用**：记录"哪个张量的哪个维度"决定了循环的范围。

**示例**：
```llvm
// 假设有 tensor<100x200xf32>
%A: tensor<100x200xf32>

// ShapeDimension { shape: %A, dimension: 0 } 表示：循环范围由 %A 的第 0 维决定（即 100）
// ShapeDimension { shape: %A, dimension: 1 } 表示：循环范围由 %A 的第 1 维决定（即 200）
```

#### FusionInfo 结构

```cpp
struct FusionInfo {
  LinalgOp originalProducer;  // 原始生产者操作
  LinalgOp fusedProducer;     // 融合后的生产者操作
};
```

**作用**：记录融合前后的生产者操作，用于调试和后续转换。

---

### 核心函数详解

**完整计算图示例**：贯穿所有函数：

```
原始代码（融合前）：
┌─────────────────────────────────────────────────────────────┐
│ %A: tensor<128x128xf32>                                     │
│ %B: tensor<128x128xf32>                                     │
│                                                             │
│ // 生产者：计算 A + 1.0                                       │
│ %1 = linalg.generic {                                       │
│   indexing_maps = [                                         │
│     affine_map<(i, j) -> (i, j)>,  // 输入 A                 │
│     affine_map<(i, j) -> (i, j)>   // 输出 %1                │
│   ]                                                         │
│   ins(%A : tensor<128x128xf32>)                             │
│   outs(%init : tensor<128x128xf32>) {                       │
│   ^bb0(%a: f32, %out: f32):                                 │
│     %result = arith.addf %a, %cst_1f : f32  // A + 1.0      │
│     linalg.yield %result : f32                              │
│ }                                                           │
│                                                             │
│ // 消费者：tile 后只计算 [32:64][32:64] 这个 32x32 的块         │
│ %2 = tensor.extract_slice %1[32, 32][32, 32][1, 1]          │
│   // 从 %1 提取：起始位置(32,32)，大小(32,32)                   │
│                                                             │
│ %3 = linalg.generic ins(%2) outs(%C) { ... }                │
└─────────────────────────────────────────────────────────────┘

融合目标：把生产者搬进循环内，只计算需要的 32x32 块
```

---

#### 函数 1: getShapeDefiningLoopRange

**问题**：给定一个循环层次（比如第 0 维循环 i），找出是哪个操作数的哪个维度决定了这个循环的范围？

```cpp
static ShapeDimension getShapeDefiningLoopRange(
    LinalgOp op,
    unsigned loopDepth,
    bool fromSubViewOpOnly = false)
```

**代码执行解析**：

```cpp
// 以 完整计算图示例 中的 linalg.generic 为例介绍
// 遍历所有操作数（输入 %A 和输出 %1）
for (OpOperand &opOperand : op->getOpOperands()) {
  // 如果要求必须来自 SubView/ExtractSlice，具体作用详见下方单独注释
  if (fromSubViewOpOnly &&
      !isa_and_nonnull<memref::SubViewOp, tensor::ExtractSliceOp>(
          opOperand.get().getDefiningOp()))
    continue;  // 跳过不符合条件的

  // 假如 opOperand = %A
  // 获取这个操作数的 IndexingMap，则 map = affine_map<(i, j) -> (i, j)>
  AffineMap map = op.getMatchingIndexingMap(&opOperand);

  // 遍历映射的每个结果（affine_map 右侧的输出索引表达式）
  // 即 map.getResults() = [{.index=0, value=i}, {.index=1, value=j}]
  // 则 en = {.index=0, value=i} 或 {.index=1, value=j}
  for (const auto &en : llvm::enumerate(map.getResults())) {
    // en.index() 是结果索引（0 或 1）
    // en.value() 是结果表达式（i 或 j）

    auto dimExpr = dyn_cast<AffineDimExpr>(en.value());
    // 不是简单的维度引用，例如 affine_map<(i, j) -> (i + 1, j) 中的 i+1，则跳过
    if (!dimExpr) continue;

    // 检查这个维度是否等于我们要找的 loopDepth
    // 下方表达式等于：loopDepth == dimExpr.getPosition()
    // dimExpr.getPosition(): 表示在 affine_map 左侧表达式中的索引
    // en.index() 表示在 affine_map 右侧表达式中的索引
    if (loopDepth == cast<AffineDimExpr>(en.value()).getPosition()) {
      // 若找到，则返回 {操作数, 维度索引}
      // 当loopDepth=0，返回：{ shape: %A, dimension: 0 }，表示：第0层循环上限为 %A.dim[0]
      // 当loopDepth=1，返回：{ shape: %A, dimension: 1 }，表示：第1层循环上限为 %A.dim[1]
      return ShapeDimension{opOperand.get(),
                            static_cast<unsigned>(en.index())};
    }
  }
}
```

**fromSubViewOpOnly的作用**

```cpp
// 参数 fromSubViewOpOnly = true 时的效果：

// ========== 场景 1: 操作数直接是张量 ==========
%1 = linalg.generic ins(%A : tensor<128x128xf32>) outs(%init) { ... }
// 当遍历到操作数 %A 时：
//   %A.getDefiningOp() 返回 nullptr（%A 是函数参数，没有定义操作）
//   isa_and_nonnull<...>(nullptr) 返回 false
//   if 条件为 true，执行 continue，跳过 %A

// ========== 场景 2: 操作数来自 ExtractSliceOp ==========
%slice = tensor.extract_slice %A[32, 32][32, 32][1, 1]
%1 = linalg.generic ins(%slice : tensor<32x32xf32>) outs(%init) { ... }
// 当遍历到操作数 %slice 时：
//   %slice.getDefiningOp() 返回 tensor::ExtractSliceOp
//   isa_and_nonnull<ExtractSliceOp>(...) 返回 true
//   if 条件为 false，不跳过，继续处理 %slice

// ========== 为什么要这样设计？==========
// Fusion 流程中，只有通过 extract_slice/subview 获取的操作数
// 才是"可以被融合的部分"，直接使用的张量（如函数参数）
// 不应该被用来推断循环范围，因为它们可能是完整的张量，
// 而不是已经 tile 过的切片。
```

---

#### 函数 2: getTiledOperands

**问题**：获取切分Tile块需要用到的所有操作数。

```cpp
static SmallVector<Value> getTiledOperands(LinalgOp producer) {
  return producer->getOperands();
}
```

**示例**：

```llvm
// 生产者操作：
%1 = linalg.generic
  ins(%A : tensor<128x128xf32>, %B : tensor<128x128xf32>)
  outs(%init : tensor<128x128xf32>)

// getTiledOperands(producer) 返回：
// [%A, %B, %init]
// 这些是需要创建 Slice 的操作数
```

---

#### 函数 3: fuse（主融合函数）

**问题**：给定一个 producer op 和要融合的循环范围，创建一个只计算指定范围的"克隆版本"。

```cpp
static LinalgOp fuse(OpBuilder &b, LinalgOp producer,
                     const DenseMap<unsigned, Range> &fusedLoopsAndRanges)
```

**代码逐行解析**：

```cpp
// ========== 输入 ==========
// 原始producer（计算整个 128x128）：
%1 = linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i, j)>
  ]
  ins(%A : tensor<128x128xf32>)
  outs(%init : tensor<128x128xf32>) {
  ^bb0(%a: f32, %out: f32):
    %r = arith.addf %a, %cst : f32
    linalg.yield %r
}

// fusedLoopsAndRanges = {
//   0: {offset: 32, size: 32, stride: 1},  // 第 0 维循环
//   1: {offset: 32, size: 32, stride: 1}   // 第 1 维循环
// }
// 解释：只计算 [32:64][32:64] 这个 32x32 块

// ========== 步骤 1: 准备循环信息 ==========
SmallVector<OpFoldResult> ivs, tileSizes, sizeBounds;
SmallVector<Range> loopRanges;
Location loc = producer.getLoc();

// producer.getNumLoops() 返回 2（因为 indexing_map 左侧有 (i, j) 两个循环）
// 遍历所有循环维度
for (unsigned i = 0, e = producer.getNumLoops(); i < e; ++i) {
  // i = 0: 获取第 0 层循环由哪个操作数的哪个维度决定
  //       调用 getShapeDefiningLoopRange(producer, 0)
  //       返回 { shape: %A, dimension: 0 }
  auto shapeDim = getShapeDefiningLoopRange(producer, i);

  // 获取这个维度的实际大小
  // createFoldedDimOp(b, loc, %A, 0) 返回 %A 第 0 维的大小，即 128
  // 生成结果类似：%dim = tensor.dim %A, %c0
  OpFoldResult dim = createFoldedDimOp(b, loc, shapeDim.shape,
                                       shapeDim.dimension);
  sizeBounds.push_back(dim);  // 第一次循环：sizeBounds = [128]

  // 检查这个循环是否要被融合
  auto it = fusedLoopsAndRanges.find(i);
  // i = 0 时，it 指向 fusedLoopsAndRanges[0] = {offset: 32, size: 32, stride: 1}
  if (it != fusedLoopsAndRanges.end()) {
    // 这是一个被融合的循环，使用融合范围
    ivs.push_back(it->second.offset);     // ivs = [32]
    tileSizes.push_back(it->second.size); // tileSizes = [32]
    loopRanges.push_back(it->second);     // loopRanges = [{32, 32, 1}]
  } else {
    // 未被融合的循环，使用完整范围
    tileSizes.push_back(b.getIndexAttr(0));  // 0 表示不融合
    loopRanges.push_back(Range{b.getIndexAttr(0), dim,
                                b.getIndexAttr(1)});
  }

  // i = 1: 同样处理
  //       getShapeDefiningLoopRange(producer, 1) 返回 { shape: %A, dimension: 1 }
  //       dim = 128
  //       sizeBounds = [128, 128]
  //       ivs = [32, 32]
  //       tileSizes = [32, 32]
  //       loopRanges = [{32, 32, 1}, {32, 32, 1}]
}

// 循环结束后：
// 回顾本代码中第19行内容：只计算 [32:64][32:64] 这个 32x32 块，所以：
// ivs = [32, 32] // ivs = induction variables（索引变量），存储每个循环的起始偏移量(i+32, j+32) 
// tileSizes = [32, 32]
// sizeBounds = [128, 128]
// loopRanges = [{32, 32, 1}, {32, 32, 1}]

// ========== 步骤 2: 计算子范围（创建切片）==========
SmallVector<Value> clonedShapes;
// getTiledOperands(producer) 返回 [%A, %init]
// makeTiledShapes 会为每个操作数创建 tensor.extract_slice
clonedShapes.append(makeTiledShapes(
    b, loc, producer, getTiledOperands(producer),
    ivs, tileSizes, sizeBounds,
    /**omitPartialTileCheck=*/false));

// 对于 %A: tensor<128x128xf32>
//   创建 %A_tile = tensor.extract_slice %A[32, 32][32, 32][1, 1]
//   结果类型：tensor<32x32xf32>
//
// 对于 %init: tensor<128x128xf32>
//   创建 %init_tile = tensor.extract_slice %init[32, 32][32, 32][1, 1]
//   结果类型：tensor<32x32xf32>
//
// clonedShapes = [%A_tile, %init_tile]

// ========== 步骤 3: 确定结果类型 ==========
SmallVector<Type, 4> resultTypes;
// producer->getNumResults() 返回 1（linalg.generic 有 1 个输出）
resultTypes.reserve(producer->getNumResults());
// firstInitOperandIdx = 1（因为 %A 是第 0 个操作数，%init 是第 1 个）
int64_t firstInitOperandIdx =
    producerDpsInits.getAsOperandRange().getBeginOperandIndex();
for (int64_t i = 0, e = producer->getNumResults(); i < e; ++i) {
  // i = 0, clonedShapes[1] = %init_tile，类型是 tensor<32x32xf32>
  resultTypes.push_back(clonedShapes[firstInitOperandIdx + i].getType());
}
// resultTypes = [tensor<32x32xf32>]

// ========== 步骤 4: 克隆producer ==========
LinalgOp clonedOp = clone(b, producer, resultTypes, clonedShapes);
// clone 会复制原始操作的所有信息，并用新操作数替换旧操作数：
// %1_tile = linalg.generic {
//   indexing_maps = [
//     affine_map<(i, j) -> (i, j)>,
//     affine_map<(i, j) -> (i, j)>
//   ]
//   ins(%A_tile : tensor<32x32xf32>)
//   outs(%init_tile : tensor<32x32xf32>) {
//   ^bb0(%a: f32, %out: f32):
//     %r = arith.addf %a, %cst : f32
//     linalg.yield %r
// }
// 现在 %1_tile 只计算 32x32 的块！

// ========== 步骤 5: 调整索引偏移 ==========
// offsetIndices 的具体作用：调整操作内部的 linalg.index 操作
//
// 什么是 linalg.index？
//   在 linalg.generic 等操作内部，可以使用 linalg.index op 获取当前循环索引
//   例如：%i = linalg.index 0  // 获取第 0 维的循环索引
//
// 举例说明问题：
//   ========== 原始操作（计算整个 128x128）==========
//   %1 = linalg.generic {
//     indexing_maps = [affine_map<(i, j) -> (i, j)>,
//                      affine_map<(i, j) -> (i, j)>]
//     ins(%A : tensor<128x128xf32>)
//     outs(%init : tensor<128x128xf32>) {
//     ^bb0(%a: f32, %out: f32):
//       %i = linalg.index 0  // 获取第 0 维索引，范围 [0, 128)
//       %j = linalg.index 1  // 获取第 1 维索引，范围 [0, 128)
//       // 假设有条件逻辑：只在 i > 64 时才计算
//       %cond = arith.cmpi sgt, %i, %c64 : index
//       %r = arith.select %cond, %a, %cst : f32
//       linalg.yield %r
//   }
//
//   ========== 融合后的问题（只计算 [32:64][32:64]）==========
//   %1_tile = linalg.generic {
//     ins(%A_tile : tensor<32x32xf32>)
//     outs(%init_tile : tensor<32x32xf32>) {
//     ^bb0(%a: f32, %out: f32):
//       %i = linalg.index 0  // 问题：这里返回什么？
//       %j = linalg.index 1  // 问题：这里返回什么？
//       // 如果不调整，%i 仍然返回 [0, 32)，范围不对！
//       // 原来的条件 "i > 64" 永远不会满足，逻辑就错了
//       ...
//   }
//
//   ========== offsetIndices 的修复 ==========
//   offsetIndices 会做以下转换：
//     %i = linalg.index 0
//   -->
//     %i_offset = arith.addi %i, %c32 : index
//     // 然后把所有使用 %i 的地方替换为 %i_offset
//
//   修复后：
//     ^bb0(%a: f32, %out: f32):
//       %i_local = linalg.index 0        // 返回 [0, 32)，局部索引
//       %i = arith.addi %i_local, %c32   // 返回 [32, 64)，全局索引
//       %j_local = linalg.index 1
//       %j = arith.addi %j_local, %c32
//       // 现在条件 "i > 64" 可以正确判断了
//       %cond = arith.cmpi sgt, %i, %c64 : index
//       ...
//
//   ========== 源码分析 ==========
//   for (IndexOp indexOp : linalgOp.getBlock()->getOps<IndexOp>()) {
//     // 找到每个 linalg.index op
//     // 例如：%i = linalg.index 0
//
//     OpFoldResult applied = makeComposedFoldedAffineApply(
//         b, indexOp.getLoc(), index + offset,
//         {indexOp.getResult(), offsets[indexOp.getDim()]});
//     // 创建 affine_apply: %i + %c32
//
//     b.replaceUsesWithIf(indexOp, materialized, ...);
//     // 把所有使用 %i 的地方替换为 (%i + 32)
//   }

SmallVector<OpFoldResult> allIvs = llvm::to_vector(
    llvm::map_range(loopRanges, [&](Range range) {
      return range.offset;  // 提取每个范围的偏移
    }));
// allIvs = [32, 32]
offsetIndices(b, clonedOp, allIvs);

return clonedOp;  // 返回融合后的操作 %1_tile
```

---

#### 函数 4: getRangeFromOperandShape

**问题**：从切片操作数中提取范围信息。

```cpp
static Range getRangeFromOperandShape(OpBuilder &b, Location loc,
                                      Value shapedOperand, unsigned dim)
```

**代码逐行解析**：

```cpp
// ========== 输入 ==========
// 输入切片：
%slice = tensor.extract_slice %A[32, 16][32, 32][1, 1]
  // 从 %A[32:64][16:48] 提取 32x32 的块

// 调用：getRangeFromOperandShape(b, loc, %slice, 0)
// 意思：获取第 0 维的范围

// ========== 执行过程 ==========
// 获取定义这个操作数的操作
Operation *shapeProducingOp = shapedOperand.getDefiningOp();
// shapeProducingOp 是 tensor::ExtractSliceOp

// 根据操作类型提取范围
if (auto subViewOp = dyn_cast<memref::SubViewOp>(shapeProducingOp))
  // Buffer 语义：从 SubViewOp 获取范围
  return subViewOp.getOrCreateRanges(b, loc)[dim];

if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(shapeProducingOp))
  // Tensor 语义：从 ExtractSliceOp 获取范围
  // dim = 0，获取第 0 维的范围
  return sliceOp.getOrCreateRanges(b, loc)[dim];
  // getOrCreateRanges() 返回 [{offset: 32, size: 32, stride: 1},
  //                             {offset: 16, size: 32, stride: 1}]
  // 返回：Range { offset: 32, size: 32, stride: 1 }
  // 解释：从偏移 32 开始，大小 32，步长 1

llvm_unreachable("必须是 SubViewOp 或 ExtractSliceOp");

// ========== 如果查询第 1 维 ==========
// getRangeFromOperandShape(b, loc, %slice, 1)
// 返回：Range { offset: 16, size: 32, stride: 1 }
```

---

#### 函数 5: fuse（重载版本）

**问题**：这是更高级的 fuse 函数，自动从消费者操作数推断融合范围。

```cpp
static LinalgOp fuse(OpBuilder &b, LinalgOp producerOp,
                     AffineMap producerMap, OpOperand &consumerOpOperand)
```

**代码逐行解析**：

```cpp
// ========== 输入 ==========
// 生产者（输出维度到循环的映射）：
producerMap = affine_map<(i, j) -> (i, j)>
// 解释：生产者的第 0 个输出维度对应循环 i，第 1 个对应循环 j

// 消费者使用的切片：
%slice = tensor.extract_slice %producer[32, 32][32, 32][1, 1]

// ========== 执行过程 ==========
DenseMap<unsigned, Range> fusedLoopsAndRanges;
Value shapedOperand = consumerOpOperand.get();  // shapedOperand = %slice

// 遍历生产者索引映射的每个结果
// producerMap.getResults() = [i, j]
for (const auto &en : llvm::enumerate(producerMap.getResults())) {
  // en.index() 是结果索引（0, 1, ...）
  // en.value() 是结果表达式（如 i, j）

  // 第一次循环：en.index() = 0, en.value() = i
  // 获取这个表达式对应的循环位置
  unsigned posInProducerLoop =
      cast<AffineDimExpr>(en.value()).getPosition();
  // i.getPosition() = 0
  // posInProducerLoop = 0

  // 从消费者切片中获取对应维度的范围
  // en.index() = 0，获取切片的第 0 维范围
  fusedLoopsAndRanges[posInProducerLoop] =
      getRangeFromOperandShape(
          b, consumerOpOperand.getOwner()->getLoc(),
          shapedOperand,           // %slice
          en.index());             // 0
  // fusedLoopsAndRanges[0] = {offset: 32, size: 32, stride: 1}

  // 第二次循环：en.index() = 1, en.value() = j
  // j.getPosition() = 1
  // posInProducerLoop = 1
  // 获取切片的第 1 维范围
  // fusedLoopsAndRanges[1] = {offset: 32, size: 32, stride: 1}
}

// fusedLoopsAndRanges = {
//   0: {offset: 32, size: 32, stride: 1},
//   1: {offset: 32, size: 32, stride: 1}
// }

// 调用主 fuse 函数执行融合
return fuse(b, producerOp, fusedLoopsAndRanges);
// 结果：创建只计算 [32:64][32:64] 的克隆版本
```

---

#### 函数 6: getProducerOfTensor

**问题**：沿着 use-def 链向上追溯，找到真正产生这个张量的操作。

```cpp
static void getProducerOfTensor(Value tensor, OpResult &opResult)
```

**代码逐行解析**：

```cpp
// ========== 场景 1: 直接生产者 ==========
// %1 = linalg.generic ins(%A) outs(%init) { ... }
// getProducerOfTensor(%1)

while (true) {
  // 情况 1: 直接由 LinalgOp 定义
  if (auto linalgOp = tensor.getDefiningOp<LinalgOp>()) {
    // 找到了！这个操作就是生产者
    // tensor = %1，由 linalg.generic 定义
    opResult = cast<OpResult>(tensor);  // 转换为 OpResult
    // opResult = linalg.generic 的第 0 个结果
    return;
  }
  // 返回：linalg.generic 操作的第 0 个结果

  // ========== 场景 2: 通过切片链接 ==========
  // %1 = linalg.generic ins(%A) outs(%init) { ... }
  // %slice = tensor.extract_slice %1[10, 10][32, 32][1, 1]
  // getProducerOfTensor(%slice)

  // 情况 2: 通过 ExtractSliceOp 链接
  if (auto sliceOp = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
    // 第一次循环：tensor = %slice，由 ExtractSliceOp 定义
    // 获取切片的源，继续追溯
    tensor = sliceOp.getSource();
    // tensor = %1
    continue;  // 继续循环

    // 第二次循环：tensor = %1，由 linalg.generic 定义
    // 进入情况 1，返回 linalg.generic 的第 0 个结果
  }

  // ========== 场景 3: 通过循环迭代参数 ==========
  // %1 = linalg.generic ins(%A) outs(%init) { ... }
  // %2 = scf.for %i = 0 to 10 iter_args(%arg = %1) {
  //   %3 = linalg.generic ins(%arg) outs(%init2) { ... }
  //   scf.yield %3
  // }
  // getProducerOfTensor(%arg)

  // 情况 3: 通过 scf::For 迭代参数
  if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
    // 第一次循环：tensor = %arg，是 BlockArgument
    if (auto forOp = blockArg.getDefiningOp<scf::ForOp>()) {
      // %arg 由 scf.for 定义
      // 获取循环的初始值，继续追溯
      // blockArg.getArgNumber() = 0（%arg 是第 0 个迭代参数）
      // forOp.getInitArgs()[0] = %1
      tensor = forOp.getInitArgs()[blockArg.getArgNumber()];
      // tensor = %1
      continue;  // 继续循环

      // 第二次循环：tensor = %1，由 linalg.generic 定义
      // 进入情况 1，返回 linalg.generic 的第 0 个结果
    }
  }

  // 无法找到生产者（可能来自函数参数等）
  return;
}
```

---

#### 函数 7: fuseProducerOfTensor（公共 API）

**问题**：这是主要的融合入口点，整合所有步骤。

```cpp
FailureOr<FusionInfo> mlir::linalg::fuseProducerOfTensor(
    OpBuilder &b, OpOperand &consumerOpOperand)
```

**代码逐行解析**：

```cpp
// ========== 输入：融合前 ==========
// consumerOpOperand 是消费者操作的第 0 个输入（%1_tile）
func.func @example(%A: tensor<128x128xf32>,
                   %C: tensor<128x128xf32>) {
  // 生产者：计算整个 128x128
  %1 = linalg.generic ins(%A) outs(%init) { ... }

  // 消费者所在的循环（已经 tile 过）
  %2 = scf.for %ii = 0 to 128 step 32 iter_args(%arg = %C) {
    %3 = scf.for %jj = 0 to 128 step 32 iter_args(%arg2 = %arg) {
      // 切片：只使用 [ii:ii+32][jj:jj+32]
      %1_tile = tensor.extract_slice %1[%ii, %jj][32, 32][1, 1]

      // 消费者操作
      %result = linalg.generic ins(%1_tile) outs(%tile) { ... }
      // ...
    }
  }
}

// ========== 步骤 1: 查找生产者 ==========
Value inputTensor = consumerOpOperand.get();
// inputTensor = %1_tile
OpResult producerOpResult;
getProducerOfTensor(inputTensor, producerOpResult);
// 沿着 use-def 链追溯：
//   %1_tile 由 ExtractSliceOp 定义，获取源 %1
//   %1 由 linalg.generic 定义，找到了！
// producerOpResult = %1 (linalg.generic 的第 0 个结果)

if (!producerOpResult) {
  return failure();  // 找不到生产者，无法融合
}

// ========== 步骤 2: 验证操作类型 ==========
auto producerOp = dyn_cast<LinalgOp>(producerOpResult.getOwner());
// producerOp = %1 的定义操作（linalg.generic）
if (!producerOp) return failure();  // 生产者不是 LinalgOp

LinalgOp consumerOp = dyn_cast<LinalgOp>(consumerOpOperand.getOwner());
// consumerOp = 消费者操作（linalg.generic）
if (!consumerOp) return failure();  // 消费者不是 LinalgOp

// ========== 步骤 3: 验证必须是 ExtractSliceOp ==========
auto sliceOp = inputTensor.getDefiningOp<tensor::ExtractSliceOp>();
// sliceOp = %1_tile 的定义操作（tensor::ExtractSliceOp）
if (!sliceOp) {
  return failure();  // 不是 extract_slice，无法融合
}
// 这是因为融合依赖于切片来知道需要计算的范围

// ========== 步骤 4: 检查是否已融合 ==========
if (consumerOpOperand.get().getParentBlock() ==
    producerOpResult.getParentBlock())
  // %1_tile 所在的块（循环内）≠ %1 所在的块（函数顶部）
  // 条件为 false，不返回，可以融合
  return failure();  // 已经在同一基本块中，已经融合过了

// ========== 步骤 5: 在消费者之前插入融合的生产者 ==========
OpBuilder::InsertionGuard g(b);  // 保存插入位置
b.setInsertionPoint(consumerOp);  // 在消费者之前插入

// 获取生产者的输出操作数（dpsInit）
// producerOpResult.getResultNumber() = 0
OpOperand *opOperand =
    producerOp.getDpsInitOperand(producerOpResult.getResultNumber());
// opOperand = %init（生产者的输出操作数）

// 执行融合
LinalgOp fusedProducer =
    fuse(b, producerOp,
         producerOp.getMatchingIndexingMap(opOperand),  // 索引映射
         consumerOpOperand);  // 消费者的操作数
// 调用 fuse 函数，创建融合后的操作：
// %1_fused = linalg.generic ins(%A_tile) outs(%init_tile)

// ========== 步骤 6: 处理 Rank Reduction ==========
Value def = fusedProducer->getResult(producerOpResult.getResultNumber());
// def = %1_fused（类型：tensor<32x32xf32>）
Type consumerType = consumerOpOperand.get().getType();
// consumerType = %1_tile 的类型（tensor<32x32xf32>）

if (cast<ShapedType>(consumerType).getRank() !=
    cast<ShapedType>(def.getType()).getRank()) {
  // 维度数量不匹配，说明发生了降维
  // 例如：tensor<32x32x1xf32> -> tensor<32x32xf32>
  // 本例：3 != 2 为 false，跳过
  llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
  def = tensor::dropGivenUnitDims(b, fusedProducer.getLoc(),
                                  def, droppedDims);
}

// ========== 步骤 7: 类型转换（如果需要）==========
if (consumerType != def.getType())
  // 类型不匹配，插入转换操作
  // 本例：tensor<32x32xf32> == tensor<32x32xf32>，跳过
  def = b.create<tensor::CastOp>(fusedProducer.getLoc(),
                                 consumerType, def);

// ========== 步骤 8: 替换使用 ==========
consumerOpOperand.set(def);  // 用融合后的结果替换原操作数
// 消费者操作的第 0 个输入从 %1_tile 替换为 %1_fused

return FusionInfo{cast<LinalgOp>(producerOpResult.getOwner()),
                  fusedProducer};

// ========== 输出：融合后 ==========
func.func @example(%A: tensor<128x128xf32>,
                   %C: tensor<128x128xf32>) {
  %2 = scf.for %ii = 0 to 128 step 32 iter_args(%arg = %C) {
    %3 = scf.for %jj = 0 to 128 step 32 iter_args(%arg2 = %arg) {
      // 切片输入 A
      %A_tile = tensor.extract_slice %A[%ii, %jj][32, 32][1, 1]
      %init_tile = tensor.empty() : tensor<32x32xf32>

      // 融合的生产者：只计算需要的 32x32 块
      %1_fused = linalg.generic ins(%A_tile) outs(%init_tile) { ... }

      // 消费者直接使用融合的结果
      %result = linalg.generic ins(%1_fused) outs(%tile) { ... }
      // ...
    }
  }
}

// 好处：
// 1. 消除了中间张量 %1 的 128x128 计算
// 2. 只计算循环内实际需要的 32x32 块
// 3. 减少内存访问，提高缓存利用率
```

---

## Fusion 与 Tiling 的配合

### TileAndFuse 工作流

```
1. Tiling 消费者操作
   └─> 生成 extract_slice 操作

2. 查找 extract_slice 的生产者
   └─> 通过 getProducerOfTensor

3. 融合生产者到消费者循环内
   └─> 使用 fuseProducerOfTensor

4. 重复直到没有更多可融合的操作
```

### 完整示例

```mlir
// ========== 原始代码 ==========
func.func @example(%A: tensor<128x128xf32>,
                   %B: tensor<128x128xf32>,
                   %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // 生产者 1
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i, j)>
    ]
    ins(%A : tensor<128x128xf32>)
    outs(%init1 : tensor<128x128xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %tmp = arith.addf %arg0, %cst : f32
      linalg.yield %tmp : f32
  }

  // 生产者 2
  %2 = linalg.matmul ins(%1, %B) outs(%init2)

  // 消费者
  %3 = linalg.generic ins(%2) outs(%C) {
    ^bb0(%arg0: f32, %arg1: f32):
      %res = arith.mulf %arg0, %cst2 : f32
      linalg.yield %res : f32
  }

  return %3 : tensor<128x128xf32>
}

// ========== 步骤 1: Tile 消费者 (tile_size = 32) ==========
%3 = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) {
  %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) {
    // 提取切片
    %2_tile = tensor.extract_slice %2[%ii, %jj][32, 32][1, 1]
    %C_tile = tensor.extract_slice %arg5[%ii, %jj][32, 32][1, 1]

    // 消费者（平铺后）
    %3_tile = linalg.generic ins(%2_tile) outs(%C_tile) { ... }

    %result3 = tensor.insert_slice %3_tile into %arg5[%ii, %jj][32, 32][1, 1]
    scf.yield %result3
  }
  scf.yield %result2
}

// ========== 步骤 2: 融合生产者 2 (matmul) ==========
// 检测到 %2_tile 由 extract_slice 定义，融合 matmul
%3 = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) {
  %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) {
    // 融合的 matmul（只计算需要的 tile）
    %1_tile = tensor.extract_slice %1[%ii, 0][32, 128][1, 1]
    %B_tile = tensor.extract_slice %B[0, %jj][128, 32][1, 1]
    %output_tile = tensor.extract_slice %arg5[%ii, %jj][32, 32][1, 1]

    %2_tile = linalg.matmul
      ins(%1_tile, %B_tile)
      outs(%output_tile)

    %3_tile = linalg.generic ins(%2_tile) outs(%output_tile) { ... }

    %result3 = tensor.insert_slice %3_tile into %arg5[%ii, %jj][32, 32][1, 1]
    scf.yield %result3
  }
  scf.yield %result2
}

// ========== 步骤 3: 融合生产者 1 (generic add) ==========
// 进一步融合 %1_tile 的生产者
%3 = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) {
  %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) {
    // 融合的 add
    %A_tile = tensor.extract_slice %A[%ii, 0][32, 128][1, 1]
    %init_tile = tensor.empty... : tensor<32x128xf32>
    %1_tile = linalg.generic ins(%A_tile) outs(%init_tile) {
      ^bb0(%arg0: f32, %arg1: f32):
        %tmp = arith.addf %arg0, %cst : f32
        linalg.yield %tmp : f32
    }

    %B_tile = tensor.extract_slice %B[0, %jj][128, 32][1, 1]
    %output_tile = tensor.extract_slice %arg5[%ii, %jj][32, 32][1, 1]

    %2_tile = linalg.matmul ins(%1_tile, %B_tile) outs(%output_tile)

    %3_tile = linalg.generic ins(%2_tile) outs(%output_tile) { ... }

    %result3 = tensor.insert_slice %3_tile into %arg5[%ii, %jj][32, 32][1, 1]
    scf.yield %result3
  }
  scf.yield %result2
}
```

---

### Fusion 决策树

```
问题: 应该使用哪种 Fusion 策略?

1. 是否在 TileAndFuse 流程中?
   YES -> 使用 Fusion.cpp 的 fuseProducerOfTensor

2. 是逐元素操作链吗?
   YES -> 检查 areElementwiseOpsFusable
         如果可融合 -> 使用 ElementwiseOpFusion.cpp

3. 有 Pad 操作吗?
   YES -> 检查生产者是否为全并行 GenericOp
         如果是 -> 使用 FusePadOpWithLinalgProducer

4. 需要部分计算吗?
   YES -> 使用 Tiling + Fusion
   NO -> 考虑完全融合
```
