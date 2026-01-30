# MLIR ElementwiseOpFusion 优化模式技术分析

本文介绍 `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp` 中的三种关键优化模式：
1. `populateFoldReshapeOpsByExpansionPatterns`
2. `tensor::populateBubbleUpExpandShapePatterns`
3. `populateConstantFoldLinalgOperations`

---

## 1. populateFoldReshapeOpsByExpansionPatterns

### 核心原理

**维度扩展融合**：通过扩展 Linalg 操作的循环维度来融合 reshape 操作，消除中间的 reshape/collapse_shape 节点，减少内存访问开销。

该模式包含两个核心 Pattern：
- `FoldWithProducerReshapeOpByExpansion`：融合 producer 的 `collapse_shape` 与 consumer
- `FoldReshapeWithGenericOpByExpansion`：融合 producer generic 与 consumer 的 `expand_shape`

### 触发条件

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp:563
static bool isFusableWithReshapeByDimExpansion(LinalgOp linalgOp,
                                               OpOperand *fusableOpOperand) {
  // 条件1: 所有 indexing maps 必须是 projected permutation
  // 条件2: 融合的张量不能是标量
  // 条件3: 必须是纯 tensor 语义
  return linalgOp.hasPureTensorSemantics() &&
         llvm::all_of(linalgOp.getIndexingMaps().getValue(),
                      [](AffineMap map) { return map.isProjectedPermutation(); });
}
```

### 示例讲解

#### Before (未优化)

```mlir
// 原始: 2D tensor (16x64) -> collapse -> (16x1) -> generic op
%0 = tensor.collapse_shape %arg0 [[0, 1]] // 16x64 -> 16
%1 = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>,
                   affine_map<(d0) -> (d0)>]
  ins(%0 : tensor<16xf32>)
  outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
}
```

#### After (优化后)

```mlir
// 融合后: generic op 直接在原始 2D tensor 上操作
%1 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,   // 扩展为 2D
                   affine_map<(d0, d1) -> (d0)>]        // 输出保持 1D
  ins(%arg0 : tensor<16x64xf32>)
  outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
}
```

### 优化效果

- **消除内存拷贝**：不需要生成 collapse_shape 的中间张量
- **提升并行度**：扩展后的循环维度可以更好地利用并行硬件
- **减少访存**：融合后的操作可以复用缓存行

---

## 2. tensor::populateBubbleUpExpandShapePatterns

### 核心原理

**上浮变换**：当 `expand_shape` 的 producer 是 `collapse_shape` 时，如果两者的 reassociation 索引"平行"（parallel），则可以交换它们的位置，使 `expand_shape` 向上移动。

这样做的目的是让 `expand_shape` 能够与其他模式（如上述的扩展融合模式）配合，进一步优化。

### 触发条件

```cpp
// mlir/lib/Dialect/Tensor/Transforms/ReshapePatterns.cpp:169
// 两个 reshape 操作平行的条件：
// 1. reassociation 索引大小相同，或
// 2. collapse 或 expand 的 reassociation 大小为 1
for (auto [expandReassociation, collapseReassociation] :
     llvm::zip_equal(expandReInds, collapseReInds)) {
  if (collapseReassociation.size() == expandReassociation.size()) {
    // 验证静态形状是否一致
    continue;
  }
  if (collapseReassociation.size() != 1 && expandReassociation.size() != 1)
    return failure();  // 不平行，无法上浮
}
```

### 示例讲解

#### Before (未优化)

```mlir
// 场景: (16x64) -> collapse -> (16) -> expand -> (4x4)
%0 = tensor.collapse_shape %arg0 [[0], [1]]  // 16x64 -> 16x1 (实际 16)
%1 = tensor.expand_shape %0 [[0, 1], [2]]    // 16 -> 4x4x1 (实际 4x4)
```

#### After (上浮后)

```mlir
// 交换: (16x64) -> expand -> (16x4x16) -> collapse -> (4x4)
%0 = tensor.expand_shape %arg0 [[0], [1, 2]]    // 16x64 -> 16x4x16
%1 = tensor.collapse_shape %0 [[0, 1], [2]]     // 16x4x16 -> 4x16 (实际 4x4)
```

### 优化效果

- **为融合创造条件**：上浮后的 expand_shape 可能与更上层的 linalg 操作融合
- **消除冗余 reshape**：某些情况下 collapse 和 expand 会相互抵消

---

## 3. populateConstantFoldLinalgOperations

### 核心原理

**常量折叠**：当 Linalg 操作（如 transpose）的输入全部是编译时常量时，直接在编译期计算出结果常量，替换整个计算操作。

目前实现的 Pattern：
- `FoldConstantTranspose`：专门处理 transpose 操作

### 实现机制

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ConstantFold.cpp:265
struct FoldConstantTranspose : public FoldConstantBase<FoldConstantTranspose> {
  // 1. 验证 indexing maps 只有一个输入和一个输出
  // 2. 验证 region 只包含 yield op
  // 3. yield 直接返回输入（无实际计算）
  // 4. 根据 indexing maps 重排常量元素
}
```

常量重排使用**索引去线性化**技术：

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ConstantFold.cpp:181
auto computeRemappedLinearIndex = [&](int linearIndex) {
  // 线性索引 -> 多维索引
  for (int dim = loopBounds.size() - 1; dim >= 0; --dim) {
    indices[dim] = totalCount % loopBounds[dim];
    totalCount /= loopBounds[dim];
  }
  // 根据 indexing maps 映射到输入/输出的多维索引
  // 再转回线性索引进行访问
};
```

### 示例讲解

#### Before (未优化)

```mlir
// 编译期常量转置
%0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
%1 = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d1, d0)>,  // transpose: (i,j) -> (j,i)
    affine_map<(d0, d1) -> (d0, d1)>
  ]
  ins(%0 : tensor<2x2xf32>)
  outs(%init : tensor<2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
}
```

#### After (常量折叠后)

```mlir
// 直接计算好的转置结果
%0 = arith.constant dense<[[1.0, 3.0], [2.0, 4.0]]> : tensor<2x2xf32>
```

### 优化效果

- **零运行时开销**：整个操作在编译期完成
- **减少代码体积**：消除循环和控制流
- **便于后续优化**：常量传播可继续向上传递

---

## 三种优化的协同工作

在 `LinalgElementwiseOpFusionPass` 中，这些模式按以下顺序应用：

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp:2301
populateElementwiseOpsFusionPatterns(patterns, defaultControlFn);
populateFoldReshapeOpsByExpansionPatterns(patterns, defaultControlFn);
tensor::populateBubbleUpExpandShapePatterns(patterns);      // 上浮 expand
// ... canonicalization ...
populateConstantFoldLinalgOperations(patterns, defaultControlFn);  // 常量折叠
```

### 协同流程示例

```
原始 IR:
  [Constant 2x4] -> Transpose -> [Constant 4x2] -> Reshape -> [4x2] -> Generic -> ...
        |                      |
        +---(1)常量折叠--------+            (3)上浮 expand
                                      |
                            (2)扩展融合
```

1. **常量折叠**先处理 Constant Transpose
2. **扩展融合**处理 Reshape + Generic
3. **上浮 expand**为融合创造更多机会

---

## 总结

| 优化模式 | 核心技术 | 目标 | 适用场景 |
|---------|---------|------|---------|
| FoldReshapeByExpansion | 维度扩展融合 | 消除 reshape 中间结果 | Linalg + reshape 链 |
| BubbleUpExpandShape | 变换上浮 | 为融合创造条件 | collapse + expand 链 |
| ConstantFold | 编译期求值 | 消除运行时计算 | 常量输入的 Linalg |

这三种优化通过减少内存访问、消除冗余操作和编译期求值，共同提升 MLIR 程序的性能。
