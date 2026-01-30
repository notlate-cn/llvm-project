# MLIR Tensor方言Transform完全指南

本文档详细介绍MLIR Tensor方言中的所有Transform变换，包括其作用、技术原理和实例演示。

## 目录

1. [子集操作优化](#1-子集操作优化)
   - [MergeConsecutiveInsertExtractSlicePatterns](#11-mergeconsecutiveinsertextractslicepatterns)
   - [FoldTensorSubsetOps](#12-foldtensorsubsetops)

2. [形状变换](#2-形状变换)
   - [ReshapePatterns (RankReductionPatterns)](#21-reshapepatterns-rankreductionpatterns)
   - [ExtractSliceFromReshapeUtils](#22-extractslicefromreshapeutils)

3. [常量折叠](#3-常量折叠)
   - [RewriteAsConstant](#31-rewriteasconstant)
   - [EmptyOpPatterns](#32-emptyoppatterns)

4. [操作分解](#4-操作分解)
   - [ConcatOpPatterns](#41-concatoppatterns)

5. [Tiling和融合](#5-tiling和融合)
   - [SwapExtractSliceWithProducerPatterns](#51-swapextractslicewithproducerpatterns)

6. [独立性变换](#6-独立性变换)
   - [IndependenceTransforms](#61-independencetransforms)

7. [运行时支持](#7-运行时支持)
   - [RuntimeOpVerification](#71-runtimeopverification)

8. [接口实现](#8-接口实现)
   - [BufferizableOpInterface](#81-bufferizableopinterface)
   - [SubsetInsertionOpInterface](#82-subsetinsertionopinterface)

---

## 1. 子集操作优化

### 1.1 MergeConsecutiveInsertExtractSlicePatterns

**文件**: `MergeConsecutiveInsertExtractSlicePatterns.cpp`

**作用**: 合并连续的 tensor.extract_slice 和 tensor.insert_slice 操作

**技术原理**:

#### ExtractSlice合并

使用 `affine::mergeOffsetsSizesAndStrides()` 合并两个连续的slice操作：

**组合规则**:
- **新偏移**: `new_offset = prev_offset + next_offset * prev_stride`
- **新大小**: `new_size = next_size`
- **新步长**: `new_stride = prev_stride * next_stride`

#### InsertSlice合并

**合并条件**:
- 两个insert_slice都必须有单位步长 (unit stride)
- 第一个insert_slice必须是rank-reducing操作
- 类型必须是静态形状

#### 冗余Rank扩展消除

**模式1**: ExtractSlice消除InsertSlice的rank扩展
```mlir
// 插入扩展rank
%0 = tensor.insert_slice %src into %dest[0, 0] [1, 1, 5, 10] [1, 1, 1, 1]
    : tensor<5x10xf32> into tensor<1x1x5x10xf32>

// 提取时移除rank
%1 = tensor.extract_slice %0[0, 0, 2, 3] [1, 1, 2, 2] [1, 1, 1, 1]
    : tensor<1x1x5x10xf32> to tensor<2x2xf32>

// 可以简化为直接从源提取
%1 = tensor.extract_slice %src[2, 3] [2, 2] [1, 1]
    : tensor<5x10xf32> to tensor<2x2xf32>
```

**模式2**: InsertSlice消除ExtractSlice的rank缩减
```mlir
// 提取时缩减rank
%0 = tensor.extract_slice %in[0, 0] [1, 8] [1, 1]
    : tensor<2x8xf32> to tensor<8xf32>

// 插入时扩展rank
%1 = tensor.insert_slice %0 into %dest[0, 0] [1, 8] [1, 1]
    : tensor<8xf32> into tensor<1x8xf32>

// 可以简化为直接提取到目标类型
%1 = tensor.extract_slice %in[0, 0] [1, 8] [1, 1]
    : tensor<2x8xf32> to tensor<1x8xf32>
```

**核心实现**:
```cpp
struct MergeConsecutiveExtractSlice : public OpRewritePattern<ExtractSliceOp> {
  LogicalResult matchAndRewrite(ExtractSliceOp nextOp,
                                PatternRewriter &rewriter) const {
    auto prevOp = nextOp.getSource().getDefiningOp<ExtractSliceOp>();
    if (!prevOp) return failure();

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, nextOp.getLoc(), prevOp, nextOp,
            prevOp.getDroppedDims(), newOffsets, newSizes, newStrides)))
      return failure();

    rewriter.replaceOpWithNewOp<ExtractSliceOp>(nextOp, nextOp.getType(),
                                                prevOp.getSource(), newOffsets,
                                                newSizes, newStrides);
  }
};
```

**使用场景**:
- 简化嵌套的切片操作
- 消除冗余的rank变换
- 为向量化做准备

---

### 1.2 FoldTensorSubsetOps

**文件**: `FoldTensorSubsetOps.cpp`

**作用**: 将tensor子集操作与生产者/消费者融合

**技术原理**:

#### TransferRead融合

**模式**: `vector.transfer_read(tensor.extract_slice %x[...])` → 直接从%x读取

**前置条件**:
```cpp
preconditionsFoldExtractOrInsertWithTransferOp() {
  // 1. 不能有越界的维度
  if (xferOp.hasOutOfBoundsDim()) return failure();

  // 2. 不能有mask
  if (xferOp.getMask()) return failure();

  // 3. 必须有单位步长
  if (!extractOrInsertSliceOp.hasUnitStride()) return failure();
}
```

**变换示例**:
```mlir
// 融合前
%slice = tensor.extract_slice %src[%off0, %off1][%size0, %size1][1, 1]
    : tensor<100x100xf32> to tensor<10x10xf32>
%vec = vector.transfer_read %slice[%c0, %c0]
    {in_bounds = [true, true]} : tensor<10x10xf32>, vector<10x10xf32>

// 融合后
%vec = vector.transfer_read %src[%off0, %off1]
    {in_bounds = [true, true]} : tensor<100x100xf32>, vector<10x10xf32>
```

#### TransferWrite融合

**模式**: `tensor.insert_slice(vector.transfer_write(...))` → 直接写入目标

**变换示例**:
```mlir
// 融合前
%written = vector.transfer_write %vec, %init[%c0, %c0]
    {in_bounds = [true, true]} : vector<10x10xf32>, tensor<10x10xf32>
%result = tensor.insert_slice %written into %dest[%off0, %off1][10, 10][1, 1]
    : tensor<10x10xf32> into tensor<100x100xf32>

// 融合后
%result = vector.transfer_write %vec, %dest[%off0, %off1]
    {in_bounds = [true, true]} : vector<10x10xf32>, tensor<100x100xf32>
```

#### InsertSlice链合并

**模式**: 连续的insert_slice可以合并为一个

**变换示例**:
```mlir
// 合并前
%partial = tensor.insert_slice %tile1 into %init[0, 0][16, 16][1, 1]
    : tensor<16x16xf32> into tensor<32x32xf32>
%result = tensor.insert_slice %tile2 into %partial[16, 0][16, 32][1, 1]
    : tensor<16x32xf32> into tensor<32x32xf32>

// 合并后（当第二个insert覆盖第一个的结果区域时）
%result = tensor.insert_slice %tile2 into %init[16, 0][16, 32][1, 1]
    : tensor<16x32xf32> into tensor<32x32xf32>
```

**核心模式**:
```cpp
struct TransferReadOfExtractSliceOpFolder
    : public OpRewritePattern<vector::TransferReadOp> {
  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const {
    // 调整transfer_read的索引以考虑extract_slice的偏移
    // 创建新的transfer_read直接从源读取
  }
};
```

**使用场景**:
- 向量化优化
- 消除中间tensor
- 提高内存访问效率

---

## 2. 形状变换

### 2.1 ReshapePatterns (RankReductionPatterns)

**文件**: 实际为 `RankReductionPatterns.cpp`

**作用**: 优化reshape操作与extract/insert_slice的交互

**技术原理**:

#### Cancel模式

当reshape和slice操作的效果可以相互抵消时，将其合并：

**Expand + ExtractSlice → ExtractSlice**
```mlir
// 变换前
%expanded = tensor.expand_shape %src [[0, 1], [2]]
    : tensor<12x4xf32> into tensor<3x4x4xf32>
%slice = tensor.extract_slice %expanded[0, 1, 0][1, 2, 4][1, 1, 1]
    : tensor<3x4x4xf32> to tensor<2x4xf32>

// 变换后（当extract在重新关联的组内是连续的）
%slice = tensor.extract_slice %src[4, 0][8, 4][1, 1]
    : tensor<12x4xf32> to tensor<2x4xf32>
```

#### Bubble Up模式

将extract_slice移过reshape操作，重新计算offsets/sizes：

**ExtractSlice + CollapseShape**
```mlir
// 变换前
%collapsed = tensor.collapse_shape %src [[0, 1], [2]]
    : tensor<4x5x6xf32> into tensor<20x6xf32>
%slice = tensor.extract_slice %collapsed[2, 0][10, 6][1, 1]
    : tensor<20x6xf32> to tensor<10x6xf32>

// 变换后（slice移到collapse之前）
%slice = tensor.extract_slice %src[0, 2, 0][4, 5, 6][1, 1, 1]
    : tensor<4x5x6xf32> to tensor<4x5x6xf32>
%collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
    : tensor<4x5x6xf32> into tensor<10x6xf32>
```

**核心算法**:
```cpp
// 检查slice在重关联组内的连续性
bool areExtractedSliceDimsContiguous(
    ArrayRef<ReassociationIndices> reassociationIndices,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) {

  for (const ReassociationIndices &indices : reassociationIndices) {
    // 找第一个非unit size的维度
    // 验证后续维度提取完整大小
    // 检查stride为1
  }
}
```

#### Bubble Up Expand Through Collapse

当expand和collapse操作的重关联组不相交时，可以交换它们：

```mlir
// 变换前
%collapsed = tensor.collapse_shape %src [[0], [1, 2]]
    : tensor<4x12x4xf32> into tensor<4x48xf32>
%expanded = tensor.expand_shape %collapsed [[0], [1, 2]]
    : tensor<4x48xf32> into tensor<4x6x8xf32>

// 变换后（当collapse的[1,2]和expand的[1,2]不相交时）
%expanded = tensor.expand_shape %src [[0], [1], [2]]
    : tensor<4x12x4xf32> into tensor<4x6x2x4xf32>
%collapsed = tensor.collapse_shape %expanded [[0], [1, 2], [3]]
    : tensor<4x6x2x4xf32> into tensor<4x6x8xf32>
```

**使用场景**:
- 简化reshape-slice链
- 为向量化准备正确的形状
- 优化张量操作序列

---

### 2.2 ExtractSliceFromReshapeUtils

**文件**: `ExtractSliceFromReshapeUtils.cpp`

**作用**: 替换reshape结果的slice为源张量的聚合slice操作

**技术原理**:

#### 核心思想

`extract_slice(collapse_shape(x))` → 在源张量x上执行更复杂的slice操作

#### 索引空间转换

使用 `AffineDelinearizeIndexOp` 进行索引反线性化：

```cpp
// 线性化索引 → 多维索引
%multi_dim = affine.delinearize_index %linear
    into ([%dim0, %dim1, ...]) : index
```

**核心数据结构**:
```cpp
class ExtractSliceFromCollapseHelper {
  // collapse_shape的输入形状
  SmallVector<OpFoldResult> collapseShapeInputShape;

  // collapse_shape的输出形状
  SmallVector<OpFoldResult> collapseShapeOutputShape;

  // extract_slice的参数
  SmallVector<Range> sliceParams;

  // 被线性化的维度标记
  llvm::SmallBitVector linearizedDimensions;
};
```

#### 简化模式

**Rank-Reducing ExtractSlice + CollapseShape**:

```mlir
// 变换前
%collapsed = tensor.collapse_shape %src [[0, 1], [2]]
    : tensor<10x20x30xf32> into tensor<200x30xf32>
%slice = tensor.extract_slice %collapsed[5, 0][100, 30][1, 1]
    : tensor<200x30xf32> to tensor<100x30xf32>

// 变换后（简化为直接在源上的slice）
%slice = tensor.extract_slice %src[2, 10, 0][5, 20, 30][1, 1, 1]
    : tensor<10x20x30xf32> to tensor<5x20x30xf32>
```

**核心函数**:
```cpp
FailureOr<Value> ExtractSliceFromCollapseHelper::create(
    RewriterBase &rewriter,
    tensor::CollapseShapeOp collapseOp,
    tensor::ExtractSliceOp sliceOp) {

  // 1. 验证slice参数与collapse的兼容性
  // 2. 计算在源张量上的等价slice参数
  // 3. 创建新的extract_slice操作
  // 4. 可选：创建后续的collapse_shape
}
```

#### 复杂情况：部分线性化

当slice只覆盖部分线性化维度时，需要更复杂的处理：

```mlir
// 源: 4x5x6 (120 elements)
// collapse [0,1] → 20x6 (120 elements)
// extract [10:20, 0:6] → 10x6 (60 elements)

// 需要计算源上的等价slice
// 源的第10-19行对应行索引2-4（每行5个元素）
// 最终slice: [2:5, 0:6] → 3x6 (18 elements)
```

**使用场景**:
- 消除reshape操作
- 优化内存访问模式
- 为后续变换简化IR

---

## 3. 常量折叠

### 3.1 RewriteAsConstant

**文件**: `RewriteAsConstant.cpp`

**作用**: 将tensor操作重写为常量

**技术原理**:

#### GenerateOp折叠

**模式**: `tensor.generate` yield常量 → `arith.constant`

**前置条件**:
- Tensor类型必须有静态形状
- Yield的值必须是常量

**变换示例**:
```mlir
// 折叠前
%result = tensor.generate %x, %y {
  ^bb0(%arg0: index, %arg1: index):
    %c42 = arith.constant 42 : i32
    tensor.yield %c42 : i32
} : tensor<10x20xi32>

// 折叠后
%result = arith.constant dense<42> : tensor<10x20xi32>
```

#### PadOp折叠

**模式**: 常量input + 常量pad → 新常量属性

**算法流程**:
1. 计算输出形状：`output_size = input_size + low_pad + high_pad`
2. 用pad值完整初始化输出
3. 使用索引空间转换复制原始数据

**索引空间转换函数**:
```cpp
int64_t transformIndexSpace(ArrayRef<int64_t> inputShape,
                            ArrayRef<int64_t> outputStrides,
                            int64_t srcLinearIndex) {
  // 避免中间分配的linearize/delinearize序列
  // 直接从源线性索引计算目标线性索引
  int64_t dstLinearIndex = 0;
  for (int64_t dim = inputShape.size() - 1; dim >= 0; --dim) {
    auto [quotient, remainder] = std::div(srcLinearIndex, inputShape[dim]);
    srcLinearIndex = quotient;
    dstLinearIndex += outputStrides[dim] * remainder;
  }
  return dstLinearIndex;
}
```

**变换示例**:
```mlir
// 折叠前
%input = arith.constant dense<1> : tensor<2x2xi32>
%pad_val = arith.constant 0 : i32
%padded = tensor.pad %input low[1, 1] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %pad_val : i32
} : tensor<2x2xi32> to tensor<4x4xi32>

// 折叠后
%padded = arith.constant dense<
  [[0, 0, 0, 0],
   [0, 1, 1, 0],
   [0, 1, 1, 0],
   [0, 0, 0, 0]]
> : tensor<4x4xi32>
```

**核心实现**:
```cpp
template <typename ElemType, typename AttrType>
Value constantFoldPadOp(PatternRewriter &rewriter, Location loc,
                        DenseElementsAttr input, AttrType padValue,
                        ArrayRef<int64_t> padLow, ArrayRef<int64_t> padHigh) {
  // 1. 计算输出形状
  auto newShape = ...;
  int64_t outputSize = computeProduct(newShape);

  // 2. 用pad值初始化
  SmallVector<ElemType> values(outputSize, padValue.getValue());

  // 3. 计算步长和起始偏移
  SmallVector<int64_t> outputStrides = computeStrides(newShape);
  int64_t startingOffset = linearize(padLow, outputStrides);

  // 4. 复制输入数据
  for (auto [inputIndex, inputValue] : llvm::enumerate(*inputValues)) {
    auto outputIndex = transformIndexSpace(oldShape, outputStrides, inputIndex);
    values[outputIndex + startingOffset] = inputValue;
  }

  // 5. 创建常量属性
  auto newAttr = DenseElementsAttr::get(newType, values);
  return materializeConstant(rewriter, newAttr, newType, loc);
}
```

**使用场景**:
- 编译时求值
- 减少运行时计算
- 常量传播优化

---

### 3.2 EmptyOpPatterns

**文件**: `EmptyOpPatterns.cpp`

**作用**: 折叠与tensor.empty相关的操作

**技术原理**:

#### Reshape折叠

**模式**: `reshape(empty)` → 直接创建正确大小的empty

**变换示例**:
```mlir
// 折叠前
%empty = tensor.empty() : tensor<12xf32>
%reshaped = tensor.collapse_shape %empty [[0]]
    : tensor<12xf32> to tensor<12xf32>

// 折叠后
%reshaped = tensor.empty() : tensor<12xf32>
```

#### ExtractSlice折叠

**模式**: `extract_slice(empty)` → 更小维度的empty

**变换示例**:
```mlir
// 折叠前
%empty = tensor.empty() : tensor<100x100xf32>
%slice = tensor.extract_slice %empty[10, 20][30, 40][1, 1]
    : tensor<100x100xf32> to tensor<30x40xf32>

// 折叠后
%slice = tensor.empty() : tensor<30x40xf32>
```

#### Concat折叠

**模式**: 全empty操作数的concat → 单个empty

**变换示例**:
```mlir
// 折叠前
%empty1 = tensor.empty() : tensor<10x32xf32>
%empty2 = tensor.empty() : tensor<20x32xf32>
%concat = tensor.concat %empty1, %empty2
    {dimension = 0} : (tensor<10x32xf32>, tensor<20x32xf32>) -> tensor<30x32xf32>

// 折叠后
%concat = tensor.empty() : tensor<30x32xf32>
```

**核心实现**:
```cpp
struct FoldEmptyTensorWithReshapeOp<ReshapeOp>
    : public OpRewritePattern<ReshapeOp> {
  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const {
    auto emptyOp = reshapeOp.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) return failure();

    // 重新计算结果形状
    ReifiedRankedShapedTypeDims resultShapes;
    if (failed(reifyResultShapes(rewriter, reshapeOp, resultShapes)))
      return failure();

    // 直接创建正确大小的empty
    auto newEmpty = rewriter.create<tensor::EmptyOp>(
        reshapeOp.getLoc(),
        resultShapes[0],
        reshapeOp.getResult().getType().getElementType());
    rewriter.replaceOp(reshapeOp, newEmpty.getResult());
    return success();
  }
};
```

**使用场景**:
- 消除冗余操作
- 延迟内存分配
- 简化IR

---

## 4. 操作分解

### 4.1 ConcatOpPatterns

**文件**: `ConcatOpPatterns.cpp`

**作用**: 将tensor.concat操作分解为更基础的操作

**技术原理**:

#### 分解算法

将 `tensor.concat` 分解为 `tensor.empty` + 链式 `tensor.insert_slice`

**分解示例**:
```mlir
// 分解前
%0 = tensor.concat %a, %b, %c
    {dimension = 1} : (tensor<10x20xf32>, tensor<10x30xf32>, tensor<10x15xf32>)
    -> tensor<10x65xf32>

// 分解后
%empty = tensor.empty() : tensor<10x65xf32>
%partial1 = tensor.insert_slice %a into %empty[0, 0][10, 20][1, 1]
    : tensor<10x20xf32> into tensor<10x65xf32>
%partial2 = tensor.insert_slice %b into %partial1[0, 20][10, 30][1, 1]
    : tensor<10x30xf32> into tensor<10x65xf32>
%result = tensor.insert_slice %c into %partial2[0, 50][10, 15][1, 1]
    : tensor<10x15xf32> into tensor<10x65xf32>
```

**偏移计算**:
```cpp
SmallVector<OpFoldResult> computeConcatOffsets(
    ArrayRef<OpFoldResult> sizes,
    int64_t concatDim) {

  SmallVector<OpFoldResult> offsets;
  OpFoldResult currentOffset = rewriter.getIndexAttr(0);

  for (Value size : sizes) {
    offsets.push_back(currentOffset);
    // 当前偏移 += 当前维度大小
    currentOffset = makeComposedAffineApply(
        rewriter, loc,
        rewriter.getAffineSymbolExpr(0) + rewriter.getAffineSymbolExpr(1),
        {currentOffset, size});
  }
  return offsets;
}
```

**核心实现**:
```cpp
struct DecomposeTensorConcatOp : public OpRewritePattern<ConcatOp> {
  LogicalResult matchAndRewrite(ConcatOp concatOp,
                                PatternRewriter &rewriter) const {
    // 调用ConcatOp的decomposeOperation方法
    FailureOr<SmallVector<Value>> decomposed =
        concatOp.decomposeOperation(rewriter);

    if (failed(decomposed))
      return failure();

    // 替换原操作
    rewriter.replaceOp(concatOp, (*decomposed)[0]);
    return success();
  }
};
```

**使用场景**:
- 降级到底层操作
- 与bufferization集成
- 代码生成准备

---

## 5. Tiling和融合

### 5.1 SwapExtractSliceWithProducerPatterns

**文件**: `SwapExtractSliceWithProducerPatterns.cpp`

**作用**: 将extract_slice与实现TilingInterface的生产者交换，实现tile+fusion

**技术原理**:

#### Producer Tiling

**模式**: `extract_slice(producer(...))` → `tiled_producer(...)`

**前置条件**:
- 生产者必须实现 `TilingInterface`
- 只支持stride=1的情况

**变换流程**:
```cpp
FailureOr<TilingResult> replaceExtractSliceWithTiledProducer(
    OpBuilder &builder,
    tensor::ExtractSliceOp sliceOp,
    OpResult producer) {

  auto producerOp = dyn_cast<TilingInterface>(producer.getOwner());
  if (!producerOp) return failure();

  // 验证stride为1
  if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
    return failure();

  // 调用producer的tiling接口直接生成切片结果
  FailureOr<TilingResult> tiledResult =
      producerOp.generateResultTileValue(
          builder,
          producer.getResultNumber(),
          sliceOp.getMixedOffsets(),
          sliceOp.getMixedSizes());

  return tiledResult;
}
```

**变换示例**:
```mlir
// 变换前
%matmul = linalg.matmul %A, %B
    : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
%tile = tensor.extract_slice %matmul[32, 32][64, 64][1, 1]
    : tensor<128x128xf32> to tensor<64x64xf32>

// 变换后（直接生成tile）
%tile = linalg.matmul
    ins(%A[32:96, 0:128], %B[0:128, 32:96])
    outs(%init : tensor<64x64xf32>)
    : tensor<64x64xf32>
```

#### Consumer Tiling

**模式**: 多个 `insert_slice` → 单个tiled consumer

**变换示例**:
```mlir
// 变换前（多个insert_slice写入同一操作的不同tile）
%tile0 = tensor.extract_slice %input[0, 0][32, 32][1, 1] : ...
%tile1 = tensor.extract_slice %input[0, 32][32, 32][1, 1] : ...
%tile2 = tensor.extract_slice %input[32, 0][32, 32][1, 1] : ...
%tile3 = tensor.extract_slice %input[32, 32][32, 32][1, 1] : ...

%partial0 = tensor.insert_slice %tile0 into %init[0, 0][32, 32][1, 1]
%partial1 = tensor.insert_slice %tile1 into %partial0[0, 32][32, 32][1, 1]
%partial2 = tensor.insert_slice %tile2 into %partial1[32, 0][32, 32][1, 1]
%result = tensor.insert_slice %tile3 into %partial2[32, 32][32, 32][1, 1]

// 变换后（直接在输出上操作）
%result = consumer.tiled(
    ins(%tile0, %tile1, %tile2, %tile3),
    outs(%init)
)
```

#### Rank-Reducing处理

当extract_slice是rank-reducing时，需要保留额外的slice：

```cpp
llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
if (droppedDims.any()) {
  // 创建rank-reducing slice以保持类型一致
  SmallVector<OpFoldResult> offsets(sourceRank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(sourceRank, rewriter.getIndexAttr(1));

  auto newSliceOp = builder.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(),
      sliceOp.getType(),
      tiledResult->tiledValues[0],
      offsets,
      sliceOp.getMixedSizes(),
      strides);
  tiledResult->tiledValues[0] = newSliceOp;
}
```

**使用场景**:
- 循环Tiling
- Producer-Consumer融合
- 缓存局部性优化

---

## 6. 独立性变换

### 6.1 IndependenceTransforms

**文件**: `IndependenceTransforms.cpp`

**作用**: 使操作独立于某些值，用于循环携带依赖消除

**技术原理**:

#### 独立上界计算

使用 `ValueBoundsConstraintSet::computeIndependentBound()` 计算不依赖于某些值的上界

```cpp
std::optional<int64_t> computeIndependentBound(
    Value value,
    Value independentOf) {
  // 使用约束求解找到不依赖于independentOf的值
  // 返回安全的静态上界
}
```

#### PadOp独立化

**模式**: 创建新pad（使用独立边界）+ extract_slice恢复原始大小

**变换示例**:
```mlir
// 变换前（pad大小依赖于循环变量%i）
%padded = tensor.pad %source low[0] high[%i] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %c0 : f32
} : tensor<100xf32> to tensor<?x100xf32>

// 变换后（使用静态上界）
%padded_large = tensor.pad %source low[0] high[%max_i] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %c0 : f32
} : tensor<100xf32> to tensor<200xf32>

%actual = tensor.extract_slice %padded_large[0, 0][100, 100+%i][1, 1]
    : tensor<200xf32> to tensor<?x100xf32>
```

#### EmptyOp独立化

**模式**: 创建新empty（使用独立大小）+ extract_slice

**变换示例**:
```mlir
// 变换前
%empty = tensor.empty(%dynamic_size) : tensor<?xf32>

// 变换后
%empty_large = tensor.empty(%max_size) : tensor<100xf32>
%empty = tensor.extract_slice %empty_large[0][%dynamic_size][1]
    : tensor<100xf32> to tensor<?xf32>
```

**核心实现**:
```cpp
FailureOr<Value> buildIndependentOp(
    OpBuilder &builder,
    tensor::PadOp padOp,
    ValueRange independenceOperands) {

  // 1. 计算每个动态维度的独立上界
  SmallVector<OpFoldResult> independentBound;
  for (Value size : padOp.getMixedHighPad()) {
    if (auto value = dyn_cast<Value>(size)) {
      auto bound = ValueBoundsConstraintSet::computeIndependentBound(
          builder, padOp.getLoc(), value,
          /*independentOf=*/independenceOperands);
      independentBound.push_back(bound.value());
    } else {
      independentBound.push_back(size);
    }
  }

  // 2. 创建新的pad操作
  auto newPadOp = cast<PadOp>(builder.clone(*padOp.getOperation()));
  // ... 设置独立上界

  // 3. 创建extract_slice恢复原始大小
  auto sliceOp = builder.create<tensor::ExtractSliceOp>(...);

  return sliceOp.getResult();
}
```

**使用场景**:
- 循环并行化
- 软件流水线
- 多缓冲优化

---

## 7. 运行时支持

### 7.1 RuntimeOpVerification

**文件**: `RuntimeOpVerification.cpp`

**作用**: 为tensor操作生成运行时验证代码

**技术原理**:

#### 边界检查生成

**基础检查**: `0 <= index < dim_size`

```cpp
Value generateInBoundsCheck(OpBuilder &builder, Location loc,
                            Value value, Value lb, Value ub) {
  Value inBounds1 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, value, lb);
  Value inBounds2 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, value, ub);
  return builder.create<arith::AndIOp>(loc, inBounds1, inBounds2);
}
```

#### 各操作的验证

**DimOp验证**:
```mlir
// 插入验证
%rank = tensor.rank %tensor : tensor<?x?xf32>
%check0 = arith.cmpi sle, %dim_index, %rank : index
%check1 = arith.cmpi sge, %dim_index, 0 : index
%valid = arith.andi %check0, %check1 : i1
cf.assert %valid, "dim index out of bounds" : i1
%result = tensor.dim %tensor[%dim_index]
```

**ExtractSlice验证**:
```mlir
// 为每个维度生成检查
%dim0 = tensor.dim %source, 0 : tensor<?x?xf32>
%lb0_check = arith.cmpi sge, %offset0, 0 : index
%ub0_check = arith.cmpi sle, (%offset0 + (%size0 - 1) * %stride0), %dim0 : index
%dim0_valid = arith.andi %lb0_check, %ub0_check : i1
// ... 对所有维度重复
cf.assert %all_valid, "slice out of bounds" : i1
```

**CastOp验证**:
```mlir
// 检查rank匹配
%src_rank = tensor.rank %source : tensor<...>
%dst_rank = tensor.rank %result_type : tensor<...>
%rank_match = arith.cmpi eq, %src_rank, %dst_rank : index

// 对于静态形状，检查维度大小
%dim_check = arith.cmpi eq, %actual_dim, %static_dim : index
cf.assert %all_checks, "cast shape mismatch" : i1
```

**接口实现**:
```cpp
struct ExtractSliceOpInterface : RuntimeVerifiableOpInterface::ExternalModel<...> {
  LogicalResult generateVerification(Operation *op, OpBuilder &builder) {
    auto sliceOp = cast<tensor::ExtractSliceOp>(op);

    // 为每个维度生成边界检查
    SmallVector<Value> checks;
    for (auto [offset, size, stride] :
         llvm::zip(sliceOp.getMixedOffsets(),
                   sliceOp.getMixedSizes(),
                   sliceOp.getMixedStrides())) {
      // 生成: 0 <= offset && offset + (size-1)*stride < dim_size
      Value check = generateDimBoundsCheck(builder, offset, size, stride);
      checks.push_back(check);
    }

    // 组合所有检查
    Value finalCheck = combineChecks(builder, checks);
    builder.create<cf::AssertOp>(op->getLoc(), finalCheck,
                                 "extract_slice out of bounds");
    return success();
  }
};
```

**使用场景**:
- 调试
- 动态形状验证
- 安全检查

---

## 8. 接口实现

### 8.1 BufferizableOpInterface

**文件**: `BufferizableOpInterfaceImpl.cpp`

**作用**: 为Tensor操作实现BufferizableOpInterface，支持tensor→memref转换

**技术原理**:

#### 核心接口方法

```cpp
struct BufferizableOpInterface {
  // 分析结果是否缓冲化
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand);
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand);

  // 获取缓冲区类型
  FailureOr<BaseMemRefType> getBufferType(Operation *op, Value value,
                                          const BufferizationOptions &options);

  // 执行缓冲化转换
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options);
};
```

#### 各操作的实现

**ExtractSliceOp**:
```cpp
struct ExtractSliceOpInterface : public BufferizableOpInterface::ExternalModel<...> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, ...) {
    auto sliceOp = cast<tensor::ExtractSliceOp>(op);

    // 转换为memref.subview
    Value buffer = getBuffer(rewriter, sliceOp.getSource());
    auto subviewOp = rewriter.create<memref::SubViewOp>(
        sliceOp.getLoc(),
        buffer,
        sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());

    results.push_back(subviewOp.getResult());
    return success();
  }
};
```

**InsertSliceOp**:
```cpp
LogicalResult bufferize(Operation *op, RewriterBase &rewriter, ...) {
  auto insertSliceOp = cast<tensor::InsertSliceOp>(op);

  // 获取source和dest的缓冲区
  Value sourceBuffer = getBuffer(rewriter, insertSliceOp.getSource());
  Value destBuffer = getBuffer(rewriter, insertSliceOp.getDest());

  // 创建subview + copy
  auto destSubview = rewriter.create<memref::SubViewOp>(
      destBuffer,
      insertSliceOp.getMixedOffsets(),
      insertSliceOp.getMixedSizes(),
      insertSliceOp.getMixedStrides());

  rewriter.create<memref::CopyOp>(
      insertSliceOp.getLoc(),
      sourceBuffer,
      destSubview);

  results.push_back(destBuffer);
  return success();
}
```

**PadOp**:
```cpp
LogicalResult bufferize(Operation *op, RewriterBase &rewriter, ...) {
  auto padOp = cast<tensor::PadOp>(op);

  // 1. 分配结果buffer
  Value tensorAlloc = allocateBuffer(rewriter, padOp.getResult());

  // 2. 使用linalg.map填充pad值
  Value filledBuffer = lowerGenerateLikeOpBody(
      rewriter, padOp.getLoc(), tensorAlloc,
      padOp.getMixedLowPad(), padOp.getMixedHighPad(),
      padOp.getBodyRegion());

  // 3. 插入原始数据
  rewriter.create<tensor::InsertSliceOp>(
      padOp.getLoc(),
      padOp.getSource(),
      filledBuffer,
      padOp.getMixedLowPad(),
      /*sizes=*/padOp.getSourceType().getShape(),
      /*strides=*/padOp.getType().getShape());

  results.push_back(filledBuffer);
  return success();
}
```

**ConcatOp**:
```cpp
LogicalResult bufferize(Operation *op, RewriterBase &rewriter, ...) {
  auto concatOp = cast<tensor::ConcatOp>(op);

  // 1. 分配结果buffer
  Value resultBuffer = allocateBuffer(rewriter, concatOp.getResult());

  // 2. 为每个operand创建subview并复制
  int64_t concatDim = concatOp.getDimension();
  int64_t offset = 0;
  for (Value operand : concatOp.getInputs()) {
    Value operandBuffer = getBuffer(rewriter, operand);

    SmallVector<OpFoldResult> offsets(resultRank, 0);
    offsets[concatDim] = rewriter.getIndexAttr(offset);

    SmallVector<OpFoldResult> sizes = getShape(operand);
    SmallVector<OpFoldResult> strides(resultRank, 1);

    auto subview = rewriter.create<memref::SubViewOp>(
        resultBuffer, offsets, sizes, strides);
    rewriter.create<memref::CopyOp>(concatOp.getLoc(),
                                     operandBuffer, subview);

    offset += getSize(operand, concatDim);
  }

  results.push_back(resultBuffer);
  return success();
}
```

**使用场景**:
- Tensor到Memref的转换
- 内存分配优化
- 与Bufferization pass集成

---

### 8.2 SubsetInsertionOpInterface

**文件**: `SubsetInsertionOpInterfaceImpl.cpp`

**作用**: 为subset操作实现接口，支持bufferization分析

**技术原理**:

#### SubsetOpInterface

提供 `getAccessedHyperrectangularSlice()`，返回访问的超矩形切片

```cpp
struct HyperrectangularSlice {
  Value source;                    // 源tensor/memref
  SmallVector<OpFoldResult> offsets;  // 偏移
  SmallVector<OpFoldResult> sizes;    // 大小
  SmallVector<OpFoldResult> strides;  // 步长
};

struct ExtractSliceOpSubsetOpInterface {
  HyperrectangularSlice getAccessedHyperrectangularSlice(Operation *op);
};
```

#### SubsetExtractionOpInterface

```cpp
struct ExtractSliceOpSubsetExtractionOpInterface {
  OpOperand *getSourceOperand(Operation *op) {
    return &op->getOpOperand(0);  // source operand
  }
};
```

#### SubsetInsertionOpInterface

```cpp
struct InsertSliceOpSubsetInsertionOpInterface {
  OpOperand *getSourceOperand(Operation *op) {
    return &cast<tensor::InsertSliceOp>(op).getSourceMutable();
  }

  OpOperand *getDestinationOperand(Operation *op) {
    return &cast<tensor::InsertSliceOp>(op).getDestMutable();
  }

  // 从目标提取对应的slice
  std::optional<Value> buildSubsetExtraction(
      Operation *op,
      OpBuilder &builder,
      Location loc) {

    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return builder.create<tensor::ExtractSliceOp>(
        loc,
        insertSliceOp.getSourceType(),
        insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
  }

  // 返回构建提取所需的所有值
  SmallVector<Value> getValuesNeededToBuildSubsetExtraction(Operation *op) {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return {insertSliceOp.getDest()};
  }
};
```

**使用场景**:
- Bufferization分析
- 别名分析
- 依赖分析

---

## 总结

MLIR Tensor方言的12个Transform文件涵盖了以下优化领域：

| 类别 | Transform | 主要用途 |
|------|-----------|----------|
| **子集操作** | MergeConsecutiveInsertExtractSlice, FoldTensorSubsetOps | 合并和融合切片操作 |
| **形状变换** | ReshapePatterns, ExtractSliceFromReshapeUtils | 优化reshape与slice交互 |
| **常量折叠** | RewriteAsConstant, EmptyOpPatterns | 编译时求值 |
| **操作分解** | ConcatOpPatterns | 降级到基础操作 |
| **Tiling融合** | SwapExtractSliceWithProducer | Producer-Consumer融合 |
| **独立性** | IndependenceTransforms | 循环依赖消除 |
| **运行时** | RuntimeOpVerification | 边界检查生成 |
| **接口** | BufferizableOpInterface, SubsetInsertionOpInterface | Bufferization支持 |

### 典型优化流水线

```
原始IR
    ↓
[RewriteAsConstant]        // 常量折叠
    ↓
[EmptyOpPatterns]          // Empty优化
    ↓
[MergeConsecutiveInsertExtractSlice]  // 合并子集操作
    ↓
[SwapExtractSliceWithProducer]        // Tiling + 融合
    ↓
[BufferizableOpInterface]  // Tensor → MemRef
    ↓
代码生成
```

### 关键数据结构

```cpp
// 切片参数
struct Range {
  OpFoldResult offset;
  OpFoldResult size;
  OpFoldResult stride;
};

// 超矩形切片
struct HyperrectangularSlice {
  Value source;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

// Tiling结果
struct TilingResult {
  SmallVector<Value> tiledValues;      // Tiled后的值
  SmallVector<Operation *> tiledOps;   // 新生成的操作
};
```

这些变换可以单独使用或组合使用，形成强大的tensor优化能力，是MLIR编译器堆栈中tensor级别优化的核心组件。
