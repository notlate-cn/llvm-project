# MLIR Linalg 方言 Tiling 机制全面分析

## 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [Tiling.cpp 核心函数详解](#tilingcpp-核心函数详解)
4. [Tiling 相关 Transforms](#tiling-相关-transforms)
5. [TilingInterface 接口](#tilinginterface-接口)
6. [高级 Tiling 技术](#高级-tiling-技术)
7. [实践示例](#实践示例)
8. [总结](#总结)

---

## 概述

**Tiling（平铺/分块）** 是一种重要的循环优化技术，通过将大的迭代空间分割成较小的块（tiles）来提高数据局部性、启用并行化，并为融合创造机会。

### 文件位置

```
mlir/lib/Dialect/Linalg/Transforms/
├── Tiling.cpp                    # 核心 Tiling 实现
├── TilingInterfaceImpl.cpp       # TilingInterface 实现
├── SplitReduction.cpp            # 归约维度分割
├── PadTilingInterface.cpp        # Padding 与 Tiling 结合
├── Fusion.cpp                    # 融合优化（与 Tiling 配合）
└── ElementwiseOpFusion.cpp       # 逐元素操作融合
```

### Tiling 的核心思想

```
原始操作:
  for i in 0..N:
    for j in 0..M:
      A[i,j] = ...

平铺后 (tile_size = 64):
  for ii in 0..N step 64:
    for jj in 0..M step 64:
      for i in ii..min(ii+64, N):
        for j in jj..min(jj+64, M):
          A[i,j] = ...
```

---

## 核心概念

### 1. 基本数据结构

#### Range 结构

```cpp
struct Range {
  OpFoldResult offset;  // 起始偏移
  OpFoldResult size;    // 大小
  OpFoldResult stride;  // 步长，通常为 1
};
```

#### TiledLinalgOp 结构

```cpp
struct TiledLinalgOp {
  LinalgOp op;                      // 平铺后的操作
  SmallVector<Operation *, 8> loops;  // 生成的循环嵌套
  SmallVector<Value, 4> tensorResults; // 张量结果
};
```

### 2. 循环类型枚举

```cpp
enum class LinalgTilingLoopType {
  Loops = 0,           // scf::ForOp - 顺序循环
  AffineLoops = 1,     // affine::ForOp - 仿射循环
  ParallelLoops = 2    // scf::ParallelOp - 并行循环
};
```

### 3. Tiling 配置选项

```cpp
struct LinalgTilingOptions {
  // 计算平铺大小的函数
  TileSizeComputationFunction tileSizeComputationFunction;

  // 循环交换向量：重新排序平铺后的循环
  SmallVector<unsigned, 4> interchangeVector;

  // 循环类型
  LinalgTilingLoopType loopType;

  // 分布式执行选项
  std::optional<LinalgLoopDistributionOptions> distribution;

  // 需要剥离的循环
  SmallVector<int64_t> peeledLoops;
};
```

---

## Tiling.cpp 核心函数详解

### 1. makeTiledLoopRanges

**功能**: 创建平铺后的循环范围

```cpp
std::tuple<SmallVector<Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(RewriterBase &b, Location loc, AffineMap map,
                    ArrayRef<OpFoldResult> allShapeSizes,
                    ArrayRef<OpFoldResult> allTileSizes)
```

**实现原理**:

```cpp
// 1. 应用仿射映射获取循环顺序的形状大小
SmallVector<OpFoldResult> shapeSizes =
    makeComposedFoldedMultiResultAffineApply(b, loc, map, allShapeSizes);

// 2. 移除零大小的 tile（不平铺的维度）
LoopIndexToRangeIndexMap loopIndexToRangeIndex;
for (int idx = 0, e = tileSizes.size(), zerosCount = 0; idx < e; ++idx) {
  if (getConstantIntValue(tileSizes[idx - zerosCount]) == 0) {
    shapeSizes.erase(shapeSizes.begin() + idx - zerosCount);
    tileSizes.erase(tileSizes.begin() + idx - zerosCount);
    ++zerosCount;
    continue;
  }
  loopIndexToRangeIndex[idx] = idx - zerosCount;
}

// 3. 创建范围: {offset=0, size=shapeSize, stride=tileSize}
for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx)
  res.push_back(Range{b.getIndexAttr(0), shapeSizes[idx], tileSizes[idx]});
```

**示例**:

```cpp
// 原始: 1024x1024 矩阵，tileSizes = [32, 64]
// 结果:
// loopRanges[0] = {0, 1024, 32}  // 外层循环 i: 0..1024 step 32
// loopRanges[1] = {0, 1024, 64}  // 外层循环 j: 0..1024 step 64
```

---

### 2. transformIndexOps

**功能**: 转换索引操作以适应平铺

```cpp
void transformIndexOps(RewriterBase &b, LinalgOp op,
                      SmallVectorImpl<Value> &ivs,
                      const LoopIndexToRangeIndexMap &loopIndexToRangeIndex)
```

**实现**:

```cpp
// 1. 为所有循环创建 IV 数组（包括未平铺的）
SmallVector<Value> allIvs(op.getNumLoops(), nullptr);
for (auto en : enumerate(allIvs)) {
  auto rangeIndex = loopIndexToRangeIndex.find(en.index());
  if (rangeIndex == loopIndexToRangeIndex.end())
    continue;  // 未平铺的循环保持 nullptr
  en.value() = ivs[rangeIndex->second];
}

// 2. 偏移所有 linalg.index 操作
offsetIndices(b, op, getAsOpFoldResult(allIvs));
```

**作用**:

```cpp
// 平铺前:
linalg.index 0  // 返回原始循环索引 i

// 平铺后（tile_size=32）:
i = iv_outer * 32 + iv_inner
linalg.index 0  // 应返回 i，需要偏移
```

---

### 3. computeStaticContinuousTileSizes

**功能**: 计算连续的静态平铺大小（用于完全覆盖迭代空间）

```cpp
FailureOr<StaticContinuousTileSizeSpecification>
computeStaticContinuousTileSizes(LinalgOp op, unsigned dimension,
                                 unsigned targetSize)
```

**算法**:

```
输入: loopRange = 100, targetSize = 32

步骤 1: 主 tile_size = 32, trip_count = 100 / 32 = 3
       覆盖: 32 * 3 = 96

步骤 2: remainder = 100 % 32 = 4
       下一个 tile_size = 16 (32 中的最大 2 的幂的一半)
       trip_count = 4 / 16 = 0 (跳过)

步骤 3: 下一个 tile_size = 8
       trip_count = 4 / 8 = 0 (跳过)

步骤 4: 下一个 tile_size = 4
       trip_count = 4 / 4 = 1
       覆盖: 96 + 4 * 1 = 100 ✓

结果: tileSizes = [32, 4], tripCounts = [3, 1]
```

**代码**:

```cpp
int64_t tileSize = targetSize;
spec.tileSizes.push_back(tileSize);
spec.tripCounts.push_back(loopRange / tileSize);

int64_t remainderChunk = loopRange % tileSize;

// 对余数部分递归分解
while (tileSize > 1 && remainderChunk != 0) {
  uint64_t maxPower = llvm::bit_floor(tileSize);
  tileSize = maxPower == tileSize ? maxPower >> 1 : maxPower;

  int64_t tripCount = remainderChunk / tileSize;
  if (tripCount > 0) {
    spec.tileSizes.push_back(tileSize);
    spec.tripCounts.push_back(tripCount);
  }

  remainderChunk = remainderChunk % tileSize;
}
```

---

### 4. computeMultiTileSizes

**功能**: 计算多大小平铺（两种 tile size 的组合）

```cpp
FailureOr<MultiSizeSpecification>
computeMultiTileSizes(OpBuilder &builder, LinalgOp op,
                      unsigned dimension,
                      OpFoldResult targetSize,
                      OpFoldResult divisor,
                      bool emitAssertions)
```

**数学推导**:

```
目标: 找到 s 和 s+divisor，使得:
  s * u + (s + divisor) * v = tripCount

计算:
  b = tripCount floordiv divisor
  t = (targetSize + divisor - 1) floordiv divisor
  d = (b + t - 1) floordiv t
  s = (b floordiv d) * divisor
  v = b % d
  u = d - v

其中:
  - s: 低 tile size
  - s + divisor: 高 tile size
  - u: 低 tile 的数量
  - v: 高 tile 的数量
```

**示例**:

```
输入: tripCount = 100, targetSize = 32, divisor = 8

计算:
  b = 100 / 8 = 12
  t = (32 + 8 - 1) / 8 = 4
  d = (12 + 4 - 1) / 4 = 3
  s = (12 / 3) * 8 = 32
  v = 12 % 3 = 0
  u = 3 - 0 = 3

验证: 32 * 3 + (32 + 8) * 0 = 96 ≠ 100 ✗ (无法完全覆盖)

返回: failure()
```

---

### 5. tileLinalgOpImpl

**功能**: 核心平铺实现（模板函数）

```cpp
template <typename LoopTy>
FailureOr<TiledLinalgOp>
tileLinalgOpImpl(RewriterBase &b, LinalgOp op,
                 ArrayRef<OpFoldResult> tileSizes,
                 const LinalgTilingOptions &options)
```

**执行流程**:

```cpp
// ========== 步骤 1: 构建平铺循环范围 ==========
SmallVector<OpFoldResult> allShapeSizes =
    op.createFlatListOfOperandDims(b, op.getLoc());
AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();

auto [loopRanges, loopIndexToRangeIndex] = makeTiledLoopRanges(
    b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);

// ========== 步骤 2: 处理循环交换 ==========
if (!options.interchangeVector.empty()) {
  // 重新计算交换向量（移除零 tile 的维度）
  // 应用逆置换映射
  // 重新排列 loopRanges 和 iteratorTypes
}

// ========== 步骤 3: 处理分布式执行 ==========
if (options.distribution) {
  // 收集并行循环范围
  // 调用 procInfo 回调获取处理器信息
  // 更新分布信息
}

// ========== 步骤 4: 创建平铺循环 ==========
auto tiledLoopBodyBuilder = [&](OpBuilder &builder, Location loc,
                                ValueRange localIvs,
                                ValueRange operandValuesToUse) {
  // a. 应用逆交换到 IVs
  SmallVector<Value, 4> interchangedIvs;
  if (!options.interchangeVector.empty()) {
    for (AffineExpr result : invPermutationMap.getResults())
      interchangedIvs.push_back(
          ivs[cast<AffineDimExpr>(result).getPosition()]);
  }

  // b. 创建平铺形状
  SmallVector<Value> tiledOperands = makeTiledShapes(
      b, loc, op, valuesToTile,
      getAsOpFoldResult(interchangedIvs), tileSizes,
      sizeBounds, /*omitPartialTileCheck=*/false);

  // c. 克隆操作
  SmallVector<Type> resultTensorTypes =
      getTensorOutputTypes(op, tiledOperands);
  res = clone(b, op, resultTensorTypes, tiledOperands);

  // d. 插入切片回
  tensorResults = insertSlicesBack(builder, loc, op, tiledOperands,
                                   res->getResults());
  return scf::ValueVector(tensorResults.begin(), tensorResults.end());
};

GenerateLoopNest<LoopTy>::doit(b, op.getLoc(), loopRanges, op,
                               iteratorTypes, tiledLoopBodyBuilder, procInfo);

// ========== 步骤 5: 转换索引操作 ==========
transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

// ========== 步骤 6: 收集生成的循环 ==========
for (auto iv : ivs) {
  if (isa<BlockArgument>(iv)) {
    loops.push_back(cast<BlockArgument>(iv).getOwner()->getParentOp());
  }
}

return TiledLinalgOp{res, loops, tensorResults};
```

---

### 6. calculateTileOffsetsAndSizes

**功能**: 为 forall 并行循环计算平铺偏移和大小

```cpp
static void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, scf::ForallOp forallOp,
    ArrayRef<OpFoldResult> numThreads, SmallVector<Range> loopRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes)
```

**实现逻辑**:

```cpp
for (unsigned loopIdx = 0, threadIdIdx = 0; loopIdx < nLoops; ++loopIdx) {
  // 退化情况: 无线程分配，取整个域
  if (overflow || isZero) {
    tiledOffsets.push_back(loopRanges[loopIdx].offset);
    tiledSizes.push_back(loopRanges[loopIdx].size);
    continue;
  }

  // 平铺情况: 计算偏移和大小
  Value threadId = threadIds[threadIdIdx];

  // 每线程的固定最大大小
  OpFoldResult tileSizePerThread =
      nominalTileSizes ? (*nominalTileSizes)[loopIdx]
      : makeComposedFoldedAffineApply(
          b, loc, m.ceilDiv(n),
          {size, nonZeroNumThreads[threadIdIdx]});

  // 动态偏移: offset + threadId * tileSizePerThread
  OpFoldResult offsetPerThread = makeComposedFoldedAffineApply(
      b, loc, i + j * m, {offset, threadId, tileSizePerThread});

  // 动态上界（处理边界）
  OpFoldResult residualTileSize = makeComposedFoldedAffineApply(
      b, loc, i + j * m - n,
      {offset, nonZeroNumThreads[threadIdIdx], tileSizePerThread, size});

  // 使用 min 确保不越界
  tileSizePerThread = buildMin(b, loc, {sizeMinusOffsetPerThread,
                                        tileSizePerThread});

  // 可选: 使用 max 确保非负
  if (!omitTileOffsetBoundsCheck)
    tileSizePerThread = buildMax(b, loc, {b.getIndexAttr(0), tileSizePerThread});

  tiledOffsets.push_back(offsetPerThread);
  tiledSizes.push_back(tileSizePerThread);
  ++threadIdIdx;
}
```

---

### 7. tileReductionUsingForall

**功能**: 使用 forall 并行化归约维度

```cpp
FailureOr<ForallReductionTilingResult>
tileReductionUsingForall(RewriterBase &b,
                         PartialReductionOpInterface op,
                         ArrayRef<OpFoldResult> numThreads,
                         ArrayRef<OpFoldResult> tileSizes,
                         std::optional<ArrayAttr> mapping)
```

**完整流程**:

```cpp
// ========== 步骤 1: 验证和准备 ==========
- 检查是否是 LinalgOp
- 检查只有一个归约维度
- 归约维度必须映射到线程

// ========== 步骤 2: 创建部分归约的初始张量 ==========
SmallVector<Value> initTensors =
    op.generateInitialTensorForPartialReduction(b, loc, numThreads,
                                                reductionDims);

// ========== 步骤 3: 创建 forall 循环 ==========
scf::ForallOp forallOp = b.create<scf::ForallOp>(
    loc, materializedNonZeroNumThreads, initTensors, mapping);

// ========== 步骤 4: 计算平铺偏移和大小 ==========
calculateTileOffsetsAndSizes(b, loc, forallOp, numThreads,
                             iterationDomain, ...,
                             tiledOffsets, tiledSizes);

// ========== 步骤 5: 克隆并平铺操作 ==========
{
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(forallOp.getTerminator());

  // a. 提取平铺的目标操作数
  for (Value initOperand : destinationStyleOp.getDpsInits()) {
    tiledDpsInitOperands.push_back(b.create<tensor::ExtractSliceOp>(...));
  }

  // b. 克隆操作并更新 init 操作数
  Operation *clonedOp = b.clone(*op.getOperation());
  b.modifyOpInPlace(clonedOp, [&]() {
    for (auto [initOperandPtr, tiledInitValue] :
         zip_equal(cast<DestinationStyleOpInterface>(clonedOp)
                      .getDpsInitsMutable(),
                  tiledDpsInitOperands)) {
      initOperandPtr.set(tiledInitValue);
    }
  });

  // c. 平铺克隆的操作
  if (tileSizes.empty()) {
    // 使用 TilingInterface
    tilingResult = cast<TilingInterface>(clonedOp)
                       .getTiledImplementation(b, tiledOffsets, tiledSizes);
  } else {
    // 使用 Linalg 专用平铺
    tiledLinalg = tileLinalgOpImpl<scf::ForOp>(b, clonedOp, tileSizes, options);
  }

  b.eraseOp(clonedOp);
}

// ========== 步骤 6: 插入部分归约结果 ==========
for (auto [index, result, bbArg] :
     zip(seq<unsigned>(0, dest.size()), tilingResults, destBbArgs)) {
  // 计算结果位置
  tilingInterfaceOp.getResultTilePosition(
      b, index, tiledOffsets, tiledSizes, resultOffsets, resultSizes);

  // 插入并行切片
  b.create<tensor::ParallelInsertSliceOp>(
      loc, result, bbArg, resultOffsetsRank, resultSizesRank, strides);
}

// ========== 步骤 7: 合并部分归约 ==========
b.setInsertionPointAfter(forallOp);
FailureOr<MergeResult> mergeResult =
    op.mergeReductions(b, loc, forallOp->getResults(), reductionDims);

// ========== 步骤 8: 替换原操作 ==========
b.replaceOp(op, mergeResult->replacements);

return ForallReductionTilingResult{...};
```

**示例**:

```mlir
// 平铺前:
%0 = linalg.generic {
  ^bb0(%arg0: f32, %arg1: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
} ins(%in : tensor<128xf32>)
  outs(%init : tensor<128xf32>)

// 平铺后 (numThreads = [4]):
%partials = scf.forall (%iv) in (4) init (
  %c0, %c0, %c0, %c0  // 4 个部分归约的初始值
) {
  // 每个线程处理 32 个元素
  %tiled = linalg.generic ... : tensor<32xf32>
  tensor.parallel_insert_slice %tiled into ...
}

// 合并部分归约
%result = linalg.generic ins(%partials...) ...
```

---

## Tiling 相关 Transforms

### 1. SplitReduction - 归约维度分割

**文件**: `SplitReduction.cpp`

**功能**: 将归约维度分割为并行和归约两部分

```cpp
FailureOr<SplitReductionResult> splitReduction(
    RewriterBase &b, LinalgOp op,
    const ControlSplitReductionFn &controlSplitReductionFn,
    bool useAlloc)
```

**原理**:

```
原始归约:
  for i in 0..M:
    for k in 0..N:  // 归约维度
      C[i] += A[i,k] * B[k]

分割后 (ratio = 4):
  for i in 0..M:
    for kk in 0..N/4:  // 外层并行
      for k in 0..4:    // 内层归约
        C_partial[i,kk] += A[i,kk*4+k] * B[kk*4+k]

  for i in 0..M:
    for k in 0..N/4:
      C[i] += C_partial[i,k]
```

**好处**:
- 外层循环可以并行化
- 提高数据局部性
- 为向量化创造机会

---

### 2. PadTilingInterface - 填充与平铺结合

**文件**: `PadTilingInterface.cpp`

**功能**: 在平铺前填充张量以避免边界检查

```cpp
FailureOr<Value> padTensorOp(RewriterBase &rewriter,
                             RankedTensorType resultType,
                             Value paddedTensor,
                             Value tensor,
                             const PadTilingInterfaceOptions &options)
```

**computePaddedShape 函数**:

```cpp
SmallVector<OpFoldResult> computePaddedShape(
    RewriterBase &rewriter, TypedValue<RankedTensorType> v,
    AffineMap indexingMap, ArrayRef<OpFoldResult> indexingSizes,
    const PadTilingInterfaceOptions &options)
```

**实现**:

```cpp
// 1. 获取完整秩的填充规格
SmallVector<OpFoldResult> paddingSizes =
    getFullRankPaddingSizes(rewriter, indexingSizes, options);

// 2. 对每个操作数维度，累加贡献项
for (const auto &enResults : enumerate(indexingMap.getResults())) {
  SmallVector<OpFoldResult> terms;
  for (size_t paddingDim = 0; paddingDim != paddingSizes.size();
       ++paddingDim) {
    // 计算该填充维度对当前操作数维度的贡献
    AffineExpr expr = ...;
    OpFoldResult term = affine::makeComposedFoldedAffineApply(
        rewriter, loc, expr, {indexingSizes[paddingDim],
                              paddingSizes[paddingDim]});
    terms.push_back(term);
  }

  // 3. 使用 affine.sum 累加所有贡献
  paddedShape[resultIndex] = affine::makeComposedFoldedAffineMax(
      rewriter, loc, AffineMap::getMultiDimIdentityMap(...), terms);
}

return paddedShape;
```

**示例**:

```mlir
// 原始: 100x100 矩阵，tile_size = 32
// 需要: 3x3 = 9 个 tile，包含边界处理

// 填充后: 128x128 矩阵
// 需要: 4x4 = 16 个 tile，无需边界处理
%0 = tensor.pad %in low[0,0] high[28,28] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
} : tensor<100x100xf32> to tensor<128x128xf32>
```

---

### 3. Fusion - 融合优化

**文件**: `Fusion.cpp`

**功能**: 在平铺后融合生产者-消费者操作

```cpp
FailureOr<FusionInfo> fuseProducerOfTensor(OpBuilder &b,
                                          OpOperand &consumerOpOperand)
```

**与 Tiling 的配合**:

```
1. Tile 消费者操作 -> 生成 extract_slice
2. 检测 extract_slice 的生产者
3. 融合生产者到消费者循环内
4. 只计算需要的 tile
```

**示例**:

```mlir
// Tiling 后:
%1 = linalg.matmul ... // 完整计算
%2 = tensor.extract_slice %1[0, 0][32, 32][1, 1]
%3 = linalg.generic %2 { ... }

// 融合后:
%1_tiled = linalg.matmul ... // 只计算 [0:32][0:32]
%3 = linalg.generic %1_tiled { ... }
```

---

## TilingInterface 接口

### 概述

`TilingInterface` 是 MLIR 中用于统一不同操作平铺行为的接口。

### 核心方法

```cpp
class TilingInterface {
public:
  /// 获取循环迭代器类型
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op);

  /// 获取迭代域范围
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b);

  /// 生成平铺实现
  FailureOr<TilingResult> getTiledImplementation(
      Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes);

  /// 获取结果 tile 位置
  LogicalResult getResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes);
};
```

### LinalgOp 的实现

**文件**: `TilingInterfaceImpl.cpp`

```cpp
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<OpFoldResult> allShapesSizes =
        linalgOp.createFlatListOfOperandDims(b, b.getLoc());
    AffineMap map = linalgOp.getShapesToLoopsMap();

    return llvm::to_vector(llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          b, b.getLoc(), loopExpr, allShapesSizes);
      return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
    }));
  }

  FailureOr<TilingResult> getTiledImplementation(
      Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);

    // 创建平铺形状
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, b.getLoc(), linalgOp, linalgOp->getOperands(),
        offsets, sizes, {}, true);

    // 克隆操作
    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(linalgOp, tiledOperands);
    Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);

    // 偏移索引
    offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

    return TilingResult{{tiledOp},
                        SmallVector<Value>(tiledOp->getResults()),
                        generatedSlices};
  }
};
```

---

## 高级 Tiling 技术

### 1. 循环交换 (Interchange)

**目的**: 重新排序平铺后的循环以优化内存访问模式

```cpp
LinalgTilingOptions options;
options.setInterchange({1, 0});  // 交换最外两层循环
```

**示例**:

```mlir
// 原始循环顺序:
// for ii:
//   for jj:
//     for i:
//       for j:

// 交换后:
// for jj:
//   for ii:
//     for i:
//       for j:
```

### 2. 分布式执行

**配置**:

```cpp
struct LinalgLoopDistributionOptions {
  ProcInfoCallBackFn procInfo;  // 返回 {procId, nprocs} 对
};

struct ProcInfo {
  Value procId;
  Value nprocs;
  DistributionMethod distributionMethod;
};
```

**分布方法**:

```cpp
enum class DistributionMethod {
  Cyclic,                    // 循环分布
  CyclicNumProcsGeNumIters,  // 处理器数 >= 迭代数
  CyclicNumProcsEqNumIters,  // 处理器数 == 迭代数
  None
};
```

### 3. Peel 循环

**目的**: 剥离边界 tile 以简化主循环

```cpp
LinalgTilingOptions options;
options.setPeeledLoops({0, 1});  // 剥离前两层循环
```

---

## 实践示例

### 示例 1: 矩阵乘法平铺

**原始代码**:

```mlir
func.func @matmul(%A: tensor<128x128xf32>,
                  %B: tensor<128x128xf32>,
                  %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
```

**平铺配置**:

```cpp
LinalgTilingOptions options;
options.setTileSizes({32, 32, 8});  // M, N, K 维度
options.setLoopType(LinalgTilingLoopType::Loops);
options.setInterchange({0, 1, 2});
```

**平铺后**:

```mlir
func.func @matmul_tiled(%A: tensor<128x128xf32>,
                        %B: tensor<128x128xf32>,
                        %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // 外层循环 (tile 循环)
  %result = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) -> tensor<128x128xf32> {
    %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) -> tensor<128x128xf32> {
      %result3 = scf.for %kk = 0 to 128 step 8 iter_args(%arg6 = %arg5) -> tensor<128x128xf32> {

        // 提取 tile
        %A_tile = tensor.extract_slice %A[%ii, %kk][32, 8][1, 1]
        %B_tile = tensor.extract_slice %B[%kk, %jj][8, 32][1, 1]
        %C_tile = tensor.extract_slice %arg6[%ii, %jj][32, 32][1, 1]

        // 内层 matmul (tile 大小)
        %C_updated = linalg.matmul ins(%A_tile, %B_tile)
                                   outs(%C_tile) -> tensor<32x32xf32>

        // 插入回结果
        %result4 = tensor.insert_slice %C_updated into %arg6[%ii, %jj][32, 32][1, 1]
        scf.yield %result4 : tensor<128x128xf32>
      }
      scf.yield %result3 : tensor<128x128xf32>
    }
    scf.yield %result2 : tensor<128x128xf32>
  }
  return %result : tensor<128x128xf32>
}
```

---

### 示例 2: 并行归约平铺

**原始归约**:

```mlir
%0 = linalg.generic {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // 输入
    affine_map<(i) -> (i)>   // 输出
  ]
  iterator_types = ["reduction"]
  ins(%in: tensor<1024xf32>)
  outs(%init: tensor<1xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
} -> tensor<1xf32>
```

**使用 tileReductionUsingForall**:

```cpp
// 配置 4 个线程
SmallVector<OpFoldResult> numThreads = {b.getIndexAttr(4)};

// 执行平铺
FailureOr<ForallReductionTilingResult> result =
    tileReductionUsingForall(b, genericOp, numThreads,
                            /*tileSizes=*/{}, /*mapping=*/std::nullopt);
```

**平铺后**:

```mlir
// 1. 初始化部分归约结果
%init0 = arith.constant 0.0 : f32
%init1 = arith.constant 0.0 : f32
%init2 = arith.constant 0.0 : f32
%init3 = arith.constant 0.0 : f32

// 2. 并行执行部分归约
%partials:4 = scf.forall (%threadId) in (4) init (%init0, %init1, %init2, %init3) {
  // 计算该线程的范围
  %offset = arith.muli %threadId, %256 : index
  %size = arith.minsi %256, %1024_minus_offset

  // 提取输入切片
  %in_slice = tensor.extract_slice %in[%offset][%size][1]

  // 部分归约
  %partial = linalg.generic ... ins(%in_slice) outs(%init_for_thread) ...

  // 并行插入
  tensor.parallel_insert_slice %partial into ...
}

// 3. 合并部分归约
%final = linalg.generic ins(%partials#0, %partials#1, %partials#2, %partials#3)
                        outs(%final_init) {
  ^bb0(%a: f32, %b: f32, %c: f32, %d: f32, %acc: f32):
    %s1 = arith.addf %a, %b : f32
    %s2 = arith.addf %s1, %c : f32
    %s3 = arith.addf %s2, %d : f32
    linalg.yield %s3 : f32
}
```

---

### 示例 3: 多大小平铺

```cpp
// 计算多大小平铺
FailureOr<MultiSizeSpecification> spec =
    computeMultiTileSizes(b, matmulOp, 0,  // M 维度
                         b.getIndexAttr(64),  // targetSize
                         b.getIndexAttr(16),  // divisor
                         true);

// spec->lowTileSize = 48
// spec->highTileSize = 64
// spec->lowTripCount = 2  // 使用 48x2
// spec->highTripCount = 1 // 使用 64x1

// 验证: 48*2 + 64*1 = 160
```

---

## 总结

### Tiling 优化策略决策树

```
问题: 如何选择合适的 Tiling 策略?

1. 数据局部性?
   YES -> 使用标准 Tiling
   NO -> 考虑其他优化

2. 需要并行化?
   YES ->
     - 并行维度: 使用 scf.parallel 或 scf.forall
     - 归约维度: 使用 tileReductionUsingForall

3. 边界处理开销大?
   YES -> 使用 PadTilingInterface 预填充

4. 需要完全覆盖迭代空间?
   YES ->
     - 使用 computeContinuousTileSizes (多 tile size)
     - 或使用 computeMultiTileSizes (两种 tile size)

5. 需要融合生产者?
   YES -> 先 Tiling，再使用 fuseProducerOfTensor

6. 归约是性能瓶颈?
   YES -> 使用 splitReduction 分割归约维度
```

### Tiling 文件关系图

```
Tiling.cpp (核心实现)
├── makeTiledLoopRanges        # 创建循环范围
├── tileLinalgOpImpl           # 平铺实现
├── tileReductionUsingForall   # 归约平铺
├── computeMultiTileSizes      # 多大小计算
└── computeContinuousTileSizes # 连续大小计算

TilingInterfaceImpl.cpp
├── getIterationDomain         # 迭代域
├── getTiledImplementation    # 平铺实现
└── getResultTilePosition     # 结果位置

SplitReduction.cpp
└── splitReduction             # 分割归约

PadTilingInterface.cpp
└── computePaddedShape         # 计算填充形状

Fusion.cpp
└── fuseProducerOfTensor       # 融合生产者

ElementwiseOpFusion.cpp
├── areElementwiseOpsFusable   # 可融合性检查
└── fuseElementwiseOps         # 逐元素融合
```

### 性能优化建议

1. **Tile Size 选择**:
   - 通常 32-64 适合 CPU 缓存行
   - 考虑硬件 SIMD 宽度
   - GPU 上考虑 warp/wavefront 大小

2. **循环顺序**:
   - 使用 interchange 优化内存访问
   - 将最内层循环向量化

3. **边界处理**:
   - 小数组: 使用 peel 循环
   - 大数组: 使用 padding 避免边界检查

4. **并行化**:
   - 识别并行维度
   - 使用 forall 或 parallel 循环
   - 归约使用部分归约策略

5. **融合时机**:
   - Tiling 后立即融合
   - 避免融合破坏数据局部性
