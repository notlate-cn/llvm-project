# MLIR Mesh方言Transform完全指南

本文档详细介绍MLIR Mesh方言中的所有Transform变换，包括其作用、技术原理和实例演示。

## 目录

1. [概述](#1-概述)
2. [ShardingPropagation](#2-shardingpropagation)
3. [Simplifications](#3-simplifications)
4. [Spmdization](#4-spmdization)
5. [Transforms](#5-transforms)
6. [使用场景和最佳实践](#6-使用场景和最佳实践)

---

## 1. 概述

MLIR Mesh方言用于表示和转换分布式机器学习计算，特别是针对大规模张量操作的并行化。该方言支持在多设备mesh拓扑上自动分片和执行张量操作。

### Mesh方言核心概念

**Mesh（网格）**: 表示计算设备的拓扑结构，通常是一个多维数组

```mlir
mesh.mesh @mesh_2d(rank = 2, size = 4x8)  // 4x8 = 32个设备的2D网格
```

**Sharding（分片）**: 描述张量如何在mesh上分布

```mlir
// 将张量的第0维沿mesh轴0分片，第1维沿mesh轴1分片
#mesh.shard<@mesh_2d, [[0], [1]]>
```

**SPMD（单程序多数据）**: 所有设备执行相同的程序，但操作不同的数据片段

### Transform文件概览

| 文件 | 功能 | 关键操作 |
|------|------|----------|
| `ShardingPropagation.cpp` | 分片属性传播 | 自动推导最优数据布局 |
| `Simplifications.cpp` | 简化优化 | 常量折叠、All-Reduce消除 |
| `Spmdization.cpp` | SPMD代码生成 | 分布式执行代码生成 |
| `Transforms.cpp` | 操作降级 | 高级操作到底层转换 |

---

## 2. ShardingPropagation

**文件**: `ShardingPropagation.cpp`

**作用**: 自动传播分片(sharding)属性，为所有张量操作推导最优的数据布局

### 2.1 核心原理

分片传播通过分析数据依赖关系，自动为张量添加`mesh.shard`注解，最小化设备间通信。

#### 重分片需求级别

```cpp
enum class ReshardingRequirementKind {
  NO_RESHARDING = 0,                              // 无需重分片（最优）
  NO_RESHARDING_FOR_EXPLICIT_ANNOTATIONS,         // 非显式注解无需重分片
  RESHARDING_FOR_EXPLICIT_ANNOTATIONS             // 需要重分片（最差）
};
```

#### 优先级规则

1. **最高优先级**: 必须满足的分片约束(`mustShardings`)
2. **中等优先级**: 可选的分片建议(`optionalShardings`)
3. **最低优先级**: 完全复制(`replicated`)

### 2.2 核心算法

#### 生成所有可能的分片组合

```cpp
// 使用DFS生成所有可能的分片属性组合
static SmallVector<std::vector<MeshSharding>>
getOrderedPossibleShardingAttrs(
    ArrayRef<MeshSharding> mustShardings,
    ArrayRef<MeshSharding> optionalShardings) {

  // mustShardings = [shard0, None]
  // optionalShardings = [None, shard1]
  // 结果: [[shard0, shard1], [shard0, None]]

  std::function<void(size_t)> dfsCreateShardingAttrs = [&](size_t i) {
    if (i == mustShardings.size()) {
      allShardingAttrs.push_back(curShardingAttrs);
      return;
    }

    if (mustShardings[i]) {
      // 必须使用指定的分片
      curShardingAttrs.push_back(mustShardings[i]);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
    } else if (optionalShardings[i]) {
      // 尝试可选分片或无分片
      curShardingAttrs.push_back(optionalShardings[i]);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();

      curShardingAttrs.push_back({});
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
    } else {
      // 无分片约束
      curShardingAttrs.push_back({});
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
    }
  };

  dfsCreateShardingAttrs(0);
  return allShardingAttrs;
}
```

#### 计算重分片需求

```cpp
ReshardingRequirementKind getReshardingRequirementKind(
    Operation *op,
    const std::vector<MeshSharding> &operandAndResultShardings) {

  ReshardingRequirementKind res = NO_RESHARDING;

  // 检查操作数的分片兼容性
  for (auto [operand, sharding] :
       llvm::zip_equal(op->getOperands(), operandShardings)) {
    ShardOp shardOp = dyn_cast<ShardOp>(operand.getDefiningOp());
    if (!shardOp) continue;

    bool needsResharding = (sharding != shardOp.getSharding());
    bool isExplicitAnnotationForThisOp = shardOp.getAnnotateForUsers();

    if (needsResharding) {
      if (isExplicitAnnotationForThisOp) {
        // 显式注解不匹配，需要重分片
        return RESHARDING_FOR_EXPLICIT_ANNOTATIONS;
      }
      res = NO_RESHARDING_FOR_EXPLICIT_ANNOTATIONS;
    }
  }

  // 检查结果的分片兼容性
  // ...类似逻辑...

  return res;
}
```

#### 选择最优分片选项

```cpp
ShardingOption selectShardingOption(
    Operation *op,
    ArrayRef<ShardingOption> options) {

  // 按优先级排序
  // 1. NO_RESHARDING
  // 2. NO_RESHARDING_FOR_EXPLICIT_ANNOTATIONS
  // 3. RESHARDING_FOR_EXPLICIT_ANNOTATIONS

  auto bestOption = llvm::min_element(options,
    [](const ShardingOption &a, const ShardingOption &b) {
      return a.reshardingRequirementKind < b.reshardingRequirementKind;
    });

  return *bestOption;
}
```

### 2.3 遍历策略

```cpp
enum PropagationOrder {
  Backward,        // 反向传播（从结果到操作数）
  Forward,         // 正向传播（从操作数到结果）
  ForwardBackward, // 先正向后反向
  BackwardForward  // 先反向后正向
};
```

### 2.4 实例演示

**示例: 矩阵乘法的自动分片**

```mlir
// 原始代码（无分片注解）
func.func @matmul(%A: tensor<128x256xf32>,
                  %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %C = linalg.matmul %A, %B
      : (tensor<128x256xf32>, tensor<256x512xf32>)
      -> tensor<128x512xf32>
  return %C : tensor<128x512xf32>
}

// 定义2D mesh
mesh.mesh @mesh_2d(rank = 2, size = 4x8)

// ShardingPropagation Pass后
// 自动添加分片注解
func.func @matmul(%A: tensor<128x256xf32>,
                  %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  // A按mesh轴0分片，B按mesh轴1分片
  %A_sharded = mesh.shard %A
      <@mesh_2d, [[0], []]>  // 第0维分片，第1维复制
      : tensor<128x256xf32>

  %B_sharded = mesh.shard %B
      <@mesh_2d, [[] , [1]]>  // 第0维复制，第1维分片
      : tensor<256x512xf32>

  %C_sharded = linalg.matmul %A_sharded, %B_sharded
      : (tensor<128x256xf32>, tensor<256x512xf32>)
      -> tensor<128x512xf32>

  // 输出按两个轴都分片
  %C_annotated = mesh.shard %C_sharded annotate_for_users
      <@mesh_2d, [[0], [1]]>
      : tensor<128x512xf32>

  return %C_annotated : tensor<128x512xf32>
}
```

### 2.5 使用场景

- **自动并行化**: 无需手动指定数据布局
- **通信优化**: 最小化all-reduce和all-gather操作
- **分布式训练**: 自动处理模型并行的数据分布

---

## 3. Simplifications

**文件**: `Simplifications.cpp`

**作用**: 简化和折叠Mesh方言的操作，优化IR

### 3.1 All-Reduce同态简化

当算术操作后跟All-Reduce时，如果该操作与All-Reduce的归约类型兼容，可以消除All-Reduce。

#### 支持的操作映射

```cpp
// 加法 + Sum All-Reduce → 消除All-Reduce
template <typename OpTy>
struct AllReduceHomomorphic : OpRewritePattern<OpTy> {
  // 检查操作结果用户是否为AllReduce(Sum)
  // 如果是，消除All-Reduce
};

// 支持的映射关系:
// arith.AddFOp, arith.AddIOp          → ReductionKind::Sum
// arith.MinimumFOp, MinSIOp, MinUIOp  → ReductionKind::Min
// arith.MaximumFOp, MaxSIOp, MaxUIOp  → ReductionKind::Max
```

#### 示例

```mlir
// 简化前
%sum = mesh.all_reduce %partial
      <@mesh, [0], sum> : tensor<128xf32>
%result = arith.addf %sum, %bias : tensor<128xf32>

// 简化后（消除All-Reduce，将加法作用到partial上）
// 注: 这需要确保加法的另一个操作数在所有设备上相同
%result = arith.addf %partial, %bias : tensor<128xf32>
```

### 3.2 Mesh Shape折叠

```cpp
struct MeshShapeFolder : OpRewritePattern<MeshShapeOp> {
  LogicalResult matchAndRewrite(MeshShapeOp op,
                                PatternRewriter &rewriter) const {
    MeshOp mesh = symbolTableCollection.lookupNearestSymbolFrom<MeshOp>(op, op.getMesh());
    if (!mesh) return failure();

    SmallVector<Value> foldedDims;
    for (auto [dim, size] : llvm::zip(op.getResults(), mesh.getShape())) {
      if (size == ShapedType::kDynamic) {
        // 保留动态维度
        foldedDims.push_back(dim);
      }
      // 静态维度被折叠，不再需要查询
    }

    if (foldedDims.size() == op.getNumResults()) {
      return failure();  // 没有可折叠的
    }

    // 创建新的查询操作，只保留动态维度
    auto newOp = rewriter.create<MeshShapeOp>(op.getLoc(),
                                              mesh.getSymName(),
                                              op.getAxes());
    // 更新使用...
  }
};
```

#### 示例

```mlir
// 折叠前
mesh.mesh @mesh_2d(rank = 2, size = 4x8)
%shape = mesh.mesh_shape @mesh_2d[0, 1]  // [%c4, %c8]
%dim0 = tensor.extract %shape[0] : tensor<2xindex>
%dim1 = tensor.extract %shape[1] : tensor<2xindex>

// 折叠后（静态形状直接使用常量）
%dim0 = arith.constant 4 : index
%dim1 = arith.constant 8 : index

// 如果mesh有动态维度
mesh.mesh @mesh_hybrid(rank = 2, size = 4x?)
%shape = mesh.mesh_shape @mesh_hybrid[0, 1]  // [%c4, %dynamic]
// 折叠后只查询动态维度
%dim0 = arith.constant 4 : index
%dynamic_dim = mesh.mesh_shape @mesh_hybrid[1]
```

### 3.3 使用场景

- **编译时优化**: 消除冗余的集体通信操作
- **常量传播**: 将静态mesh信息内联到代码中
- **性能提升**: 减少运行时开销

---

## 4. Spmdization

**文件**: `Spmdization.cpp`

**作用**: 将单程序代码转换为SPMD（单程序多数据）形式的分布式执行代码

### 4.1 核心概念

**SPMD化**: 将每个操作转换为在分片数据上执行的版本

**Resharding**: 当操作需要不同数据布局时，插入通信操作

### 4.2 Partial Axes处理

处理部分归约(partial reduction)的All-Reduce操作。

```cpp
// sourceSharding = <@mesh, [[0]], partial = sum[0]>
// targetSharding = <@mesh, [[]]>
// 需要对mesh轴0执行All-Reduce(Sum)

static std::tuple<TypedValue<ShapedType>, MeshSharding>
handlePartialAxesDuringResharding(
    OpBuilder &builder,
    MeshSharding sourceSharding,
    MeshSharding targetSharding,
    TypedValue<ShapedType> sourceShard) {

  // 找出需要All-Reduce的轴
  llvm::SmallVector<MeshAxis> allReduceMeshAxes;
  for (auto axis : sourceSharding.getPartialAxes()) {
    if (!targetSharding.getPartialAxes().contains(axis)) {
      allReduceMeshAxes.push_back(axis);
    }
  }

  if (allReduceMeshAxes.empty()) {
    return {sourceShard, sourceSharding};
  }

  // 插入All-Reduce操作
  TypedValue<ShapedType> resultValue = builder.create<AllReduceOp>(
      sourceShard.getLoc(),
      sourceShard.getType(),
      sourceSharding.getMeshAttr().getLeafReference(),
      allReduceMeshAxes,
      sourceShard,
      sourceSharding.getPartialType()).getResult();

  // 更新分片属性
  MeshSharding resultSharding = MeshSharding::get(
      sourceSharding.getMeshAttr(),
      sourceSharding.getSplitAxes(),
      remainingPartialAxes,
      sourceSharding.getPartialType());

  return {resultValue, resultSharding};
}
```

#### 示例

```mlir
// SPMD化前
%partial = "some_op"() : () -> tensor<128xf32>
// %partial有partial sum分片，需要在后续使用前All-Reduce
%result = "consumer"(%partial) : (tensor<128xf32>) -> tensor<128xf32>

// SPMD化后
%partial = "some_op"() : () -> tensor<128xf32>
// 插入All-Reduce
%full = mesh.all_reduce %partial
      <@mesh, [0], sum> : tensor<128xf32>
%result = "consumer"(%full) : (tensor<128xf32>) -> tensor<128xf32>
```

### 4.3 Resharding操作

#### 检测轴变换模式

**1. Split Last Axis**: `[[0,1]] → [[0,1,2]]`

```cpp
// 检测: [[0, 1]] -> [[0, 1, 2]]
static std::optional<std::tuple<int64_t, MeshAxis>>
detectSplitLastAxisInResharding(
    MeshSharding sourceSharding,
    MeshSharding targetSharding) {

  for (size_t tensorAxis = 0; tensorAxis < targetSharding.getSplitAxes().size();
       ++tensorAxis) {

    // 检查是否在末尾增加了一个mesh轴
    if (sourceSharding.getSplitAxes().size() > tensorAxis) {
      if (sourceSharding.getSplitAxes()[tensorAxis].size() + 1 ==
          targetSharding.getSplitAxes()[tensorAxis].size()) {
        // 验证前面的轴相同
        if (llvm::equal(sourceSharding.getSplitAxes()[tensorAxis],
                        targetSharding.getSplitAxes()[tensorAxis].drop_back())) {
          return {tensorAxis, targetSharding.getSplitAxes()[tensorAxis].back()};
        }
      }
    }
  }
  return std::nullopt;
}
```

**实现**: 使用`mesh.all_slice`操作

```cpp
// [[0, 1]] -> [[0, 1, 2]]
// 将复制的张量沿新轴分片

static std::tuple<TypedValue<ShapedType>, MeshSharding>
splitLastAxisInResharding(
    ImplicitLocOpBuilder &builder,
    MeshSharding sourceSharding,
    TypedValue<ShapedType> sourceShard,
    MeshOp mesh,
    int64_t splitTensorAxis,
    MeshAxis splitMeshAxis) {

  // 创建all_slice操作
  TypedValue<ShapedType> targetShard = builder.create<AllSliceOp>(
      sourceShard, mesh,
      ArrayRef<MeshAxis>(splitMeshAxis),
      splitTensorAxis).getResult();

  // 更新分片属性
  MeshSharding targetSharding = targetShardingInSplitLastAxis(
      builder.getContext(), sourceSharding, splitTensorAxis, splitMeshAxis);

  return {targetShard, targetSharding};
}
```

**2. Unsplit Last Axis**: `[[0,1,2]] → [[0,1]]`

```cpp
// 检测: [[0, 1, 2]] -> [[0, 1]]
static std::optional<std::tuple<int64_t, MeshAxis>>
detectUnsplitLastAxisInResharding(
    MeshSharding sourceSharding,
    MeshSharding targetSharding) {

  for (size_t tensorAxis = 0; tensorAxis < sourceSharding.getSplitAxes().size();
       ++tensorAxis) {

    // 检查是否在末尾移除了一个mesh轴
    if (targetSharding.getSplitAxes().size() > tensorAxis) {
      if (sourceSharding.getSplitAxes()[tensorAxis].size() ==
          targetSharding.getSplitAxes()[tensorAxis].size() + 1) {
        if (llvm::equal(sourceSharding.getSplitAxes()[tensorAxis].drop_back(),
                        targetSharding.getSplitAxes()[tensorAxis])) {
          return {tensorAxis, sourceSharding.getSplitAxes()[tensorAxis].back()};
        }
      }
    }
  }
  return std::nullopt;
}
```

**实现**: 使用`mesh.all_gather`操作

```cpp
// [[0, 1, 2]] -> [[0, 1]]
// 收集分片的数据

static std::tuple<TypedValue<ShapedType>, MeshSharding>
unsplitLastAxisInResharding(
    ImplicitLocOpBuilder &builder,
    MeshSharding sourceSharding,
    TypedValue<ShapedType> sourceShard,
    MeshOp mesh,
    int64_t unsplitTensorAxis,
    MeshAxis unsplitMeshAxis) {

  // 创建all_gather操作
  TypedValue<ShapedType> targetShard = builder.create<AllGatherOp>(
      sourceShard, mesh,
      ArrayRef<MeshAxis>(unsplitMeshAxis),
      unsplitTensorAxis).getResult();

  // 更新分片属性
  MeshSharding targetSharding = targetShardingInUnsplitLastAxis(
      builder.getContext(), sourceSharding, unsplitTensorAxis);

  return {targetShard, targetSharding};
}
```

**3. Move Split Axis**: `[[0,1],[2]] → [[0],[1,2]]`

```cpp
// 检测: [[0],[1,2]] -> [[0,1],[2]]
// 将分片从一个张量轴移到另一个轴
// 实现: all_slice + all_gather的组合
```

### 4.4 通用Resharding

```cpp
FailureOr<std::tuple<Value, MeshSharding>> reshard(
    ImplicitLocOpBuilder &builder,
    Value value,
    MeshSharding sourceSharding,
    MeshSharding targetSharding) {

  // 1. 尝试特殊模式优化
  if (auto result = trySplitLastAxisInResharding(...))
    return result;

  if (auto result = tryUnsplitLastAxisInResharding(...))
    return result;

  // 2. 回退到通用方法
  // 先handlePartialAxes
  auto [afterPartial, partialSharding] =
      handlePartialAxesDuringResharding(builder, sourceSharding,
                                       targetSharding, value);

  // 然后对每个轴进行reshape
  // 这里只支持1D mesh
  return reshardOn1DMesh(builder, afterPartial,
                         partialSharding, targetSharding);
}
```

### 4.5 SPMD化核心流程

```cpp
LogicalResult spmdizeFuncOp(
    FunctionLikeOpInterface funcOp,
    SymbolTableCollection &symbolTableCollection) {

  // 1. 推断分块参数类型
  SmallVector<Type> shardedArgumentTypes;
  for (auto [arg, shardAttr] :
       llvm::zip_equal(funcOp.getArguments(), shardings)) {
    Type shardedType = getShardedType(arg.getType(), shardAttr);
    shardedArgumentTypes.push_back(shardedType);
  }

  // 2. 更新函数签名
  funcOp.setType(FunctionType::get(..., shardedArgumentTypes, ...));

  // 3. 转换函数体
  for (Block &block : funcOp.getBlocks()) {
    if (failed(spmdizeBlock(block, ...))) {
      return failure();
    }
  }

  return success();
}

LogicalResult spmdizeBlock(
    Block &block,
    IRMapping &mapping,
    SpmdizationState &state) {

  for (Operation &op : block) {
    // 1. Remap操作数
    SmallVector<Value> remappedOperands;
    for (Value operand : op.getOperands()) {
      remappedOperands.push_back(mapping.lookupOrDefault(operand));
    }

    // 2. SPMD化操作
    if (failed(spmdizeOperation(&op, remappedOperands, ...))) {
      return failure();
    }
  }

  return success();
}

LogicalResult spmdizeOperation(
    Operation *op,
    ValueRange remappedOperands,
    IRMapping &mapping,
    SpmdizationState &state) {

  // 1. 获取操作的ShardingInterface
  auto shardingInterface = dyn_cast<ShardingInterface>(op);
  if (!shardingInterface) {
    // 不支持分片的操作，确保所有输入是复制的
    return ensureAllReplicated(op, remappedOperands, ...);
  }

  // 2. 获取分片属性
  SmallVector<MeshSharding> shardings =
      shardingInterface.getShardingAttr(remappedOperands, op, state);

  // 3. Reshard操作数到所需的布局
  SmallVector<Value> reshardedOperands;
  for (auto [operand, srcSharding, tgtSharding] :
       llvm::zip(remappedOperands, sourceShardings, targetShardings)) {
    auto [resharded, _] = reshard(builder, operand,
                                   srcSharding, tgtSharding);
    reshardedOperands.push_back(resharded);
  }

  // 4. 克隆操作
  IRMapping operandMapping;
  for (auto [old, new] : llvm::zip(op.getOperands(), reshardedOperands)) {
    operandMapping.map(old, new);
  }

  Operation *spmdizedOp = op.clone(operandMapping);

  // 5. 映射结果
  for (auto [old, new] : llvm::zip(op.getResults(), spmdizedOp->getResults())) {
    mapping.map(old, new);
  }

  return success();
}
```

### 4.6 实例演示

**示例: 矩阵乘法的SPMD化**

```mlir
// 原始代码（单设备）
func.func @matmul(%A: tensor<128x256xf32>,
                  %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %C = linalg.matmul ins(%A, %B: tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%init: tensor<128x512xf32>) -> tensor<128x512xf32>
  return %C : tensor<128x512xf32>
}

// 添加分片注解
func.func @matmul_sharded(%A: tensor<128x256xf32>,
                          %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %A_sharded = mesh.shard %A <@mesh, [[0], []]> : tensor<128x256xf32>
  %B_sharded = mesh.shard %B <@mesh, [[] , [1]]> : tensor<256x512xf32>
  %C_sharded = linalg.matmul ins(%A_sharded, %B_sharded) ...
  %C_annotated = mesh.shard %C_sharded <@mesh, [[0], [1]]> annotate_for_users
  return %C_annotated : tensor<128x512xf32>
}

// SPMD化后（每个设备执行的代码）
func.func @matmul_spmd(%A_local: tensor<32x256xf32>,  // 128/4=32
                       %B_local: tensor<256x64xf32>)  // 512/8=64
    -> tensor<32x64xf32> {

  // 本地矩阵乘法
  %C_local = linalg.matmul
      ins(%A_local, %B_local: tensor<32x256xf32>, tensor<256x64xf32>)
      outs(%init_local: tensor<32x64xf32>)
      -> tensor<32x64xf32>

  return %C_local : tensor<32x64xf32>
}

// 在mesh轴0上4个设备，mesh轴1上8个设备，总共32个设备
// 每个设备执行相同的代码，但操作不同的数据切片
```

### 4.7 使用场景

- **分布式训练**: 自动生成数据并行的训练代码
- **模型并行**: 处理无法放入单设备内存的大模型
- **流水线并行**: 生成层间并行的执行代码

---

## 5. Transforms

**文件**: `Transforms.cpp`

**作用**: 将高级Mesh操作降级为底层可执行形式

### 5.1 ProcessMultiIndexOp降级

将`mesh.process_multi_index`降级为使用`mesh.process_linear_index`和`mesh.mesh_shape`的实现。

```cpp
struct ProcessMultiIndexOpLowering : OpRewritePattern<ProcessMultiIndexOp> {
  LogicalResult matchAndRewrite(ProcessMultiIndexOp op,
                                PatternRewriter &rewriter) const {
    MeshOp mesh = getMesh(op, symbolTableCollection);
    if (!mesh) return failure();

    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());

    // 1. 获取线性索引
    Value linearIndex = builder.create<ProcessLinearIndexOp>(mesh);

    // 2. 获取mesh形状
    ValueRange meshShape = builder.create<MeshShapeOp>(mesh).getResults();

    // 3. 反线性化得到多维索引
    SmallVector<Value> completeMultiIndex =
        builder.create<affine::AffineDelinearizeIndexOp>(linearIndex, meshShape)
            .getMultiIndex();

    // 4. 提取需要的轴
    SmallVector<Value> multiIndex;
    ArrayRef<MeshAxis> opMeshAxes = op.getAxes();

    if (opMeshAxes.empty()) {
      // 返回所有轴
      multiIndex = completeMultiIndex;
    } else {
      // 返回指定轴
      for (MeshAxis axis : opMeshAxes) {
        multiIndex.push_back(completeMultiIndex[axis]);
      }
    }

    rewriter.replaceAllUsesWith(op.getResults(), multiIndex);
    return success();
  }
};
```

#### 示例

```mlir
// 降级前
mesh.mesh @mesh_2d(rank = 2, size = 4x8)
%i, %j = mesh.process_multi_index @mesh_2d axes([0, 1])
    : () -> (index, index)

// 降级后
%linear = mesh.process_linear_index @mesh_2d : () -> index
%shape = mesh.mesh_shape @mesh_2d axes([0, 1]) : () -> (index, index)
// %shape = [4, 8]
%i_full, %j_full = affine.delinearize_index %linear into [%shape#0, %shape#1]
    : () -> (index, index)
%i = %i_full  // 直接使用，因为请求的轴是[0,1]
%j = %j_full
```

### 5.2 AllSliceOp降级

将`mesh.all_slice`降级为`tensor.extract_slice`操作。

```cpp
struct AllSliceOpLowering : OpRewritePattern<AllSliceOp> {
  LogicalResult matchAndRewrite(AllSliceOp op,
                                PatternRewriter &rewriter) const {
    MeshOp mesh = getMesh(op, symbolTableCollection);
    if (!mesh) return failure();

    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());

    // 1. 获取进程组内的线性索引
    Operation::result_range processInGroupMultiIndex =
        builder.create<ProcessMultiIndexOp>(mesh.getSymName(), op.getMeshAxes())
            .getResults();

    // 2. 获取进程组大小
    Value processGroupSize =
        createCollectiveProcessGroupSize(mesh, op.getMeshAxes(), builder);

    // 3. 验证张量轴尺寸可被进程组大小整除
    int64_t sliceAxis = op.getSliceAxis().getSExtValue();
    Value operandSliceAxisSize =
        builder.create<tensor::DimOp>(op.getOperand(), sliceAxis);
    Value mod = builder.create<arith::RemUIOp>(
        operandSliceAxisSize, processGroupSize);
    Value isDivisible = builder.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, mod,
        builder.create<arith::ConstantOp>(builder.getIndexAttr(0)));

    builder.create<cf::AssertOp>(isDivisible,
        "Slicing axis size not divisible by process group size");

    // 4. 计算每个进程的切片大小
    Value resultSliceAxisSize =
        builder.create<arith::DivUIOp>(operandSliceAxisSize, processGroupSize);

    // 5. 计算本进程的切片偏移
    OpFoldResult processInGroupLinearIndex = affine::linearizeIndex(
        llvm::to_vector_of<OpFoldResult>(processInGroupMultiIndex),
        // 进程组形状
        builder.create<MeshShapeOp>(mesh.getSymName(), op.getMeshAxes()).getResults(),
        builder);

    OpFoldResult offset = ArithBuilder(builder, builder.getLoc())
        .mul(processInGroupLinearIndex, resultSliceAxisSize);

    // 6. 创建extract_slice操作
    SmallVector<OpFoldResult> offsets(op.getType().getRank(),
                                       builder.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(op.getType().getRank(),
                                       builder.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;

    for (int64_t i = 0; i < op.getType().getRank(); ++i) {
      if (i == sliceAxis) {
        offsets[i] = offset;
        sizes.push_back(resultSliceAxisSize);
      } else {
        Value dimSize = builder.create<tensor::DimOp>(op.getOperand(), i);
        sizes.push_back(dimSize);
      }
    }

    Value slice = builder.create<tensor::ExtractSliceOp>(
        op.getOperand(), offsets, sizes, strides);

    // 7. 可能的类型转换
    Value newResult = builder.create<tensor::CastOp>(
        op.getResult().getType(), slice);

    rewriter.replaceAllUsesWith(op.getResult(), newResult);
    return success();
  }
};
```

#### 示例

```mlir
// 降级前
mesh.mesh @mesh_2d(rank = 2, size = 4x8)
%tensor = tensor.empty() : tensor<128x256xf32>
%sliced = mesh.all_slice %tensor
      mesh(@mesh_2d) axes([0, 1]) axis(1)
      : tensor<128x256xf32> to tensor<128x32xf32>
// 在8个进程上沿轴1分片，每个进程得到256/8=32大小的切片

// 降级后
// 进程(0,0): offset = 0*32 = 0
// 进程(0,1): offset = 1*32 = 32
// 进程(0,2): offset = 2*32 = 64
// ...

%i, %j = mesh.process_multi_index @mesh_2d axes([0, 1])
%linear_idx = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%i, %j)
%slice_size = arith.divui 256, 8 : index  // 32
%offset = arith.muli %linear_idx, %slice_size : index
%sliced = tensor.extract_slice %tensor[0, %offset][128, 32][1, 1]
    : tensor<128x256xf32> to tensor<128x32xf32>
```

### 5.3 辅助函数

```cpp
// 计算进程组大小（各轴大小的乘积）
TypedValue<IndexType> createCollectiveProcessGroupSize(
    MeshOp mesh,
    ArrayRef<MeshAxis> axes,
    ImplicitLocOpBuilder &builder) {

  Operation::result_range meshShape =
      builder.create<MeshShapeOp>(mesh, axes).getResults();

  return cast<TypedValue<IndexType>>(arith::createProduct(
      builder, builder.getLoc(),
      llvm::to_vector_of<Value>(meshShape),
      builder.getIndexType()));
}

// 从多维索引创建线性索引
TypedValue<IndexType> createProcessLinearIndex(
    StringRef mesh,
    ValueRange processInGroupMultiIndex,
    ArrayRef<MeshAxis> meshAxes,
    ImplicitLocOpBuilder &builder) {

  Operation::result_range processGroupShape =
      builder.create<MeshShapeOp>(mesh, meshAxes).getResult();

  OpFoldResult processInGroupLinearIndex = affine::linearizeIndex(
      llvm::to_vector_of<OpFoldResult>(processInGroupMultiIndex),
      llvm::to_vector_of<OpFoldResult>(processGroupShape),
      builder);

  auto res = dyn_cast<Value>(processInGroupLinearIndex);
  if (!res) {
    res = builder.create<arith::ConstantIndexOp>(
        cast<IntegerAttr>(cast<Attribute>(processInGroupLinearIndex)).getInt());
  }

  return cast<TypedValue<IndexType>>(res);
}
```

### 5.4 使用场景

- **代码生成**: 为不支持高级mesh操作的runtime生成代码
- **调试**: 理解分布式操作的底层实现
- **优化**: 为特定硬件定制实现

---

## 6. 使用场景和最佳实践

### 6.1 典型编译流程

```
原始MLIR代码
    ↓
[ShardingPropagation]     // 自动推导分片
    ↓
[Simplifications]         // 优化和折叠
    ↓
[Spmdization]             // 生成SPMD代码
    ↓
[Transforms]              // 降级到底层操作
    ↓
目标后端代码生成
```

### 6.2 实际应用示例

#### 示例1: 分布式矩阵乘法

```mlir
// 1. 定义mesh
mesh.mesh @mesh_2d(rank = 2, size = 4x8)

// 2. 原始函数
func.func @distributed_matmul(%A: tensor<1024x2048xf32>,
                               %B: tensor<2048x4096xf32>)
    -> tensor<1024x4096xf32> {
  %C = linalg.matmul ins(%A, %B) outs(%init: tensor<1024x4096xf32>)
  return %C : tensor<1024x4096xf32>
}

// 3. 添加分片提示（可选）
func.func @distributed_matmul_sharded(%A: tensor<1024x2048xf32>,
                                      %B: tensor<2048x4096xf32>)
    -> tensor<1024x4096xf32> {
  %A_shard = mesh.shard %A <@mesh_2d, [[0], []]> : tensor<1024x2048xf32>
  %B_shard = mesh.shard %B <@mesh_2d, [[] , [1]]> : tensor<2048x4096xf32>

  %C = linalg.matmul ins(%A_shard, %B_shard)
      outs(%init: tensor<1024x4096xf32>)

  %C_shard = mesh.shard %C <@mesh_2d, [[0], [1]]> annotate_for_users
  return %C_shard : tensor<1024x4096xf32>
}

// 4. 运行ShardingPropagation
// 自动推导最优分片（与手动添加的相同）

// 5. 运行Spmdization
// 每个设备执行:
// - 设备(i,j)处理: A[i*256:(i+1)*256, :] × B[:, j*512:(j+1)*512]
// - 输出: C[i*256:(i+1)*256, j*512:(j+1)*512]
```

#### 示例2: 数据并行训练

```mlir
mesh.mesh @mesh_1d(rank = 1, size = 8)  // 8个设备

func.func @data_parallel_train(%weights: tensor<784x128xf32>,
                               %inputs: tensor<128x784xf32>,
                               %gradients: tensor<128x128xf32>)
    -> tensor<784x128xf32> {

  // 权重在所有设备上复制
  %w_shard = mesh.shard %weights <@mesh_1d, [[]]>

  // 输入按batch维度分片
  %x_shard = mesh.shard %inputs <@mesh_1d, [[0]]>  // 128/8=16 per device

  // 前向传播
  %output = "forward"(%w_shard, %x_shard) : (tensor<784x128xf32>, tensor<16x784xf32>)

  // 计算梯度（部分结果）
  %grad_partial = "backward"(%output, %x_shard)
      : (tensor<16x128xf32>, tensor<16x784xf32>) -> tensor<784x128xf32>

  // All-Reduce聚合梯度
  %grad_full = mesh.all_reduce %grad_partial
      <@mesh_1d, [0], sum> : tensor<784x128xf32>

  // 更新权重
  %new_weights = "update"(%w_shard, %grad_full) : tensor<784x128xf32>

  return %new_weights : tensor<784x128xf32>
}
```

### 6.3 最佳实践

1. **优先使用自动传播**: 让ShardingPropagation自动推导分片
2. **显式注解关键张量**: 只在必要时手动添加分片注解
3. **理解通信成本**: All-Reduce和All-Gather是昂贵的操作
4. **选择合适的mesh拓扑**: 2D mesh适合矩阵操作，1D mesh适合数据并行
5. **验证静态形状**: 确保张量维度可被mesh大小整除

### 6.4 性能优化技巧

**1. 减少Resharding**

```mlir
// 避免频繁的分片切换
// 不好:
%A = shard <[[0]]>
%B = shard <[[], [1]]>
%C = op(%A, %B)  // 需要Resharding

// 好:
%A = shard <[[0]]>
%B = shard <[[0]]>
%C = op(%A, %B)  // 无需Resharding
```

**2. 利用Partial Reduction**

```mlir
// 延迟All-Reduce到最后
// 不好:
%partial = "compute"(%input)
%full = all_reduce %partial
%result = "process"(%full)

// 好:
%partial = "compute"(%input)  // 保持partial状态
%processed = "process"(%partial)  // 如果支持partial输入
%full = all_reduce %processed  // 最后才All-Reduce
```

**3. 选择合适的Mesh维度**

```mlir
// 对于2D张量操作
// 矩阵乘法: (M,K) × (K,N)
// M沿mesh轴0分片，N沿mesh轴1分片
mesh.mesh @mesh(rank = 2, size = [M_mesh, N_mesh])

// 对于卷积
// batch沿mesh轴0，channel沿mesh轴1
mesh.mesh @mesh(rank = 2, size = [batch_mesh, channel_mesh])
```

---

## 总结

MLIR Mesh方言提供了一套完整的分布式机器学习编译框架：

| Transform | 功能 | 输入 | 输出 |
|-----------|------|------|------|
| **ShardingPropagation** | 分片传播 | 单设备代码 | 带分片注解的代码 |
| **Simplifications** | 简化优化 | 带分片的代码 | 优化后的代码 |
| **Spmdization** | SPMD化 | 带分片的代码 | 多设备SPMD代码 |
| **Transforms** | 操作降级 | 高级操作 | 底层可执行代码 |

### 关键数据结构

```cpp
// 分片属性
struct MeshSharding {
  MeshAttr mesh;                  // Mesh引用
  SmallVector<MeshAxesAttr> splitAxes;  // 各张量轴的mesh分片
  SmallVector<MeshAxis> partialAxes;    // Partial归约轴
  ReductionKind partialType;            // 归约类型
};

// SPMD化状态
struct SpmdizationState {
  DenseMap<Value, MeshSharding> shardingMap;  // 值到分片的映射
  SymbolTableCollection symbolTableCollection;
};

// 重分片结果
struct ReshardingResult {
  Value reshardedValue;
  MeshSharding newSharding;
};
```

这套框架使得MLIR能够自动将单设备机器学习程序编译为高效的分布式执行代码，是现代分布式深度学习编译器的核心基础设施。
