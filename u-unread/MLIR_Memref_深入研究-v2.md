# MLIR MemRef方言深入研究

## 一、MemRef方言概述

### 设计目的和核心概念

MemRef（Memory Reference）方言是MLIR中用于表示内存引用的核心方言。它提供了一种抽象的方式来描述和操作多维内存缓冲区，而不依赖于具体的内存分配方式或硬件细节。

**核心概念**：

1. **MemRef类型**：表示内存引用的抽象类型，包含以下关键属性：
   - `element type`：元素类型（如f32, i32等）
   - `shape`：形状信息（静态维度和动态维度）
   - `layout map`：布局映射（通常用仿射表示）
   - `address space`：地址空间
   - `memory space`：内存空间

2. **布局映射**：使用仿射映射描述内存布局，支持：
   - 恒等布局：`affine_map<(d0, d1) -> (d0, d1)>`
   - 压缩布局：`affine_map<(i, j) -> (i * 8 + j)>`
   - 分块布局：`affine_map<(i) -> (i floordiv 4, i mod 4)>`

3. **内存空间**：支持不同的内存空间（如GPU的共享内存、全局内存等）

### 与MLIR整体架构的关系

MemRef方言在MLIR架构中处于核心位置：
1. **中间表示层**：提供了高级的内存抽象，位于具体硬件实现之上
2. **平台无关性**：不依赖于特定的内存分配器（如malloc/alloca）
3. **可优化性**：通过布局映射和变换能力，支持编译时优化
4. **多方言协同**：与Affine、Arith、Vector、SCF、Func等方言紧密协作

### 与Affine方言的关系

MemRef与Affine方言的协作是MLIR内存管理的核心：
1. **地址计算**：Affine方言计算访问索引，MemRef提供被访问的内存缓冲区
2. **布局映射**：MemRef类型包含仿射布局映射，Affine操作使用这些映射进行地址转换
3. **优化协同**：共享循环变换和分析能力，联合进行内存访问模式分析和并行化

## 二、核心Operations

### 操作分类

#### 1. 内存分配操作
- `memref.alloc`：堆内存分配
- `memref.alloca`：栈内存分配
- `memref.realloc`：重新分配内存
- `memref.dealloc`：释放内存

#### 2. 内存访问操作
- `memref.load`：从缓冲区加载数据
- `memref.store`：向缓冲区存储数据

#### 3. 内存视图操作
- `memref.subview`：创建子视图（rank-reducing）
- `memref.reinterpret_cast`：重新解释内存布局
- `memref.cast`：类型转换
- `memref.reshape`：改变形状（不复制数据）
- `memref.expand_shape`：扩展维度
- `memref.collapse_shape`：合并维度
- `memref.transpose`：转置视图

#### 4. 元数据查询操作
- `memref.dim`：查询维度大小
- `memref.rank`：查询张量秩

#### 5. 元数据提取操作
- `memref.extract_strided_metadata`：提取步幅和偏移量元数据

#### 6. 内存空间操作
- `memref.memory_space_cast`：内存空间转换

#### 7. DMA操作（异构系统）
- `memref.dma_start`：开始DMA传输
- `memref.dma_wait`：等待DMA完成

#### 8. 全局变量
- `memref.global`：声明全局变量
- `memref.get_global`：获取全局变量引用

#### 9. 原子操作
- `memref.atomic_rmw`：原子读-改-写操作
- `memref.generic_atomic_rmw`：通用的原子RMW操作

#### 10. 复制操作
- `memref.copy`：内存复制

#### 11. 假设操作
- `memref.assume_alignment`：假设对齐
- `memref.memory_space_cast`：内存空间转换

#### 12. 其他操作
- `memref.alloc_time`：获取分配时间
- `memref.prefetch`：数据预取

## 三、Pass详解（重点）

### 3.1 ExpandOpsPass - 操作扩展Pass

**功能**：将高层次的MemRef操作转换为更基础的操作。

**主要转换**：
- `memref.reshape` → `memref.reinterpret_cast`（当形状静态时）

**实现细节**（位于`ExpandOps.cpp`）：
```cpp
// 转换reshape为reinterpret_cast
struct MemRefReshapeOpConverter : public OpRewritePattern<memref::ReshapeOp> {
  LogicalResult matchAndRewrite(memref::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    // 计算sizes和strides
    // 使用affine表达式计算stride
    // 创建reinterpret_cast操作
  }
};
```

**优化效果**：
- 将动态形状计算转换为显式的大小和步幅计算
- 为后续优化提供更清晰的IR表示

### 3.2 NormalizeMemRefsPass - 内存规范化Pass

**功能**：将所有具有非平凡布局映射的MemRef转换为恒等布局。

**核心算法**（位于`NormalizeMemRefs.cpp`）：

1. **可规范化性分析**：
```cpp
bool areMemRefsNormalizable(func::FuncOp funcOp) {
  // 检查函数中所有MemRef类型是否可规范化
  // 只有load/store/dealloc/call/return等操作的use才能规范化
}
```

2. **函数签名更新**：
```cpp
void updateFunctionSignature(func::FuncOp funcOp, ModuleOp moduleOp) {
  // 更新函数参数和返回类型的MemRef布局
  // 需要同时更新所有调用点
}
```

3. **操作结果规范化**：
```cpp
Operation *createOpResultsNormalized(func::FuncOp funcOp, Operation *oldOp) {
  // 为操作的MemRef结果创建恒等布局版本
}
```

**转换示例**：
```mlir
// 转换前
#map = affine_map<(i) -> (i floordiv 4, i mod 4)>
%alloc = memref.alloc() : memref<16xf32, #map>

// 转换后
%alloc = memref.alloc() : memref<4x4xf32>
%flat = affine.apply affine_map<(i, j) -> (i * 4 + j)> (%i, %j)
```

**优化效果**：
- 简化后续分析（恒等布局更容易分析）
- 为向量化、并行化等优化铺平道路

### 3.3 FoldMemRefAliasOpsPass - 别名操作折叠Pass

**功能**：将对子视图的加载/存储折叠为对原始MemRef的加载/存储。

**核心模式**（位于`FoldMemRefAliasOps.cpp`）：

1. **ExpandShape折叠**：
```cpp
// %0 = memref<12x42xf32>
// %1 = memref.expand_shape %0 [[0, 1], [2]]
// load %1[%i1, %i2, %i3]
// =>
// load %0[6 * i1 + i2, %i3]
```

2. **CollapseShape折叠**：
```cpp
// %0 = memref<2x6x42xf32>
// %1 = memref.collapse_shape %0 [[0, 1], [2]]
// load %1[%i1, %i2]
// =>
// load %0[%i1 / 6, %i1 % 6, %i2]
```

3. **SubView折叠**：
```cpp
// 将subview的访问转换为对源memref的直接访问
// 需要计算组合的偏移量和步幅
```

**算法流程**：
```cpp
static LogicalResult resolveSourceIndicesExpandShape(
    Location loc, PatternRewriter &rewriter,
    memref::ExpandShapeOp expandShapeOp, ValueRange indices,
    SmallVectorImpl<Value> &sourceIndices, bool startsInbounds) {
  // 遍历reassociation groups
  // 对每个group计算线性化索引
  // 使用affine.linearize_index op
}
```

**优化效果**：
- 减少间接访问层级
- 提高内存访问效率
- 为后续优化提供更清晰的访问模式

### 3.4 FlattenMemRefsPass - 内存扁平化Pass

**功能**：将多维MemRef操作转换为一维MemRef操作。

**核心算法**（位于`FlattenMemRefs.cpp`）：

1. **线性化计算**：
```cpp
static std::pair<Value, Value> getFlattenMemrefAndOffset(
    OpBuilder &rewriter, Location loc, Value source, ValueRange indices) {
  // 提取步幅元数据
  memref::ExtractStridedMetadataOp stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);

  // 计算线性化索引
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(...);

  // 创建一维reinterpret_cast
  return std::make_pair(
      rewriter.create<memref::ReinterpretCastOp>(...),
      getValueFromOpFoldResult(rewriter, loc, linearizedIndices));
}
```

2. **操作重写**：
- Load/Store操作：添加线性化索引
- SubView操作：计算新的offset/size/stride
- Copy操作：更新源和目标

**转换示例**：
```mlir
// 转换前
%0 = memref.alloc() : memref<4x8xf32>
%1 = memref.load %0[%i, %j] : memref<4x8xf32>

// 转换后
%0 = memref.alloc() : memref<32xf32>
%idx = arith.muli %i, %c8 : index
%linear_idx = arith.addi %idx, %j : index
%1 = memref.load %0[%linear_idx] : memref<32xf32>
```

**优化效果**：
- 简化地址计算
- 提高缓存利用率
- 为SIMD向量化铺路

### 3.5 MultiBufferPass - 多缓冲优化Pass

**功能**：通过数组扩展消除循环迭代之间的临时分配依赖。

**核心算法**（位于`MultiBuffer.cpp`）：

1. **候选识别**：
```cpp
// 查找在循环内分配的MemRef
// 检查是否有完整的写覆盖（overrideBuffer）
// 检查是否可以使用多缓冲
```

2. **缓冲区扩展**：
```cpp
FailureOr<memref::AllocOp> multiBuffer(
    RewriterBase &rewriter, memref::AllocOp allocOp,
    unsigned multiBufferingFactor, bool skipOverrideAnalysis) {

  // 1. 获取原始分配大小
  SmallVector<OpFoldResult> originalSizes = allocOp.getMixedSizes();

  // 2. 创建新的分配（多倍大小）
  // 在新维度上扩展
  SmallVector<OpFoldResult> newSizes = originalSizes;
  newSizes.insert(newSizes.begin(),
                  rewriter.getIndexAttr(multiBufferingFactor));

  // 3. 创建新分配并包装在subview中
  // 返回原始大小的subview
}
```

3. **索引更新**：
```cpp
// 更新所有使用点，添加循环归纳变量作为索引
// %new_alloc = memref.alloc(%factor, %size) : memref<?xsize>
// %subview = memref.subview %new_alloc[%iv, 0] [%c1, %size] [1, 1]
```

**转换示例**：
```mlir
// 转换前
affine.for %i = 0 to 100 {
  %temp = memref.alloc() : memref<128xf32>
  // 使用%temp...
  memref.dealloc %temp : memref<128xf32>
}

// 转换后（factor=2）
%temp = memref.alloc() : memref<2x128xf32>
affine.for %i = 0 to 100 {
  %idx = arith.remsi %i, %c2 : index
  %subview = memref.subview %temp[%idx, 0] [1, 128] [1, 1]
  // 使用%subview...
}
memref.dealloc %temp : memref<2x128xf32>
```

**优化效果**：
- 减少内存分配/释放开销
- 消除迭代间依赖，提高并行性
- 软件流水化的基础

### 3.6 ExpandStridedMetadataPass - 步幅元数据扩展Pass

**功能**：将修改MemRef元数据的操作展开为显式的元数据计算序列。

**核心数据结构**（位于`ExpandStridedMetadata.cpp`）：
```cpp
struct StridedMetadata {
  Value basePtr;
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};
```

**主要转换**：

1. **SubView元数据解析**：
```cpp
// 从 subview(memref, subOffset, subSizes, subStrides) 计算：
// baseBuffer, baseOffset, baseSizes, baseStrides = extract_strided_metadata(memref)
// strides#i = baseStrides#i * subStrides#i
// offset = baseOffset + sum(subOffset#i * baseStrides#i)
// sizes = subSizes
```

2. **Explicit计算生成**：
- 使用Affine表达式显式计算偏移和步幅
- 生成`extract_strided_metadata`操作

**优化效果**：
- 使元数据计算显式化
- 为后续优化提供更多分析信息
- 简化后端代码生成

### 3.7 ComposeSubView - 子视图组合Pass

**功能**：将嵌套的SubView操作组合为单个SubView。

**核心算法**（位于`ComposeSubView.cpp`）：

1. **模式匹配**：
```cpp
struct ComposeSubViewOpPattern : public OpRewritePattern<memref::SubViewOp> {
  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    // 检查源是否是SubView
    auto sourceOp = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!sourceOp) return failure();
  }
};
```

2. **组合计算**：
```cpp
// 步幅：strides[i] = sourceStrides[i] * opStrides[i]
// 偏移：offset[i] = sourceOffset[i] + opOffset[i] * sourceStrides[i]
// 大小：取最终的大小（最小）
```

**转换示例**：
```mlir
// 转换前
%0 = memref.subview %base[10, 20] [5, 5] [1, 1] : ...
%1 = memref.subview %0[2, 3] [2, 2] [1, 1] : ...

// 转换后
%1 = memref.subview %base[12, 23] [2, 2] [1, 1] : ...
```

**优化效果**：
- 减少中间SubView操作
- 简化访问路径
- 提高分析效率

### 3.8 ResolveShapedTypeResultDimsPass - 形状维度解析Pass

**功能**：通过`InferShapedTypeOpInterface`解析`memref.dim`操作。

**核心算法**（位于`ResolveShapedTypeResultDims.cpp`）：

1. **Dim操作折叠**：
```cpp
template <typename OpTy>
struct DimOfShapedTypeOpInterface : public OpRewritePattern<OpTy> {
  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    // 获取操作实现的InferShapedTypeOpInterface
    auto shapedTypeOp = dyn_cast<InferShapedTypeOpInterface>(...);

    // Reify返回类型形状
    SmallVector<Value> reifiedResultShapes;
    shapedTypeOp.reifyReturnTypeShapes(..., reifiedResultShapes);

    // 从形状中提取维度
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(dimOp, resultShape, index);
  }
};
```

**优化效果**：
- 消除运行时dim查询
- 使形状信息在编译时可用
- 提高类型推断能力

### 3.9 AllocationOpInterfaceImpl - 分配操作接口

**功能**：为MemRef分配操作提供`AllocationOpInterface`实现。

**核心实现**（位于`AllocationOpInterfaceImpl.cpp`）：

```cpp
struct DefaultAllocationInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAllocationInterface, memref::AllocOp> {
  // 构建dealloc操作
  static std::optional<Operation *> buildDealloc(OpBuilder &builder, Value alloc) {
    return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc).getOperation();
  }

  // 构建clone操作
  static std::optional<Value> buildClone(OpBuilder &builder, Value alloc) {
    return builder.create<bufferization::CloneOp>(alloc.getLoc(), alloc).getResult();
  }

  // 获取提升类型
  static HoistingKind getHoistingKind() {
    return HoistingKind::Loop | HoistingKind::Block;
  }

  // 构建提升后的alloc
  static std::optional<Operation *> buildPromotedAlloc(OpBuilder &builder, Value alloc) {
    return builder.create<memref::AllocaOp>(...);
  }
};
```

**优化效果**：
- 支持分配提升优化
- 支持内存到寄存器提升
- 与bufferization框架集成

### 3.10 ExtractAddressComputations - 地址计算提取Pass

**功能**：将有偏移量的加载/存储重写为对Subview的加载/存储。

**核心模式**（位于`ExtractAddressComputations.cpp`）：

```mlir
// 转换前
%val = memref.load %base[%off0, %off1, %i, %j]

// 转换后
%subview = memref.subview %base[%off0, %off1] [1, 1] [1, 1]
%val = memref.load %subview[0, 0, %i, %j]
```

**支持的Op**：
- `memref.LoadOp`
- `memref.StoreOp`
- `nvgpu.LdMatrixOp`
- `vector.LoadOp`
- `vector.TransferReadOp`
- `vector.TransferWriteOp`

**优化效果**：
- 分离地址计算和数据访问
- 提高地址计算重用机会
- 为向量化优化提供基础

### 3.11 IndependenceTransforms - 独立性变换Pass

**功能**：使操作独立于特定的依赖值。

**核心算法**（位于`IndependenceTransforms.cpp`）：

```cpp
static FailureOr<OpFoldResult> makeIndependent(
    OpBuilder &b, Location loc, OpFoldResult ofr, ValueRange independencies) {
  // 使用ValueBoundsConstraintSet计算独立边界
  AffineMap boundMap;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeIndependentBound(
          boundMap, mapOperands, presburger::BoundType::UB,
          ofr, independencies, /*closedUB=*/true)))
    return failure();

  // 物化计算出的边界
  return affine::materializeComputedBound(b, loc, boundMap, mapOperands);
}

// 应用示例：使Alloca大小独立于循环变量
FailureOr<Value> buildIndependentOp(OpBuilder &b, memref::AllocaOp allocaOp,
                                   ValueRange independencies) {
  // 计算独立的上界大小
  // 创建新的Alloca
  // 包装在SubView中
}
```

**优化效果**：
- 允许分配提升到循环外
- 减少分配次数
- 提高并行性

### 3.12 其他Pass

#### ExpandReallocPass
- 功能：展开realloc操作
- 转换：`realloc` → `alloc + copy + dealloc`

#### ReifyResultShapesPass
- 功能：为形状操作实现形状反演
- 用途：使运行时形状信息可用于编译时分析

#### RuntimeOpVerificationPass
- 功能：添加运行时验证操作
- 用途：调试和验证IR正确性

#### EmulateNarrowTypePass
- 功能：模拟窄类型操作
- 转换：将不支持窄类型的目标代码转换为宽类型模拟

#### EmulateWideIntPass
- 功能：模拟宽整数操作
- 转换：将超宽整数分解为多个机器字

#### BufferViewFlowOpInterfaceImpl
- 功能：实现缓冲区视图流分析接口
- 用途：别名分析和优化

## 四、Pass依赖关系

### 典型的Pass流水线

```
1. ExpandStridedMetadataPass
   ↓ (使元数据显式化)
2. ExpandOpsPass
   ↓ (展开高层次操作)
3. ComposeSubView
   ↓ (组合视图操作)
4. NormalizeMemRefsPass
   ↓ (规范化布局)
5. FoldMemRefAliasOpsPass
   ↓ (折叠别名操作)
6. FlattenMemRefsPass
   ↓ (扁平化内存)
7. MultiBufferPass
   ↓ (多缓冲优化)
8. ExtractAddressComputations
   ↓ (提取地址计算)
9. (后续向量化、并行化等优化)
```

### Pass选择建议

| 优化目标 | 推荐Pass |
|---------|---------|
| 简化内存布局 | NormalizeMemRefsPass |
| 减少间接访问 | FoldMemRefAliasOpsPass |
| 提高缓存效率 | FlattenMemRefsPass |
| 循环并行化 | MultiBufferPass |
| 向量化准备 | ExtractAddressComputations + FlattenMemRefsPass |
| 分配优化 | AllocationOpInterface + IndependenceTransforms |

## 五、接口和Trait

### 关键接口

1. **InferShapedTypeOpInterface**
   - 推断形状类型操作的结果形状
   - 用于动态形状的编译时推断

2. **AllocationOpInterface**
   - 内存分配操作的统一接口
   - 支持分配提升、克隆等优化

3. **OffsetSizeAndStrideOpInterface**
   - 描述偏移-大小-步长模式的接口
   - 用于Subview等操作

4. **ViewLikeOpInterface**
   - 视图操作的通用接口
   - 支持视图操作的统一处理

5. **BufferViewFlowOpInterface**
   - 缓冲区视图流分析
   - 用于别名分析

### 关键Trait

1. **MemRefsNormalizable**
   - 标记可规范化的操作
   - NormalizeMemRefsPass使用

2. **SameOperandsAndResultShape**
   - 操作数和结果形状相同
   - 用于形状推断

3. **OperandsAreShapeConvertible**
   - 操作数可转换为形状
   - 用于reshape等操作

## 六、测试用例解析

### 重要测试场景

#### 1. 基本操作测试
- `ops.mlir`：所有基本操作
- `expand-ops.mlir`：操作扩展测试
- `expand-strided-metadata.mlir`：元数据扩展测试

#### 2. Pass测试
- `canonicalize.mlir`：标准化测试
- `normalize-memrefs.mlir`：规范化测试
- `fold-memref-alias-ops.mlir`：别名折叠测试
- `multibuffer.mlir`：多缓冲测试
- `flattened-memref.mlir`：扁平化测试

#### 3. 优化效果测试
- `mem2reg.mlir`：内存到寄存器提升
- `compose-subview.mlir`：Subview组合测试
- `resolve-shaped-type-result-dims.mlir`：形状解析测试

### 使用模式示例

#### 多缓冲优化
```mlir
// 优化前
affine.for %i = 0 to 100 {
  %temp = memref.alloc() : memref<128xf32>
  // ... 使用%temp
  memref.dealloc %temp
}

// 应用MultiBufferPass后
%temp = memref.alloc() : memref<2x128xf32>
affine.for %i = 0 to 100 {
  %idx = arith.uremi %i, %c2 : index
  %sub = memref.subview %temp[%idx] [1] [1] : ... to memref<128xf32>
  // ... 使用%sub
}
memref.dealloc %temp
```

## 七、总结

MLIR MemRef方言的Pass系统提供了全面的内存优化能力：

1. **层次化设计**：从高层次的Normalize到低层次的Flatten
2. **渐进式优化**：每个Pass专注于特定转换
3. **可组合性**：Pass可以灵活组合形成优化流水线
4. **与Affine深度集成**：共享分析和优化能力
5. **丰富的接口**：支持定制化扩展

通过合理使用这些Pass，可以显著提高内存访问效率，为后续的向量化、并行化等优化奠定基础。
