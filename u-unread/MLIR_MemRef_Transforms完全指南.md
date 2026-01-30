# MLIR MemRef方言Transform完全指南

本文档详细介绍MLIR MemRef方言中的所有Transform变换，包括其作用、技术原理和实例演示。

## 目录

1. [视图和形状操作](#1-视图和形状操作)
   - [ComposeSubView](#11-composesubview)
   - [ExpandOps](#12-expandops)
   - [ExpandStridedMetadata](#13-expandstridedmetadata)
   - [ExtractAddressComputations](#14-extractaddresscomputations)
   - [FoldMemRefAliasOps](#15-foldmemrefaliasops)

2. [类型模拟](#2-类型模拟)
   - [EmulateNarrowType](#21-emulatenarrowtype)
   - [EmulateWideInt](#22-emulatewideint)

3. [内存布局优化](#3-内存布局优化)
   - [FlattenMemRefs](#31-flattenmemrefs)
   - [NormalizeMemRefs](#32-normalizememrefs)
   - [MultiBuffer](#33-multibuffer)

4. [内存管理](#4-内存管理)
   - [ExpandRealloc](#41-expandrealloc)
   - [IndependenceTransforms](#42-independencetransforms)

5. [形状和元数据](#5-形状和元数据)
   - [ReifyResultShapes](#51-reifyresultshapes)
   - [ResolveShapedTypeResultDims](#52-resolveshapedtyperesultdims)

6. [运行时支持](#6-运行时支持)
   - [RuntimeOpVerification](#61-runtimeopverification)

7. [接口实现](#7-接口实现)
   - [AllocationOpInterface](#71-allocationopinterface)
   - [BufferViewFlowOpInterface](#72-bufferviewflowopinterface)

---

## 1. 视图和形状操作

### 1.1 ComposeSubView

**文件**: `ComposeSubView.cpp`

**作用**: 组合嵌套的 subview 操作（即 subview of subview 变为单个 subview）

**技术原理**:

将多个 subview 操作合并为一个等价的 subview，通过组合其偏移量、大小和步长：

- **偏移量组合**: `Offset by m and Stride by k` followed by `Offset by n` == `Offset by m + n * k`
- **大小组合**: 取最终 subview 的大小，忽略其他（"Take m values" followed by "Take n values" == "Take n values"）
- **步长组合**: 输出步长 = 源步长 × 操作步长

**限制条件**:
- 源 SubViewOp 不能是降秩操作（rank-reducing）
- 只支持静态大小
- 支持静态和动态偏移量

**示例**:

转换前:
```mlir
%base = memref.alloc() : memref<100x100xf32>
%view1 = memref.subview %base[10, 0] [50, 100] [1, 1]
    : memref<100x100xf32> to memref<50x100xf32>
%view2 = memref.subview %view1[0, 5] [50, 20] [1, 1]
    : memref<50x100xf32> to memref<50x20xf32>
```

转换后:
```mlir
%base = memref.alloc() : memref<100x100xf32>
%view2 = memref.subview %base[10, 5] [50, 20] [1, 1]
    : memref<100x100xf32> to memref<50x20xf32>
```

**使用场景**:
- 简化嵌套的视图操作
- 减少间接层次
- 为其他优化做准备

---

### 1.2 ExpandOps

**文件**: `ExpandOps.cpp`

**作用**: 扩展 MemRef 操作 - 将复杂操作分解为更简单的序列

**技术原理**:

**Reshape 扩展**: 将具有静态大小形状的 `memref.reshape` 转换为 `memref.reinterpret_cast`

**步长计算**: 从后向前计算步长，每个维度的步长等于其后所有维度大小的乘积

**动态大小处理**: 对于动态维度，从 shape 输入加载大小值

**核心转换**:
```cpp
// memref.reshape %src(%shape)
// 转换为 memref.reinterpret_cast，需要计算:
// - offset: 0
// - sizes: 从shape参数获取
// - strides: 从后向前计算
```

**示例**:

转换前:
```mlir
%shape = arith.constant [2, 3, 4] : index
%reshaped = memref.reshape %src(%shape)
    : (memref<24xf32>) -> memref<2x3x4xf32>
```

转换后:
```mlir
%reshaped = memref.reinterpret_cast %src to
    offset: [0], sizes: [2, 3, 4], strides: [12, 4, 1]
    : memref<24xf32> to memref<2x3x4xf32>
```

**使用场景**:
- 将高级reshape操作降级为底层操作
- 为代码生成做准备
- 简化分析

---

### 1.3 ExpandStridedMetadata

**文件**: `ExpandStridedMetadata.cpp`

**作用**: 扩展步长元数据操作 - 将修改 memref 元数据的操作展开为更易分析的构造

**技术原理**:

使用 affine 表达式显式计算步长、偏移和大小，使元数据操作的效果可被分析。

**Subview 展开**:
```cpp
// 新步长: newStrides#i = baseStrides#i * subStrides#i
// 新偏移: offset = baseOffset + sum(subOffsets#i * baseStrides#i)
// 新大小: sizes = subSizes
```

**ExpandShape 展开**:
```cpp
// 扩展大小: expandedSizes#i = baseSizes#groupId / product(expandShapeSizes#j for j != i)
// 扩展步长: expandedStrides#i = origStrides#reassDim * product(expandShapeSizes#j for j <= i)
```

**CollapseShape 展开**:
```cpp
// 折叠大小: collapsedSize = prod(origSizes#i in group)
// 折叠步长: collapsedStride = 最内层维度的步长
```

**Alloc 展开**: 计算恒等步长布局

**数据结构**:
```cpp
struct StridedMetadata {
  Value basePtr;
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};
```

**示例 - SubView**:

转换前:
```mlir
%subview = memref.subview %base[%off0, %off1] [%size0, %size1] [%stride0, %stride1]
    : memref<100x100xf32> to memref<50x50xf32>
```

转换后:
```mlir
%base_ptr, %base_offset, %base_sizes, %base_strides =
    memref.extract_strided_metadata %base

// 计算新步长
%new_stride0 = affine.apply affine_map<(s0, s1) -> (s0 * s1)>
    (%base_strides#0, %stride0)

// 计算新偏移
%new_offset = affine.apply affine_map<(s0, s1, s2) -> (s0 + s1*s2 + s3*s4)>
    (%base_offset, %off0, %base_strides#0, %off1, %base_strides#1)

%subview = memref.reinterpret_cast %base_ptr
    offset: [%new_offset], sizes: [%size0, %size1],
    strides: [%new_stride0, ...]
```

**使用场景**:
- 使元数据操作显式化
- 便于依赖分析
- 为后续优化提供基础

---

### 1.4 ExtractAddressComputations

**文件**: `ExtractAddressComputations.cpp`

**作用**: 提取地址计算 - 将带偏移的加载/存储重写为使用 subview 的加载/存储

**技术原理**:

**变换模式**: `load base[off0, off1, ...]` => `load (subview base[off0, ...])[0, ...]`

- 创建从原始偏移量开始、大小为 1 的 subview
- 重写后的操作使用零索引
- 使用模板实现通用重写逻辑

**支持的操作**:
- `memref::LoadOp`, `memref::StoreOp`
- `nvgpu::LdMatrixOp`
- `vector::TransferReadOp`, `vector::TransferWriteOp`

**核心模式**:
```cpp
template <typename LoadOpTy, typename StoreOpTy>
struct LoadStoreLikeOpRewriter {
  LogicalResult matchAndRewrite(OpTy op, ...) const {
    // 提取偏移量
    // 创建 subview
    // 用零索引重写操作
  }
};
```

**示例**:

转换前:
```mlir
%val = memref.load %base[%i + 5, %j * 2]
    : memref<100x100xf32>
```

转换后:
```mlir
%subview = memref.subview %base[%i + 5, %j * 2] [1, 1] [1, 1]
    : memref<100x100xf32> to memref<1x1xf32>
%val = memref.load %subview[0, 0]
    : memref<1x1xf32>
```

**使用场景**:
- 分离地址计算和数据访问
- 优化局部性
- 为向量化做准备

---

### 1.5 FoldMemRefAliasOps

**文件**: `FoldMemRefAliasOps.cpp`

**作用**: 折叠 MemRef 别名操作 - 将从 subview 加载/存储折叠为从原始 memref 加载/存储

**技术原理**:

通过解析索引，消除中间的别名操作层：

- **SubView 折叠**: 解析索引到源 memref 的索引
- **ExpandShape 折叠**: 使用 `AffineLinearizeIndexOp` 线性化索引
- **CollapseShape 折叠**: 使用 `AffineDelinearizeIndexOp` 反线性化索引

**支持的操作**:
```cpp
// 加载操作
LoadOpOfSubViewOpFolder, LoadOpOfExpandShapeOpFolder,
LoadOpOfCollapseShapeOpFolder

// 存储操作
StoreOpOfSubViewOpFolder

// SubView嵌套
SubViewOfSubViewFolder
```

**示例 - SubView折叠**:

转换前:
```mlir
%subview = memref.subview %base[10, 20] [30, 40] [1, 1]
    : memref<100x100xf32> to memref<30x40xf32>
%val = memref.load %subview[%i, %j]
    : memref<30x40xf32>
```

转换后:
```mlir
%val = memref.load %base[10 + %i, 20 + %j]
    : memref<100x100xf32>
```

**示例 - ExpandShape折叠**:

转换前:
```mlir
%expanded = memref.expand_shape %base [[0, 1], [2]]
    : memref<12x4xf32> into memref<3x4x4xf32>
%val = memref.load %expanded[%i, %j, %k]
```

转换后:
```mlir
%linear_idx = %i * 16 + %j * 4 + %k  // 线性化索引
%val = memref.load %base[%linear_idx / 4, %linear_idx % 4]
```

**使用场景**:
- 消除别名操作
- 减少间接访问
- 提高访问效率

---

## 2. 类型模拟

### 2.1 EmulateNarrowType

**文件**: `EmulateNarrowType.cpp`

**作用**: 窄类型模拟 - 将窄位宽类型（如 i4）模拟为更宽的类型（如 i8）

**技术原理**:

**线性化**: 将多维 memref 线性化为一维，元素类型从窄位宽变为宽位宽

**加载操作**:
1. 加载宽类型数据
2. 计算位偏移: `(index % (dstBits/srcBits)) * srcBits`
3. 右移到正确位置
4. 应用掩码提取特定位

**存储操作**:
1. 计算位偏移
2. 生成掩码: `~(((1 << srcBits) - 1) << bitOffset)`
3. 使用原子 RMW 操作进行位级写入
   - `andi`: 清除目标位
   - `ori`: 设置新值

**索引计算**:
```cpp
// 线性化索引: linear_index = floor(original_index / (dstBits / srcBits))
// 位偏移: bit_offset = (original_index % (dstBits / srcBits)) * srcBits
```

**核心转换类**:
```cpp
template <typename OpTy>
struct ConvertMemRefAllocation final : OpConversionPattern<OpTy> {
  // 转换 AllocOp/AllocaOp
  // 计算线性化大小
  // 创建新分配
};

struct ConvertMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  // 加载宽类型数据
  // 右移和掩码提取窄类型值
};

struct ConvertMemrefStore final : OpConversionPattern<memref::StoreOp> {
  // 使用 AtomicRMWOp 进行位级写入
};
```

**限制**:
- `dstBits % srcBits == 0`
- 步长必须为 1

**示例**:

转换前 (i4类型):
```mlir
%alloc = memref.alloc() : memref<10xi4>
%val = memref.load %alloc[%i] : memref<10xi4>
memref.store %new_val, %alloc[%i] : memref<10xi4>
```

转换后 (i8模拟):
```mlir
%alloc = memref.alloc() : memref<5xi8>  // 10个i4需要5个i8

// 加载
%byte_idx = affine.apply affine_map<(s0) -> (s0 floordiv 2)>(%i)
%bit_offset = affine.apply affine_map<(s0) -> ((s0 mod 2) * 4)>(%i)
%byte = memref.load %alloc[%byte_idx] : memref<5xi8>
%shifted = arith.shrui %byte, %bit_offset : i8
%val = arith.andi %shifted, 0xF : i8

// 存储
%clear_mask = arith.xori arith.shli 0xF, %bit_offset, -1 : i8
%old_byte = memref.load %alloc[%byte_idx] : memref<5xi8>
%cleared = arith.andi %old_byte, %clear_mask : i8
%new_byte_shifted = arith.shli %new_val, %bit_offset : i8
%final = arith.ori %cleared, %new_byte_shifted : i8
memref.store %final, %alloc[%byte_idx] : memref<5xi8>
```

**使用场景**:
- 硬件不支持窄位宽时的模拟
- 压缩存储
- 位级数据处理

---

### 2.2 EmulateWideInt

**文件**: `EmulateWideInt.cpp`

**作用**: 宽整数操作模拟 - 将不支持的超宽整数类型分解为支持的类型

**技术原理**:

**类型转换**: 将整数元素宽度大于 `widestIntSupported` 的 memref 转换为支持的宽度

**依赖 Arith**: 依赖 `arith::WideIntEmulationConverter` 和相关的算术模式

**简单转换**: 直接转换 memref 类型，元素类型相应调整

**核心实现**:
```cpp
struct ConvertMemRefAlloc : OpConversionPattern<memref::AllocOp> {
  LogicalResult matchAndRewrite(memref::AllocOp op, ...) {
    // 直接转换类型
    auto newType = typeConverter->convertType<MemRefType>(op.getType());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newType, ...);
  }
};
```

**示例**:

转换前 (假设最大支持64位):
```mlir
%alloc = memref.alloc() : memref<10xi128>
%val = memref.load %alloc[%i] : memref<10xi128>
```

转换后 (分解为两个i64):
```mlir
%alloc = memref.alloc() : memref<10xvector<2xi64>>
%vec = vector.load %alloc[%i] : memref<10xvector<2xi64>>
// 后续算术操作会将i128操作分解为i64操作序列
```

**使用场景**:
- 硬件不支持超宽整数时的模拟
- 大整数运算
- 密码学应用

---

## 3. 内存布局优化

### 3.1 FlattenMemRefs

**文件**: `FlattenMemRefs.cpp`

**作用**: 扁平化 MemRef - 将多秩 memref 相关操作展平为一维 memref 操作

**技术原理**:

**线性化**: 使用 `ExtractStridedMetadataOp` 获取元数据，计算线性化偏移和大小

**操作重写**:
- **AllocOp/AllocaOp**: 创建一维分配，然后用 reinterpret_cast 恢复原始类型
- **LoadOp/StoreOp**: 使用线性化索引访问一维 memref
- **Vector 操作**: 类似处理

**限制**:
- 要求 identity 或 strided 布局
- Transfer 操作要求 inbounds 访问和 identity/minor_identity 排列映射

**核心函数**:
```cpp
std::tuple<Value, OpFoldResult> getFlattenMemrefAndOffset(
    OpBuilder &builder, Location loc, Value memref,
    ArrayRef<OpFoldResult> indices);
```

**示例**:

转换前:
```mlir
%alloc = memref.alloc() : memref<10x20x30xf32>
%val = memref.load %alloc[%i, %j, %k] : memref<10x20x30xf32>
```

转换后:
```mlir
// 分配时保持原始类型
%alloc = memref.alloc() : memref<10x20x30xf32>

// 访问时线性化
%base, %offset, %sizes, %strides =
    memref.extract_strided_metadata %alloc
%linear_offset = %i * %strides#0 + %j * %strides#1 + %k * %strides#2
%flat = memref.reinterpret_cast %base
    offset: [%linear_offset], sizes: [1], strides: [1]
    : (memref<10x20x30xf32>) -> memref<1xf32>
%val = memref.load %flat[0] : memref<1xf32>
```

**使用场景**:
- 简化内存访问模式
- 为SIMD优化做准备
- 与底层API交互

---

### 3.2 NormalizeMemRefs

**文件**: `NormalizeMemRefs.cpp`

**作用**: 规范化 MemRef - 将 memref 转换为恒等布局映射

**技术原理**:

**函数间分析**:
1. 识别所有可规范化的函数
2. 调用/被调用非规范化函数的函数也被视为不可规范化

**规范化过程**:
1. 更新函数参数类型
2. 规范化 AllocOp, AllocaOp, ReinterpretCastOp
3. 更新函数返回类型
4. 更新调用点

**布局映射**: 使用 AffineMap 处理非恒等布局

**核心函数**:
```cpp
bool areMemRefsNormalizable(
    FuncOp funcOp,
    llvm::SmallDenseMap<func::FuncOp, bool> &normalizedFuncs);

LogicalResult normalizeFuncOpMemRefs(
    FuncOp funcOp,
    llvm::DenseMap<Type, Type> &typeMapping);
```

**示例**:

转换前:
```mlir
func.func @foo(%arg0: memref<10x10xf32, affine_map<(d0, d1) -> (d0 + d1)>>) {
  // 使用混合布局
  %val = memref.load %arg0[%i, %j] : memref<10x10xf32, affine_map<(d0, d1) -> (d0 + d1)>>
  return
}
```

转换后:
```mlir
func.func @foo(%arg0: memref<10x10xf32>) {
  // 使用identity布局，访问时显式计算偏移
  %linear_idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
  %base, %offset, %sizes, %strides =
      memref.extract_strided_metadata %arg0
  %flat = memref.reinterpret_cast %base
      offset: [%linear_idx], sizes: [1], strides: [1]
  %val = memref.load %flat[0]
  return
}
```

**使用场景**:
- 标准化内存布局
- 简化代码生成
- 与只支持identity布局的后端兼容

---

### 3.3 MultiBuffer

**文件**: `MultiBuffer.cpp`

**作用**: 多缓冲变换 - 通过扩展数组移除循环迭代间对临时分配的依赖

**技术原理**:

**候选循环识别**: 查找所有用户都在同一个循环中的分配

**多缓冲分配**: 创建新的 memref 类型，第一维为多缓冲因子

**模索引**: 计算每个迭代的缓冲区索引:
```cpp
buffer_idx = ((%iv - %lb) / %step) % %mb_factor
```

**Subview 创建**: 创建访问特定切片的 subview

**使用替换**: 递归替换所有使用，处理 subview 嵌套

**核心算法**:
```cpp
FailureOr<memref::AllocOp> multiBuffer(
    RewriterBase &rewriter,
    memref::AllocOp allocOp,
    unsigned multiBufferingFactor,
    bool skipOverrideAnalysis) {

  // 1. 检查条件: 所有用户在同一循环中
  // 2. 创建多缓冲类型: [factor] + original_shape
  // 3. 在循环内计算模索引
  // 4. 创建访问切片的 subview
  // 5. 替换所有使用
}
```

**示例**:

转换前 (循环携带依赖):
```mlir
%temp = memref.alloc() : memref<100xf32>
scf.for %i = 0 to 100 {
  memref.copy %src[%i], %temp : memref<100xf32>
  %val = "compute"(%temp) : (memref<100xf32>) -> f32
  memref.store %val, %dest[%i] : memref<100xf32>
}
// 每次迭代都依赖前一次迭代的 %temp
```

转换后 (多缓冲，factor=2):
```mlir
%temp = memref.alloc() : memref<2x100xf32>
scf.for %i = 0 to 100 {
  // 计算循环内的缓冲区索引
  %buffer_idx = affine.apply affine_map<((d0) -> ((d0) mod 2))>(%i)
  %slice = memref.subview %temp[%buffer_idx, 0] [1, 100] [1, 1]
      : memref<2x100xf32> to memref<100xf32>

  memref.copy %src[%i], %slice : memref<100xf32>
  %val = "compute"(%slice) : (memref<100xf32>) -> f32
  memref.store %val, %dest[%i] : memref<100xf32>
}
// 迭代0使用buffer[0]，迭代1使用buffer[1]，迭代2使用buffer[0]...
// 移除了循环携带依赖
```

**使用场景**:
- 软件流水线
- 并行化
- 循环携带依赖消除
- GPU kernel 优化

---

## 4. 内存管理

### 4.1 ExpandRealloc

**文件**: `ExpandRealloc.cpp`

**作用**: 扩展 memref.realloc 操作 - 将 realloc 分解为其组成操作

**技术原理**:

**条件分配和复制**:
1. 比较当前缓冲区大小与请求大小
2. 如果旧缓冲区较小，分配新缓冲区，复制数据，释放旧缓冲区
3. 如果旧缓冲区足够大，使用 reinterpret_cast 调整大小

**实现**: 使用 `scf.if` 实现条件逻辑

**核心模式**:
```cpp
struct ExpandReallocOpPattern : OpRewritePattern<memref::ReallocOp> {
  LogicalResult matchAndRewrite(memref::ReallocOp op, ...) {
    // 提取源缓冲区大小
    // 创建条件: if (source_size < requested_size)
    // then分支: 分配新缓冲区，复制数据
    // else分支: reinterpret_cast
  }
};
```

**示例**:

转换前:
```mlir
%result = memref.realloc %source[%new_size]
    : (memref<?xf32>, index) -> memref<?xf32>
```

转换后:
```mlir
%source_size = memref.dim %source, 0 : memref<?xf32>
%result = scf.if (%source_size < %new_size) -> (memref<?xf32>) {
  // 需要重新分配
  %new_alloc = memref.alloc(%new_size) : memref<?xf32>
  memref.copy %source, %new_alloc : memref<?xf32>
  memref.dealloc %source : memref<?xf32>
  scf.yield %new_alloc : memref<?xf32>
} else {
  // 可以重用现有缓冲区
  %view = memref.reinterpret_cast %source
      to offset: [0], sizes: [%new_size], strides: [1]
  scf.yield %view : memref<?xf32>
}
```

**使用场景**:
- 代码生成（目标平台不支持realloc）
- 显式内存管理
- 优化分析

---

### 4.2 IndependenceTransforms

**文件**: `IndependenceTransforms.cpp`

**作用**: 独立性变换 - 使操作独立于某些值

**技术原理**:

**独立大小计算**: 使用 `ValueBoundsConstraintSet` 计算独立上界

**Subview 包装**: 创建新分配，然后用 subview 包装以获得原始形状

**类型传播**: 使用 `UnrealizedConversionCastOp` 处理类型不匹配

**Alloc 到 Alloca 转换**: 将有匹配 dealloc 的 alloc 转换为 alloca

**核心函数**:
```cpp
std::pair<Value, Value> makeIndependent(
    OpBuilder &builder,
    Operation *op,
    OpOperand &operand,
    independenceTransform::TransformState &state);

FailureOr<memref::AllocaOp> allocToAlloca(
    RewriterBase &rewriter,
    memref::AllocOp allocOp);
```

**示例**:

转换前:
```mlir
%size = "unknown_size"() : () -> index
%alloc = memref.alloc(%size) : memref<?xf32>
```

转换后:
```mlir
// 计算独立上界
%static_size = arith.constant 100 : index
%independent_alloc = memref.alloc(%static_size) : memref<100xf32>

// 使用subview包装获得原始类型
%cast = memref.subview %independent_alloc[0][%size][1]
    : memref<100xf32> to memref<?xf32>
```

**Alloc到Alloca转换**:

转换前:
```mlir
%alloc = memref.alloc() : memref<100xf32>
// ... 使用 %alloc ...
memref.dealloc %alloc : memref<100xf32>
```

转换后:
```mlir
%alloca = memref.alloca() : memref<100xf32>
// ... 使用 %alloca ...
// (不需要dealloc)
```

**使用场景**:
- 依赖分析
- 生命周期优化
- 栈分配优化
- 并行化准备

---

## 5. 形状和元数据

### 5.1 ReifyResultShapes

**文件**: `ReifyResultShapes.cpp`

**作用**: 具体化结果形状 - 为 `ReifyRankedShapedTypeOpInterface` 操作具体化结果形状

**技术原理**:

**形状具体化**: 调用 `reifyResultShapes` 获取形状

**类型更新**: 根据具体化的形状更新结果类型

**操作克隆**: 克隆操作并更新结果类型

**转换插入**: 插入 cast 操作以保持 IR 一致性

**限制**: 当前只支持 `tensor::PadOp` 和 `tensor::ConcatOp`

**核心流程**:
```cpp
LogicalResult reifyOpResultShapes(
    Operation *op,
    ReificationCallbackFn reificationCallback);

// 对于每个操作结果:
// 1. 调用 reifyResultShapes 获取形状值
// 2. 更新结果类型为静态形状
// 3. 插入 cast 从动态类型到静态类型
```

**示例**:

转换前:
```mlir
%padded = tensor.pad %source low[0] high[%pad_amount] {
  ^bb0(%arg0: index):
    tensor.yield %c0 : f32
} : tensor<?xf32> to tensor<?xf32>
%dim = tensor.dim %padded, 0 : tensor<?xf32>
```

转换后:
```mlir
// 形状被具体化为计算值
%original_dim = tensor.dim %source, 0 : tensor<?xf32>
%static_size = arith.addi %original_dim, %pad_amount : index

// 结果类型变为静态
%padded_static = tensor.pad %source low[0] high[%pad_amount] {
  ^bb0(%arg0: index):
    tensor.yield %c0 : f32
} : tensor<?xf32> to tensor<100xf32>  // 具体化大小

// Cast保持类型兼容
%padded = tensor.cast %padded_static : tensor<100xf32> to tensor<?xf32>

// dim操作可以被折叠
%dim = arith.constant 100 : index  // 替换原始的 tensor.dim
```

**使用场景**:
- 形状推断
- 类型静态化
- 边界检查消除
- 优化循环边界

---

### 5.2 ResolveShapedTypeResultDims

**文件**: `ResolveShapedTypeResultDims.cpp`

**作用**: 解析结果维度操作 - 使用 `InferShapedTypeOpInterface` 解析 memref.dim 操作

**技术原理**:

**Dim 折叠**: 使用 `reifyReturnTypeShapes` 获取形状，然后提取维度

**迭代参数**: 在 `scf.forall` 中，将 `%arg0` 的 dim 替换为对应初始参数的 dim

**边界检查**: 确保维度索引不越界

**核心模式**:
```cpp
// 对于实现 InferShapedTypeOpInterface 的操作
struct DimOfShapedTypeOpInterface
    : public OpRewritePattern<memref::DimOp> {
  LogicalResult matchAndRewrite(memref::DimOp dimOp, ...) {
    // 调用 reifyReturnTypeShapes
    // 提取维度
    // 替换 dim 操作
  }
};

// 对于 scf.forall 的迭代参数
struct IterArgsToInitArgs : public OpRewritePattern<memref::DimOp> {
  // 将 iter_args 的 dim 替换为 init_args 的 dim
};
```

**示例**:

转换前:
```mlir
%alloc = memref.alloc(%size) : memref<?xf32>
%d = memref.dim %alloc, 0 : memref<?xf32>
```

转换后:
```mlir
// dim操作直接使用size值
%d = %size  // 假设size是索引值

// 或如果size需要转换:
%d = arith.index_cast %size : i32 to index
```

**Forall迭代参数示例**:

转换前:
```mlir
scf.forall (%arg0) in (%size) shared_outs(%init = %output) -> (memref<?xf32>) {
  %d = memref.dim %arg0, 0 : memref<?xf32>
  // 使用 %d
  scf.forall.in_parallel {
    tensor.parallel_insert %val into %output[...]
  }
}
```

转换后:
```mlir
scf.forall (%arg0) in (%size) shared_outs(%init = %output) -> (memref<?xf32>) {
  %d = %size  // 直接使用初始大小
  // 使用 %d
  scf.forall.in_parallel {
    tensor.parallel_insert %val into %output[...]
  }
}
```

**使用场景**:
- 消除冗余的dim操作
- 形状传播优化
- 静态形状推断

---

## 6. 运行时支持

### 6.1 RuntimeOpVerification

**文件**: `RuntimeOpVerification.cpp`

**作用**: 运行时操作验证 - 为 MemRef 操作生成运行时验证代码

**技术原理**:

**边界检查**: 生成 `0 <= index < dim_size` 的断言

**对齐验证**: 检查指针对齐

**类型验证**: 检查秩、维度大小、偏移和步长

**SubView 验证**: 验证偏移和切片不越界

**使用 `cf::AssertOp`**: 生成运行时断言

**接口实现**:
```cpp
struct AssumeAlignmentOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};

struct CastOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};

struct LoadStoreOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};

struct SubViewOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};
```

**示例 - Load验证**:

转换前:
```mlir
%val = memref.load %memref[%i, %j] : memref<100x100xf32>
```

转换后:
```mlir
// 运行时边界检查
%dim0 = memref.dim %memref, 0 : memref<100x100xf32>
%dim1 = memref.dim %memref, 1 : memref<100x100xf32>
%check0 = arith.cmpi slt, %i, %dim0 : index
%check1 = arith.cmpi slt, %j, %dim1 : index
%check2 = arith.cmpi sge, %i, 0 : index
%check3 = arith.cmpi sge, %j, 0 : index
%valid = arith.andi %check0, %check1, %check2, %check3 : i1
cf.assert %valid, "index out of bounds" : i1

%val = memref.load %memref[%i, %j] : memref<100x100xf32>
```

**示例 - SubView验证**:

转换前:
```mlir
%subview = memref.subview %base[%off0, %off1] [%size0, %size1] [1, 1]
    : memref<100x100xf32> to memref<10x20xf32>
```

转换后:
```mlir
// 验证偏移不越界
%dim0 = memref.dim %base, 0 : memref<100x100xf32>
%dim1 = memref.dim %base, 1 : memref<100x100xf32>

%off0_ok = arith.cmpi sle, %off0, %dim0 : index
%off1_ok = arith.cmpi sle, %off1, %dim1 : index
%size0_ok = arith.cmpi sle, (%off0 + %size0), %dim0 : index
%size1_ok = arith.cmpi sle, (%off1 + %size1), %dim1 : index

%all_ok = arith.andi %off0_ok, %off1_ok, %size0_ok, %size1_ok : i1
cf.assert %all_ok, "subview out of bounds" : i1

%subview = memref.subview %base[%off0, %off1] [%size0, %size1] [1, 1]
    : memref<100x100xf32> to memref<10x20xf32>
```

**使用场景**:
- 调试
- 安全检查
- 动态验证
- 边界条件检测

---

## 7. 接口实现

### 7.1 AllocationOpInterface

**文件**: `AllocationOpInterfaceImpl.cpp`

**作用**: 为 MemRef 操作实现 `AllocationOpInterface`

**技术原理**:

**接口方法**:
- `buildDealloc`: 为 AllocOp/ReallocOp 构建相应的 DeallocOp
- `buildClone`: 构建克隆操作
- `getHoistingKind`: 指定提升类型（Loop/Block）
- `buildPromotedAlloc`: 将 AllocOp 提升为 AllocaOp

**实现的接口**:
```cpp
// AllocOp
struct DefaultAllocationInterface
    : public AllocationOpInterface::ExternalModel<...> {
  std::optional<Operation *> buildDealloc(OpBuilder &builder, Value alloc);
  std::optional<Value> buildClone(OpBuilder &builder, Value alloc);
  HoistingKind getHoistingKind();
  std::optional<Value> buildPromotedAlloc(OpBuilder &builder,
                                          Value alloc);
};

// AllocaOp
struct DefaultAutomaticAllocationHoistingInterface
    : public AllocationOpInterface::ExternalModel<...> {
  HoistingKind getHoistingKind();
};

// ReallocOp
struct DefaultReallocationInterface
    : public AllocationOpInterface::ExternalModel<...> {
  std::optional<Operation *> buildDealloc(OpBuilder &builder, Value alloc);
  std::optional<Value> buildClone(OpBuilder &builder, Value alloc);
};
```

**示例 - buildDealloc**:

```cpp
// 对于 AllocOp
Operation *DefaultAllocationInterface::buildDealloc(OpBuilder &builder, Value alloc) {
  return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc);
}

// 使用场景
%alloc = memref.alloc() : memref<100xf32>
// ... 使用 %alloc ...
%dealloc = AllocationOpInterface::buildDealloc(builder, %alloc)
// %dealloc 是 memref.dealloc %alloc
```

**示例 - buildClone**:

```cpp
Value DefaultAllocationInterface::buildClone(OpBuilder &builder, Value alloc) {
  MemRefType type = cast<MemRefType>(alloc.getType());
  Operation *clone = builder.create<memref::AllocOp>(
      alloc.getLoc(), type,
      getAsOpFoldResult(alloc.getDefiningOp()->getOperands()));
  builder.create<memref::CopyOp>(alloc.getLoc(), alloc, clone->getResult(0));
  return clone->getResult(0);
}

// 使用场景
%original = memref.alloc() : memref<100xf32>
%cloned = AllocationOpInterface::buildClone(builder, %original)
// %cloned 是 %original 的副本
```

**示例 - buildPromotedAlloc**:

```cpp
Value DefaultAllocationInterface::buildPromotedAlloc(
    OpBuilder &builder, Value alloc) {
  MemRefType type = cast<MemRefType>(alloc.getType());
  return builder.create<memref::AllocaOp>(alloc.getLoc(), type);
}

// 使用场景: 将堆分配提升为栈分配
%heap_alloc = memref.alloc() : memref<100xf32>
%stack_alloc = AllocationOpInterface::buildPromotedAlloc(builder, %heap_alloc)
// %stack_alloc 是 memref.alloca() : memref<100xf32>
```

**使用场景**:
- 自动内存管理
- 分配提升优化
- 克隆操作生成
- 与Bufferization pass集成

---

### 7.2 BufferViewFlowOpInterface

**文件**: `BufferViewFlowOpInterfaceImpl.cpp`

**作用**: 为 ReallocOp 实现 `BufferViewFlowOpInterface`

**技术原理**:

**依赖关系**: realloc 的结果可能依赖于源操作数

**终端缓冲区**: realloc 可能返回新分配的缓冲区，因此是终端缓冲区

**接口方法**:
```cpp
struct ReallocOpInterface
    : public BufferViewFlowOpInterface::ExternalModel<...> {
  void populateDependencies(Operation *op,
                           BufferViewFlowAnalysis::DependenyMap &dependencies);

  bool mayBeTerminalBuffer(Operation *op, Value value);
};
```

**populateDependencies**:
```cpp
void ReallocOpInterface::populateDependencies(
    Operation *op,
    BufferViewFlowAnalysis::DependencyMap &dependencies) {
  auto reallocOp = cast<memref::ReallocOp>(op);
  // realloc 的结果依赖于:
  // 1. 源操作数（如果重用）
  // 2. 新分配（如果重新分配）
  dependencies[reallocOp.getResult()].push_back(reallocOp.getSource());
}
```

**mayBeTerminalBuffer**:
```cpp
bool ReallocOpInterface::mayBeTerminalBuffer(Operation *op, Value value) {
  auto reallocOp = cast<memref::ReallocOp>(op);
  // 如果realloc返回新分配，则是终端缓冲区
  return true;
}
```

**使用场景**:
- 缓冲区生命周期分析
- 别名分析
- 内存优化
- 死代码消除

---

## 总结

MLIR MemRef方言的17个Transform涵盖了以下优化领域：

| 类别 | Transform | 主要用途 |
|------|-----------|----------|
| **视图操作** | ComposeSubView, ExpandOps, ExpandStridedMetadata, ExtractAddressComputations, FoldMemRefAliasOps | 简化视图层次 |
| **类型模拟** | EmulateNarrowType, EmulateWideInt | 硬件类型支持 |
| **布局优化** | FlattenMemRefs, NormalizeMemRefs, MultiBuffer | 内存访问优化 |
| **内存管理** | ExpandRealloc, IndependenceTransforms | 生命周期优化 |
| **形状元数据** | ReifyResultShapes, ResolveShapedTypeResultDims | 形状推断 |
| **运行时** | RuntimeOpVerification | 安全检查 |
| **接口** | AllocationOpInterface, BufferViewFlowOpInterface | 框架集成 |

### 典型优化流水线

```
原始代码
    ↓
[ExpandStridedMetadata]  // 展开元数据
    ↓
[FoldMemRefAliasOps]     // 折叠别名
    ↓
[ComposeSubView]         // 组合视图
    ↓
[NormalizeMemRefs]       // 规范化布局
    ↓
[MultiBuffer]            // 多缓冲优化
    ↓
[FlattenMemRefs]         // 扁平化
    ↓
代码生成
```

### 关键数据结构

```cpp
// 步长元数据
struct StridedMetadata {
  Value basePtr;
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

// 线性化信息
struct LinearizedMemRefInfo {
  OpFoldResult linearizedOffset;
  OpFoldResult linearizedSize;
};

// 多缓冲配置
struct MultiBufferConfig {
  unsigned factor;
  bool skipOverrideAnalysis;
};
```

这些变换可以单独使用或组合使用，形成强大的内存优化能力。
