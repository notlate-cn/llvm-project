# LLVM MLIR Vector方言Transform技术详解

本文档详细梳理LLVM MLIR中Vector方言的所有Transform Pass的作用、技术原理和应用场景。

**目录路径**: `mlir/lib/Dialect/Vector/Transforms/`

---

## 目录

1. [核心向量操作Lower变换](#核心向量操作Lower变换)
2. [特殊优化Transform](#特殊优化Transform)
3. [接口实现](#接口实现)
4. [总结与最佳实践](#总结与最佳实践)

---

## 核心向量操作Lower变换

### 1. Lower Vector Contract (向量收缩降级)

**文件**: `LowerVectorContract.cpp`

#### 1.1 作用
将高级的`vector.contract`操作(通用张量收缩/矩阵乘法)降级为更基础的向量操作。

#### 1.2 技术原理

**核心概念**: Vector Contract是MLIR中表示通用张量收缩的操作,包含:
- **Indexing Maps**: 定义操作数和结果的访问模式
- **Iterator Types**: 指定每个维度是parallel还是reduction

#### 降级策略
1. **OuterProduct**: 降级为外积形式
2. **MatmulOp**: 转换为矩阵乘法
3. **Dot**: 降级为点积
4. **FlatTranspose**: 通过转置实现

#### 1.3 实例演示

**降级前**:
```mlir
%c = vector.contract {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // LHS访问模式
    affine_map<(i,j,k) -> (k,j)>,  // RHS访问模式
    affine_map<(i,j,k) -> (i,j)>   // ACC访问模式
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} %A, %B, %acc : vector<4x8xf32>, vector<8x16xf32> into vector<4x16xf32>
```

**降级后** (OuterProduct):
```mlir
%result = %acc
affine.for %k = 0 to 8 {
  %a_col = vector.extract %A[%k] : vector<8xf32>
  %b_row = vector.extract %B[%k] : vector<16xf32>
  %outer = vector.outerproduct %a_col, %b_row, %result
  %result = %outer
}
```

**效果**: 转换为硬件更友好的操作序列

---

### 2. Lower Vector Transfer (向量传输降级)

**文件**: `LowerVectorTransfer.cpp`

#### 2.1 作用
优化`vector.transfer_read`和`vector.transfer_write`操作的permutation_map,降级为标准访问模式。

#### 2.2 技术原理

#### Permutation Map优化
- 将非恒等映射转换为恒等映射 + transpose
- 消除broadcasting维度
- 调整in_bounds属性

#### 2.3 实例演示

**优化前**:
```mlir
%v = vector.transfer_read %memref[%i, %j]
  {permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
  : memref<100x200xf32>, vector<16x8xf32>
```

**优化后**:
```mlir
%tmp = vector.transfer_read %memref[%i, %j]
  {permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
  : memref<100x200xf32>, vector<8x16xf32>
%v = vector.transpose %tmp, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
```

**效果**: 分离数据访问和布局变换

---

### 3. Lower Vector Transpose (向量转置降级)

**文件**: `LowerVectorTranspose.cpp`

#### 3.1 作用
将`vector.transpose`降级为shuffle操作或其他更基础的向量操作。

#### 3.2 技术原理

#### 多种降级策略
1. **Shuffle1D**: 1D向量的shuffle实现
2. **Shuffle16x16**: 针对16x16矩阵的优化shuffle序列
3. **EltWise**: 基于extract/insert的逐元素实现

#### 16x16转置优化
使用特殊的unpack和shuffle指令序列模拟硬件转置:
- `UnpackLoPd` / `UnpackHiPd`
- `UnpackLoPs` / `UnpackHiPs`
- 4x128位lane shuffle

#### 3.3 实例演示

**Shuffle1D变换**:
```mlir
// 2x3 转置为 3x2, 线性化为1D
%in = vector<6xf32>  // [a,b,c,d,e,f] 表示 [[a,b,c], [d,e,f]]

// 降级为shuffle
%out = vector.shuffle %in, %in [0, 3, 1, 4, 2, 5]
// 结果: [a,d,b,e,c,f] 表示 [[a,d], [b,e], [c,f]]
```

**16x16转置** (使用AVX512指令模式):
```mlir
%0 = createUnpackLoPd(%v0, %v1)
%1 = createUnpackHiPd(%v0, %v1)
// ... 8步shuffle操作
%result = create4x128BitShuffle(%temp, ...)
```

**效果**: 充分利用SIMD硬件的shuffle能力

---

### 4. Lower Vector Broadcast (向量广播降级)

**文件**: `LowerVectorBroadcast.cpp`

#### 4.1 作用
渐进式降级`vector.broadcast`操作。

#### 4.2 技术原理

#### 三阶段降级
1. **Scalar to Vector**: 标量到1D向量(splat)
2. **Rank Stretching**: 增加前导维度
3. **Dimension Stretching**: 扩展特定维度

#### 4.3 实例演示

**阶段1**: Scalar → Vector
```mlir
// 前
%v = vector.broadcast %scalar : f32 to vector<4xf32>

// 后
%v = vector.splat %scalar : vector<4xf32>
```

**阶段2**: 增加维度
```mlir
// 前
%v = vector.broadcast %in : vector<4xf32> to vector<3x4xf32>

// 后
%result = arith.constant dense<0> : vector<3x4xf32>
%v0 = vector.insert %in, %result[0]
%v1 = vector.insert %in, %v0[1]
%v2 = vector.insert %in, %v1[2]
```

**阶段3**: 扩展维度
```mlir
// 前: 扩展最后一维 2 → 4
%out = vector.broadcast %in : vector<3x2xf32> to vector<3x4xf32>

// 后: 使用extract/insert复制
affine.for %i = 0 to 3 {
  %row = vector.extract %in[%i] : vector<2xf32>
  %e0 = vector.extract %row[0] : f32
  %e1 = vector.extract %row[1] : f32
  %new_row = vector.from_elements %e0, %e1, %e0, %e1 : vector<4xf32>
  %result = vector.insert %new_row, %result[%i]
}
```

---

### 5. Lower Vector Gather (向量聚集降级)

**文件**: `LowerVectorGather.cpp`

#### 5.1 作用
降级`vector.gather`操作(不规则索引访问)。

#### 5.2 技术原理

#### 降级策略
1. **多维展开**: 将多维gather展开为1维
2. **移除stride**: 通过memref.collapse_shape
3. **标量化**: 转换为条件标量load循环

#### 5.3 实例演示

**降级前**:
```mlir
%result = vector.gather %base[%c0][%indices], %mask, %passthru
  : memref<100xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>
  into vector<8xf32>
```

**降级后**:
```mlir
%result = %passthru
affine.for %i = 0 to 8 {
  %idx = vector.extract %indices[%i] : i32
  %m = vector.extract %mask[%i] : i1
  %val = scf.if %m {
    %loaded = memref.load %base[%idx] : memref<100xf32>
    scf.yield %loaded
  } else {
    %pass = vector.extract %passthru[%i] : f32
    scf.yield %pass
  }
  %result = vector.insert %val, %result[%i]
}
```

**效果**: 转换为可执行的标量循环

---

### 6. Lower Vector Multi-Reduction (多维归约降级)

**文件**: `LowerVectorMultiReduction.cpp`

#### 6.1 作用
降级`vector.multi_reduction`(跨多维度的归约操作)。

#### 6.2 技术原理

#### 两种策略
1. **InnerParallel**: 保持内层并行,外层归约
2. **InnerReduction**: 先内层归约,后外层

使用transpose在两种形式间转换。

#### 6.3 实例演示

**降级前**:
```mlir
// 对维度[0, 2]做add归约: vector<4x8x16xf32> → vector<8xf32>
%result = vector.multi_reduction <add>, %input, %acc [0, 2]
  : vector<4x8x16xf32> to vector<8xf32>
```

**降级后** (InnerReduction):
```mlir
// 步骤1: 先归约维度2
%tmp = vector.multi_reduction <add>, %input, %cst [2]
  : vector<4x8x16xf32> to vector<4x8xf32>

// 步骤2: 再归约维度0
%result = vector.multi_reduction <add>, %tmp, %acc [0]
  : vector<4x8xf32> to vector<8xf32>
```

**效果**: 分解为单维度归约链

---

### 7. Lower Vector Shape Cast (向量形状转换降级)

**文件**: `LowerVectorShapeCast.cpp`

#### 7.1 作用
降级`vector.shape_cast`(元素数量不变的reshape)。

#### 7.2 技术原理

#### 三种特殊化策略
1. **Drop Leading Unit Dims**: 移除前导1维度
2. **GCD Decomposition**: 基于最大公约数分解
3. **Scalable Vector**: 处理可伸缩向量

#### 7.3 实例演示

**策略1**: 移除unit维度
```mlir
// 前
%out = vector.shape_cast %in : vector<1x1x4x8xf32> to vector<4x8xf32>

// 后
%t1 = vector.extract %in[0] : vector<1x4x8xf32>
%out = vector.extract %t1[0] : vector<4x8xf32>
```

**策略2**: GCD分解
```mlir
// 前: 2x3 → 3x2
%out = vector.shape_cast %in : vector<2x3xf32> to vector<3x2xf32>

// 后: 通过GCD(2,3)=1, 先flatten再reshape
%flat = vector.shape_cast %in : vector<2x3xf32> to vector<6xf32>
%out = vector.shape_cast %flat : vector<6xf32> to vector<3x2xf32>
```

---

### 8. Lower Vector Scan (向量扫描降级)

**文件**: `LowerVectorScan.cpp`

#### 8.1 作用
降级`vector.scan`操作(前缀和/后缀和等扫描算法)。

#### 8.2 技术原理

使用`extract_strided_slice`和`insert_strided_slice`实现迭代累积。

#### 8.3 实例演示

**降级前**:
```mlir
// inclusive scan (前缀和)
%sum, %last = vector.scan <add>, %input, %init {inclusive = true, reduction_dim = 0}
  : vector<4xf32>, f32
```

**降级后**:
```mlir
%acc = %init
%result = arith.constant dense<0.0> : vector<4xf32>
affine.for %i = 0 to 4 {
  %elem = vector.extract %input[%i] : f32
  %acc = arith.addf %acc, %elem : f32
  %result = vector.insert %acc, %result[%i]
}
%last = %acc
```

---

### 9. Lower Vector Interleave (向量交叉降级)

**文件**: `LowerVectorInterleave.cpp`

#### 9.1 作用
降级`vector.interleave`和`vector.deinterleave`操作。

#### 9.2 技术原理

**Interleave**: 交替合并两个向量的元素
**Deinterleave**: 分离交叉的元素

#### 9.3 实例演示

**Interleave**:
```mlir
// 前
%result = vector.interleave %lhs, %rhs : vector<4xf32>
// %lhs = [a, b, c, d], %rhs = [e, f, g, h]

// 后 (使用shuffle)
%result = vector.shuffle %lhs, %rhs [0, 4, 1, 5, 2, 6, 3, 7]
// 结果: [a, e, b, f, c, g, d, h]
```

**Deinterleave**:
```mlir
// 前
%even, %odd = vector.deinterleave %input : vector<8xf32>
// %input = [a, b, c, d, e, f, g, h]

// 后
%even = vector.shuffle %input, %input [0, 2, 4, 6]  // [a, c, e, g]
%odd = vector.shuffle %input, %input [1, 3, 5, 7]   // [b, d, f, h]
```

---

### 10. Lower Vector Mask (向量掩码降级)

**文件**: `LowerVectorMask.cpp`

#### 10.1 作用
降级掩码创建和应用操作。

#### 10.2 技术原理

#### 处理的操作
- `vector.create_mask`: 动态创建掩码
- `vector.constant_mask`: 常量掩码
- `vector.mask`: 掩码应用

#### 10.3 实例演示

**create_mask降级**:
```mlir
// 前
%mask = vector.create_mask %dim0, %dim1 : vector<4x8xi1>

// 后: 降维处理
%mask_1d = vector.create_mask %dim1 : vector<8xi1>
%result = arith.constant dense<false> : vector<4x8xi1>
affine.for %i = 0 to 4 {
  %cond = arith.cmpi slt, %i, %dim0
  %row = scf.if %cond {
    scf.yield %mask_1d
  } else {
    %zero = arith.constant dense<false> : vector<8xi1>
    scf.yield %zero
  }
  %result = vector.insert %row, %result[%i]
}
```

---

### 11. Lower Vector BitCast (向量位转换降级)

**文件**: `LowerVectorBitCast.cpp`

#### 11.1 作用
降级`vector.bitcast`(类型重新解释)操作。

#### 11.2 技术原理

将多维bitcast降级为1D操作,通过extract/insert实现。

#### 11.3 实例演示

```mlir
// 前: 重新解释位模式
%out = vector.bitcast %in : vector<4xi32> to vector<8xi16>

// 后: 通过flatten实现
%flat_in = vector.shape_cast %in : vector<4xi32> to vector<4xi32>
%flat_out = vector.bitcast %flat_in : vector<4xi32> to vector<8xi16>
%out = vector.shape_cast %flat_out : vector<8xi16> to vector<8xi16>
```

---

### 12. Lower Vector Step (向量步进降级)

**文件**: `LowerVectorStep.cpp`

#### 12.1 作用
降级`vector.step`操作(创建索引向量 [0, 1, 2, ...])。

#### 12.2 实例演示

```mlir
// 前
%step = vector.step : vector<4xindex>

// 后
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c3 = arith.constant 3 : index
%result = vector.from_elements %c0, %c1, %c2, %c3 : vector<4xindex>
```

---

## 特殊优化Transform

### 13. Vector Unroll (向量展开)

**文件**: `VectorUnroll.cpp`

#### 13.1 作用
将向量操作展开到目标硬件的native向量大小。

#### 13.2 技术原理

#### 核心概念
- **Target Shape**: 硬件原生支持的向量形状
- **Shape Ratio**: 原始形状与目标形状的比率
- **Traversal Order**: 展开的遍历顺序

#### 展开策略
通过`VectorUnrollOpInterface`确定可展开操作,计算切片偏移,生成多个小向量操作。

#### 13.3 实例演示

```mlir
// 前: 大向量操作 (目标硬件只支持vector<4xf32>)
%result = arith.addf %a, %b : vector<16xf32>

// 后: 展开为4个操作
%s0_a = vector.extract_strided_slice %a {offsets=[0], sizes=[4], strides=[1]}
%s0_b = vector.extract_strided_slice %b {offsets=[0], sizes=[4], strides=[1]}
%s0_r = arith.addf %s0_a, %s0_b : vector<4xf32>

%s1_a = vector.extract_strided_slice %a {offsets=[4], sizes=[4], strides=[1]}
%s1_b = vector.extract_strided_slice %b {offsets=[4], sizes=[4], strides=[1]}
%s1_r = arith.addf %s1_a, %s1_b : vector<4xf32>

// ... 类似处理 [8:12], [12:16]

%result = vector.insert_strided_slice %s0_r, %init {offsets=[0], strides=[1]}
%result = vector.insert_strided_slice %s1_r, %result {offsets=[4], strides=[1]}
// ...
```

**效果**: 适配硬件向量宽度,提高执行效率

---

### 14. Vector Distribute (向量分布)

**文件**: `VectorDistribute.cpp`

#### 14.1 作用
将向量操作分布到并行执行单元(如GPU warp的不同lane)。

#### 14.2 技术原理

#### 核心概念
- **Warp Execution**: GPU warp内的并行执行模型
- **Distribution Map**: 隐式的维度到线程的映射
- **Sequential vs Distributed**: 区分顺序维度和分布维度

#### 分布策略
1. 识别分布维度
2. 创建per-thread的切片
3. 使用临时buffer在parallel/sequential边界传递数据

#### 14.3 实例演示

```mlir
// 前: warp-level操作
%result = gpu.warp_execute_on_lane_0(%laneid) -> (vector<1x16xf32>) {
  %full = arith.addf %a, %b : vector<32x16xf32>
  gpu.yield %full : vector<32x16xf32>
}

// 后: 分布到各个lane
// 隐式map: (d0, d1) -> (d0), 维度0分布到32个lane
%tid_a = vector.extract_strided_slice %a[%laneid*1] : vector<1x16xf32>
%tid_b = vector.extract_strided_slice %b[%laneid*1] : vector<1x16xf32>
%result = arith.addf %tid_a, %tid_b : vector<1x16xf32>
```

**效果**: 充分利用GPU硬件并行性

---

### 15. Vector Linearize (向量线性化)

**文件**: `VectorLinearize.cpp`

#### 15.1 作用
将多维向量线性化为1D向量。

#### 15.2 技术原理

使用TypeConverter将所有向量类型转换为1D,并相应调整操作。

#### 15.3 实例演示

```mlir
// 前
%c = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
%result = arith.addi %a, %c : vector<2x2xi32>

// 后
%c_linear = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
%a_linear = vector.shape_cast %a : vector<2x2xi32> to vector<4xi32>
%result_linear = arith.addi %a_linear, %c_linear : vector<4xi32>
%result = vector.shape_cast %result_linear : vector<4xi32> to vector<2x2xi32>
```

**效果**: 简化后续lowering,统一处理维度

---

### 16. Vector Emulate Narrow Type (窄类型模拟)

**文件**: `VectorEmulateNarrowType.cpp`

#### 16.1 作用
使用宽类型模拟硬件不支持的窄整数类型(如i4用i8模拟)。

#### 16.2 技术原理

#### 模拟策略
1. **Container Type**: 使用更宽的容器类型
2. **Packing**: 多个窄元素打包到一个宽元素
3. **Mask Compression**: 压缩掩码以匹配容器数量

例如: 8个i4元素可以打包到2个i16元素中。

#### 16.3 实例演示

```mlir
// 前: 硬件不支持i4
%result = arith.addi %a, %b : vector<8xi4>

// 后: 使用i8模拟(每个i8容纳2个i4)
// 转换类型
%a_i8 = <pack> %a : vector<8xi4> to vector<4xi8>
%b_i8 = <pack> %b : vector<8xi4> to vector<4xi8>

// 在i8上操作(需要mask和shift)
%result_i8 = <emulated_add> %a_i8, %b_i8 : vector<4xi8>

// 转换回来
%result = <unpack> %result_i8 : vector<4xi8> to vector<8xi4>
```

**效果**: 支持任意位宽的整数向量

---

### 17. Vector Drop Lead Unit Dim (丢弃前导单元维度)

**文件**: `VectorDropLeadUnitDim.cpp`

#### 17.1 作用
消除向量操作的前导单元维度(大小为1的维度)。

#### 17.2 技术原理

提取前导维度,对降维后的向量执行操作,再broadcast回原始形状。

#### 17.3 实例演示

```mlir
// 前
%result = arith.addf %a, %b : vector<1x1x8xf32>

// 后
%a_drop = vector.extract %a[0, 0] : vector<8xf32>
%b_drop = vector.extract %b[0, 0] : vector<8xf32>
%r_drop = arith.addf %a_drop, %b_drop : vector<8xf32>
%tmp = vector.broadcast %r_drop : vector<8xf32> to vector<1x8xf32>
%result = vector.broadcast %tmp : vector<1x8xf32> to vector<1x1x8xf32>
```

**效果**: 减少不必要的维度处理

---

### 18. Vector Transfer Op Transforms (传输操作优化)

**文件**: `VectorTransferOpTransforms.cpp`

#### 18.1 作用
优化`vector.transfer_read`和`vector.transfer_write`操作。

#### 18.2 技术原理

#### 优化模式
1. **Dead Store Elimination**: 消除无用的写操作
2. **Store-to-Load Forwarding**: 直接传递store的值到load

使用DominanceInfo和AliasAnalysis分析。

#### 18.3 实例演示

**Store-to-Load Forwarding**:
```mlir
// 前
vector.transfer_write %v, %mem[%i] : vector<4xf32>, memref<100xf32>
%loaded = vector.transfer_read %mem[%i] : memref<100xf32>, vector<4xf32>

// 后: 直接使用%v
// vector.transfer_write仍然保留
vector.transfer_write %v, %mem[%i] : vector<4xf32>, memref<100xf32>
%loaded = %v  // 直接替换
```

---

### 19. Vector Transfer Split (传输分割)

**文件**: `VectorTransferSplitRewritePatterns.cpp`

#### 19.1 作用
将transfer操作分割为in-bounds快速路径和out-of-bounds慢速路径。

#### 19.2 技术原理

检查访问边界,对于可能越界的访问,使用条件执行。

#### 19.3 实例演示

```mlir
// 前: 可能越界的读取
%v = vector.transfer_read %mem[%i], %pad
  : memref<?xf32>, vector<8xf32>

// 后: 分割为快速/慢速路径
%is_inbounds = arith.cmpi sle, %i+8, %memsize
%v = scf.if %is_inbounds -> vector<8xf32> {
  // 快速路径: 直接读取
  %fast = vector.transfer_read %mem[%i] {in_bounds = [true]}
  scf.yield %fast
} else {
  // 慢速路径: 使用临时buffer处理padding
  %tmp = memref.alloca() : memref<8xf32>
  <copy with padding>
  %slow = vector.load %tmp[0]
  scf.yield %slow
}
```

**效果**: 快速路径无边界检查,提高性能

---

### 20. Vector Mask Elimination (掩码消除)

**文件**: `VectorMaskElimination.cpp`

#### 20.1 作用
消除可证明为全true的掩码。

#### 20.2 技术原理

使用界限分析判断`create_mask`的所有元素是否必为true。

#### 20.3 实例演示

```mlir
// 前
%c8 = arith.constant 8 : index
%mask = vector.create_mask %c8 : vector<8xi1>
// 已知: 掩码大小固定为8,create_mask参数也是8

// 后: 优化为常量
%mask = arith.constant dense<true> : vector<8xi1>
```

---

### 21. Vector Insert/Extract Strided Slice Rewrite

**文件**: `VectorInsertExtractStridedSliceRewritePatterns.cpp`

#### 21.1 作用
降级`vector.insert_strided_slice`和`vector.extract_strided_slice`。

#### 21.2 技术原理

#### 策略
1. **Different Rank**: 递归decompose
2. **Same Rank, All Strides=1**: 转换为shuffle

#### 21.3 实例演示

```mlir
// extract_strided_slice with strides=1
// 前
%slice = vector.extract_strided_slice %v
  {offsets = [2], sizes = [4], strides = [1]}
  : vector<8xf32> to vector<4xf32>

// 后: 转换为shuffle
%slice = vector.shuffle %v, %v [2, 3, 4, 5]
  : vector<8xf32>, vector<8xf32>
```

---

### 22. Vector Emulate Masked Load/Store (掩码加载/存储模拟)

**文件**: `VectorEmulateMaskedLoadStore.cpp`

#### 22.1 作用
模拟硬件不支持的masked load/store操作。

#### 22.2 技术原理

标量化为条件load/store循环。

#### 22.3 实例演示

```mlir
// 前
%v = vector.maskedload %mem[%base], %mask, %passthru
  : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>

// 后
%result = %passthru
affine.for %i = 0 to 4 {
  %m = vector.extract %mask[%i] : i1
  %elem = scf.if %m -> f32 {
    %idx = arith.addi %base, %i
    %loaded = memref.load %mem[%idx]
    scf.yield %loaded
  } else {
    %pass = vector.extract %passthru[%i]
    scf.yield %pass
  }
  %result = vector.insert %elem, %result[%i]
}
```

---

### 23. Lower Vector To/From Elements to Shuffle Tree

**文件**: `LowerVectorToFromElementsToShuffleTree.cpp`

#### 23.1 作用
将`vector.from_elements`降级为平衡的shuffle树。

#### 23.2 技术原理

构建二叉树结构的shuffle操作,减少延迟。

#### 23.3 实例演示

```mlir
// 前
%v = vector.from_elements %e0, %e1, %e2, %e3 : vector<4xf32>

// 后: 二叉树shuffle
%s0 = vector.splat %e0 : vector<4xf32>
%s1 = vector.splat %e1 : vector<4xf32>
%s2 = vector.splat %e2 : vector<4xf32>
%s3 = vector.splat %e3 : vector<4xf32>

// Level 1
%l1_0 = vector.shuffle %s0, %s1 [0, 5, 2, 3] // [e0, e1, ?, ?]
%l1_1 = vector.shuffle %s2, %s3 [0, 1, 6, 7] // [?, ?, e2, e3]

// Level 2
%v = vector.shuffle %l1_0, %l1_1 [0, 1, 6, 7] // [e0, e1, e2, e3]
```

---

### 24. Vector Transforms (通用向量变换)

**文件**: `VectorTransforms.cpp`

#### 24.1 作用
提供通用的向量到向量转换模式。

#### 24.2 关键模式

##### MultiReduce to Contract
将`mul + multi_reduction`转换为`vector.contract`

```mlir
// 前
%prod = arith.mulf %a, %b : vector<8x16xf32>
%result = vector.multi_reduction <add>, %prod [1]
  : vector<8x16xf32> to vector<8xf32>

// 后
%result = vector.contract {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0)>
  ],
  iterator_types = ["parallel", "reduction"]
} %a, %b, %acc : vector<8x16xf32>, vector<8x16xf32> into vector<8xf32>
```

##### Combine Contract with Transpose
消除contract操作数的transpose

```mlir
// 前
%t = vector.transpose %a, [1, 0] : vector<16x8xf32> to vector<8x16xf32>
%c = vector.contract {map_a, map_b, map_c} %t, %b, %acc

// 后: 直接调整indexing map
%c = vector.contract {adjusted_map_a, map_b, map_c} %a, %b, %acc
```

---

## 接口实现

### 25. Bufferizable Op Interface

**文件**: `BufferizableOpInterfaceImpl.cpp`

#### 作用
实现vector.transfer操作的bufferization接口,支持从tensor到memref的转换。

```mlir
// 前: tensor形式
%result = vector.transfer_read %tensor[%i] : tensor<?xf32>, vector<4xf32>

// 后: memref形式
%mem = bufferization.to_memref %tensor
%result = vector.transfer_read %mem[%i] : memref<?xf32>, vector<4xf32>
```

---

### 26. Subset Op Interface

**文件**: `SubsetOpInterfaceImpl.cpp`

#### 作用
为vector.transfer操作提供subset操作接口,用于并行化和fusion分析。

---

## 总结与最佳实践

### Vector Lowering Pipeline推荐顺序

```
1. VectorTransforms           → 合并高级模式(contract融合等)
2. VectorUnroll               → 展开到目标大小
3. VectorLinearize (可选)     → 线性化多维向量
4. LowerVectorContract        → 降级contract
5. LowerVectorMultiReduction  → 降级多维归约
6. LowerVectorTranspose       → 降级转置
7. LowerVectorBroadcast       → 降级广播
8. LowerVectorGather          → 降级gather
9. LowerVectorMask            → 降级掩码
10. LowerVectorTransfer       → 优化传输操作
11. VectorTransferSplit       → 分割边界路径
12. LowerVectorShapeCast      → 降级形状转换
13. VectorEmulateNarrowType   → 模拟窄类型(如需要)
14. VectorEmulateMaskedLoadStore → 模拟masked操作(如需要)
```

### 性能优化建议

#### 1. 目标硬件适配
- 使用`VectorUnroll`匹配硬件向量寄存器大小
- 启用`VectorTranspose::Shuffle16x16`以利用AVX512
- 针对GPU使用`VectorDistribute`

#### 2. 内存访问优化
- 使用`VectorTransferSplit`分离快慢路径
- 启用`VectorTransferOpTransforms`进行store-to-load forwarding
- 考虑`VectorLinearize`以简化内存布局

#### 3. 掩码优化
- 启用`VectorMaskElimination`消除冗余掩码
- 使用`VectorEmulateNarrowType`的掩码压缩

#### 4. 特殊化策略
- 对于矩阵乘法,优先使用`vector.contract`
- 对于规则访问,避免`vector.gather`,使用`vector.transfer_read`
- 对于小向量,考虑完全展开(scalarization)

### 适用场景

#### 高性能计算
- 使用contract表达GEMM
- 利用multi_reduction表达复杂归约
- 通过unroll匹配硬件SIMD宽度

#### 深度学习
- Conv2D → im2col + contract
- Pooling → multi_reduction
- Batch操作 → broadcast + elementwise

#### GPU编程
- 使用VectorDistribute实现warp-level编程
- 利用masked操作实现条件计算
- 通过transfer操作优化shared memory访问

### 常见陷阱

1. **过早展开**: 在高层优化前展开会错过融合机会
2. **忽略掩码成本**: masked操作可能比分支更慢
3. **不匹配的向量大小**: 未对齐硬件可能导致性能下降
4. **过度线性化**: 某些维度信息对优化有用

### 调试技巧

```bash
# 查看某个pass的效果
mlir-opt input.mlir -pass-pipeline='builtin.module(func.func(lower-vector-contract))'

# 组合多个pass
mlir-opt input.mlir \
  -lower-vector-contract \
  -lower-vector-transpose \
  --vector-transfer-to-scf

# 使用-debug-only查看详细信息
mlir-opt input.mlir -lower-vector-contract -debug-only=vector-contract-lowering
```

---

## 关键数据结构与API

### VectorUnrollOptions
```cpp
struct UnrollVectorOptions {
  // 目标向量形状
  std::function<std::optional<SmallVector<int64_t>>(Operation*)> nativeShape;

  // 过滤哪些操作可以展开
  std::function<LogicalResult(Operation*)> filterConstraint;

  // 遍历顺序
  std::function<std::optional<SmallVector<int64_t>>(Operation*)>
    traversalOrderCallback;
};
```

### VectorTransposeLowering
```cpp
enum class VectorTransposeLowering {
  EltWise = 0,        // 逐元素
  Shuffle1D,          // 1D shuffle
  Shuffle16x16,       // 16x16优化
};
```

### VectorContractLowering
```cpp
enum class VectorContractLowering {
  OuterProduct = 0,   // 外积
  MatmulOp,          // 矩阵乘
  Dot,               // 点积
  FlatTranspose,     // 平铺转置
};
```

---

## 扩展阅读

### 相关文档
- MLIR Vector Dialect: https://mlir.llvm.org/docs/Dialects/Vector/
- Codegen Strategy: https://mlir.llvm.org/docs/VectorOps/
- GPU Distribution: https://mlir.llvm.org/docs/Dialects/GPU/

### 学术论文
- Polyhedral compilation
- Halide scheduling
- TVM tensor expression

### 示例代码
查看MLIR测试用例:
- `mlir/test/Dialect/Vector/vector-contract-transforms.mlir`
- `mlir/test/Dialect/Vector/vector-transfer-flatten.mlir`
- `mlir/test/Integration/Dialect/Vector/` (端到端测试)

---

**文档版本**: LLVM 主干分支 (2026-01)
**维护者**: MLIR Vector Dialect团队
**许可证**: Apache 2.0 with LLVM Exception
