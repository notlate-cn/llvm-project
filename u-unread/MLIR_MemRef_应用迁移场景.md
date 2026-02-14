# MLIR MemRef 应用迁移场景

## 应用迁移场景

MemRef方言的设计遵循抽象分层和关注点分离原则，使其核心能力可以跨不同硬件平台和应用领域迁移。本章通过三个典型场景，展示MemRef的内存抽象、布局变换和多内存空间等特性如何应用于GPU计算、稀疏计算和分布式系统。

---

### 场景1：GPU内存管理迁移

#### 迁移背景

深度学习和高性能计算对GPU加速的需求日益增长。将CPU端的内存管理迁移到GPU需要处理**异构内存空间**和**显式同步**两个核心挑战。MemRef通过`memory_space`属性提供了原生支持。

#### 原始场景：CPU内存分配

```mlir
// CPU端标准的MemRef分配
%0 = memref.alloc() : memref<1024x1024xf32>
%1 = memref.subview %0[0, 0][512, 512][1, 1] : memref<1024x1024xf32> to memref<512x512xf32>
```

**特征**：
- 默认内存空间（`memory_space = 0`或未指定）
- 自动内存管理
- 隐式缓存一致性

#### 新场景：GPU设备内存管理

```mlir
// GPU Workgroup共享内存 - 所有线程可见
%workgroup_mem = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>

// GPU Private内存 - 单线程私有
%private_mem = memref.alloc() : memref<16xf32, #gpu.address_space<private>>

// GPU Global内存 - 跨kernel可见
%global_mem = memref.alloc() : memref<256x256xf32, #gpu.address_space<global>>

// 子视图操作保持一致
%sub = memref.subview %global_mem[0, 0][128, 128][1, 1]
  : memref<256x256xf32, #gpu.address_space<global>>
   to memref<128x128xf32, #gpu.address_space<global>>
```

#### 不变原理

1. **内存抽象统一性**：
   - MemRef类型系统完全兼容GPU内存空间
   - 子视图（`subview`）、广播（`broadcast`）、转置（`transpose`）等操作语义不变
   - 类型检查和形状推断机制保持一致

2. **布局表示能力**：
   - Affine Map布局描述同样适用于GPU内存
   - Strided layout可表示GPU coalesced access pattern

3. **访问操作接口**：
   - `memref.load` / `memref.store`操作保持不变
   - 索引计算逻辑与硬件无关

#### 需要修改的部分

1. **Memory Space属性扩展**：

```cpp
// GPU定义的地址空间枚举 (mlir/include/mlir/Dialect/GPU/IR/GPUBase.td)
def GPU_AddressSpaceGlobal : I32EnumAttrCase<"Global", 1, "global">;
def GPU_AddressSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 2, "workgroup">;
def GPU_AddressSpacePrivate : I32EnumAttrCase<"Private", 3, "private">;
```

2. **Lowering规则适配**：

```mlir
// CPU Lowering: 转换为LLVM alloca/malloc
// NVVM Lowering: 转换为地址空间3的LLVM IR
// ROCDL Lowering: 转换为AMDGPU local memory

// Example from mlir/test/Conversion/GPUCommon/memory-attrbution.mlir
// Workgroup memory lowering:
// NVVM: llvm.mlir.global internal @buffer() addr_space = 3
// ROCDL: llvm.mlir.global internal @buffer() addr_space = 3
```

3. **同步机制添加**：

```mlir
// 需要显式barrier同步（GPU方言提供）
gpu.barrier
memref.load %workgroup_mem[%idx] : memref<4xf32, #gpu.address_space<workgroup>>
```

#### WHY这样迁移

**架构优势**：
- MemRef的`memory_space`是一个**开放设计**，通过Attribute机制支持任意扩展
- GPU方言只需定义自己的AddressSpaceAttr，无需修改MemRef核心类型系统
- Lowering pass根据`memory_space`生成对应硬件指令，实现前端抽象与后端实现解耦

**工程收益**：
- 前端代码复用：相同的数据结构操作代码可同时生成CPU和GPU版本
- 类型安全保证：编译期检查内存空间一致性（如不能将workgroup memref传给global参数）
- 渐进式迁移：可先迁移核心计算到GPU，保持内存分配逻辑不变

---

### 场景2：稀疏张量表示迁移

#### 迁移背景

机器学习和科学计算中，稀疏数据普遍存在（如推荐系统矩阵、社交网络图）。传统稠密存储会浪费大量空间并引入无效计算。MemRef的布局抽象为稀疏格式提供了理论基础。

#### 原始场景：稠密MemRef存储

```mlir
// 稠密矩阵存储 - 1024x1024元素全部存储
%dense = memref.alloc() : memref<1024x1024xf32>

// 矩阵乘法，即使大部分为零也要计算
%result = linalg.matmul ins(%dense, %dense: memref<1024x1024xf32>)
                        outs(%output: memref<1024x1024xf32>)
```

**问题**：
- 存储开销：O(n²)无论稀疏度
- 计算浪费：对零元素进行无效乘加
- 内存带宽：搬运大量零数据

#### 新场景：稀疏数据结构存储

```mlir
// 稀疏张量类型定义（使用SparseTensorEncodingAttr）
#sparse_enc = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],  // CSR格式
  dimOrdering = affine_map<(i, j) -> (i, j)>,
  pointerBitWidth = 64,
  indexBitWidth = 32
}>

// 稀疏张量内存布局（由StorageLayout管理）
// 内存中实际存储：
//   memref<?xindex>  positions-0  ; 行指针数组
//   memref<?xindex>  coordinates-1; 列索引数组
//   memref<?xf32>    values       ; 非零值数组
%sparse_tensor = tensor.empty() : tensor<1024x1024xf32, #sparse_enc>

// 稀疏矩阵乘法 - 自动跳过零元素
%sparse_result = linalg.matmul
  ins(%sparse_tensor, %sparse_tensor: tensor<1024x1024xf32, #sparse_enc>)
  outs(%output: tensor<1024x1024xf32, #sparse_enc>)
```

#### 不变原理（可复用）

1. **Strided Layout概念**：

MemRef的Affine Map本质是坐标到线性地址的映射函数，这一概念直接对应稀疏格式的**间接寻址**：

```
// 稠密MemRef: 地址 = base + (row * stride_col + col) * elem_size
// 稀疏CSR:    地址 = values[positions[row] + idx]  // idx是列索引中的偏移
```

SparseTensorStorageLayout (定义在`mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorStorageLayout.h`) 复用了相同的多级索引思想：

```cpp
// SparseTensor存储由多个MemRef组成 (StorageLayout类的foreachField方法)
// For CSR format:
//   Field 0: positions memref<?xpos>  - 行偏移数组
//   Field 1: coordinates memref<?xcrd> - 列索引数组
//   Field 2: values memref<?xeltType>  - 值数组
```

2. **视图操作兼容性**：

- 子张量切片（slice）对应positions/coordinates的范围裁剪
- 维度置换（transpose）对应`dimOrdering` Affine Map的修改

3. **动态形状支持**：

- 稀疏存储的`?`尺寸（非零元数量）与MemRef动态维度语义一致
- 运行时sizes数组管理机制可复用

#### 需要扩展的部分

1. **编码属性定义**：

```mlir
// SparseTensorEncodingAttr包含稀疏格式元数据
#sparse_enc = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "singleton" ],
  dimOrdering = affine_map<(i, j, k) -> (i, k, j)>,  // 维度重排
  pointerBitWidth = 64,   // 位置数组位宽
  indexBitWidth = 32      // 坐标数组位宽
}>
```

2. **存储布局管理器**：

`StorageLayout`类（第114-154行）提供字段遍历和索引计算：

```cpp
class StorageLayout {
  void foreachField(
      llvm::function_ref<bool(FieldIndex, SparseTensorFieldKind,
                              Level, LevelType)> callback) const;

  // 获取特定字段的MemRef索引
  FieldIndex getMemRefFieldIndex(SparseTensorFieldKind kind,
                                 std::optional<Level> lvl) const;
};
```

3. **稀疏迭代器生成**：

```mlir
// 稀疏循环展开需要特殊的迭代器
// 遍历压缩维度的非零元素
scf.for %i = %pos[%row] to %pos[%row+1] {
  %col = %coordinates[%i]
  %val = %values[%i]
  // ...计算...
}
```

#### WHY MemRef适合扩展

**设计哲学匹配**：
- MemRef的**内存抽象**本身就是"坐标→地址"的映射函数
- 稀疏格式只是将这个函数从线性映射变成非线性（通过间接索引）
- 两者都是**数据布局**的表达方式，只是复杂度不同

**实现优势**：
- 底层存储仍使用MemRef：`memref<?xindex>`（positions）、`memref<?xf32>`（values）
- 稀疏方言只需描述**如何组合**这些MemRef，而非重新发明存储类型
- SparseTensor类型继承TensorType，但内部依赖MemRef实现实际内存操作

**生态系统兼容**：
- Linalg方言可同时处理稠密和稀疏张量（通过encoding属性区分）
- 缓冲化（Bufferization）流程统一处理稀疏tensor到memref的转换

---

### 场景3：分布式内存系统迁移

#### 迁移背景

大规模并行计算（如气候模拟、分子动力学）需要跨节点分布式内存。MemRef的`memory_space`概念可扩展表示**数据分布策略**，但需要新的同步和通信原语。

#### 原始场景：单进程内存

```mlir
// 单节点内存分配
%local = memref.alloc() : memref<8192x8192xf64>

// 全局矩阵乘法
linalg.matmul ins(%local, %local: memref<8192x8192xf64>)
              outs(%result: memref<8192x8192xf64>)
```

**限制**：
- 单机内存容量限制问题规模
- 无法利用多节点计算资源
- 无通信机制

#### 新场景：分布式内存

```mlir
// 定义分布式内存空间属性
#dist_space = #distributed.address_space<rank=4, layout="block_cyclic">

// 分布式MemRef - 逻辑上是一个大矩阵，物理分布在4个节点
%distributed = memref.alloc() : memref<32768x32768xf64, #dist_space>

// 每个节点持有本地块
%local_block = distributed.get_local_block %distributed
  : memref<32768x32768xf64, #dist_space> -> memref<8192x8192xf64>

// 分布式操作需要通信
%dist_result = distributed.matmul
  ins(%distributed, %distributed: memref<32768x32768xf64, #dist_space>)
  outs(%output: memref<32768x32768xf64, #dist_space>)
  {communication = "halo_exchange", overlap = true}
```

#### 可复用的原理

1. **Memory Space抽象**：
   - 分布式节点可视为一种新的内存空间（类似GPU workgroup的扩展）
   - `#dist_space`属性描述rank数量、块大小、分布策略

2. **视图操作语义**：
   - `get_local_block`本质是带分布语义的`subview`
   - Halo区域对应带边界的扩展视图

3. **布局描述能力**：
   - Block-cyclic分布可用Affine Map描述：`(i, j) -> (rank, local_i, local_j)`

#### 需要的新原语

1. **数据分布原语**：

```mlir
// 分布策略描述
#block_cyclic = #distributed.layout<{
  block_sizes = [256, 256],
  cyclic_order = [0, 1],   // 先沿维度0循环，再沿维度1
  ranks = [0, 1, 2, 3]     // 4个rank
}>
```

2. **通信原语**：

```mlir
// Halo交换
distributed.halo_exchange %local_block[%radius, %radius]
  : memref<8192x8192xf64> -> memref<8448x8448xf64>  // 扩展边界

// 全局归约
%global_sum = distributed.all_reduce %local_sum, "sum" : f64
```

3. **同步屏障**：

```mlir
// 跨节点屏障
distributed.barrier

// 点对点通信
distributed.send %data to %rank : memref<1024xf64>
%recv = distributed.recv from %rank : memref<1024xf64>
```

#### WHY需要新原语

**本质区别**：
- GPU memory space是**同一地址空间内的不同区域**（通过地址空间ID区分）
- 分布式内存是**不同地址空间**（每个节点独立的物理内存）

**挑战**：
1. **通信开销**：分布式操作需要显式数据移动，不是简单的内存访问
2. **延迟隐藏**：需要overlap通信与计算
3. **容错**：节点失败需要恢复机制

**MemRef的可扩展性**：
- 类型系统层面：通过自定义`memory_space`属性支持
- 操作层面：分布式方言可定义新操作（如`distributed.matmul`）
- Lowering层面：生成MPI、NCCL或GASNet等通信库调用

**设计启示**：
MemRef提供了良好的**抽象边界**：类型系统处理分布描述，操作语义处理通信逻辑，Lowering处理具体实现。这使得分布式扩展不必修改核心MemRef定义，保持方言独立性。

---

## 总结

三个迁移场景展示了MemRef设计的核心优势：

| 维度 | GPU迁移 | 稀疏迁移 | 分布式迁移 |
|------|---------|----------|-----------|
| **不变原理** | 内存抽象、视图操作 | Strided layout概念、动态形状 | Memory space抽象、布局描述 |
| **修改部分** | Memory space属性、同步机制 | 编码属性、存储布局管理器 | 通信原语、分布策略 |
| **扩展机制** | AddressSpaceAttr | SparseTensorEncodingAttr | DistributedSpaceAttr |
| **核心收益** | 异构计算支持 | 稀疏性利用 | 横向扩展能力 |

**设计哲学**：MemRef通过**属性化扩展**而非继承扩展，实现了核心稳定性和应用灵活性的平衡。这种设计使得不同领域的方言可以共享类型系统，同时保留各自的领域特定优化。
