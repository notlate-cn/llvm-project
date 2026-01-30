# LLVM MLIR SparseTensor方言Transform技术详解

本文档详细梳理LLVM MLIR中SparseTensor方言的所有Transform Pass的作用、技术原理和应用场景。

**目录路径**: `mlir/lib/Dialect/SparseTensor/Transforms/`

---

## 目录

1. [核心编译Pipeline](#核心编译Pipeline)
2. [代码生成策略](#代码生成策略)
3. [优化与重写](#优化与重写)
4. [特殊硬件支持](#特殊硬件支持)
5. [辅助转换](#辅助转换)
6. [完整编译流程](#完整编译流程)

---

## 稀疏张量基础

### 稀疏存储格式

SparseTensor方言支持多种稀疏存储格式:

- **Compressed (CSR/CSC)**: 压缩稀疏行/列
- **Singleton**: 每个坐标只有一个元素
- **Loose Compressed**: 宽松压缩格式
- **N-out-of-M**: 结构化稀疏(如2:4稀疏)
- **Dense**: 密集维度
- **COO**: 坐标格式(Coordinate format)

### 编码属性

```mlir
#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

%tensor = tensor<100x200xf32, #SparseMatrix>
```

---

## 核心编译Pipeline

### 1. Sparsification (稀疏化核心Pass)

**文件**: `Sparsification.cpp` (1482行)

#### 1.1 作用
**核心编译Pass**,将带有稀疏张量注解的高级Linalg操作转换为优化的稀疏循环代码。

#### 1.2 技术原理

##### Merger Lattice算法
构建格子(lattice)表示所有可能的稀疏/密集迭代组合:

```
例如: A[i,j] = B[i,j] + C[i,j]
- B稀疏(compressed), C密集
- Lattice包含: {B稀疏 ∧ C密集}, {B稀疏}, {C密集}, {空}
```

##### 循环合成策略
1. **仿射表达式分析**:
   - 直接支持: `d0`, `d1` (简单维度索引)
   - 索引归约: 处理复杂表达式如 `d0+d1`, `d0*2`

2. **循环顺序优化**:
   ```cpp
   // 选择最优遍历顺序
   // 优先: 稀疏外层 > 密集外层 > 混合
   ```

3. **协同迭代(Co-iteration)**:
   ```mlir
   // 两个稀疏张量的同步遍历
   scf.while (%pos_a, %pos_b, ...) {
     %idx_a = memref.load %crd_a[%pos_a]
     %idx_b = memref.load %crd_b[%pos_b]
     %cmp = arith.cmpi slt, %idx_a, %idx_b
     scf.if %cmp {
       // 处理A的元素
     } else {
       scf.if %eq {
         // 处理A和B的相同索引
       } else {
         // 处理B的元素
       }
     }
   }
   ```

##### 张量表达式树
- **Hoisting**: 提升循环不变量
- **Reduction识别**: 检测sum, min, max等归约
- **半环操作**: 支持自定义归约(加法环, 乘法环)

#### 1.3 实例演示

**输入**: 稀疏矩阵向量乘法
```mlir
func.func @matvec(%A: tensor<100x200xf32, #CSR>,
                  %x: tensor<200xf32>,
                  %y: tensor<100xf32>) -> tensor<100xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,j) -> (i,j)>,  // A
      affine_map<(i,j) -> (j)>,    // x
      affine_map<(i,j) -> (i)>     // y
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%A, %x : tensor<100x200xf32, #CSR>, tensor<200xf32>)
    outs(%y : tensor<100xf32>) {
    ^bb0(%a: f32, %x_val: f32, %y_val: f32):
      %mul = arith.mulf %a, %x_val : f32
      %add = arith.addf %y_val, %mul : f32
      linalg.yield %add : f32
  } -> tensor<100xf32>
  return %result : tensor<100xf32>
}
```

**输出**: 优化的稀疏循环
```mlir
func.func @matvec(%pos: memref<?xindex>, %crd: memref<?xindex>,
                  %val: memref<?xf32>, %x: memref<200xf32>,
                  %y: memref<100xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = memref.dim %pos, %c0 : memref<?xindex>

  // 外层循环: 遍历行
  scf.for %i = %c0 to %rows step %c1 {
    %row_start = memref.load %pos[%i] : memref<?xindex>
    %row_end = memref.load %pos[%i+1] : memref<?xindex>

    // 初始化累加器
    %y_init = memref.load %y[%i] : memref<100xf32>
    %sum = scf.for %k = %row_start to %row_end step %c1
                iter_args(%acc = %y_init) -> f32 {
      %j = memref.load %crd[%k] : memref<?xindex>
      %a_val = memref.load %val[%k] : memref<?xf32>
      %x_val = memref.load %x[%j] : memref<200xf32>
      %prod = arith.mulf %a_val, %x_val : f32
      %new_acc = arith.addf %acc, %prod : f32
      scf.yield %new_acc : f32
    }
    memref.store %sum, %y[%i] : memref<100xf32>
  }
}
```

**效果**:
- 避免遍历零元素
- 优化内存访问模式
- 生成高效的嵌套循环

#### 1.4 高级特性

##### 张量扩展(Tensor Expansion)
处理稀疏输出的随机写入:

```mlir
// 扩展阶段: 创建临时密集缓冲区
%expanded = memref.alloc() : memref<200xf32>
%filled = memref.alloc() : memref<200xi1>

// 计算并填充
<sparse computation writing to %expanded>

// 压缩阶段: 提取非零元素
%nnz = <count non-zeros in %expanded>
%result_val = memref.alloc(%nnz)
%result_crd = memref.alloc(%nnz)
<compress %expanded into result arrays>
```

##### 并行化支持
```mlir
// 配置并行策略
sparse_tensor.parallel_strategy = "dense_outer_loop"

// 生成OpenMP风格并行循环
scf.parallel (%i) = (%c0) to (%rows) step (%c1) {
  <行级并行计算>
}
```

---

## 代码生成策略

### 2. SparseTensor Codegen (直接代码生成)

**文件**: `SparseTensorCodegen.cpp` (1632行)

#### 2.1 作用
将稀疏张量原语直接lowering为编译器可见的buffer操作,**不依赖运行时库**。

#### 2.2 技术原理

##### SparseTensorDescriptor模式
使用描述符管理多个buffer:

```cpp
class SparseTensorDescriptor {
  SmallVector<Value> positionBuffers;  // 位置数组
  SmallVector<Value> coordinateBuffers; // 坐标数组
  SmallVector<Value> valueBuffer;       // 值数组
  Value sizes;                          // 大小元数据
};
```

##### 类型Lowering映射

**CSR格式示例**:
```mlir
// 高级类型
tensor<100x200xf32, #CSR>

// Lowering后
!llvm.struct<(
  memref<?xindex>,  // row_pos: [0, 3, 5, ...]
  memref<?xindex>,  // col_idx: [1, 4, 7, 2, 6, ...]
  memref<?xf32>,    // values:  [1.5, 2.3, ...]
  memref<2xindex>   // sizes: [100, 200]
)>
```

##### 操作Lowering

**Load操作**:
```mlir
// 前
%val = sparse_tensor.values %tensor : tensor<100x200xf32, #CSR>

// 后
%desc = <get descriptor>
%val = <extract value buffer from desc>
```

**Position/Coordinate访问**:
```mlir
// 前
%pos = sparse_tensor.positions %tensor {level = 0}

// 后
%desc = <get descriptor>
%pos = <extract position buffer[0] from desc>
```

**动态插入(Push-back)**:
```mlir
// 前
sparse_tensor.push_back %tensor, %idx, %val

// 后
%size = memref.load %sizes[%level]
%new_size = arith.addi %size, %c1
memref.store %idx, %crd[%size]
memref.store %val, %values[%size]
memref.store %new_size, %sizes[%level]
// 可能需要realloc逻辑
```

#### 2.3 实例演示

**COO格式构建**:
```mlir
// 输入: 构建稀疏张量
%tensor = sparse_tensor.empty() : tensor<10x10xf32, #COO>
%t1 = sparse_tensor.insert %val1 into %tensor[%i1, %j1]
%t2 = sparse_tensor.insert %val2 into %t1[%i2, %j2]
%result = sparse_tensor.load %t2 hasInserts

// Lowering后
// 分配buffers
%crd_i = memref.alloc() : memref<?xindex>
%crd_j = memref.alloc() : memref<?xindex>
%values = memref.alloc() : memref<?xf32>
%nnz = memref.alloc() : memref<1xindex>

// 插入元素
%n0 = memref.load %nnz[0]
memref.store %i1, %crd_i[%n0]
memref.store %j1, %crd_j[%n0]
memref.store %val1, %values[%n0]
%n1 = arith.addi %n0, %c1
memref.store %n1, %nnz[0]

// 继续插入第二个元素...

// 最终排序(如果需要)
call @sort_coo(%crd_i, %crd_j, %values, %nnz)
```

#### 2.4 优势

- **内联优化**: LLVM可以看到buffer操作
- **死代码消除**: 未使用的buffer可被优化掉
- **别名分析**: 更好的内存优化
- **循环优化**: 支持循环融合、向量化

---

### 3. SparseTensor Conversion (运行时库转换)

**文件**: `SparseTensorConversion.cpp` (928行)

#### 3.1 作用
通过**不透明指针**和运行时库调用实现稀疏张量操作,实现简单但优化受限。

#### 3.2 技术原理

##### 不透明指针映射
```mlir
// 稀疏张量类型
tensor<100x200xf32, #CSR>

// 转换为不透明指针
!llvm.ptr  // 指向运行时管理的稀疏结构
```

##### 运行时函数调用
```cpp
// 声明外部运行时函数
llvm.func @sparseLvlSize(!llvm.ptr, index) -> index
llvm.func @sparseDimSize(!llvm.ptr, index) -> index
llvm.func @sparseInsert(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
llvm.func @sparseNewCOO(!llvm.ptr) -> !llvm.ptr
```

#### 3.3 实例演示

```mlir
// 前: 查询稀疏张量大小
%size = tensor.dim %sparse, %c0 : tensor<?x?xf32, #CSR>

// 后: 运行时库调用
%ptr = <convert tensor to opaque ptr>
%size = llvm.call @sparseDimSize(%ptr, %c0) : (!llvm.ptr, index) -> index
```

#### 3.4 对比

| 特性 | Codegen | Conversion |
|------|---------|------------|
| 实现方式 | 直接IR生成 | 运行时库调用 |
| 优化潜力 | 高 | 低 |
| 编译时间 | 长 | 短 |
| 二进制大小 | 大 | 小 |
| 依赖 | 无 | 需要运行时库 |
| 调试难度 | 中 | 易 |

---

## 优化与重写

### 4. SparseTensor Rewriting (稀疏操作重写)

**文件**: `SparseTensorRewriting.cpp` (1597行)

#### 4.1 作用
应用模式匹配优化稀疏张量操作。

#### 4.2 关键优化模式

##### 零初始化消除
```mlir
// 前: 不必要的零初始化
%empty = sparse_tensor.empty() : tensor<100x100xf32, #CSR>
%zero = linalg.fill ins(%cst_0) outs(%empty)
%result = <compute into %zero>

// 后: 直接使用empty
%result = <compute into %empty>
```

##### Sampling模式识别
```mlir
// 前: Element-wise乘法(sampling)
%C = linalg.generic {map_A, map_B, map_C}
  ins(%A, %B) outs(%C) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b
    linalg.yield %mul
}

// 后: 标记为sampling操作
%C = linalg.generic {sparse_tensor.sampling = unit}
  ins(%A, %B) outs(%C) { ... }
// 可以应用特殊优化
```

##### 三角求解模式
```mlir
// 识别稀疏三角矩阵求解
// L * x = b (L为下三角)
pattern: SpTriSolvePattern
-> 生成优化的前向/后向替代代码
```

##### 拼接优化
```mlir
// 前: 通用拼接
%concat = sparse_tensor.concatenate %A, %B {dimension = 0}

// 后: 如果A, B已排序,优化为merge操作
%concat = <merge sorted %A and %B>
```

#### 4.3 实例演示

**Sum-of-Products模式**:
```mlir
// 检测并优化: D = A * B + C * E
linalg.generic {
  ^bb0(%a, %b, %c, %e, %d):
    %mul1 = arith.mulf %a, %b
    %mul2 = arith.mulf %c, %e
    %sum = arith.addf %mul1, %mul2
    linalg.yield %sum
}

// 识别为两个contraction的和
// -> 可以分解为两个独立的稀疏乘法
%ab = sparse_matmul(%A, %B)
%ce = sparse_matmul(%C, %E)
%result = sparse_add(%ab, %ce)
```

---

### 5. SparseBuffer Rewriting (Buffer重写)

**文件**: `SparseBufferRewriting.cpp` (1430行)

#### 5.1 作用
生成高效的排序和缓冲区管理函数。

#### 5.2 技术原理

##### 排序函数生成

**Name Mangling**:
```cpp
// 生成唯一函数名
// _sparse_sort_{perm}_{type1}_{type2}_...
// 例如: _sparse_sort_perm_0_1_index_index_f32
```

**支持的排序算法**:
1. **Quick Sort**: 标准快排
2. **Hybrid Quick Sort**: 深度限制的快排+堆排
3. **Heap Sort**: 堆排序
4. **Stable Sort**: 稳定排序(归并)
5. **Partition**: 快排分区
6. **Binary Search**: 二分查找

##### COO多键排序

```mlir
// 对COO格式按(i, j)排序
func.func private @sort_coo_2d(
    %xi: memref<?xindex>,  // i坐标
    %xj: memref<?xindex>,  // j坐标
    %xv: memref<?xf32>,    // 值
    %n: index              // 元素数量
) {
  // 生成hybrid quicksort
  // 主键: i, 次键: j
  call @_sparse_hybrid_qsort_perm_0_1(
    %c0, %n, %xi, %xj, %xv
  )
}

// 实际生成的排序函数(简化版)
func.func private @_sparse_hybrid_qsort_perm_0_1(
    %lo: index, %hi: index,
    %x0: memref<?xindex>, %x1: memref<?xindex>,
    %values: memref<?xf32>
) {
  %depth_limit = <compute>
  call @qsort_body(%lo, %hi, %depth_limit, %x0, %x1, %values)
}
```

#### 5.3 实例演示

**生成排序调用**:
```mlir
// 输入: 未排序的COO数据
sparse_tensor.sort %nnz, %crd0, %crd1, %vals
  {perm_map = affine_map<(d0, d1) -> (d0, d1)>}

// 生成:
call @_sparse_sort_perm_0_1_index_index_f32(
  %nnz, %crd0, %crd1, %vals
)
```

---

### 6. Sparse Reinterpret Map (维度重解释)

**文件**: `SparseReinterpretMap.cpp` (803行)

#### 6.1 作用
消除不可直接表示的仿射表达式,通过引入辅助维度将复杂索引展开。

#### 6.2 技术原理

##### 不可容许表达式

**禁止的表达式**:
- `floordiv`: `d0 floordiv 4`
- `ceildiv`: `d0 ceildiv 4`
- `mod`: `d0 mod 4`

**为什么禁止**: 稀疏迭代器无法直接处理这些非单调映射。

##### 重解释策略

**示例**: 2D卷积的im2col变换
```mlir
// 原始: 4D -> 2D映射
// output[n, oh, ow, c] = input[n, oh+kh, ow+kw, c]
// 简化为1D例子: output[i] = input[i floordiv 4, i mod 4]

map = affine_map<(i) -> (i floordiv 4, i mod 4)>
// 不可容许!

// 重解释为:
map = affine_map<(i, j) -> (i, j)>
// 其中: i = original_i floordiv 4, j = original_i mod 4

// 添加额外循环变量j遍历[0, 4)
```

##### 算法步骤

1. **检测不可容许表达式**
2. **构造逆映射**: `level -> dims`
3. **提取辅助变量**: 例如从`d0 floordiv 4`和`d0 mod 4`提取`d0`
4. **拓扑排序**: 确定新循环顺序
5. **更新迭代器类型**: parallel/reduction标签

#### 6.3 实例演示

**Blocked格式**:
```mlir
// 前: 块稀疏 (8x8块)
#BlockCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 floordiv 8, d1 floordiv 8, d0 mod 8, d1 mod 8)
}>

// 重解释为4个独立维度
#BlockCSR_Reinterpreted = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0, d1, d2, d3)
  // d0: 块行索引
  // d1: 块列索引
  // d2: 块内行偏移[0,8)
  // d3: 块内列偏移[0,8)
}>

// 循环结构变化
// 前: for i, j in matrix
// 后: for block_i, block_j in blocks
//       for in_i, in_j in block
```

---

## 特殊硬件支持

### 7. SparseGPU Codegen (GPU代码生成)

**文件**: `SparseGPUCodegen.cpp` (1335行)

#### 7.1 作用
生成GPU内核和cuSparse库调用,支持稀疏张量的GPU加速。

#### 7.2 技术原理

##### GPU Module结构
```mlir
module {
  // Host代码
  func.func @host_function() {
    <host logic>
    gpu.launch_func @kernel::@kernel_func
  }

  // GPU Module
  gpu.module @kernel {
    gpu.func @kernel_func(...) kernel {
      <GPU kernel code>
      gpu.return
    }
  }
}
```

##### cuSparse格式检测

**支持的格式**:
- **COO**: Coordinate format
- **CSR**: Compressed Sparse Row
- **CSC**: Compressed Sparse Column
- **BSR**: Block Sparse Row

```cpp
bool isCuSparseFormat(SparseTensorType stt) {
  // 检查是否为2D
  // 检查level格式序列
  // COO: [compressed(unique), singleton(soa)]
  // CSR: [dense, compressed]
  // CSC: [compressed, dense]
  // BSR: [dense, compressed] + 块结构
}
```

##### 主机-设备数据传输

**异步传输模式**:
```mlir
// 1. 注册主机内存(pinned memory)
%token0 = gpu.wait async
%token1 = gpu.host_register %host_mem : memref<?xf32>

// 2. 分配设备内存
%dev_mem, %token2 = gpu.alloc async [%token1] : memref<?xf32>

// 3. 异步拷贝 Host -> Device
%token3 = gpu.memcpy async [%token2] %dev_mem, %host_mem

// 4. 启动kernel
%token4 = gpu.launch_func async [%token3] @kernel::@func
  blocks in (%bx, %by, %bz) threads in (%tx, %ty, %tz)
  args(%dev_mem : memref<?xf32>)

// 5. 拷贝回主机
%token5 = gpu.memcpy async [%token4] %host_mem, %dev_mem

// 6. 同步
gpu.wait [%token5]
```

#### 7.3 实例演示

**SpMV (Sparse Matrix-Vector Multiplication)**:
```mlir
// 输入: CSR格式矩阵乘法
func.func @spmv_csr(%A: tensor<1000x2000xf32, #CSR>,
                    %x: tensor<2000xf32>,
                    %y: tensor<1000xf32>) -> tensor<1000xf32> {
  %result = linalg.generic {
    indexing_maps = [map_A, map_x, map_y],
    iterator_types = ["parallel", "reduction"]
  } ins(%A, %x) outs(%y) { <matvec body> }
  return %result
}

// GPU Codegen输出
func.func @spmv_csr_gpu(...) {
  // 提取CSR结构
  %pos, %crd, %val = sparse_tensor.values %A

  // 注册和分配
  gpu.host_register %pos, %crd, %val, %x, %y
  %d_pos = gpu.alloc() : memref<?xindex>
  %d_crd = gpu.alloc() : memref<?xindex>
  %d_val = gpu.alloc() : memref<?xf32>
  %d_x = gpu.alloc() : memref<2000xf32>
  %d_y = gpu.alloc() : memref<1000xf32>

  // 拷贝到设备
  gpu.memcpy %d_pos, %pos
  gpu.memcpy %d_crd, %crd
  gpu.memcpy %d_val, %val
  gpu.memcpy %d_x, %x
  gpu.memcpy %d_y, %y

  // 调用cuSparse库(如果可用)
  // 或启动自定义kernel
  gpu.launch_func @spmv_kernel::@spmv
    blocks in (%blocks, %c1, %c1)
    threads in (%threads, %c1, %c1)
    args(%d_pos, %d_crd, %d_val, %d_x, %d_y)

  // 拷贝结果回主机
  gpu.memcpy %y, %d_y

  // 释放设备内存
  gpu.dealloc %d_pos, %d_crd, %d_val, %d_x, %d_y
}

// GPU Kernel
gpu.module @spmv_kernel {
  gpu.func @spmv(%pos: memref<?xindex>, %crd: memref<?xindex>,
                 %val: memref<?xf32>, %x: memref<2000xf32>,
                 %y: memref<1000xf32>) kernel {
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    %row = arith.addi %bid, %tid

    // 每个线程处理一行
    %start = memref.load %pos[%row]
    %end = memref.load %pos[%row + 1]

    %sum = scf.for %k = %start to %end step %c1 iter_args(%acc = %cst_0) {
      %col = memref.load %crd[%k]
      %a_val = memref.load %val[%k]
      %x_val = memref.load %x[%col]
      %prod = arith.mulf %a_val, %x_val
      %new_acc = arith.addf %acc, %prod
      scf.yield %new_acc
    }

    memref.store %sum, %y[%row]
    gpu.return
  }
}
```

#### 7.4 优化技巧

- **Warp-level优化**: 利用warp内线程协作
- **共享内存**: 缓存频繁访问的数据
- **Coalesced访问**: 合并全局内存访问
- **库调用优先**: 优先使用cuSparse/cuBLAS

---

### 8. Sparse Vectorization (稀疏向量化)

**文件**: `SparseVectorization.cpp` (690行)

#### 8.1 作用
将稀疏循环向量化,支持SIMD和可伸缩向量(SVE)。

#### 8.2 技术原理

##### 向量化配置
```cpp
struct VectorLength {
  unsigned vectorLength;  // SIMD宽度(如4, 8, 16)
  bool enableVLA;        // 可变长度向量(ARM SVE)
  bool enableSIMDIndex32; // 32位索引向量
};
```

##### 向量Mask生成

**处理trip count不整除**:
```mlir
// 原始标量循环: for i = 0 to 13
// 向量化(VL=4): for i = 0 to 12 step 4
//               + cleanup: for i = 12 to 13

// Mask方式(无cleanup loop):
%vl = arith.constant 4 : index
scf.for %i = %c0 to %c13 step %vl {
  %remaining = arith.subi %c13, %i  // 13 - i
  %is_partial = arith.cmpi slt, %remaining, %vl
  %active_lanes = arith.select %is_partial, %remaining, %vl

  %mask = vector.create_mask %active_lanes : vector<4xi1>
  // 使用mask进行条件向量化操作
}
```

##### Gather/Scatter向量化

**间接访问模式**:
```mlir
// 标量: for i, a[idx[i]]
scf.for %i = ... {
  %idx = memref.load %indices[%i]
  %val = memref.load %data[%idx]
}

// 向量化:
scf.for %i = ... step %vl {
  %idx_vec = vector.load %indices[%i] : memref<?xindex>, vector<4xindex>
  %val_vec = vector.gather %data[%base][%idx_vec], %mask, %passthru
    : memref<?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
}
```

##### 归约向量化

**水平归约**:
```mlir
// 标量归约
%sum = scf.for %i iter_args(%acc = %init) {
  %val = memref.load %data[%i]
  %new_acc = arith.addf %acc, %val
  scf.yield %new_acc
}

// 向量化归约
%vec_acc = arith.constant dense<0.0> : vector<4xf32>
%vec_acc_final = scf.for %i step %vl iter_args(%vacc = %vec_acc) {
  %vec = vector.load %data[%i] : vector<4xf32>
  %new_vacc = arith.addf %vacc, %vec : vector<4xf32>
  scf.yield %new_vacc
}
// 最终水平求和
%sum = vector.reduction <add>, %vec_acc_final : vector<4xf32> into f32
```

#### 8.3 实例演示

**稀疏矩阵向量乘法向量化**:
```mlir
// 输入: 已sparsify的CSR SpMV内层循环
scf.for %k = %row_start to %row_end step %c1 iter_args(%acc = %init) {
  %j = memref.load %crd[%k]
  %a_val = memref.load %val[%k]
  %x_val = memref.load %x[%j]    // 间接访问!
  %prod = arith.mulf %a_val, %x_val
  %new_acc = arith.addf %acc, %prod
  scf.yield %new_acc
}

// 向量化后(VL=4):
%vl = arith.constant 4 : index
%vec_zero = arith.constant dense<0.0> : vector<4xf32>
%vec_acc_init = arith.constant dense<0.0> : vector<4xf32>

%vec_acc_final = scf.for %k = %row_start to %row_end step %vl
                     iter_args(%vacc = %vec_acc_init) {
  // 生成动态mask
  %remaining = arith.subi %row_end, %k
  %trip = arith.minui %remaining, %vl
  %mask = vector.create_mask %trip : vector<4xi1>

  // 向量gather索引
  %j_vec = vector.load %crd[%k] : vector<4xindex>

  // 连续load值
  %a_vec = vector.load %val[%k] : vector<4xf32>

  // 向量gather x
  %x_vec = vector.gather %x[%c0][%j_vec], %mask, %vec_zero
    : memref<?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>

  // 向量化乘加
  %prod_vec = arith.mulf %a_vec, %x_vec : vector<4xf32>
  %new_vacc = arith.addf %vacc, %prod_vec : vector<4xf32>
  scf.yield %new_vacc
}

// 水平归约
%final_sum = vector.reduction <add>, %vec_acc_final : vector<4xf32> into f32
%result = arith.addf %init, %final_sum
```

**效果**:
- 理论加速: 4x (VL=4)
- 实际加速: 2-3x (考虑gather开销)

---

## 辅助转换

### 9. Sparse Iteration to SCF (迭代到控制流)

**文件**: `SparseIterationToScf.cpp` (459行)

#### 9.1 作用
将高级稀疏迭代器操作lowering为SCF控制流。

#### 9.2 技术原理

##### 迭代器类型
```mlir
// 高级: 稀疏迭代空间
!sparse_tensor.iterator<...>

// Lowering: 结构体
!llvm.struct<(
  index,  // position
  index,  // cursor
  index   // end
)>
```

##### CoIterate Lowering

**输入**: 多个稀疏张量的同步遍历
```mlir
sparse_tensor.coiterate (%iter_a, %iter_b) in (%space_a, %space_b) {
  ^case1(%idx_a):
    <只有A有元素>
  ^case2(%idx_b):
    <只有B有元素>
  ^case3(%idx_a, %idx_b):
    <A和B都有元素>
}
```

**输出**: 嵌套if结构
```mlir
scf.while (%pos_a, %pos_b) : (index, index) -> (index, index) {
  %valid_a = arith.cmpi slt, %pos_a, %end_a
  %valid_b = arith.cmpi slt, %pos_b, %end_b
  %continue = arith.ori %valid_a, %valid_b
  scf.condition(%continue) %pos_a, %pos_b
} do {
^bb0(%pos_a: index, %pos_b: index):
  %idx_a = memref.load %crd_a[%pos_a]
  %idx_b = memref.load %crd_b[%pos_b]

  scf.if %valid_a {
    scf.if %valid_b {
      %cmp = arith.cmpi slt, %idx_a, %idx_b
      scf.if %cmp {
        // case1: 只有A
        <process A>
        %new_pos_a = arith.addi %pos_a, %c1
        scf.yield %new_pos_a, %pos_b
      } else {
        %eq = arith.cmpi eq, %idx_a, %idx_b
        scf.if %eq {
          // case3: A和B都有
          <process A and B>
          %new_pos_a = arith.addi %pos_a, %c1
          %new_pos_b = arith.addi %pos_b, %c1
          scf.yield %new_pos_a, %new_pos_b
        } else {
          // case2: 只有B
          <process B>
          %new_pos_b = arith.addi %pos_b, %c1
          scf.yield %pos_a, %new_pos_b
        }
      }
    } else {
      // 只有A有效
      <process A>
    }
  } else {
    // 只有B有效
    <process B>
  }
  scf.yield %new_pos_a, %new_pos_b
}
```

---

### 10. Sparse Assembler (汇编器)

**文件**: `SparseAssembler.cpp` (251行)

#### 10.1 作用
提供稀疏张量与组成buffer之间的转换。

#### 10.2 技术原理

##### 分解(Disassemble)
```mlir
// 输入
func.func @compute(%tensor: tensor<100x200xf32, #CSR>) {
  <use %tensor>
}

// 转换
func.func @compute(%pos: memref<?xindex>,
                   %crd: memref<?xindex>,
                   %val: memref<?xf32>) {
  %tensor = sparse_tensor.assemble %pos, %crd, %val
  <use %tensor>
}
```

##### 组装(Assemble)
```mlir
// 返回稀疏张量
func.func @create() -> tensor<100x200xf32, #CSR> {
  %pos = <allocate and fill>
  %crd = <allocate and fill>
  %val = <allocate and fill>
  %tensor = sparse_tensor.assemble %pos, %crd, %val
  return %tensor
}

// 转换为返回tuple
func.func @create() -> (!llvm.struct<(
    memref<?xindex>, memref<?xindex>, memref<?xf32>
)>) {
  %pos = <allocate and fill>
  %crd = <allocate and fill>
  %val = <allocate and fill>
  %tuple = llvm.mlir.undef : !llvm.struct<...>
  %t1 = llvm.insertvalue %pos, %tuple[0]
  %t2 = llvm.insertvalue %crd, %t1[1]
  %t3 = llvm.insertvalue %val, %t2[2]
  return %t3
}
```

---

### 11. Storage Specifier to LLVM

**文件**: `SparseStorageSpecifierToLLVM.cpp` (358行)

#### 11.1 作用
将存储说明符(元数据)lowering为LLVM结构体。

#### 11.2 结构体布局

```mlir
// 高级
!sparse_tensor.storage_specifier<#CSR>

// LLVM结构体
!llvm.struct<(
  array<2 x i64>,  // level sizes: [rows, nnz]
  array<3 x i64>   // memory sizes: [pos_size, crd_size, val_size]
)>
```

#### 11.3 字段访问

```mlir
// 获取level size
%size = sparse_tensor.storage_specifier.get %spec[lvl_sz at 0]

// Lowering
%struct = <load spec>
%array = llvm.extractvalue %struct[0] : !llvm.struct<...>
%size = llvm.extractvalue %array[0] : !llvm.array<2 x i64>
```

---

### 12. Sparse Space Collapse (空间折叠)

**文件**: `SparseSpaceCollapse.cpp` (199行)

#### 12.1 作用
合并完美嵌套的稀疏迭代循环。

#### 12.2 示例

```mlir
// 前: 两层嵌套
sparse_tensor.iterate %iter1 in %space1 {
  sparse_tensor.iterate %iter2 in %space2 {
    <body>
  }
}

// 检查是否可折叠:
// 1. 完美嵌套
// 2. 来自同一张量的连续level
// 3. 无其他副作用

// 后: 单层迭代
sparse_tensor.iterate (%iter1, %iter2) in %collapsed_space {
  <body>
}
```

---

### 13. Stage Sparse Operations (操作分段)

**文件**: `StageSparseOperations.cpp` (73行)

#### 13.1 作用
为无序插入的稀疏操作插入排序阶段。

#### 13.2 示例

```mlir
// 前: 直接convert(可能无序)
%result = sparse_tensor.convert %input

// 后: 插入staging
%staged = sparse_tensor.stage %input
%sorted = sparse_tensor.sort %staged
%result = sparse_tensor.convert %sorted
```

---

## 完整编译流程

### Pipeline编排

**文件**: `SparseTensorPasses.cpp` (522行)

#### 推荐Pass顺序

```bash
# 完整稀疏张量编译流程

# 1. 前处理
--sparse-assembler                    # 提取组件
--sparse-reinterpret-map              # 消除复杂仿射

# 2. 核心稀疏化
--sparsification                      # 主要稀疏化pass
--sparse-tensor-rewrite               # 模式优化

# 3. 暂存和排序
--stage-sparse-operations             # 插入排序

# 4. 向量化(可选)
--sparse-vectorization="vl=8 enable-vla=false"

# 5. GPU(可选)
--sparse-gpu-codegen                  # GPU内核生成

# 6. Lowering到实现
# 选择一个:
# A. 直接代码生成(推荐)
--sparse-tensor-codegen

# B. 运行时库
--sparse-tensor-conversion

# 7. Buffer优化
--sparse-buffer-rewrite

# 8. 迭代器lowering
--lower-sparse-foreach-to-scf
--lower-sparse-iteration-to-scf

# 9. 存储说明符lowering
--sparse-storage-specifier-to-llvm

# 10. 空间折叠(可选)
--sparse-space-collapse

# 11. Bufferization
--one-shot-bufferize
--finalizing-bufferize

# 12. 标准lowering
--convert-scf-to-cf
--convert-vector-to-llvm
--convert-memref-to-llvm
--convert-func-to-llvm
--reconcile-unrealized-casts
```

### 集成Pipeline

**SparsificationAndBufferizationPass** (268行):
```cpp
// 统一处理稀疏和密集张量
pipeline {
  // 1. 预处理
  pre-sparsification-rewrite

  // 2. 转换empty → alloc_tensor
  empty-tensor-to-alloc-tensor

  // 3. One-Shot分析
  one-shot-bufferize {analysis-only}

  // 4. 稀疏化
  sparsification

  // 5. 通用bufferization
  one-shot-bufferize {bufferize-function-boundaries}

  // 6. 清理
  canonicalize
}
```

---

## 性能优化建议

### 1. 选择合适的存储格式

```mlir
// CSR: 适合行访问
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CSC: 适合列访问
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : dense)
}>

// COO: 适合随机插入
#COO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa))
}>

// Blocked: 适合块稀疏
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : dense, d1 : compressed, d2 : dense, d3 : dense)
}>
```

### 2. 启用向量化

```bash
# 标量版本
--sparsification

# 向量化版本(4-way SIMD)
--sparsification --sparse-vectorization="vl=4"

# 可伸缩向量(ARM SVE)
--sparse-vectorization="vl=4 enable-vla=true"
```

### 3. GPU加速

```bash
# 识别cuSparse兼容操作
--sparse-gpu-codegen

# 确保使用标准格式
# CSR/CSC/COO/BSR
```

### 4. 并行化

```mlir
// 注解并行策略
sparse_tensor.encoding<{
  ...,
  parallelization_strategy = #sparse_tensor.parallel<
    kind = "dense_outer_loop",
    reduction_parallel = false
  >
}>
```

### 5. 内存优化

```bash
# 使用Codegen而非Conversion
--sparse-tensor-codegen  # 启用编译器优化

# 避免不必要的拷贝
--one-shot-bufferize="allow-return-allocs-from-loops"
```

---

## 应用场景

### 科学计算
```mlir
// 有限元分析: 刚度矩阵求解
// K * u = f
%u = sparse_triangular_solve(%K, %f)
```

### 机器学习
```mlir
// 稀疏特征提取
// output = ReLU(sparse_weights * input + bias)
%matmul = sparse_matmul(%weights, %input)
%add = elementwise_add(%matmul, %bias)
%output = relu(%add)
```

### 图计算
```mlir
// PageRank: r_new = α * A^T * r + (1-α) * e
// A是稀疏邻接矩阵(CSC格式)
```

### 深度学习推理
```mlir
// 结构化稀疏(2:4)
#Structured = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : n_out_of_m(2, 4))
}>

// 适用于NVIDIA Ampere架构
```

---

## 调试技巧

### 1. 查看中间IR

```bash
# 每个pass后dump IR
mlir-opt input.mlir \
  -pass-pipeline='builtin.module(
    sparse-assembler{debug},
    sparsification{debug}
  )' \
  -mlir-print-ir-after-all
```

### 2. 验证正确性

```bash
# 添加assertions
mlir-opt input.mlir \
  --sparse-tensor-codegen \
  --test-math-polynomial-approximation \
  --verify-each
```

### 3. 性能分析

```bash
# 生成LLVM IR
mlir-opt input.mlir <passes> | \
  mlir-translate --mlir-to-llvmir | \
  opt -O3 | \
  llc -filetype=obj

# 使用perf分析
perf record ./sparse_program
perf report
```

### 4. 可视化稀疏结构

```python
# 使用Python辅助
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# 加载MLIR生成的稀疏数据
pos = np.loadtxt('pos.txt', dtype=int)
crd = np.loadtxt('crd.txt', dtype=int)
val = np.loadtxt('val.txt')

# 构造scipy稀疏矩阵
matrix = sp.csr_matrix((val, crd, pos))

# 可视化
plt.spy(matrix)
plt.show()
```

---

## 常见问题

### Q1: Codegen vs Conversion?
**A**:
- **Codegen**: 生成直接buffer操作,启用LLVM优化,性能更高,但编译慢
- **Conversion**: 使用运行时库,编译快,但性能受限

推荐生产环境使用Codegen。

### Q2: 如何处理动态稀疏性?
**A**: 使用COO格式 + convert操作:
```mlir
%coo = sparse_tensor.empty() : tensor<?x?xf32, #COO>
// 动态插入
%coo1 = sparse_tensor.insert %val into %coo[%i, %j]
// 转换为CSR
%csr = sparse_tensor.convert %coo1 : tensor<?x?xf32, #CSR>
```

### Q3: 向量化为何性能提升有限?
**A**:
- Gather操作开销大
- 稀疏不规则访问限制向量化收益
- 推荐在密集热点循环使用

### Q4: GPU加速适用场景?
**A**:
- 大规模稀疏矩阵(> 10K x 10K)
- 较高稀疏度(> 80% 零元素)
- cuSparse兼容格式
- 批处理操作

---

## 扩展阅读

### 学术论文
1. **"Sparse Tensor Algebra Compilation"** - Kjolstad et al.
2. **"The Tensor Algebra Compiler (taco)"** - MIT
3. **"Format Abstraction for Sparse Tensor Algebra Compilers"**

### 相关文档
- [MLIR SparseTensor Dialect](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)
- [TACO Documentation](http://tensor-compiler.org/)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)

### 工具链
- **taco**: Tensor Algebra Compiler
- **scipy.sparse**: Python稀疏矩阵库
- **cuSparse**: NVIDIA GPU稀疏库
- **Intel MKL Sparse**: Intel稀疏BLAS

---

## 总结

MLIR SparseTensor方言提供了完整的稀疏张量编译基础设施:

### 核心优势
✅ **格式无关**: 支持CSR/CSC/COO/BSR等多种格式
✅ **多后端**: CPU/GPU/向量化支持
✅ **优化充分**: 循环融合、向量化、并行化
✅ **类型安全**: 编译时验证稀疏属性
✅ **可组合**: 与密集计算无缝集成

### 性能特征
- **稀疏SpMV**: 3-10x vs 密集实现
- **向量化加速**: 2-4x (SIMD)
- **GPU加速**: 5-50x (大规模矩阵)

### 未来方向
- 更多硬件后端(TPU, FPGA)
- 自动格式选择
- 运行时自适应优化
- 混合精度支持

---

**文档版本**: LLVM 主干分支 (2026-01)
**维护者**: MLIR SparseTensor团队
**许可证**: Apache 2.0 with LLVM Exception
