

## 2. **循环与迭代空间优化（Loop & Iteration Space Optimization）**

### 2.1 循环融合（Loop Fusion）

#### 2.1.1 Loop Fusion & Tiling

##### 背景

在底层 IR（如 Affine 或 LLVM IR）中，**循环融合**是指将两个具有相同或兼容**迭代空间**的相邻循环合并为一个循环体的代码变换技术。

与图层面的算子融合不同，循环融合更关注**指令执行流**的优化。其核心收益不仅仅是减少全局内存访问，还包括：

1.  **提升时间局部性（Temporal Locality）**：将数据的**"定义"**与**"使用"**拉近，使其在**寄存器文件**或 **L1/L2 Cache** 中保持活跃（Hot），避免被驱逐。
2.  **减少循环控制开销**：减少了循环计数器的增量指令、条件分支跳转指令（Branch）以及未命中的分支预测惩罚。
3.  **隐式同步消除**：在并行编程（如 OpenMP 或 CUDA）中，两个独立的循环之间通常隐含一个**同步信号（Barrier）**。融合后，这个屏障被消除，减少了线程空转等待。

##### 约束

循环融合并非总是合法的。编译器必须进行严格的**依赖分析（Data Dependency Analysis）**，确保融合后的执行顺序不会破坏原有的读写依赖（例如：不能将“先读后写”的依赖变成“先写后读”）。

##### C++ 示例

```cpp
// 融合前
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++)
    A[i][j] = B[i][j] + C[i][j];

for (i = 0; i < N; i++)
  for (j = 0; j < M; j++)
    D[i][j] = A[i][j] * 2;

// 融合后
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++) {
    A[i][j] = B[i][j] + C[i][j];
    D[i][j] = A[i][j] * 2;  // A[i][j] 保持寄存器
  }
```

##### MLIR Affine 实现

```cpp
// 融合前
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %a = affine.load %A[%i, %j] : memref<100x100xf32>
    %b = affine.load %B[%i, %j] : memref<100x100xf32>
    %sum = arith.addf %a, %b : f32
    affine.store %sum, %C[%i, %j] : memref<100x100xf32>
  }
}

affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %c = affine.load %C[%i, %j] : memref<100x100xf32>
    %d = arith.mulf %c, %c : f32
    affine.store %d, %D[%i, %j] : memref<100x100xf32>
  }
}

// 融合后（-affine-loop-fusion）
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %a = affine.load %A[%i, %j] : memref<100x100xf32>
    %b = affine.load %B[%i, %j] : memref<100x100xf32>
    %sum = arith.addf %a, %b : f32
    %d = arith.mulf %sum, %sum : f32
    affine.store %d, %D[%i, %j] : memref<100x100xf32>
  }
}
```

---

### 2.2 **跨迭代状态优化（Cross-iteration State Optimization）**

此类优化关注跨越时间步（Time Step）或循环迭代（Iteration）的数据状态管理。

#### 2.2.1 Loop-carried Scalar Replacement (Accumulator Fusion)

##### 背景

在执行**归约（Reduction）**（如求和、点积）或**自回归（Autoregressive）**计算时，当前迭代的输入依赖于上一迭代的输出。

*   **传统低效模式**：每次迭代都从内存（或栈）中读取状态变量，更新后再写回内存。
*   **优化目标**：**寄存器提升（Register Promotion）**。将循环携带的状态变量（Accumulator/State）强行驻留在 CPU/GPU 的**寄存器**中，在整个循环执行期间不发生溢出（Spill）到内存的操作。

##### 收益

1.  **消除内存流量**：对于长度为 $N$ 的循环，消除 $2N$ 次内存读写。
2.  **打破延迟链**：通过寄存器直接转发数据，缩短指令间的依赖延迟（Latency Hiding）。

##### MLIR 实现方案

MLIR 的 `scf.for` 和 `affine.for` 引入了 **`iter_args`** 机制，这是显式表达寄存器驻留状态的各种 IR 中的最佳实践。它将可变变量（Mutable Variables）转化为 SSA 值（Immutable Values）的传递。

##### MLIR 示例

```cpp
// 场景：计算点积 Dot(A, B)
// 优化前（概念）：频繁读写内存
%sum_ptr = memref.alloc() : memref<f32>
scf.for %i = 0 to 1024 {
  %old_sum = memref.load %sum_ptr[] : memref<f32>
  %prod = ...
  %new_sum = arith.addf %old_sum, %prod
  memref.store %new_sum, %sum_ptr[] : memref<f32> // 瓶颈：写回内存
}

// 优化后（MLIR iter_args）：全寄存器操作
// %sum_iter 在编译后直接映射为物理寄存器
%final_sum = scf.for %i = 0 to 1024 
  iter_args(%sum_iter = %initial_sum) -> (f32) {
  
  %a = load %A[%i]
  %b = load %B[%i]
  %prod = arith.mulf %a, %b : f32
  
  // 状态更新仅发生在 SSA 值之间
  %sum_next = arith.addf %sum_iter, %prod : f32
  
  // 将新状态传递给下一次迭代
  scf.yield %sum_next : f32
}
```

#### 2.2.2 Temporal Fusion (RNN/LSTM Cell Fusion)

##### 背景

在处理循环神经网络（RNN/LSTM/GRU）时，存在严格的时间序列依赖：$State_t = f(Input_t, State_{t-1})$。
**时间融合（Temporal Fusion）** 指将一个时间步（Time Step）内的所有操作（MatMul, Activation, Gate Operations）融合为一个 Kernel，同时将**Hidden State** 保持在高速缓存或寄存器中传递给下一个时间步，而不是写回全局显存。

##### 挑战

*   **串行依赖**：时间步之间无法并行，因此融合重点在于减少每个时间步的**启动开销（Launch Overhead）**和**状态读写开销**。

##### 技术实现

在 MLIR 中，这通常表现为 `scf.for` 内部嵌套复杂的 `linalg` 操作，随后通过 **Loop Unrolling**（循环展开）和 **Invariant Code Motion**（不变量外提）来优化权重矩阵的加载。

---

### 2.3 **循环展开与流水线（Loop Unrolling & Pipelining）**

此技术通过重组循环内的指令调度，最大化硬件单元的利用率，主要解决指令流水线气泡和内存延迟问题。

#### 2.3.1 Loop Unrolling (Instruction Level Parallelism)

##### 背景

现代 CPU/GPU 拥有超标量架构（Superscalar），每个时钟周期可发射多条指令。紧凑的循环（Tight Loop）由于频繁的分支跳转检测（Branch Compare & Jump），会打断指令流水线。
**循环展开**通过复制循环体代码，减少跳转次数，并暴露更多的独立指令供硬件调度器进行**指令级并行（ILP）**优化。

##### 收益

1.  **减少分支开销**：$N$ 次迭代变成 $N/K$ 次跳转。
2.  **向量化机会**：展开后的连续访存指令更容易被合并为向量加载（Vector Load）。

##### MLIR 实现

在 `affine` 或 `scf` 方言中，可以通过属性标记或变换 Pass 显式控制展开因子。

```cpp
// 原始循环
affine.for %i = 0 to 1024 {
  %x = affine.load %A[%i]
  %y = arith.mulf %x, %c2
  affine.store %y, %A[%i]
}

// 展开因子 = 4 (Unroll Factor = 4)
// -affine-loop-unroll="unroll-factor=4"
affine.for %i = 0 to 1024 step 4 {
  // 编译器生成 4 个独立的计算链，允许 CPU 并行发射指令
  %x0 = affine.load %A[%i]
  %x1 = affine.load %A[%i+1]
  %x2 = affine.load %A[%i+2]
  %x3 = affine.load %A[%i+3]
  // ... 计算 x0...x3 ...
  // ... 存储 x0...x3 ...
}
```

#### 2.3.2 Software Pipelining (Latency Hiding)

##### 背景

在深度学习算子（如 GEMM, Attention）中，从全局内存（HBM）加载数据到片上缓存（SRAM/Register）的延迟极高（数百个时钟周期）。

如果采用 **Load $\to$ Compute** 的串行模式，计算单元在等待数据时会空转。
**软件流水线**（配合双缓冲/多缓冲 Double Buffering）将不同迭代的阶段重叠执行：在计算当前块（Tile $i$）的同时，预取下一块（Tile $i+1$）的数据。

##### 机制

流水线变换将循环重构为三个部分：

1.  **Prologue（序言）**：预取第 0 次迭代的数据。
2.  **Steady State（稳态/核）**：同时执行第 $i$ 次计算和第 $i+1$ 次加载。
3.  **Epilogue（尾声）**：完成最后一次迭代的计算。

```
Timeline:
Iter 0: [Load 0]
Iter 1:          [Comp 0] [Load 1]  <-- 并行执行 (Overlap)
Iter 2:                   [Comp 1] [Load 2]
```

##### MLIR 实现 (Async Pipelining)

MLIR 的 `scf.for` 配合 `iter_args` 和异步指令（如 `nvgpu.device_async_copy`）可以完美表达这种模式。

```cpp
// 软件流水线化后的循环结构
// 初始阶段：预加载第 0 个 Tile (Prologue)
%token0 = gpu.async_copy %Global[%c0] to %Shared[%c0] ...

// 循环携带 token 状态 (Stage $i$ 的加载句柄传递给 Stage $i+1$)
scf.for %i = 0 to %N step %TileSize 
  iter_args(%token_prev = %token0) {
  
  // 1. 发起第 $i+1$ 个 Tile 的异步加载 (Prefetch)
  %next_idx = arith.addi %i, %TileSize
  %token_next = gpu.async_copy %Global[%next_idx] to %Shared[%buffer_next] ...
  
  // 2. 等待第 $i$ 个 Tile 加载完成 (Wait)
  gpu.wait %token_prev
  
  // 3. 计算第 $i$ 个 Tile (Compute)
  // 此时计算与步骤 1 的加载是并行的
  linalg.matmul ins(%Shared[%buffer_curr] ...)
  
  // 4. 传递下一轮的 token
  scf.yield %token_next
}

// 尾声：计算最后一个 Tile (Epilogue)
gpu.wait %token_last
linalg.matmul ...
```

---

## 3. 数据布局与表示（Tensor Representation）

### 3.1 布局传播（Layout Propagation）

#### 3.1.1 Layout Transform Elimination

##### 背景

不同算子偏好不同的内存布局（如 Row-major vs Column-major）。频繁转换布局（transpose/reshape）会降低性能。

##### 策略

```
原始：Conv(NHWC) → Transpose → MatMul(NC)
优化：选择 Conv(NCHW) 直接输出 NC 布局
```

##### MLIR 实现

```mlir
// Layout-aware Linalg fusion
%output = linalg.matmul {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,  // A: Row-major
    affine_map<(m, n, k) -> (k, n)>,  // B: Column-major
    affine_map<(m, n, k) -> (m, n)>   // C: Output
  ]
} ins(%A, %B : ...) outs(%C : ...)
```

---

### 3.2 数据局部性优化

#### 3.2.1 Tile-local Fusion

##### 背景

**分块**（Tiling）后，每个 tile 内的数据可以保持在缓存中。Tile-local fusion 将操作在 tile 级别融合。

##### MLIR 示例

```mlir
// -linalg-tile + -linalg-fuse
%tiled_A = tensor.extract_slice %A[...]

%tiled_result = linalg.generic {
  // 在 tile 上执行融合操作
} ins(%tiled_A, %tiled_B : ...) outs(%tiled_C : ...)

%result = tensor.insert_slice %tiled_result into %C[...]
```

---

#### 3.2.2 Shared Memory Fusion（GPU）

##### 背景

GPU 的 **Shared Memory** 是片上高速缓存，跨线程共享。将操作融合到使用 shared memory 的 kernel 中可大幅提升性能。

##### MLIR GPU Dialect

```mlir
gpu.func @kernel_fused(%data: memref<?xf32>) {
  // 数据从 global memory 加载到 shared memory
  %shared = gpu.alloc ... attributes {memory_space = #gpu.address_space<workgroup>}

  gpu.barrier

  // 使用 shared memory 进行计算
  %result = linalg.generic ... ins(%shared : ...)

  gpu.barrier

  // 写回 global memory
  gpu.store %result, %output[...]
}
```

---

### 3.3 中间结果消除

#### 3.3.1 Intermediate Materialization Elimination

##### 背景

消除了中间张量的显式存储，使用 **in-place** 更新或寄存器复用。

##### MLIR 策略

```mlir
// 使用 buffer 语义进行 in-place 更新
func.func @inplace_add(%A: memref<?xf32>, %B: memref<?xf32>) {
  linalg.generic {__inplace_operands_attr__ = ["A"]}
  ins(%B : ...) outs(%A : ...) {
    ^bb0(%b: f32, %a: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  }
  return
}
```

---

## 4. 内存层次与多级分块（Memory Hierarchy）

### 4.1 多级分块（Multi-level Tiling）

#### 4.1.1 Register/L1/L2/Global Memory Tiling

##### 背景

现代处理器有**多级内存层次**：寄存器最快但容量小，L1/L2 缓存中等，全局内存慢但大。多级分块针对每级优化。

##### MLIR 实现

```cpp
// -linalg-tile 到不同级别
// 三层分块：register tiles, L1 tiles, L2 tiles

// 外层：L2, Tile Size=64
scf.for %i0 = 0 to 1024 step 64 {
  scf.for %j0 = 0 to 1024 step 64 {
    // 中层：L1, Tile Size=8
    scf.for %i1 = %i0 to min(%i0 + 64, 1024) step 8 {
      scf.for %j1 = %j0 to min(%j0 + 64, 1024) step 8 {
        // 内层：寄存器分块（向量化）
        %tile = linalg.matmul ... // 8x8 矩阵乘法
      }
    }
  }
}
```

---

### 4.2 内存复用优化

#### 4.2.1 Buffer Reuse & Folding

##### 背景

当多个 buffer 的生命周期不重叠时，可以**复用同一块内存**。

##### MLIR 策略

```mlir
// -mlir-bufferize + 内存分配优化
func.func @buffer_reuse() {
  %buf1 = memref.alloc() : memref<1024xf32>
  // ... 使用 buf1 ...
  memref.dealloc %buf1

  // 编译器可优化：buf2 复用 buf1 的空间
  %buf2 = memref.alloc() : memref<1024xf32>
  // ... 使用 buf2 ...
  memref.dealloc %buf2
}
```

---

### 4.3 内存规划

#### 4.3.1 Memory Planning & Allocation Fusion

##### 背景

**峰值内存占用**（Peak Memory Usage）是部署的关键约束。内存规划分析张量生命周期，优化分配策略。

##### 技术

1. **活跃度分析**（Liveness Analysis）
2. **内存复用**（Memory Reuse）
3. **算子分割**（Operator Splitting）控制峰值

---

## 5. 并行性（Parallelism）

### 5.1 线程级并行融合

#### 5.1.1 Thread-block Fusion（GPU）

##### 背景

将多个 kernel 融合到一个 **Thread Block** 中，减少 kernel launch 开销。

##### MLIR GPU 实现

```mlir
// 单个 thread block 执行多个操作
gpu.launch blocks(%bx, %by, %bz) in (%grid_x, %grid_y, %grid_z)
             threads(%tx, %ty, %tz) in (%block_x, %block_y, %block_z) {
  // Op1
  %1 = linalg.generic ...

  // Op2（融合）
  %2 = linalg.generic ... ins(%1 : ...)

  // Op3（融合）
  %3 = linalg.generic ... ins(%2 : ...)

  gpu.terminate
}
```

---

#### 5.1.2 Warp-level Fusion（GPU）

##### 背景

**Warp**（32 个线程）是 SIMT 执行的基本单位。Warp-level 融合利用 **Warp Shuffle** 指令在线程间通信。

##### MLIR/NVVM 实现

```mlir
// 使用 NVVM warp shuffle
%value = nvvm.shfl.bfly.b32 %warp_value, %lane_id : i32
```

---

### 5.2 指令级并行

#### 5.2.1 SIMD Vectorization Fusion

##### 背景

**SIMD**（Single Instruction Multiple Data）一条指令处理多个数据元素。融合后可更好地向量化。

##### MLIR Vector Dialect

```mlir
// 融合后的向量化操作
%v_a = vector.load %A[%i] : memref<?xf32>, vector<256xf32>
%v_b = vector.load %B[%i] : memref<?xf32>, vector<256xf32>

// 向量化 add + mul（FMA）
%v_sum = vector.fma %v_a, %v_b, %v_c : vector<256xf32>

vector.store %v_sum, %C[%i] : memref<?xf32>, vector<256xf32>
```

---

### 5.3 任务级并行

#### 5.3.1 Async Execution Fusion

##### 背景

使用 **CUDA Streams** 或 **异步执行**并发运行独立的操作。

##### MLIR 示例

```cpp
func.func @async_fused_ops(%A, %B, %C) {
  // Stream 1: Op1 + Op2 融合异步执行
  %token1 = async.execute {
    %1 = linalg.matmul ins(%A, %B : ...) outs(%tmp : ...)
    %2 = linalg.generic ... ins(%1 : ...)
    async.yield %2 : tensor<?xf32>
  }

  // Stream 2: Op3 独立执行
  %token2 = async.execute {
    %3 = linalg.generic ... ins(%C : ...)
    async.yield %3 : tensor<?xf32>
  }

  // 等待两个 token
  %r1 = async.await %token1 : tensor<?xf32>
  %r2 = async.await %token2 : tensor<?xf32>

  // 最终融合
  %final = linalg.generic ... ins(%r1, %r2 : ...)

  return %final
}
```

---

## 6. 硬件适配与计算-内存权衡（Hardware Adaptation）

### 6.1 Kernel融合

#### 6.1.1 Multi-operator Kernel Fusion

##### 背景

将多个算子编译为**单个 kernel**，减少 kernel launch 开销（约 5-10 μs）。

##### 决策因素

```
收益 = 启动开销节省 + 内存带宽节省
成本 = 寄存器压力增加 + 编译时间增加

融合条件：收益 > 成本
```

#### 6.1.2 Epilogue Fusion（GEMM 后处理融合）

**背景**

在矩阵乘法（GEMM）或卷积（Convolution）这类**计算密集型**算子中，计算结果通常存储在片上**寄存器**（Accumulators）中。如果将结果写回全局内存（Global Memory）后再读出进行激活或偏置加法，会浪费宝贵的内存带宽。

**核心思想**

**Epilogue Fusion** 指在 GEMM 的主循环结束后、结果写回内存之前，直接在寄存器中执行轻量级的逐元素操作（如 BiasAdd, ReLU, GeLU, Type Conversion）。

**技术原理**

在 CUTLASS 或 Triton 等后端中，Epilogue 被设计为可组合的代码片段。

```cpp
// MLIR 中通常体现为将 linalg.matmul 的输出直接作为 linalg.generic 的输入
// 编译器后端（如 IREE Codegen）需将其识别并映射为单一 Kernel

// 逻辑上的融合
%acc = linalg.matmul ins(%A, %B) ...
%res = linalg.generic ins(%acc, %bias) {
  ^bb0(%a, %b):
    %0 = arith.addf %a, %b
    %1 = arith.maxf %0, %c0  // ReLU
    linalg.yield %1
}

// 物理层代码生成（伪代码）
// 这种融合很难通过简单的循环合并实现，通常需要专门的 Epilogue 生成器
mma_op(A_frag, B_frag, C_frag) // C_frag 在寄存器中
for i in range(fragment_size):
    C_frag[i] += bias_frag[i]  // 寄存器级操作
    C_frag[i] = max(C_frag[i], 0)
store_matrix(C_frag, GlobalMemory)
```

---

### 6.2 推测性融合

#### 6.2.1 Speculative Fusion

##### 背景

**重计算**（Recomputation） vs **存储**（Storing）中间结果。某些情况下重计算比存储更高效。

##### 应用：Activation Checkpointing

```
训练时的权衡：
- 存储：占用大量内存
- 重计算：增加计算量但节省内存

Speculative Fusion 动态决策
```

---

### 6.3 专用硬件加速

#### 6.3.1 Tensor Core Fusion（NVIDIA）

##### 背景

**Tensor Core** 是 NVIDIA GPU 的矩阵乘法加速单元。融合操作以使用 Tensor Core。

##### MLIR 实现

```cpp
// 使用 NVIDIA WMMA（Warp Matrix Multiply-Accumulate）
#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>

%acc = gpu.subgroup_mma_load_matrix %C[%i, %j] : ...
%mma = gpu.subgroup_mma_compute %A_tile, %B_tile, %acc
     : (!gpu.mma_fragment, !gpu.mma_fragment, !gpu.mma_fragment) -> !gpu.mma_fragment
gpu.subgroup_mma_store_matrix %mma, %D[%i, %j]
```

*注：MLIR 中 MMA 指令集通常通过 nvgpu 或 vector dialect 映射*

---

#### 6.3.2 Systolic Array Fusion（TPU）

##### 背景

**脉动阵列**（Systolic Array）数据流式处理，数据在阵列中流动。融合操作以匹配数据流模式。

---

### 6.4 混合精度优化

#### 6.4.1 Mixed Precision Fusion

##### 背景

**FP32 → FP16/BF16/INT8** 量化可提升吞吐量、降低内存占用。

##### MLIR 实现

```mlir
// 量化感知融合
%input_f16 = arith.truncf %input_fp32 : f32 to f16
%weight_f16 = arith.truncf %weight_fp32 : f32 to f16

%accum_f16 = linalg.matmul ins(%input_f16, %weight_f16 : ...)

// 可选：最终输出转回 FP32
%output_fp32 = arith.extf %accum_f16 : f16 to f32
```

---

### 6.5 数据类型转换优化

#### 6.5.1 Type Conversion Elimination

##### 背景

消除**冗余的类型转换**，保持数据在同精度下计算。

##### 示例

```
原始：FP32 → FP16 → 计算 → FP16 → FP32
优化：保持 FP16 或 全程 FP32
```

---

## 7. 控制流与动态性（Control-flow & Dynamism）

### 7.1 分支优化融合

#### 7.1.1 Control-flow Fusion

##### 背景

**条件分支**可能导致融合边界难以跨越。分析分支可达性，融合总是执行的路径。

##### MLIR SCF Dialect

```mlir
// 融合跨分支的操作
scf.if %condition {
  %1 = linalg.generic ...  // Branch A
  scf.yield %1
} else {
  %2 = linalg.generic ...  // Branch B
  scf.yield %2
}

// 后续操作可以与分支内操作融合（如果可行）
```

---

### 7.2 动态Shape融合

#### 7.2.1 Dynamic Shape Fusion

##### 背景

**动态形状**（Dynamic Shapes）引入了编译期的维度不确定性，导致传统静态分析技术失效。现代编译器通常采用 **形状实体化（Shape Reification）** 与 **符号化分析（Symbolic Analysis）** 技术，将形状约束转化为标准**标量运算**，结合**运行时信息**动态生成融合代码。

##### MLIR Shape Dialect

```mlir
func.func @dynamic_fusion(%input: tensor<?x?xf32>) {
  // 运行时获取形状
  %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  // 基于动态形状的融合操作
  %output = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ]
  } ins(%input : tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) {
    // ...
  }

  return %output : tensor<?x?xf32>
}
```

---

### 7.3 运行时自适应

#### 7.3.1 JIT Fusion

##### 背景

**即时编译**（JIT）根据运行时信息（具体形状、硬件特性）生成最优融合 kernel。

---

#### 7.3.2 Profile-guided Fusion

##### 背景

基于**性能剖析数据**（Profile Data）决定融合策略。热点算子优先融合。

---

## 8. 跨层次联合优化（Cross-layer Co-optimization）

### 8.1 图级融合

#### 8.1.1 Graph-level Fusion

##### 背景

在**计算图级别**识别可融合的子图。

##### 技术

1. **模式匹配**（Pattern Matching）
2. **成本模型**（Cost Model）
3. **图划分**（Graph Partitioning）

---

### 8.2 代码生成融合

#### 8.2.1 Codegen-level Fusion

##### 背景

针对**目标硬件**（CPU/GPU/TPU）生成最优代码。

##### MLIR 后端

```
MLIR IR
   ↓
Linalg on Tensors
   ↓
Linalg on Buffers
   ↓
Affine/SCF Loops
   ↓
LLVM IR / SPIR-V / CUDA
   ↓
Machine Code
```

---

### 8.3 端到端联合优化

#### 8.3.1 End-to-end Auto-tuning

##### 背景

**自动调优**（Auto-tuning）搜索最优融合配置。

##### 系统

- **TVM AutoTVM/Ansor**
- **MLIR Affine Loop Tuning**
- **MLIR-based IREE 编译器**

---

## 特殊应用场景的融合策略映射

以下场景是多个理论维度的**组合应用**：

### 大语言模型优化

**涉及维度**：依赖拓扑 + 内存层次 + 并行性

**核心技术**：

- **FlashAttention**：Tile-local Fusion + Shared Memory Fusion
- **Activation Checkpointing**：Speculative Fusion
- **Tensor Parallelism**：Multi-output Fusion (QKV) + Collective Communication

**代表系统**：

- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

### 稀疏计算

**涉及维度**：数据布局 + 硬件适配

**核心技术**：

- **Sparse-Dense Fusion**：稀疏矩阵与密集操作融合
- **Structured Sparsity**：利用硬件支持的稀疏格式

**代表系统**：

- TVM Sparse
- cuSPARSE
- Intel MKL Sparse

---

### 量化部署

**涉及维度**：硬件适配 + 数据布局

**核心技术**：

- **INT8 Kernel Fusion**：量化感知融合
- **Quantization-aware Layout**：针对量化的布局优化

**代表系统**：

- NVIDIA TensorRT
- ONNX Runtime Quantization
- QNN (Qualcomm)

---

### 边缘设备

**涉及维度**：硬件适配 + 内存层次

**核心技术**：

- **Memory-constrained Fusion**：受内存限制的融合策略
- **Operator Splitting**：大算子分割以降低峰值内存

**代表系统**：

- TensorFlow Lite
- Android NNAPI
- CoreML

---

### 动态Batch

**涉及维度**：控制流 + 内存层次

**核心技术**：

- **Dynamic Shape Fusion**：动态形状支持
- **Runtime Memory Planning**：运行时内存规划

**代表系统**：

- TorchDynamo
- ONNX Runtime Dynamic Shapes
- IREE

---

## 主流编译器融合技术映射

### XLA (XLA: Accelerated Linear Algebra)

- **主要维度**：依赖拓扑 + 数据布局 + 跨层优化
- **核心融合技术**：
  - HLO Fusion (Vertical/Horizontal)
  - Layout Assignment
  - Buffer Assignment
- **适用场景**：TensorFlow, JAX 训练/推理
- **链接**：[XLA Documentation](https://www.tensorflow.org/xla)

---

### TVM (Tensor Virtual Machine)

- **主要维度**：全维度覆盖
- **核心融合技术**：
  - Tensor Expression Fusion
  - Auto-scheduling (AutoTVM/Ansor)
  - Multi-level Tiling
- **适用场景**：跨硬件部署
- **链接**：[TVM GitHub](https://github.com/apache/tvm)

---

### TensorRT

- **主要维度**：硬件适配 + 模式融合
- **核心融合技术**：
  - Layer Fusion
  - INT8 Calibration
  - Kernel Auto-tuning
- **适用场景**：NVIDIA GPU 推理
- **链接**：[TensorRT Documentation](https://developer.nvidia.com/tensorrt)

---

### Triton

- **主要维度**：内存层次 + 并行性
- **核心融合技术**：
  - Block-level Programming
  - Tile-based Fusion
  - Auto-tuning
- **适用场景**：GPU 自定义 kernel 开发
- **链接**：[Triton GitHub](https://github.com/openai/triton)

---

### MLIR (Multi-Level Intermediate Representation)

- **主要维度**：多级抽象 + 跨层优化
- **核心融合技术**：
  - Linalg Fusion
  - Affine Loop Fusion
  - Progressive Lowering
- **适用场景**：编译器基础设施
- **链接**：[MLIR GitHub](https://github.com/llvm/llvm-project/tree/main/mlir)

---

### TorchInductor

- **主要维度**：依赖拓扑 + 动态性
- **核心融合技术**：
  - Graph Pattern Matching
  - Triton Codegen
  - Dynamic Shape Support
- **适用场景**：PyTorch 2.0 推理/训练
- **链接**：[PyTorch 2.0 Inductor](https://pytorch.org/get-started/pytorch-2.0/)

---

## 参考资料

1. [MLIR Documentation](https://mlir.llvm.org/)
2. [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
3. [Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
4. [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
5. [IREE Compiler](https://iree.dev/)
6. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

---

*本文档为 AI 编译器融合技术的系统化分类与详细说明，覆盖 MLIR 实现方案与代码示例。*