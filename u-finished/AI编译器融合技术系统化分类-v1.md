# AI编译器融合技术系统化分类

## 主分类表：理论维度 → 工程实践

### 1. 依赖拓扑（Dependency Topology）

#### 1.1 垂直融合（Vertical Fusion）

##### 1.1.1 Producer-Consumer Fusion（通用场景）

- **优化目标**：消除中间物化，减少内存读写（Eliminate intermediate materialization; reduce memory R/W）
- **决策因素**：依赖链长度、内存压力
- **MLIR方案**：
  - 显式 SSA 数据流
  - 显式 iteration space（linalg / affine）
  - 可组合的 IR 变换（Pattern / Rewrite）


##### 1.1.2 Element-wise Chain Fusion（特定场景）

- **优化目标**：Fuse element-wise op chains（融合逐元素操作链）
- **决策因素**：操作数量、kernel launch开销
- **MLIR方案**：
  - 完全一致的 iteration space
  - 无 reduction / 无 loop-carried dependency
  - indexing map = identity（或 broadcast 的简单变体）


#### 1.2 水平融合（Horizontal Fusion）

##### 1.2.1 Multi-output Fusion（共享输入）
- **优化目标**：减少输入重复加载，提高并行度
- **决策因素**：输入共享度、寄存器压力
- **典型场景**：Attention Q/K/V Fusion
- **MLIR方案**：
  - Linalg：通过 `indexing_maps`描述算子的迭代空间，当两个算子共享相同的输入（`Producer`）时，`LinalgElementwiseOpFusionPass`（`-linalg-fuse-elementwise-ops`）可以识别出它们的迭代空间是重合或相关的，则把计算逻辑合并到一个 `linalg.generic` 算子中。
  - Affine：如果两个循环（`affine.for`）遍历的是相同的范围，并且都读取同一个内存块（`MemRef`），`AffineLoopFusion`（`-affine-loop-fusion`）策略可以执行"兄弟融合"（`Sibling Fusion`），将两个独立的循环合并为一个，原本需要两次启动的任务变成了一次，且提高了缓存（`Cache`）的局部性。
  - IREE：一种MLIR-based的端到端编译器。以`Attention`计算`Q/K/V`为例，IREE 并不是在 `linalg` 层简单做`op fusion`，而是在其开发的`Flow方言`层统一调度（`flow control`）。而且支持自动`Tiling`和`Dispatch`，将`Q/K/V`的生成逻辑融合进同一个分块调度中。[IREE Flow高效计算QKV](https://www.cnblogs.com/notlate-cn/p/19518938)


##### 1.2.2 Batch/SIMD Fusion

- **优化目标**：批次维度并行化
- **决策因素**：batch size、向量化机会

#### 1.3 模式融合（Pattern Fusion）

##### 1.3.1 Special Pattern Fusion

- **优化目标**：特定模式识别与合并
- **决策因素**：模式频率、硬件支持

##### 1.3.2 Multi-Head Attention Fusion

- **优化目标**：Transformer专用优化
- **决策因素**：序列长度、head数量

### 2. 循环与时间依赖（Temporal/Loop State）

#### 2.1 循环融合（Loop Fusion）
##### 2.1.1 Loop Fusion & Tiling

- **优化目标**：合并循环迭代，减少循环开销
- **决策因素**：循环嵌套深度、迭代空间

#### 2.2 循环依赖状态融合
##### 2.2.1 Loop-carried Scalar State Fusion
- **优化目标**：循环内累加器共享，减少冗余计算
- **决策因素**：状态依赖复杂度

#### 2.3 循环展开与流水线
##### 2.3.1 Loop Unrolling + Pipeline Fusion
- **优化目标**：指令级并行与流水线优化
- **决策因素**：寄存器数量、指令延迟

### 3. 数据布局与表示（Tensor Representation）

#### 3.1 布局传播（Layout Propagation）
##### 3.1.1 Layout Transform Elimination
- **优化目标**：减少reshape/transpose，提升访问效率
- **决策因素**：布局转换开销、硬件偏好

##### 3.1.2 Multi-layout Co-optimization
- **优化目标**：多布局共存协调
- **决策因素**：算子布局偏好冲突

#### 3.2 数据局部性优化
##### 3.2.1 Tile-local Fusion
- **优化目标**：利用硬件局部缓存提高数据重用
- **决策因素**：cache大小、访问模式

##### 3.2.2 Shared Memory Fusion（GPU）
- **优化目标**：GPU共享内存优化
- **决策因素**：shared memory容量

#### 3.3 中间结果消除
##### 3.3.1 Intermediate Materialization Elimination
- **优化目标**：消除临时Buffer，减少内存分配
- **决策因素**：中间张量大小、生命周期

### 4. 内存层次与多级分块（Memory Hierarchy）

#### 4.1 多级分块（Multi-level Tiling）
##### 4.1.1 Register/L1/L2/Global Memory Tiling
- **优化目标**：减少主存访问，提升cache利用率
- **决策因素**：各级缓存大小、带宽

#### 4.2 内存复用优化
##### 4.2.1 Buffer Reuse & Folding
- **优化目标**：减少内存分配，优化带宽
- **决策因素**：Buffer生命周期、对齐要求

#### 4.3 内存规划
##### 4.3.1 Memory Planning & Allocation Fusion
- **优化目标**：减少内存碎片，优化峰值占用
- **决策因素**：内存峰值、碎片化程度

### 5. 并行性（Parallelism）

#### 5.1 线程级并行融合
##### 5.1.1 Thread-block Fusion（GPU）
- **优化目标**：线程块级并行优化
- **决策因素**：线程块大小、占用率

##### 5.1.2 Warp-level Fusion（GPU）
- **优化目标**：线程束级SIMT优化
- **决策因素**：warp调度、分支发散

#### 5.2 指令级并行
##### 5.2.1 SIMD Vectorization Fusion
- **优化目标**：向量化指令融合
- **决策因素**：向量宽度、数据对齐

#### 5.3 任务级并行
##### 5.3.1 Async Execution Fusion
- **优化目标**：异步流并发
- **决策因素**：依赖关系、流数量

### 6. 硬件适配与计算-内存权衡（Hardware Adaptation）

#### 6.1 Kernel融合
##### 6.1.1 Multi-operator Kernel Fusion
- **优化目标**：多算子合并为单kernel，减少启动开销
- **决策因素**：kernel启动开销、寄存器压力

#### 6.2 推测性融合
##### 6.2.1 Speculative Fusion
- **优化目标**：权衡重计算vs存储
- **决策因素**：计算强度、内存带宽

#### 6.3 专用硬件加速
##### 6.3.1 Tensor Core Fusion（NVIDIA）
- **优化目标**：利用Tensor Core WMMA指令
- **决策因素**：矩阵尺寸、硬件支持

##### 6.3.2 Systolic Array Fusion（TPU）
- **优化目标**：脉动阵列数据流优化
- **决策因素**：数据流模式、阵列尺寸

#### 6.4 混合精度优化
##### 6.4.1 Mixed Precision Fusion
- **优化目标**：提升吞吐量，降低内存占用
- **决策因素**：精度敏感度、硬件支持

#### 6.5 数据类型转换优化
##### 6.5.1 Type Conversion Elimination
- **优化目标**：消除冗余类型转换
- **决策因素**：类型转换频率、开销

### 7. 控制流与动态性（Control-flow & Dynamism）

#### 7.1 分支优化融合
##### 7.1.1 Control-flow Fusion
- **优化目标**：减少分支执行，提高效率
- **决策因素**：分支预测准确率

##### 7.1.2 Branch Prediction Optimization
- **优化目标**：分支预测优化
- **决策因素**：分支热度、模式

#### 7.2 动态Shape融合
##### 7.2.1 Dynamic Shape Fusion
- **优化目标**：支持可变输入，减少内存浪费
- **决策因素**：shape变化频率

#### 7.3 运行时自适应
##### 7.3.1 JIT Fusion
- **优化目标**：即时编译融合
- **决策因素**：编译开销、执行频率

##### 7.3.2 Profile-guided Fusion
- **优化目标**：性能剖析引导融合
- **决策因素**：profile数据可用性

### 8. 跨层次联合优化（Cross-layer Co-optimization）

#### 8.1 图级融合
##### 8.1.1 Graph-level Fusion
- **优化目标**：子图识别与合并
- **决策因素**：图规模、模式复杂度

#### 8.2 代码生成融合
##### 8.2.1 Codegen-level Fusion
- **优化目标**：针对目标硬件优化kernel
- **决策因素**：目标架构特性

#### 8.3 端到端联合优化
##### 8.3.1 End-to-end Auto-tuning
- **优化目标**：搜索空间探索与自动调优
- **决策因素**：调优时间预算

---

## 特殊应用场景的融合策略映射

以下场景是多个理论维度的**组合应用**，非独立分类：

### 大语言模型优化
- **涉及理论维度**：依赖拓扑 + 内存层次 + 并行性
- **核心技术组合**：Attention Fusion + Activation Checkpointing + Pipeline Parallelism
- **代表系统**：FlashAttention, Megatron-LM, DeepSpeed

### 稀疏计算
- **涉及理论维度**：数据布局 + 硬件适配
- **核心技术组合**：Sparse-Dense Fusion + Structured Sparsity
- **代表系统**：TVM Sparse, cuSPARSE

### 量化部署
- **涉及理论维度**：硬件适配 + 数据布局
- **核心技术组合**：INT8 Kernel Fusion + Quantization-aware Layout
- **代表系统**：TensorRT, ONNX Runtime

### 边缘设备
- **涉及理论维度**：硬件适配 + 内存层次
- **核心技术组合**：Memory-constrained Fusion + Operator Splitting
- **代表系统**：TFLite, NNAPI

### 动态Batch
- **涉及理论维度**：控制流 + 内存层次
- **核心技术组合**：Dynamic Shape + Memory Planning
- **代表系统**：TorchDynamo, ONNX Runtime

---

## 主流编译器融合技术映射

### XLA
- **主要优化维度**：依赖拓扑 + 数据布局 + 跨层优化
- **核心融合技术**：HLO Fusion (Vertical/Horizontal), Layout Assignment, Buffer Assignment
- **适用场景**：TensorFlow, JAX训练/推理

### TVM
- **主要优化维度**：全维度覆盖
- **核心融合技术**：Tensor Expression Fusion, Auto-scheduling (AutoTVM/Ansor), Multi-level Tiling
- **适用场景**：跨硬件部署

### TensorRT
- **主要优化维度**：硬件适配 + 模式融合
- **核心融合技术**：Layer Fusion, INT8 Calibration, Kernel Auto-tuning
- **适用场景**：NVIDIA GPU推理

### Triton
- **主要优化维度**：内存层次 + 并行性
- **核心融合技术**：Block-level Programming, Tile-based Fusion, Auto-tuning
- **适用场景**：GPU自定义kernel开发

### MLIR
- **主要优化维度**：多级抽象 + 跨层优化
- **核心融合技术**：Linalg Fusion, Affine Loop Fusion, Progressive Lowering
- **适用场景**：编译器基础设施

### TorchInductor
- **主要优化维度**：依赖拓扑 + 动态性
- **核心融合技术**：Graph Pattern Matching, Triton Codegen, Dynamic Shape
- **适用场景**：PyTorch 2.0推理