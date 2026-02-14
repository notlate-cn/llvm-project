# LLVM MLIR Affine方言Transform技术详解

本文档详细梳理LLVM MLIR中Affine方言的所有Transform Pass的作用、技术原理和应用场景。

**目录路径**: `mlir/lib/Dialect/Affine/Transforms/`

---

## 1. Loop Fusion (循环融合)

**文件**: `LoopFusion.cpp`

### 1.1 作用
循环融合是一种优化技术,将具有producer-consumer或input-reuse关系的循环嵌套融合到一起,以提高数据局部性。

### 1.2 技术原理

#### 核心算法
- **Producer-Consumer融合**: 将生产者循环(写入memref)与消费者循环(读取同一memref)融合
- **Sibling融合**: 融合共享相同父循环但无依赖关系的兄弟循环
- **Greedy策略**: 使用贪心算法遍历依赖图,寻找可融合的循环对

#### 关键技术点
1. **依赖分析**: 构建MemRefDependenceGraph分析循环间的数据依赖
2. **计算切片**: 计算ComputationSliceState以确定融合的代码范围
3. **盈利性分析**: 通过以下指标判断融合是否有益:
   - 内存占用减少 (storage reduction)
   - 额外计算量 (additional compute fraction)
   - 缓存利用率
4. **Private Memref创建**: 为融合后的循环创建私有内存缓冲区

### 1.3 实例演示

**融合前**:
```mlir
affine.for %i = 0 to 256 {
  affine.store %val, %A[%i] : memref<256xf32>
}
affine.for %j = 0 to 256 {
  %v = affine.load %A[%j] : memref<256xf32>
  affine.store %v, %B[%j] : memref<256xf32>
}
```

**融合后**:
```mlir
affine.for %i = 0 to 256 {
  affine.store %val, %A[%i] : memref<256xf32>
  %v = affine.load %A[%i] : memref<256xf32>
  affine.store %v, %B[%i] : memref<256xf32>
}
```

**效果**: 消除中间缓冲区A的访问,提高缓存命中率

---

## 2. Loop Tiling (循环分块)

**文件**: `LoopTiling.cpp`

### 2.1 作用
将循环迭代空间分割成更小的块(tiles),以提高缓存利用率和数据局部性。

### 2.2 技术原理

#### Tile Size确定策略
1. **基于缓存大小**: 根据提供的cache size计算tile大小
2. **基于内存足迹**: 通过`getMemoryFootprintBytes`分析访问模式
3. **调整为除数**: 将tile size调整为trip count的除数,避免边界处理

#### 核心算法
```cpp
// tile大小计算公式
unsigned tSize = floor(pow(excessFactor, 1.0 / band.size()))
```
其中excessFactor = footprint / cacheSize

### 2.3 实例演示

**Tiling前**:
```mlir
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    // 矩阵操作
    %v = affine.load %A[%i, %j]
  }
}
```

**Tiling后** (tile size = 32):
```mlir
affine.for %i0 = 0 to 1024 step 32 {
  affine.for %j0 = 0 to 1024 step 32 {
    affine.for %i1 = 0 to 32 {
      affine.for %j1 = 0 to 32 {
        %i = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i0, %i1)
        %j = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%j0, %j1)
        %v = affine.load %A[%i, %j]
      }
    }
  }
}
```

**效果**: 提高cache复用,减少cache miss

---

## 3. Loop Unrolling (循环展开)

**文件**: `LoopUnroll.cpp`

### 3.1 作用
将循环体复制多次,减少循环控制开销,提供指令级并行机会。

### 3.2 技术原理

#### 三种展开模式
1. **Full Unroll**: 完全展开(适用于trip count已知且较小的循环)
2. **By Factor**: 按指定因子展开
3. **Up To Factor**: 展开至指定因子(不足则完全展开)

#### 实现机制
- 使用`loopUnrollByFactor`函数
- 处理循环不整除情况(epilogue)
- 支持cleanup选项移除冗余代码

### 3.3 实例演示

**展开前**:
```mlir
affine.for %i = 0 to 8 {
  %v = affine.load %A[%i]
  affine.store %v, %B[%i]
}
```

**展开后** (unroll factor = 4):
```mlir
affine.for %i = 0 to 8 step 4 {
  %v0 = affine.load %A[%i]
  affine.store %v0, %B[%i]

  %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
  %v1 = affine.load %A[%i1]
  affine.store %v1, %B[%i1]

  %i2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%i)
  %v2 = affine.load %A[%i2]
  affine.store %v2, %B[%i2]

  %i3 = affine.apply affine_map<(d0) -> (d0 + 3)>(%i)
  %v3 = affine.load %A[%i3]
  affine.store %v3, %B[%i3]
}
```

**效果**: 减少循环控制开销,增加指令级并行

---

## 4. Loop Unroll and Jam (展开并合并)

**文件**: `LoopUnrollAndJam.cpp`

### 4.1 作用
将外层循环展开,然后将内层循环合并(jam),提高寄存器复用和操作级并行。

### 4.2 技术原理

这是Loop Unrolling的高级变体,特别适合嵌套循环优化。

#### 变换过程
1. 展开外层循环
2. 将内层循环合并
3. 提高寄存器复用

### 4.3 实例演示

**变换前**:
```mlir
affine.for %i = 0 to 8 {
  S1(%i)
  S2(%i)
  affine.for %j = 0 to 16 {
    S3(%i, %j)
    S4(%i, %j)
  }
  S5(%i)
}
```

**变换后** (unroll-jam factor = 2):
```mlir
affine.for %i = 0 to 8 step 2 {
  S1(%i)
  S2(%i)
  S1(%i+1)
  S2(%i+1)

  affine.for %j = 0 to 16 {
    S3(%i, %j)
    S4(%i, %j)
    S3(%i+1, %j)
    S4(%i+1, %j)
  }

  S5(%i)
  S5(%i+1)
}
```

**效果**: 同时提高寄存器复用和向量化机会

---

## 5. Loop Coalescing (循环合并)

**文件**: `LoopCoalescing.cpp`

### 5.1 作用
将完美嵌套的多层循环合并为单层循环,简化循环结构。

### 5.2 技术原理

#### 适用条件
- 循环必须是完美嵌套(perfectly nested)
- 内层循环边界不依赖外层循环变量

#### 实现方式
使用`coalescePerfectlyNestedAffineLoops`函数

### 5.3 实例演示

**合并前**:
```mlir
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 20 {
    affine.for %k = 0 to 30 {
      // body
    }
  }
}
```

**合并后**:
```mlir
affine.for %idx = 0 to 6000 {  // 10 * 20 * 30 = 6000
  %i = affine.apply affine_map<(d0) -> (d0 floordiv 600)>(%idx)
  %j = affine.apply affine_map<(d0) -> ((d0 mod 600) floordiv 30)>(%idx)
  %k = affine.apply affine_map<(d0) -> (d0 mod 30)>(%idx)
  // body
}
```

**效果**: 简化循环控制,便于后续优化

---

## 6. Affine Loop Invariant Code Motion (LICM)

**文件**: `AffineLoopInvariantCodeMotion.cpp`

### 6.1 作用
将循环不变的代码移到循环外部,减少重复计算。

### 6.2 技术原理

#### 不变性判断
1. 操作数不依赖循环归纳变量
2. 操作数不依赖iter_args
3. 无副作用或只有affine读写操作
4. 对于affine读写,需检查别名分析

#### 提升策略
- 按innermost-first顺序处理
- 嵌套if/for操作递归检查

### 6.3 实例演示

**优化前**:
```mlir
affine.for %i = 0 to 100 {
  %c = arith.constant 10 : index
  %x = arith.addi %c, %c : index
  %v = affine.load %A[%i]
  %result = arith.muli %v, %x
  affine.store %result, %B[%i]
}
```

**优化后**:
```mlir
%c = arith.constant 10 : index
%x = arith.addi %c, %c : index
affine.for %i = 0 to 100 {
  %v = affine.load %A[%i]
  %result = arith.muli %v, %x
  affine.store %result, %B[%i]
}
```

**效果**: 减少循环内的重复计算

---

## 7. Affine Data Copy Generation (显式数据拷贝)

**文件**: `AffineDataCopyGeneration.cpp`

### 7.1 作用
自动生成显式的数据拷贝操作,将数据从慢速内存空间拷贝到快速内存空间。

### 7.2 技术原理

#### 核心概念
- **Memory Hierarchy**: 识别快速/慢速内存空间
- **Buffer Allocation**: 在快速内存中分配buffer
- **DMA Operations**: 生成DMA传输操作

#### 关键步骤
1. 分析内存访问模式
2. 计算memory footprint
3. 在合适深度插入copy操作
4. 替换原始访问为buffer访问

### 7.3 实例演示

**优化前** (从慢速DRAM访问):
```mlir
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    %v = affine.load %A[%i, %j] : memref<1024x1024xf32, 2> // space 2 = DRAM
    // compute
  }
}
```

**优化后** (通过快速cache访问):
```mlir
%buffer = memref.alloc() : memref<1024x1024xf32, 1>  // space 1 = cache
affine.dma_start %A[%c0, %c0], %buffer[%c0, %c0], ...
affine.dma_wait %tag
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    %v = affine.load %buffer[%i, %j] : memref<1024x1024xf32, 1>
    // compute
  }
}
```

**效果**: 显著减少内存访问延迟

---

## 8. Affine Parallelization (并行化)

**文件**: `AffineParallelize.cpp`

### 8.1 作用
将可并行的affine.for循环转换为affine.parallel操作。

### 8.2 技术原理

#### 并行性检测
使用`isLoopParallel`函数检查:
- 无循环携带依赖
- 归约操作可并行化

#### 嵌套控制
通过`maxNested`参数控制并行嵌套层数

### 8.3 实例演示

**并行化前**:
```mlir
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    %v = affine.load %A[%i, %j]
    %result = arith.mulf %v, %v
    affine.store %result, %B[%i, %j]
  }
}
```

**并行化后**:
```mlir
affine.parallel (%i, %j) = (0, 0) to (1024, 1024) {
  %v = affine.load %A[%i, %j]
  %result = arith.mulf %v, %v
  affine.store %result, %B[%i, %j]
}
```

**效果**: 明确并行语义,便于后端生成并行代码

---

## 9. Pipeline Data Transfer (数据传输流水线)

**文件**: `PipelineDataTransfer.cpp`

### 9.1 作用
将DMA数据传输操作与计算重叠,通过流水线技术隐藏传输延迟。

### 9.2 技术原理

#### 核心技术
1. **Double Buffering**: 使用双缓冲技术
2. **Loop Pipelining**: 将DMA start/wait与计算重叠

#### 实现步骤
1. 识别DMA start/wait操作对
2. 创建双缓冲区(添加维度2)
3. 使用`iv mod 2`选择缓冲区
4. 重排循环使传输与计算重叠

### 9.3 实例演示

**优化前**:
```mlir
affine.for %i = 0 to 100 {
  affine.dma_start %src[%i], %buf[0], %tag[0]
  affine.dma_wait %tag[0]
  // compute using %buf
}
```

**优化后**:
```mlir
%double_buf = memref.alloc() : memref<2x...>  // 双缓冲
affine.for %i = 0 to 100 {
  %buf_idx = affine.apply affine_map<(d0) -> (d0 mod 2)>(%i)
  affine.dma_start %src[%i], %double_buf[%buf_idx], %tag[%buf_idx]
  // 前一次迭代的计算与当前DMA重叠
  affine.dma_wait %tag[%buf_idx]
}
```

**效果**: 隐藏数据传输延迟,提高吞吐量

---

## 10. Super Vectorization (超级向量化)

**文件**: `SuperVectorize.cpp`

### 10.1 作用
将循环和操作向量化为目标无关的n维super-vector抽象。

### 10.2 技术原理

#### Super-Vector概念
- 不限于硬件向量寄存器大小
- 支持多维向量类型
- 使用vector.transfer抽象

#### 关键特性
1. **Pattern Matching**: 使用NestedMatcher识别向量化模式
2. **Contiguity Analysis**: 分析内存访问连续性
3. **Multi-dimensional**: 支持多维向量化

### 10.3 实例演示

**向量化前**:
```mlir
affine.for %i = 0 to 1024 {
  %v = affine.load %A[%i] : memref<1024xf32>
  %r = arith.mulf %v, %c
  affine.store %r, %B[%i]
}
```

**向量化后**:
```mlir
affine.for %i = 0 to 1024 step 128 {
  %v = vector.transfer_read %A[%i] : memref<1024xf32>, vector<128xf32>
  %r = arith.mulf %v, %vc : vector<128xf32>
  vector.transfer_write %r, %B[%i] : vector<128xf32>, memref<1024xf32>
}
```

**效果**: 充分利用SIMD指令

---

## 11. Affine Scalar Replacement (标量替换)

**文件**: `AffineScalarReplacement.cpp`

### 11.1 作用
前向传播store到load,消除中间memref,将memref访问替换为SSA值。

### 11.2 技术原理

#### 优化类型
1. **Store-to-Load Forwarding**: store值直接传递给load
2. **Redundant Load Elimination**: 消除冗余load

#### 必要条件
- 支配性分析
- 后支配性分析
- 别名分析

### 11.3 实例演示

**优化前**:
```mlir
affine.for %i = 0 to 100 {
  affine.store %val, %tmp[0]
  %v = affine.load %tmp[0]
  affine.store %v, %B[%i]
}
```

**优化后**:
```mlir
affine.for %i = 0 to 100 {
  affine.store %val, %B[%i]  // 直接使用%val,消除%tmp
}
```

**效果**: 减少内存访问,可能完全消除临时buffer

---

## 12. Loop Normalization (循环归一化)

**文件**: `AffineLoopNormalize.cpp`

### 12.1 作用
将循环归一化为标准形式:下界为0,步长为1。

### 12.2 技术原理

#### 归一化变换
- 调整lower bound到0
- 调整step到1
- 更新循环体中的IV使用

#### 支持的操作
- `affine.for`
- `affine.parallel`

### 12.3 实例演示

**归一化前**:
```mlir
affine.for %i = 10 to 100 step 5 {
  %idx = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  affine.store %val, %A[%idx]
}
```

**归一化后**:
```mlir
affine.for %i_norm = 0 to 18 {  // (100-10)/5 = 18
  %i = affine.apply affine_map<(d0) -> (d0 * 5 + 10)>(%i_norm)
  %idx = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  affine.store %val, %A[%idx]
}
```

**效果**: 简化循环分析和变换

---

## 13. Simplify Affine Structures (仿射结构简化)

**文件**: `SimplifyAffineStructures.cpp`

### 13.1 作用
简化affine map和integer set,移除冗余表达式。

### 13.2 技术原理

#### 简化策略
1. 使用`simplifyAffineExpr`简化表达式
2. 使用`MutableAffineMap`进行化简
3. 应用canonicalization patterns

### 13.3 实例演示

**简化前**:
```mlir
%0 = affine.apply affine_map<(d0) -> (d0 + 5 - 5)>(%i)
%1 = affine.apply affine_map<(d0, d1) -> (d0 * 1 + d1 * 0)>(%a, %b)
```

**简化后**:
```mlir
%0 = %i  // d0 + 5 - 5 = d0
%1 = %a  // d0 * 1 + d1 * 0 = d0
```

**效果**: 消除冗余计算,简化IR

---

## 14. Simplify Affine Min/Max (最小/最大值简化)

**文件**: `SimplifyAffineMinMax.cpp`

### 14.1 作用
通过界限分析简化affine.min/affine.max操作。

### 14.2 技术原理

#### 核心算法
1. 使用ValueBoundsConstraintSet分析
2. 对min/max的每个结果进行比较
3. 合并可证明有界限关系的表达式

### 14.3 实例演示

**简化前**:
```mlir
%0 = affine.min affine_map<(d0) -> (d0, d0 + 10, 100)>(%i)
// 如果可证明 %i < 90
```

**简化后**:
```mlir
%0 = affine.min affine_map<(d0) -> (d0, d0 + 10)>(%i)
// 移除常量100,因为它永远不是最小值
```

**效果**: 简化条件判断,可能消除min/max

---

## 15. Decompose Affine Ops (仿射操作分解)

**文件**: `DecomposeAffineOps.cpp`

### 15.1 作用
将复杂的affine.apply操作分解为更细粒度的操作。

### 15.2 技术原理

#### 分解策略
1. **重排操作数**: 按hoistability排序
2. **表达式分解**: 将二元表达式分解为多个子表达式
3. **重关联**: 按符号依赖重新关联

### 15.3 实例演示

**分解前**:
```mlir
%r = affine.apply affine_map<(s0, s1, s2) -> (s0 + s1 * s2 + s0 * s1)>(%a, %b, %c)
```

**分解后**:
```mlir
%t1 = affine.apply affine_map<(s0, s1) -> (s1 * s0)>(%b, %c)
%t2 = affine.apply affine_map<(s0, s1) -> (s0 * s1)>(%a, %b)
%t3 = affine.apply affine_map<(s0, s1) -> (s0 + s1)>(%a, %t1)
%r = affine.apply affine_map<(s0, s1) -> (s0 + s1)>(%t3, %t2)
```

**效果**: 更好的CSE和LICM机会

---

## 16. Expand Index Ops (索引操作展开)

**文件**: `AffineExpandIndexOps.cpp`

### 16.1 作用
将高级索引操作(linearize/delinearize)展开为基础算术操作。

### 16.2 技术原理

#### 支持的操作
1. **affine.linearize_index**: 多维索引→线性索引
2. **affine.delinearize_index**: 线性索引→多维索引

#### 展开算法
- Linearize: `result = idx[0] * stride[0] + idx[1] * stride[1] + ...`
- Delinearize: 使用除法和取模运算

### 16.3 实例演示

**展开前**:
```mlir
%linear = affine.linearize_index [%i, %j, %k] by (10, 20, 30)
```

**展开后**:
```mlir
%s0 = arith.constant 600 : index  // 20 * 30
%s1 = arith.constant 30 : index
%t0 = arith.muli %i, %s0
%t1 = arith.muli %j, %s1
%t2 = arith.addi %t0, %t1
%linear = arith.addi %t2, %k
```

**效果**: 转换为可优化的算术操作

---

## 17. Raise Memref Dialect (提升Memref方言)

**文件**: `RaiseMemrefDialect.cpp`

### 17.1 作用
将通用的memref.load/store操作提升为affine.load/store操作。

### 17.2 技术原理

#### 转换条件
1. 索引表达式可转换为affine expression
2. 支持的表达式: 常量、加法、乘法、循环IV

#### 转换过程
1. 分析索引表达式
2. 构造affine map
3. 替换为affine操作

### 17.3 实例演示

**提升前**:
```mlir
%idx = arith.addi %i, %c10
%v = memref.load %A[%idx] : memref<100xf32>
```

**提升后**:
```mlir
%v = affine.load %A[%i + 10] : memref<100xf32>
```

**效果**: 启用更多affine优化

---

## 18. Reify Value Bounds (值界限具体化)

**文件**: `ReifyValueBounds.cpp`

### 18.1 作用
将抽象的值界限约束具体化为可执行的affine操作。

### 18.2 技术原理

#### 核心功能
1. 计算值的上/下界
2. 生成affine.apply表达式
3. 处理动态维度

#### 应用场景
- 边界检查消除
- 循环界限推导

### 18.3 实例演示

**界限推导**:
```mlir
%dim = tensor.dim %t, %c0 : tensor<?xf32>
// 已知: %dim <= 100
%bound = affine.min affine_map<(d0) -> (d0, 100)>(%dim)
// 简化为: %bound = %dim (如果可证明 %dim <= 100)
```

---

## 总结

### 优化Pipeline建议

典型的Affine优化流程:

1. **Normalize** → 归一化循环
2. **Raise Memref** → 提升为affine操作
3. **LICM** → 循环不变代码外提
4. **Scalar Replacement** → 标量替换
5. **Loop Fusion** → 循环融合
6. **Loop Tiling** → 循环分块
7. **Parallelize** → 并行化
8. **Data Copy** → 显式数据移动
9. **Pipeline** → 数据传输流水线
10. **Vectorize** → 向量化
11. **Unroll** → 循环展开
12. **Simplify** → 简化优化

### 性能收益

- **循环融合**: 减少内存访问50-80%
- **循环分块**: 提高缓存命中率2-10x
- **数据拷贝**: 减少内存延迟3-5x
- **向量化**: SIMD加速4-16x
- **并行化**: 多核加速接近核心数

### 适用场景

Affine变换特别适合:
- 密集线性代数计算
- 图像/信号处理
- 科学计算
- 深度学习kernel优化

**注意**: Affine分析要求循环界限和访问模式是仿射的,对于不规则访问模式效果有限。
