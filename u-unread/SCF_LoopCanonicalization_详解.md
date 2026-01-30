# MLIR SCF 循环规范化技术深度解析

## 目录
- [概述](#概述)
- [背景与动机](#背景与动机)
- [核心概念](#核心概念)
- [技术原理](#技术原理)
- [优化模式详解](#优化模式详解)
- [实例详解](#实例详解)
- [约束求解机制](#约束求解机制)
- [源码解析](#源码解析)
- [性能分析](#性能分析)
- [局限性与展望](#局限性与展望)
- [总结](#总结)

---

## 概述

### 什么是循环规范化（Loop Canonicalization）？

循环规范化是 MLIR 编译器中的一个优化过程，用于**简化循环内部的表达式**，特别是那些依赖于循环归纳变量（induction variable）的仿射运算（affine min/max）和维度查询操作（dim ops）。

**核心思想**：利用循环的边界信息（lower bound、upper bound、step），通过约束求解来简化循环内部的计算。

**文件位置**: `mlir/lib/Dialect/SCF/Transforms/LoopCanonicalization.cpp`

### 简单类比

想象你在一个有围栏的操场上跑步：
- 起点：0 米（lower bound）
- 终点：100 米（upper bound）
- 步长：10 米（step）

你的位置 `i` 永远在 `[0, 10, 20, ..., 90]` 之间。

如果有人问："你离终点多远？"（即计算 `min(10, 100 - i)`）

**未优化的回答**：
- 每次都要计算 `100 - i`，然后和 `10` 比较

**优化后的回答**：
- 编译器知道在循环内部，`100 - i >= 10` 永远成立（因为步长是 10）
- 所以答案总是 `10`，不需要每次计算！

这就是循环规范化的核心价值。

---

## 背景与动机

### 为什么需要循环规范化？

在数值计算和深度学习编译中，循环内部经常出现与循环边界相关的计算：

#### 问题 1：冗余的 Min/Max 计算

**场景：分块（Tiling）处理数据**
```mlir
// 处理一个数组，每次处理 tile_size 个元素
scf.for %i = %c0 to %c100 step %c10 {
  // 计算当前块的实际大小（防止越界）
  %actual_size = affine.min affine_map<(d0)[s0] -> (10, s0 - d0)> (%i)[%c100]
  // 使用 %actual_size 进行处理...
}
```

**问题**：
- 在大多数迭代中，`%actual_size` 都是 `10`
- 只有最后一次迭代可能不足 `10`
- 但编译器每次都要计算 `min(10, 100 - i)`

**影响**：
```
循环 10 次，每次执行：
  - 1 次减法 (100 - i)
  - 1 次比较 (10 vs 100-i)
  - 1 次选择 (min)
总计：30 次额外操作

优化后：
  - 直接使用常量 10
总计：0 次额外操作
```

#### 问题 2：动态张量维度查询冗余

**场景：张量循环处理**
```mlir
%init = tensor.empty(...) : tensor<?x?xf32>  // 初始张量
%result = scf.for %i = %c0 to %c10 step %c1
    iter_args(%arg = %init) -> (tensor<?x?xf32>) {

  // 每次迭代都查询维度（即使维度不变）
  %dim0 = tensor.dim %arg, %c0 : tensor<?x?xf32>

  // 使用 %dim0 进行计算...
  %new_arg = some_operation(%arg, %dim0)
  scf.yield %new_arg
}
```

**问题**：
- 如果循环保持张量形状不变（shape-preserving）
- 每次查询 `%arg` 的维度都会得到相同结果
- 但每次迭代都要执行维度查询操作

**影响**：
```
循环 10 次，每次：
  - 1 次内存访问（读取形状元数据）
  - 1 次间接寻址

优化后：
  - 一次性查询初始维度
  - 后续直接使用缓存值
```

### 真实世界的影响

#### 深度学习示例：卷积 Tiling

```python
# PyTorch 风格的伪代码
for batch in range(0, N, tile_size):
    actual_batch = min(tile_size, N - batch)
    process_batch(data[batch:batch+actual_batch])
```

在 MLIR 中，这会生成如下 IR：
```mlir
scf.for %b = %c0 to %N step %tile_size {
  %size = affine.min affine_map<(d0)[s0, s1] -> (s0, s1 - d0)>
                               (%b)[%tile_size, %N]
  // 每次迭代都计算 min
}
```

**性能数据**（假设场景）：
```
输入：N = 1024, tile_size = 64
迭代次数：16
未优化：16 次 min 计算 = ~48 CPU 周期
优化后：0 次计算（编译期常量）
```

对于嵌套循环，节省会成倍增长！

---

## 核心概念

### SCF 方言（Structured Control Flow Dialect）

SCF 是 MLIR 中的结构化控制流方言，提供高层次的循环抽象：

#### `scf.for` - 标准 for 循环
```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%arg = %init) -> (tensor<?xf32>) {
  %new_val = some_op(%arg, %iv)
  scf.yield %new_val : tensor<?xf32>
} -> tensor<?xf32>
```

**关键元素**：
- `%iv`: 归纳变量（induction variable），范围 `[lb, ub)`，步长 `step`
- `iter_args`: 循环携带的值（类似 SSA 的 phi 节点）
- `scf.yield`: 返回值给下一次迭代或循环外

#### `scf.parallel` - 并行循环
```mlir
scf.parallel (%i, %j) = (%lb1, %lb2) to (%ub1, %ub2) step (%s1, %s2) {
  // 并行执行的循环体
  scf.yield
}
```

#### `scf.forall` - 新的并行抽象
```mlir
scf.forall (%i, %j) in (%M, %N) {
  // 现代并行循环，支持更多语义
  scf.forall.in_parallel {
    // 归约操作
  }
}
```

### 仿射运算（Affine Operations）

#### `affine.min` - 仿射最小值
```mlir
// 语法：affine.min #map(dims...)[symbols...]
%result = affine.min affine_map<(d0, d1)[s0] -> (d0, d1 - s0)> (%i, %ub)[%step]
// 计算 min(%i, %ub - %step)
```

#### `affine.max` - 仿射最大值
```mlir
%result = affine.max affine_map<(d0) -> (0, -d0)> (%i)
// 计算 max(0, -i)
```

#### 仿射映射（Affine Map）
```mlir
affine_map<(d0, d1)[s0, s1] -> (d0 * 2 + d1, s0 - d1)>
```

**组成部分**：
- `(d0, d1)`: **维度**（dimensions）- 通常是循环归纳变量
- `[s0, s1]`: **符号**（symbols）- 循环不变量（如边界、步长）
- `-> (...)`: **表达式**（affine expressions）- 线性组合

**限制**：
- 表达式必须是仿射的（线性 + 常数）
- 支持：加法、减法、常数乘法、floor/ceil 除法
- 不支持：乘法（变量间）、取模、非线性运算

### 张量维度操作

#### `tensor.dim` - 查询张量维度
```mlir
%d0 = tensor.dim %tensor, %c0 : tensor<?x?xf32>
// 查询第 0 维的大小（运行时值）
```

#### `memref.dim` - 查询内存引用维度
```mlir
%d1 = memref.dim %memref, %c1 : memref<?x?xf32>
```

### 形状保持性（Shape Preservation）

**定义**：如果循环的每次迭代产生的张量形状与输入相同，则称该循环**形状保持**。

**示例 1：形状保持** ✅
```mlir
scf.for %i = ... iter_args(%arg = %init) -> (tensor<?x?xf32>) {
  %slice = tensor.extract_slice %arg[0, 0][10, 10][1, 1]
  %processed = some_op(%slice)
  %result = tensor.insert_slice %processed into %arg[0, 0][10, 10][1, 1]
  scf.yield %result  // 形状与 %init 相同
}
```

**示例 2：非形状保持** ❌
```mlir
scf.for %i = ... iter_args(%arg = %init) -> (tensor<?x?xf32>) {
  %expanded = tensor.expand_shape %arg [[0, 1], [2]]
  scf.yield %expanded  // 形状改变了！
}
```

---

## 技术原理

### 整体架构

循环规范化由三个主要优化模式组成：

```
LoopCanonicalization Pass
├── AffineOpSCFCanonicalizationPattern
│   ├── 目标：affine.min / affine.max
│   └── 策略：基于循环边界的约束求解
├── DimOfIterArgFolder
│   ├── 目标：tensor.dim / memref.dim (作用于 iter_args)
│   └── 策略：替换为初始参数的 dim 操作
└── DimOfLoopResultFolder
    ├── 目标：tensor.dim / memref.dim (作用于循环结果)
    └── 策略：替换为初始参数的 dim 操作
```

### 核心思想：约束求解

**基本原理**：
1. 收集循环边界信息构建约束系统
2. 将约束系统应用于 affine min/max 表达式
3. 使用 Presburger 算术简化表达式
4. 如果简化成功，替换原操作

**约束系统示例**：
```
循环：scf.for %i = %c0 to %c100 step %c10

约束：
  - %i >= 0              (下界)
  - %i < 0 + 10 * ((100 - 0 - 1) floordiv 10) + 1 = 100  (上界)
  - %i 是 10 的倍数      (步长)

表达式：min(10, 100 - %i)

求解：
  由于 %i < 100，所以 100 - %i > 0
  由于 %i 是 10 的倍数，且 %i <= 90，所以 100 - %i >= 10
  因此 min(10, 100 - %i) = 10 (常量)
```

### 约束系统构建

**代码位置**: `AffineCanonicalizationUtils.cpp:77-134`

#### 上界计算公式
```cpp
// 通用公式（多次迭代）
iv < lb + step * ((ub - lb - 1) floordiv step) + 1

// 单次迭代优化
if (lb + step >= ub) {
  iv < lb + 1
}
```

**直觉理解**：
```
循环：for i = 0 to 100 step 10

最后一次迭代的 i 值：
  = lb + step * floor((ub - lb - 1) / step)
  = 0 + 10 * floor((100 - 0 - 1) / 10)
  = 0 + 10 * floor(99 / 10)
  = 0 + 10 * 9
  = 90

所以 i < 90 + 10 = 100 ✓
```

#### 约束表示：整数多面体（Integer Polyhedron）

**数学表示**：
```
约束系统 = {(iv, lb, ub) ∈ ℤ³ | A × [iv, lb, ub]ᵀ + b ≥ 0}

例如：
  iv >= lb  →  [1, -1, 0] × [iv, lb, ub]ᵀ + 0 ≥ 0
  iv < ub   →  [-1, 0, 1] × [iv, lb, ub]ᵀ + (-1) ≥ 0
```

**实现**：使用 MLIR 的 `FlatAffineValueConstraints` 类

---

## 优化模式详解

### 模式 1：Affine Min/Max 规范化

**Pattern**: `AffineOpSCFCanonicalizationPattern`

#### 工作流程

```
┌─────────────────────────────────────────┐
│  发现 affine.min/max 操作                │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  遍历操作数，查找归纳变量                 │
│  - 调用 matchForLikeLoop()               │
│  - 获取 lb, ub, step                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  构建约束系统                            │
│  - 调用 addLoopRangeConstraints()       │
│  - 为每个 IV 添加边界约束                │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  简化表达式                              │
│  - 调用 simplifyConstrainedMinMaxOp()   │
│  - Presburger 算术求解                   │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    成功 ▼           ▼ 失败
┌────────────┐  ┌──────────┐
│ 替换为     │  │ 保持不变  │
│ affine.apply│ │          │
└────────────┘  └──────────┘
```

#### 关键函数调用链

```cpp
// 1. Pattern 入口
AffineOpSCFCanonicalizationPattern::matchAndRewrite(op, rewriter)
  ↓
// 2. 调用通用规范化函数
canonicalizeMinMaxOpInLoop(rewriter, op, matchForLikeLoop)
  ↓
// 3. 遍历操作数，匹配循环
for (Value operand : op->getOperands()) {
  matchForLikeLoop(operand, lb, ub, step)  // 获取循环信息
  ↓
  addLoopRangeConstraints(constraints, iv, lb, ub, step)  // 添加约束
}
  ↓
// 4. 简化表达式
canonicalizeMinMaxOp(rewriter, op, constraints)
  ↓
simplifyConstrainedMinMaxOp(op, constraints)  // Affine 分析
  ↓
// 5. 替换操作
rewriter.replaceOpWithNewOp<AffineApplyOp>(op, simplifiedMap, operands)
```

#### 支持的循环类型

**`matchForLikeLoop` 函数** (`AffineCanonicalizationUtils.cpp:31-62`)：

```cpp
LogicalResult matchForLikeLoop(Value iv, OpFoldResult &lb,
                                OpFoldResult &ub, OpFoldResult &step) {
  // 1. scf.for
  if (scf::ForOp forOp = scf::getForInductionVarOwner(iv)) {
    lb = forOp.getLowerBound();
    ub = forOp.getUpperBound();
    step = forOp.getStep();
    return success();
  }

  // 2. scf.parallel
  if (scf::ParallelOp parOp = scf::getParallelForInductionVarOwner(iv)) {
    // 查找对应维度的 iv
    for (unsigned idx = 0; idx < parOp.getNumLoops(); ++idx) {
      if (parOp.getInductionVars()[idx] == iv) {
        lb = parOp.getLowerBound()[idx];
        ub = parOp.getUpperBound()[idx];
        step = parOp.getStep()[idx];
        return success();
      }
    }
  }

  // 3. scf.forall
  if (scf::ForallOp forallOp = scf::getForallOpThreadIndexOwner(iv)) {
    // 类似处理
  }

  return failure();
}
```

### 模式 2：Iter Arg 的 Dim 折叠

**Pattern**: `DimOfIterArgFolder<tensor::DimOp>` / `DimOfIterArgFolder<memref::DimOp>`

#### 转换逻辑

**原始代码**：
```mlir
%init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
scf.for %i = ... iter_args(%arg = %init) -> (tensor<?x?xf32>) {
  %dim = tensor.dim %arg, %c0  // 查询 iter_arg 的维度
  // 使用 %dim...
  scf.yield %new_arg
}
```

**优化后**：
```mlir
%init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
scf.for %i = ... iter_args(%arg = %init) -> (tensor<?x?xf32>) {
  %dim = tensor.dim %init, %c0  // 查询初始值的维度（循环不变）
  // 使用 %dim...
  scf.yield %new_arg
}
```

**好处**：
- `%init` 是循环不变量，编译器可以提升到循环外
- 减少循环内的内存访问

#### 实现细节

**代码位置**: `LoopCanonicalization.cpp:87-107`

```cpp
template <typename OpTy>
LogicalResult DimOfIterArgFolder<OpTy>::matchAndRewrite(
    OpTy dimOp, PatternRewriter &rewriter) const {

  // 1. 检查 dimOp 的源是否为 block argument
  auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
  if (!blockArg) return failure();

  // 2. 检查 block 的父操作是否为 scf.for
  auto forOp = dyn_cast<ForOp>(blockArg.getParentBlock()->getParentOp());
  if (!forOp) return failure();

  // 3. 检查循环是否形状保持（shape-preserving）
  if (!isShapePreserving(forOp, blockArg.getArgNumber() - 1))
    return failure();

  // 4. 获取对应的初始参数
  Value initArg = forOp.getTiedLoopInit(blockArg)->get();

  // 5. 替换 dimOp 的源
  rewriter.modifyOpInPlace(dimOp, [&]() {
    dimOp.getSourceMutable().assign(initArg);
  });

  return success();
}
```

#### 形状保持性分析

**函数**: `isShapePreserving` (`LoopCanonicalization.cpp:38-61`)

**策略**：向后追踪 `scf.yield` 的值，检查它是否最终来自原始 iter_arg

**支持的模式**：

1. **直接返回 iter_arg**
```mlir
scf.for ... iter_args(%arg = %init) {
  scf.yield %arg  // ✅ 形状保持
}
```

2. **通过 tensor.insert_slice**
```mlir
scf.for ... iter_args(%arg = %init) {
  %slice = tensor.extract_slice %arg[...][...][...]
  %processed = some_op(%slice)
  %result = tensor.insert_slice %processed into %arg[...][...][...]
  scf.yield %result  // ✅ 形状保持（目标是 %arg）
}
```

3. **嵌套 for 循环**
```mlir
scf.for ... iter_args(%arg1 = %init) {
  %result = scf.for ... iter_args(%arg2 = %arg1) {
    scf.yield %arg2  // 内层形状保持
  }
  scf.yield %result  // ✅ 外层也形状保持
}
```

**代码逻辑**：
```cpp
static bool isShapePreserving(ForOp forOp, int64_t arg) {
  Value value = forOp.getYieldedValues()[arg];

  while (value) {
    // 情况 1: 直接是 iter_arg
    if (value == forOp.getRegionIterArgs()[arg])
      return true;

    OpResult opResult = dyn_cast<OpResult>(value);
    if (!opResult) return false;

    // 情况 2: 通过特定操作
    value = TypeSwitch<Operation *, Value>(opResult.getOwner())
      .Case<tensor::InsertSliceOp>([&](auto op) {
        return op.getDest();  // 目标张量
      })
      .Case<ForOp>([&](ForOp nestedFor) {
        // 递归检查嵌套循环
        return isShapePreserving(nestedFor, opResult.getResultNumber())
          ? nestedFor.getInitArgs()[opResult.getResultNumber()]
          : Value();
      })
      .Default([&](auto) { return Value(); });
  }

  return false;
}
```

### 模式 3：Loop Result 的 Dim 折叠

**Pattern**: `DimOfLoopResultFolder<tensor::DimOp>` / `DimOfLoopResultFolder<memref::DimOp>`

#### 转换逻辑

**原始代码**：
```mlir
%init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
%result = scf.for %i = ... iter_args(%arg = %init) -> (tensor<?x?xf32>) {
  // 循环体
  scf.yield %new_arg
}
%dim = tensor.dim %result, %c0  // 查询循环结果的维度
```

**优化后**：
```mlir
%init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
%result = scf.for %i = ... iter_args(%arg = %init) -> (tensor<?x?xf32>) {
  // 循环体
  scf.yield %new_arg
}
%dim = tensor.dim %init, %c0  // 查询初始值的维度
```

**代码位置**: `LoopCanonicalization.cpp:132-149`

---

## 实例详解

### 实例 1：基本 Min 规范化

#### 原始 MLIR
```mlir
func.func @scf_for_canonicalize_min(%A : memref<i64>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  scf.for %i = %c0 to %c4 step %c2 {
    // 计算 min(2, 4 - %i)
    %1 = affine.min affine_map<(d0, d1)[] -> (2, d1 - d0)> (%i, %c4)
    %2 = arith.index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}
```

#### 执行过程分析

**迭代 0**: `%i = 0`
```
min(2, 4 - 0) = min(2, 4) = 2
```

**迭代 1**: `%i = 2`
```
min(2, 4 - 2) = min(2, 2) = 2
```

**观察**：所有迭代的结果都是 `2`！

#### 约束求解过程

**步骤 1：构建约束系统**
```
变量：
  d0 = %i (维度)
  s0 = %c4 (符号)

约束：
  d0 >= 0                           // 下界
  d0 < 0 + 2 * ((4 - 0 - 1) / 2) + 1 = 4  // 上界

简化：
  0 <= d0 < 4
  d0 ∈ {0, 2}  (因为 step = 2)
```

**步骤 2：分析表达式**
```
表达式：min(2, s0 - d0) = min(2, 4 - d0)

由于 d0 ∈ {0, 2}：
  当 d0 = 0: min(2, 4) = 2
  当 d0 = 2: min(2, 2) = 2

结论：表达式恒等于 2
```

**步骤 3：简化**
```
原操作：
  %1 = affine.min affine_map<(d0, d1) -> (2, d1 - d0)> (%i, %c4)

简化为：
  %1 = arith.constant 2 : index
```

#### 优化后的 MLIR
```mlir
func.func @scf_for_canonicalize_min(%A : memref<i64>) {
  %c2 = arith.constant 2 : i64

  scf.for %i = %c0 to %c4 step %c2 {
    // 直接使用常量 2
    memref.store %c2, %A[]: memref<i64>
  }
  return
}
```

**性能提升**：
- 消除：2 次减法、2 次比较、2 次选择
- 增加：0（常量折叠）
- 净提升：~6 条指令

---

### 实例 2：嵌套循环规范化

#### 原始 MLIR
```mlir
func.func @scf_for_loop_nest_canonicalize_min(%A : memref<i64>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index

  scf.for %i = %c0 to %c4 step %c2 {
    scf.for %j = %c0 to %c6 step %c3 {
      // 计算 min(5, (4 - %i) + (6 - %j))
      %1 = affine.min affine_map<(d0, d1, d2, d3)[] -> (5, d1 + d3 - d0 - d2)>
                                (%i, %c4, %j, %c6)
      %2 = arith.index_cast %1: index to i64
      memref.store %2, %A[]: memref<i64>
    }
  }
  return
}
```

#### 迭代空间分析

**外层循环**：`%i ∈ {0, 2}`
**内层循环**：`%j ∈ {0, 3}`

**所有组合**：
```
(%i, %j) = (0, 0): min(5, 4 + 6 - 0 - 0) = min(5, 10) = 5
(%i, %j) = (0, 3): min(5, 4 + 6 - 0 - 3) = min(5, 7)  = 5
(%i, %j) = (2, 0): min(5, 4 + 6 - 2 - 0) = min(5, 8)  = 5
(%i, %j) = (2, 3): min(5, 4 + 6 - 2 - 3) = min(5, 5)  = 5
```

**结论**：恒等于 `5`

#### 约束系统（多归纳变量）

```
变量：
  d0 = %i, d1 = %j (维度)
  s0 = %c4, s1 = %c6 (符号)

约束：
  0 <= d0 < 4, d0 ∈ {0, 2}
  0 <= d1 < 6, d1 ∈ {0, 3}
  s0 = 4, s1 = 6

表达式：min(5, s0 + s1 - d0 - d1) = min(5, 10 - d0 - d1)

边界分析：
  最小值：d0 = 2, d1 = 3 → 10 - 2 - 3 = 5
  最大值：d0 = 0, d1 = 0 → 10 - 0 - 0 = 10

因此：10 - d0 - d1 ∈ [5, 10]
     min(5, 10 - d0 - d1) = 5
```

#### 优化后的 MLIR
```mlir
func.func @scf_for_loop_nest_canonicalize_min(%A : memref<i64>) {
  %c5 = arith.constant 5 : i64

  scf.for %i = %c0 to %c4 step %c2 {
    scf.for %j = %c0 to %c6 step %c3 {
      memref.store %c5, %A[]: memref<i64>
    }
  }
  return
}
```

---

### 实例 3：部分规范化

#### 原始 MLIR
```mlir
func.func @scf_for_canonicalize_partly(%A : memref<i64>) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index

  scf.for %i = %c1 to %c256 step %c16 {
    // 计算 min(256, 256 - %i)
    %1 = affine.min affine_map<(d0) -> (256, 256 - d0)> (%i)
    %2 = arith.index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}
```

#### 为什么只能部分规范化？

**迭代分析**：
```
%i ∈ {1, 17, 33, ..., 241}

256 - %i ∈ {255, 239, 223, ..., 15}

对比：
  当 %i = 1:   min(256, 255) = 255
  当 %i = 17:  min(256, 239) = 239
  当 %i = 241: min(256, 15)  = 15

结果不是常量！
```

**但可以简化一半**：
```
min(256, 256 - d0)

由于 d0 >= 1，所以 256 - d0 <= 255 < 256
因此 min(256, 256 - d0) = 256 - d0
```

#### 优化后的 MLIR
```mlir
func.func @scf_for_canonicalize_partly(%A : memref<i64>) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index

  scf.for %i = %c1 to %c256 step %c16 {
    // 简化为 affine.apply（消除 min）
    %1 = affine.apply affine_map<(d0) -> (256 - d0)> (%i)
    %2 = arith.index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}
```

**收益**：
- 消除了 `min` 比较操作
- 仍需要减法，但避免了分支

---

### 实例 4：Tensor Dim 折叠

#### 原始 MLIR
```mlir
func.func @tensor_dim_of_iter_arg(%t : tensor<?x?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %result, %dim = scf.for %i = %c0 to %c10 step %c1
      iter_args(%arg = %t, %d = %c0) -> (tensor<?x?xf32>, index) {
    // 每次迭代都查询 %arg 的第 0 维
    %dim = tensor.dim %arg, %c0 : tensor<?x?xf32>
    scf.yield %arg, %dim : tensor<?x?xf32>, index
  }

  return %dim : index
}
```

#### 形状保持性验证

**追踪 yield 值**：
```
scf.yield %arg, %dim

%arg 来自哪里？
  → iter_args(%arg = %t, ...)

循环体中 %arg 被修改了吗？
  → 没有，直接 yield

结论：形状保持 ✅
```

#### 优化后的 MLIR
```mlir
func.func @tensor_dim_of_iter_arg(%t : tensor<?x?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %result, %dim = scf.for %i = %c0 to %c10 step %c1
      iter_args(%arg = %t, %d = %c0) -> (tensor<?x?xf32>, index) {
    // 查询初始值 %t 的维度（循环不变）
    %dim = tensor.dim %t, %c0 : tensor<?x?xf32>
    scf.yield %arg, %dim : tensor<?x?xf32>, index
  }

  return %dim : index
}
```

**后续优化**：
- 循环不变量提升（Loop Invariant Code Motion）会将 `tensor.dim %t, %c0` 移到循环外
- 最终 `%dim` 在所有迭代中都是相同的常量

---

### 实例 5：InsertSlice 的形状保持

#### 原始 MLIR
```mlir
func.func @tensor_dim_of_iter_arg_insertslice(
    %t : tensor<?x?xf32>,
    %t2 : tensor<10x10xf32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %result, %dim = scf.for %i = %c0 to %c10 step %c1
      iter_args(%arg = %t, %d = %c0) -> (tensor<?x?xf32>, index) {

    %dim = tensor.dim %arg, %c0 : tensor<?x?xf32>

    // 插入切片操作
    %2 = tensor.insert_slice %t2 into %arg[0, 0] [10, 10] [1, 1]
        : tensor<10x10xf32> into tensor<?x?xf32>
    %3 = tensor.insert_slice %t2 into %2[1, 1] [10, 10] [1, 1]
        : tensor<10x10xf32> into tensor<?x?xf32>

    scf.yield %3, %dim : tensor<?x?xf32>, index
  }

  return %dim : index
}
```

#### 形状保持性分析

**追踪链**：
```
yield %3

%3 来自：tensor.insert_slice %t2 into %2[...]
       → 目标是 %2

%2 来自：tensor.insert_slice %t2 into %arg[...]
       → 目标是 %arg

%arg 来自：iter_args(%arg = %t, ...)

结论：%3 的形状 = %2 的形状 = %arg 的形状 = %t 的形状
     形状保持 ✅
```

**关键代码**：
```cpp
value = TypeSwitch<Operation *, Value>(opResult.getOwner())
  .Case<InsertSliceOp>([&](InsertSliceOp op) {
    return op.getDest();  // 返回目标张量，继续追踪
  })
```

#### 优化后
```mlir
// 同样将 tensor.dim %arg 替换为 tensor.dim %t
%dim = tensor.dim %t, %c0
```

---

### 实例 6：非规范化情况

#### 情况 1：步长不能整除
```mlir
func.func @scf_for_not_canonicalizable_1(%A : memref<i64>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  scf.for %i = %c1 to %c4 step %c2 {
    // 计算 min(2, 4 - %i)
    %1 = affine.min affine_map<(d0)[s0] -> (2, s0 - d0)> (%i)[%c4]
    %2 = arith.index_cast %1: index to i64
    memref.store %2, %A[]: memref<i64>
  }
  return
}
```

**为什么不能规范化？**
```
%i ∈ {1, 3}

迭代 1: %i = 1 → min(2, 4 - 1) = min(2, 3) = 2
迭代 2: %i = 3 → min(2, 4 - 3) = min(2, 1) = 1

结果不同！不能简化为常量
```

#### 情况 2：非形状保持
```mlir
func.func @tensor_dim_of_iter_arg_no_canonicalize(
    %t : tensor<?x?xf32>,
    %t2 : tensor<?x?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %result, %dim = scf.for %i = %c0 to %c10 step %c1
      iter_args(%arg = %t, %d = %c0) -> (tensor<?x?xf32>, index) {
    %dim = tensor.dim %arg, %c0 : tensor<?x?xf32>
    // 直接 yield %t2（不同的张量！）
    scf.yield %t2, %dim : tensor<?x?xf32>, index
  }

  return %dim : index
}
```

**形状保持性检查失败**：
```
yield %t2

%t2 是外部参数，不是 %arg！
追踪链断裂 ❌
不能优化
```

---

## 约束求解机制

### Presburger 算术简介

**定义**：Presburger 算术是一阶逻辑的一个子集，只包含：
- 整数常量
- 加法和减法
- 乘以常数
- 比较运算（<, ≤, =, ≥, >）
- 逻辑连接词（∧, ∨, ¬, →）
- 量词（∀, ∃）

**关键性质**：
- **可判定性**：所有 Presburger 公式都可以判定真假
- **不完备性**：不包含乘法（变量间）、除法、指数

**在 MLIR 中的应用**：
- 仿射表达式都是 Presburger 公式
- 可以自动化简化和求解

### 整数多面体表示

**几何直觉**：
```
约束 { 0 <= x < 10, 0 <= y < 10 }
在平面上是一个正方形：

y
10│        ┌────────┐
  │        │        │
  │        │        │
  │        │        │
 0└────────┴────────┴── x
  0                 10
```

**代数表示**：
```
Ax + b >= 0

其中：
  x = [iv, lb, ub]ᵀ (变量向量)
  A = 约束系数矩阵
  b = 常数向量
```

**示例**：
```
约束：0 <= iv < 10

矩阵形式：
┌              ┐   ┌    ┐     ┌   ┐
│  1   0   0  │   │ iv │     │ 0 │
│ -1   0   1  │ × │ lb │  +  │ -1│  >= 0
└              ┘   │ ub │     └   ┘
                   └    ┘

展开：
  1*iv + 0*lb + 0*ub + 0 >= 0  →  iv >= 0
 -1*iv + 0*lb + 1*ub - 1 >= 0  →  iv < ub
```

### 简化算法流程

**高层流程**：
```
┌─────────────────────────────────┐
│ 输入：affine.min/max 操作        │
│       + 约束系统                 │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│ 构建 AffineValueMap             │
│ - affine map                    │
│ - operands (dims + symbols)     │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│ 枚举所有候选表达式               │
│ (min/max 的每个分支)            │
└──────────────┬──────────────────┘
               │
         ┌─────┴─────┐
         │  for each │
         │  expression│
         └─────┬─────┘
               │
┌──────────────▼──────────────────┐
│ 检查是否在约束下恒定             │
│ - 添加表达式作为临时约束         │
│ - 检查多面体是否为空             │
└──────────────┬──────────────────┘
               │
         ┌─────┴─────┐
         │           │
    是   ▼           ▼ 否
┌─────────────┐ ┌─────────────┐
│ 该表达式    │ │ 继续下一个  │
│ 恒定成立    │ │             │
└──────┬──────┘ └─────────────┘
       │
┌──────▼──────────────────────────┐
│ 生成 affine.apply               │
│ (用简化后的表达式)               │
└─────────────────────────────────┘
```

**详细示例**：
```mlir
// 输入
%r = affine.min affine_map<(d0) -> (10, 100 - d0)> (%i)

// 约束：0 <= %i < 90 (步长 10)

// 候选表达式：
//   expr1 = 10
//   expr2 = 100 - d0

// 检查 expr1:
//   在约束下，是否 expr1 <= expr2？
//   即：10 <= 100 - d0
//      d0 <= 90
//   由于 d0 < 90 恒成立，所以 expr1 <= expr2 ✓

// 检查是否 expr1 是最小值（排除其他候选）
//   只有两个候选，且 expr1 <= expr2
//   所以 min(...) = expr1 = 10

// 输出
%r = affine.apply affine_map<() -> (10)> ()
   = arith.constant 10 : index
```

### FlatAffineValueConstraints API

**关键方法**：

```cpp
class FlatAffineValueConstraints {
public:
  // 添加变量
  unsigned appendDimVar(Value val);      // 添加维度变量
  unsigned appendSymbolVar(Value val);   // 添加符号变量

  // 添加约束
  void addBound(BoundType type, unsigned pos, int64_t value);
  void addBound(BoundType type, unsigned pos, AffineMap map);
  void addInequality(ArrayRef<int64_t> coeffs);

  // 查询
  bool isEmpty();  // 约束是否矛盾（无解）

  // 访问
  unsigned getNumDimVars();
  unsigned getNumSymbolVars();
  unsigned getNumCols();  // 总列数 = dims + symbols + 1(常数项)
};
```

**使用示例**：
```cpp
FlatAffineValueConstraints cstr;

// 添加 %i 为维度
unsigned dimI = cstr.appendDimVar(iv);  // 返回 0

// 添加 %lb, %ub 为符号
unsigned symLb = cstr.appendSymbolVar(lb);  // 返回 0
unsigned symUb = cstr.appendSymbolVar(ub);  // 返回 1

// 添加下界约束：iv >= lb
// 表示为：1*iv - 1*lb + 0 >= 0
SmallVector<int64_t> ineqLb(cstr.getNumCols(), 0);
ineqLb[dimI] = 1;     // iv 的系数
ineqLb[symLb] = -1;   // lb 的系数
cstr.addInequality(ineqLb);

// 添加上界约束：iv < ub
// 表示为：-1*iv + 1*ub - 1 >= 0
SmallVector<int64_t> ineqUb(cstr.getNumCols(), 0);
ineqUb[dimI] = -1;
ineqUb[symUb] = 1;
ineqUb[cstr.getNumCols() - 1] = -1;  // 常数项
cstr.addInequality(ineqUb);
```

---

## 源码解析

### Pass 入口

**文件**: `LoopCanonicalization.cpp:163-174`

```cpp
struct SCFForLoopCanonicalization
    : public impl::SCFForLoopCanonicalizationBase<SCFForLoopCanonicalization> {
  void runOnOperation() override {
    auto *parentOp = getOperation();  // 获取要优化的操作
    MLIRContext *ctx = parentOp->getContext();

    // 创建模式集合
    RewritePatternSet patterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);

    // 应用贪心重写
    if (failed(applyPatternsGreedily(parentOp, std::move(patterns))))
      signalPassFailure();
  }
};
```

**Pass 注册**：
```cpp
std::unique_ptr<Pass> mlir::createSCFForLoopCanonicalizationPass() {
  return std::make_unique<SCFForLoopCanonicalization>();
}
```

### 模式注册

**文件**: `LoopCanonicalization.cpp:176-185`

```cpp
void mlir::scf::populateSCFForLoopCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<
    // Affine min/max 规范化
    AffineOpSCFCanonicalizationPattern<affine::AffineMinOp>,
    AffineOpSCFCanonicalizationPattern<affine::AffineMaxOp>,

    // Tensor/MemRef dim 折叠
    DimOfIterArgFolder<tensor::DimOp>,
    DimOfIterArgFolder<memref::DimOp>,
    DimOfLoopResultFolder<tensor::DimOp>,
    DimOfLoopResultFolder<memref::DimOp>
  >(ctx);
}
```

### 约束构建核心函数

**文件**: `AffineCanonicalizationUtils.cpp:77-134`

```cpp
LogicalResult scf::addLoopRangeConstraints(
    FlatAffineValueConstraints &cstr,
    Value iv, OpFoldResult lb, OpFoldResult ub, OpFoldResult step) {

  Builder b(iv.getContext());

  // 1. 检查步长是否为常量（限制）
  auto stepInt = getConstantIntValue(step);
  if (!stepInt)
    return failure();  // 不支持动态步长

  // 2. 添加变量到约束系统
  unsigned dimIv = cstr.appendDimVar(iv);

  auto lbv = llvm::dyn_cast_if_present<Value>(lb);
  unsigned symLb = lbv ? cstr.appendSymbolVar(lbv)
                       : cstr.appendSymbolVar(/*num=*/1);

  auto ubv = llvm::dyn_cast_if_present<Value>(ub);
  unsigned symUb = ubv ? cstr.appendSymbolVar(ubv)
                       : cstr.appendSymbolVar(/*num=*/1);

  // 3. 如果 lb/ub 是常量，添加等式约束
  std::optional<int64_t> lbInt = getConstantIntValue(lb);
  std::optional<int64_t> ubInt = getConstantIntValue(ub);

  if (lbInt)
    cstr.addBound(BoundType::EQ, symLb, *lbInt);
  if (ubInt)
    cstr.addBound(BoundType::EQ, symUb, *ubInt);

  // 4. 添加下界约束：iv >= lb
  SmallVector<int64_t> ineqLb(cstr.getNumCols(), 0);
  ineqLb[dimIv] = 1;
  ineqLb[symLb] = -1;
  cstr.addInequality(ineqLb);

  // 5. 计算上界表达式
  AffineExpr ivUb;

  if (lbInt && ubInt && (*lbInt + *stepInt >= *ubInt)) {
    // 单次迭代：iv < lb + 1
    ivUb = b.getAffineSymbolExpr(symLb - cstr.getNumDimVars()) + 1;
  } else {
    // 多次迭代：iv < lb + step * floor((ub - lb - 1) / step) + 1
    AffineExpr exprLb = lbInt
      ? b.getAffineConstantExpr(*lbInt)
      : b.getAffineSymbolExpr(symLb - cstr.getNumDimVars());

    AffineExpr exprUb = ubInt
      ? b.getAffineConstantExpr(*ubInt)
      : b.getAffineSymbolExpr(symUb - cstr.getNumDimVars());

    ivUb = exprLb + 1 +
           (*stepInt * ((exprUb - exprLb - 1).floorDiv(*stepInt)));
  }

  // 6. 添加上界约束
  auto map = AffineMap::get(
    /*dimCount=*/cstr.getNumDimVars(),
    /*symbolCount=*/cstr.getNumSymbolVars(),
    /*result=*/ivUb);

  return cstr.addBound(BoundType::UB, dimIv, map);
}
```

**关键点**：

1. **动态步长限制**：
```cpp
auto stepInt = getConstantIntValue(step);
if (!stepInt) return failure();
```
因为整数多面体不支持半仿射（semi-affine）表达式。

2. **OpFoldResult 处理**：
```cpp
auto lbv = llvm::dyn_cast_if_present<Value>(lb);
unsigned symLb = lbv ? cstr.appendSymbolVar(lbv)
                     : cstr.appendSymbolVar(/*num=*/1);
```
`OpFoldResult` 可能是 `Value`（运行时）或 `Attribute`（编译时常量）。

3. **单次迭代优化**：
```cpp
if (lbInt && ubInt && (*lbInt + *stepInt >= *ubInt)) {
  ivUb = exprLb + 1;
}
```
如果 `lb + step >= ub`，循环最多执行一次。

### 形状保持性检查

**文件**: `LoopCanonicalization.cpp:38-61`

```cpp
static bool isShapePreserving(ForOp forOp, int64_t arg) {
  assert(arg < static_cast<int64_t>(forOp.getNumResults()));

  // 获取 yield 的值
  Value value = forOp.getYieldedValues()[arg];

  while (value) {
    // 情况 1: 直接是 iter_arg
    if (value == forOp.getRegionIterArgs()[arg])
      return true;

    // 必须是操作的结果
    OpResult opResult = dyn_cast<OpResult>(value);
    if (!opResult)
      return false;

    // 情况 2: 通过特定操作链
    value = llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())

      // InsertSliceOp: 返回目标张量
      .template Case<InsertSliceOp>([&](InsertSliceOp op) {
        return op.getDest();
      })

      // 嵌套 ForOp: 递归检查
      .template Case<ForOp>([&](ForOp nestedFor) {
        unsigned resNum = opResult.getResultNumber();
        if (isShapePreserving(nestedFor, resNum))
          return nestedFor.getInitArgs()[resNum];
        return Value();
      })

      // 其他操作: 不支持
      .Default([&](auto) { return Value(); });
  }

  return false;
}
```

**设计模式**：
- 使用 `TypeSwitch` 进行操作类型分派
- 递归处理嵌套循环
- 保守策略：不确定时返回 `false`

---

## 性能分析

### 理论分析

#### Affine Min/Max 优化收益

**未优化成本**：
```
affine.min #map(operands...)
  = evaluate_map(operands)  // N 次表达式求值
  + min(results)            // N-1 次比较
  + select_min              // 1 次选择

假设 N=2（最常见）:
  - 2 次表达式求值（可能包括加减乘除）
  - 1 次比较
  - 1 次条件移动

总计: ~5-10 CPU 周期（取决于表达式复杂度）
```

**优化后成本**：
```
arith.constant C
  = 0 周期（编译时常量）

或

affine.apply #simplified_map(operands)
  = 1 次表达式求值
  = ~2-4 CPU 周期
```

**净收益**：
- 常量情况：100% 节省（5-10 周期 → 0）
- 简化情况：50-80% 节省（5-10 周期 → 2-4）

#### Dim 操作优化收益

**未优化成本**：
```
tensor.dim %tensor, %dim_idx
  = load_metadata_ptr      // 1 次内存加载
  + offset_calculation      // 1-2 次算术运算
  + load_dimension_value    // 1 次内存加载

总计: ~20-30 CPU 周期（受缓存影响）
```

**优化后成本**：
```
tensor.dim %init_tensor, %dim_idx
  = 循环不变量提升后移到循环外
  = 1 次执行，循环内 0 开销
```

**净收益**：
- 循环 N 次：(N-1) × 20-30 周期

### 真实场景测试

#### 场景 1：图像 Tiling 处理

**代码**：
```mlir
func.func @image_tiling(%img : tensor<1024x1024xf32>,
                         %out : tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index

  %result = scf.for %i = %c0 to %c1024 step %c64 iter_args(%arg = %out) {
    %result_inner = scf.for %j = %c0 to %c1024 step %c64
        iter_args(%arg_inner = %arg) {

      // 计算 tile 大小（未优化时每次计算）
      %tile_h = affine.min affine_map<(d0)[s0, s1] -> (s0, s1 - d0)>
                                     (%i)[%c64, %c1024]
      %tile_w = affine.min affine_map<(d0)[s0, s1] -> (s0, s1 - d0)>
                                     (%j)[%c64, %c1024]

      // 提取并处理 tile
      %tile = tensor.extract_slice %img[%i, %j][%tile_h, %tile_w][1, 1]
      %processed = some_processing(%tile)
      %updated = tensor.insert_slice %processed into %arg_inner[%i, %j]
                                    [%tile_h, %tile_w][1, 1]

      scf.yield %updated
    }
    scf.yield %result_inner
  }
  return %result
}
```

**性能估算**：
```
外层循环: 1024 / 64 = 16 次
内层循环: 1024 / 64 = 16 次
总迭代: 16 × 16 = 256 次

未优化:
  - 每次迭代: 2 次 affine.min (tile_h, tile_w)
  - 总计: 256 × 2 × 8 周期 = 4096 周期

优化后:
  - affine.min 全部简化为常量 64
  - 总计: 0 周期

节省: 4096 周期 ≈ 1-2 微秒 (现代 CPU)
```

对于嵌套更深或图像更大的情况，收益会成倍增长！

#### 场景 2：动态批处理

**代码**：
```mlir
func.func @dynamic_batch_process(%data : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %data) {
    // 未优化时每次查询维度
    %dim0 = tensor.dim %arg, %c0
    %dim1 = tensor.dim %arg, %c1

    // 使用维度信息
    %processed = process_with_dims(%arg, %dim0, %dim1)
    scf.yield %processed
  }
  return %result
}
```

**性能估算**：
```
循环: 10 次迭代

未优化:
  - 每次迭代: 2 次 tensor.dim
  - 每次 dim: ~25 周期 (内存访问)
  - 总计: 10 × 2 × 25 = 500 周期

优化后:
  - dim 操作替换为 tensor.dim %data (循环不变)
  - 提升到循环外: 2 × 25 = 50 周期
  - 总计: 50 周期

节省: 450 周期 ≈ 90% 改进
```

### 编译时间影响

**模式应用开销**：
```
每个模式匹配:
  - DimOfIterArgFolder: O(追踪链长度)
  - AffineOpSCFCanonicalizationPattern: O(约束系统求解)

约束求解复杂度:
  - 变量数: O(归纳变量个数)
  - 约束数: O(归纳变量个数 × 2)
  - 求解: O(N³) (单纯形法最坏情况)

实践中:
  - 大多数循环: < 5 个归纳变量
  - 求解时间: < 1ms
  - 可接受的编译时间开销
```

---

## 局限性与展望

### 当前限制

#### 1. 动态步长不支持

**原因**：
```cpp
auto stepInt = getConstantIntValue(step);
if (!stepInt)
  return failure();
```

整数多面体只支持仿射约束，不能表达：
```
iv_next = iv + dynamic_step
```

**影响**：
```mlir
// ❌ 不能优化
%step = some_computation()
scf.for %i = %c0 to %c100 step %step {
  %r = affine.min #map(%i)[%step]
  ...
}
```

**潜在解决方案**：
- 使用半仿射分析（Semi-affine analysis）
- 或者为常见步长模式添加特殊处理

#### 2. 形状保持性检查保守

**当前支持**：
- `tensor.insert_slice`
- 嵌套 `scf.for`

**不支持**：
```mlir
// ❌ 不支持：条件分支
scf.for ... iter_args(%arg = %init) {
  %result = scf.if %cond {
    scf.yield %arg  // 形状保持
  } else {
    scf.yield %arg  // 形状保持
  }
  scf.yield %result  // 但 isShapePreserving 返回 false
}

// ❌ 不支持：自定义操作（即使保持形状）
scf.for ... iter_args(%arg = %init) {
  %result = my_custom_op %arg  // 可能保持形状，但无法验证
  scf.yield %result
}
```

**改进方向**：
- 添加形状保持性接口（ShapePreservingOpInterface）
- 支持更多内置操作（scf.if, scf.while）
- 允许用户标注自定义操作

#### 3. 复杂仿射表达式

**限制示例**（测试中的注释）：
```mlir
// 当前无法简化（需要半仿射规范化）
%ub = affine.apply affine_map<(d0) -> (42 * d0)> (%step)
scf.for %i = %c0 to %ub step %step {
  %r = affine.min affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
                            (%step, %ub, %i)
  // 理论上 %r = %step（恒定），但当前无法推导
}
```

**问题**：
```
(step * 42 - i) / step 理论上等于 41 (当 i 接近上界时)
但需要除法规范化，当前不支持
```

#### 4. 跨函数优化

**限制**：
```mlir
func.func @caller(%t : tensor<?x?xf32>) {
  %r = scf.for ... iter_args(%arg = %t) {
    %processed = call @callee(%arg)
    scf.yield %processed
  }
}

func.func @callee(%input : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // 即使这里保持形状，caller 中无法推导
  return %input
}
```

**需要**：
- 过程间分析（Interprocedural analysis）
- 函数摘要（Function summaries）

### 未来工作方向

#### 1. 扩展到更多循环类型

**`scf.while` 支持**：
```mlir
scf.while (%arg = %init) : (tensor<?xf32>) -> tensor<?xf32> {
  %cond = ...
  scf.condition(%cond) %arg : tensor<?xf32>
} do {
  ^bb0(%arg: tensor<?xf32>):
  %dim = tensor.dim %arg, %c0  // 可以优化为 tensor.dim %init
  ...
  scf.yield %new_arg
}
```

**挑战**：while 循环的迭代次数不确定

#### 2. 结合其他优化

**循环融合（Loop Fusion）**：
```mlir
// 融合前
scf.for %i = ... {
  %r1 = affine.min #map1(%i)
}
scf.for %i = ... {
  %r2 = affine.min #map2(%i)
}

// 融合后 + 规范化
scf.for %i = ... {
  // 两个 min 都可能简化
}
```

**循环 Tiling**：
```mlir
// Tiling 后自动应用规范化
scf.for %ii = ... {
  scf.for %i = ... {
    // 内层循环的边界计算可以简化
    %inner_ub = affine.min #map(%i, %ii)
  }
}
```

#### 3. 基于 Polyhedral 的全局优化

**当前**：局部模式匹配

**未来**：
- 构建全局 Polyhedral 模型
- 联合优化多个循环
- 自动发现最优循环变换

**示例工具**：
- Pluto（自动并行化）
- Polly（LLVM 的 polyhedral 优化器）

#### 4. 机器学习辅助优化

**思路**：
- 训练模型预测优化效果
- 避免昂贵的约束求解（某些情况）
- 指导优化策略选择

```python
# 伪代码
def should_canonicalize(loop_info):
    features = extract_features(loop_info)
    benefit = ml_model.predict(features)
    return benefit > threshold
```

---

## 总结

### 核心价值

循环规范化优化体现了现代编译器的几个关键思想：

1. **利用领域知识**
   - 循环的结构化特性（边界、步长）
   - 仿射表达式的数学性质

2. **约束求解**
   - 将编译问题转化为数学问题
   - 利用 Presburger 算术的可判定性

3. **保守正确性**
   - 只在能证明安全时才优化
   - 不确定时保持原样

4. **渐进式优化**
   - 高层次优化创造低层次优化机会
   - 配合其他 Pass 形成优化流水线

### 适用场景总结

**强烈推荐**：
- ✅ 图像/视频处理的分块算法
- ✅ 深度学习的批处理循环
- ✅ 科学计算的多维数组遍历
- ✅ 任何带常量步长的规则循环

**效果有限**：
- ⚠️ 动态步长循环
- ⚠️ 不规则迭代空间
- ⚠️ 复杂半仿射表达式
- ⚠️ 跨函数边界的优化

### 学习要点

**对于 MLIR 学习者**：
1. 理解 SCF 方言的语义（特别是 iter_args）
2. 掌握仿射表达式和映射
3. 学习 Pattern Rewriting 框架
4. 了解约束求解基础

**对于编译器开发者**：
1. 如何设计可组合的优化模式
2. 平衡优化收益与编译时间
3. 保守分析 vs. 激进优化的权衡
4. 测试驱动开发的重要性

**对于性能优化工程师**：
1. 编写编译器友好的代码
2. 理解哪些模式可以优化
3. 如何验证优化是否生效
4. 性能分析和 Profiling

### 相关资源

**MLIR 官方文档**：
- [SCF Dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
- [Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)

**学术背景**：
- [Presburger Arithmetic](https://en.wikipedia.org/wiki/Presburger_arithmetic)
- [Polyhedral Compilation](https://polyhedral.info/)
- [Integer Linear Programming](https://en.wikipedia.org/wiki/Integer_programming)

**相关工具**：
- [isl (Integer Set Library)](http://isl.gforge.inria.fr/)
- [Polly](https://polly.llvm.org/)
- [MLIR Toys Tutorial](https://mlir.llvm.org/docs/Tutorials/)

**论文**：
- "MLIR: A Compiler Infrastructure for the End of Moore's Law" (CGO 2020)
- "Polyhedral Compilation: A Comprehensive Overview" (Survey)

---

## 附录：完整优化流程图

```
                     输入 MLIR 代码
                           │
                           ▼
         ┌─────────────────────────────────┐
         │  SCFForLoopCanonicalization Pass│
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │  贪心重写驱动器                 │
         │  (Greedy Pattern Rewrite Driver)│
         └────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │  应用所有模式   │
         │  直到不动点     │
         └────────┬────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌──────────┐ ┌─────────┐ ┌──────────┐
│Affine    │ │Dim of   │ │Dim of    │
│Min/Max   │ │IterArg  │ │LoopResult│
│Pattern   │ │Pattern  │ │Pattern   │
└────┬─────┘ └────┬────┘ └────┬─────┘
     │            │           │
     ▼            ▼           ▼
┌─────────────────────────────────┐
│  matchAndRewrite                │
│  ├─ 检查是否匹配                │
│  ├─ 构建约束/检查形状保持       │
│  ├─ 简化表达式                  │
│  └─ 替换操作                    │
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        │  成功?      │
        └──────┬──────┘
               │
         Yes ──┴── No
          │         │
          ▼         ▼
    ┌─────────┐ ┌────────┐
    │替换操作 │ │保持不变│
    └────┬────┘ └────┬───┘
         │           │
         └─────┬─────┘
               │
               ▼
         ┌────────────┐
         │ 继续迭代   │
         │ (直到收敛) │
         └─────┬──────┘
               │
               ▼
         输出优化后的 MLIR
```

---

**文档版本**: v1.0
**最后更新**: 2026-01-15
**MLIR 版本**: LLVM 主分支
**分析文件**: `mlir/lib/Dialect/SCF/Transforms/LoopCanonicalization.cpp`

---

*本文档旨在为 MLIR 学习者和编译器开发者提供深入理解。如有疑问，请参考 MLIR 官方文档或社区讨论。*
