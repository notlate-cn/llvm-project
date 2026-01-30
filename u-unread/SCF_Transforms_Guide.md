# MLIR SCF方言Transform详解

本文档详细介绍MLIR SCF (Structured Control Flow) 方言中的所有Transform变换，包括其作用、技术原理和实例演示。

## 目录

1. [循环转换](#1-循环转换)
   - [ForToWhile](#11-fortowhile)
   - [ForallToFor](#12-foralltofor)
   - [ForallToParallel](#13-foralltoparallel)
   - [UpliftWhileToFor](#14-upliftwhiletofor)

2. [循环优化](#2-循环优化)
   - [LoopPipelining](#21-looppipelining)
   - [LoopRangeFolding](#22-looprangefolding)
   - [LoopSpecialization](#23-loopspecialization)
   - [RotateWhileLoop](#24-rotatewhileloop)
   - [WrapInZeroTripCheck](#25-wrapinzerotripcheck)

3. [并行循环处理](#3-并行循环处理)
   - [ParallelLoopCollapsing](#31-parallelloopcollapsing)
   - [ParallelLoopFusion](#32-parallelloopfusion)
   - [ParallelLoopTiling](#33-parallellooptiling)

4. [规范化和类型转换](#4-规范化和类型转换)
   - [LoopCanonicalization](#41-loopcanonicalization)
   - [StructuralTypeConversions](#42-structuraltypeconversions)
   - [Buffer相关接口](#43-buffer相关接口)

5. [通用接口平铺](#5-通用接口平铺)
   - [TileUsingInterface](#51-tileusinginterface)

---

## 1. 循环转换

### 1.1 ForToWhile

**文件**: `ForToWhile.cpp`

**作用**: 将 `scf.for` 循环转换为 `scf.while` 循环。

**技术原理**:
- 将归纳变量(IV)作为第一个循环携带值
- 在 "before" 区域中创建循环条件（比较IV与上界）
- 在 "after" 区域中内联原始for循环体，并添加IV增量操作
- 修改所有yield操作以包含递增的归纳变量

**实现代码结构**:
```cpp
struct ForLoopLoweringPattern : public OpRewritePattern<ForOp> {
  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // 构建WhileOp，将IV作为第一个iter_arg
    // before区域：条件检查 (IV < ub)
    // after区域：原始循环体 + IV递增
  }
};
```

**示例**:

转换前:
```mlir
scf.for %arg0 = %lb to %ub step %step {
  %result = "some.op"(%arg0) : (index) -> tensor<?xf32>
}
```

转换后:
```mlir
%0:2 = scf.while : (index, tensor<?xf32>) -> (index, tensor<?xf32>) {
  // before region: 条件检查
  scf.condition(%arg0 < %ub) %arg0, %arg1
} do {
  ^bb0(%arg2: index, %arg3: tensor<?xf32>):
  // after region: 循环体
  %result = "some.op"(%arg2) : (index) -> tensor<?xf32>
  %next = arith.addi %arg2, %step : index
  scf.yield %next, %result : index, tensor<?xf32>
}
```

**使用场景**:
- 需要更灵活的循环条件时
- 循环上下界或步长需要在循环体内修改时
- 作为其他变换的中间表示

---

### 1.2 ForallToFor

**文件**: `ForallToFor.cpp`

**作用**: 将 `scf.forall` 操作转换为嵌套的 `scf.for` 循环。

**技术原理**:
- 使用 `scf::buildLoopNest` 构建嵌套的scf.for循环
- 提取forall操作的下界(lbs)、上界(ubs)和步长(steps)
- 将forall循环体内联到最内层的for循环中
- 替换归纳变量使用

**核心函数**:
```cpp
LogicalResult forallToForLoop(RewriterBase &rewriter, scf::ForallOp forallOp,
                              SmallVectorImpl<Operation *> *results) {
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);
  LoopNest loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps);

  // 内联forall体到最内层循环
  Block *innermostBlock = loopNest.loops.back().getBody();
  rewriter.inlineBlockBefore(forallOp.getBody(), innermostBlock, ...);
}
```

**示例**:

转换前:
```mlir
scf.forall (%arg0, %arg1) in (%lb0, %lb1) to (%ub0, %ub1) step (%step0, %step1) {
  "use.ivs"(%arg0, %arg1) : (index, index) -> ()
}
```

转换后:
```mlir
scf.for %arg0 = %lb0 to %ub0 step %step0 {
  scf.for %arg1 = %lb1 to %ub1 step %step1 {
    "use.ivs"(%arg0, %arg1) : (index, index) -> ()
  }
}
```

**使用场景**:
- 将并行的forall循环降级为串行的for循环
- 调试和验证
- 作为其他变换的前置步骤

---

### 1.3 ForallToParallel

**文件**: `ForallToParallel.cpp`

**作用**: 将 `scf.forall` 操作转换为 `scf.parallel` 操作。

**技术原理**:
- 仅支持完全缓冲化的scf.forall操作
- 将forall转换为parallel op
- 将terminator替换为scf.reduce
- 传播mapping属性（如果存在）

**核心函数**:
```cpp
LogicalResult forallToParallelLoop(RewriterBase &rewriter, scf::ForallOp forallOp,
                                   scf::ParallelOp *result) {
  // 检查outputs为空
  if (!forallOp.getOutputs().empty())
    return failure();

  // 创建parallel op并内联region
  // 替换terminator为reduce op
}
```

**示例**:

转换前:
```mlir
scf.forall (%arg0) in (%lb) to (%ub) step (%step) {
  %val = "compute"(%arg0) : (index) -> f32
  scf.forall.in_parallel {
    scf.reduce %val parallel_handler @reduce_add
  }
}
```

转换后:
```mlir
scf.parallel (%arg0) = (%lb) to (%ub) step (%step) {
  %val = "compute"(%arg0) : (index) -> f32
  scf.reduce @reduce_add %val : f32
}
```

**使用场景**:
- 将forall转换为parallel循环以利用特定的并行后端
- 与使用parallel循环的现有代码集成

---

### 1.4 UpliftWhileToFor

**文件**: `UpliftWhileToFor.cpp`

**作用**: 将符合模式的 `scf.while` 循环转换为 `scf.for` 循环。

**技术原理**:
- 识别while循环的归纳模式：
  - before块包含单个比较操作
  - 比较一个块参数与外部常量
  - after块递增该参数
- 提取lb、ub、step
- 创建for循环并重新映射参数

**核心函数**:
```cpp
FailureOr<scf::ForOp> upliftWhileToForLoop(RewriterBase &rewriter, scf::WhileOp loop) {
  // 验证before块: 单个cmp操作
  auto condOp = dyn_cast<arith::CmpIOp>(whileOp.getConditionBlock()->getTerminator());
  // 验证after块: addi操作定义IV递增
  // 提取lb, ub, step
  // 创建scf.for并内联after块
}
```

**示例**:

转换前:
```mlir
%0:2 = scf.while : (index) -> (index, i32) {
  scf.condition(%arg0 < %ub) %arg0 : index
} do {
  ^bb0(%arg1: index):
  %val = "compute"(%arg1) : (index) -> i32
  %next = arith.addi %arg1, %step : index
  scf.yield %next, %val : index, i32
}
```

转换后:
```mlir
%result:2 = scf.for %arg0 = %lb to %ub step %step
    iter_args(%arg1 = %init) -> (index, i32) {
  %val = "compute"(%arg0) : (index) -> i32
  scf.yield %val : i32
}
```

**使用场景**:
- 优化符合for循环模式的while循环
- 简化循环结构便于进一步优化
- 自动循环标准化

---

## 2. 循环优化

### 2.1 LoopPipelining

**文件**: `LoopPipelining.cpp`

**作用**: 实现循环软件流水线（Software Pipelining）优化。

**技术原理**:
- 将循环操作分配到不同的流水线阶段
- 生成prologue（阶段[0; i]的操作）
- 生成kernel（主循环，跨阶段值作为循环参数）
- 生成epilogue（阶段[i; maxStage]的操作）
- 支持动态循环和谓词执行

**数据结构**:
```cpp
struct LoopPipelinerInternal {
  struct LiverangeInfo {
    unsigned lastUseStage = 0;
    unsigned defStage = 0;
  };

  ForOp forOp;
  unsigned maxStage = 0;
  DenseMap<Operation *, unsigned> stages;  // 操作到阶段的映射
  std::vector<Operation *> opOrder;        // 操作顺序
  bool dynamicLoop;                         // 是否动态循环
  bool peelEpilogue;                        // 是否剥离epilogue
};
```

**流水线结构**:

```
原始循环:
for i = 0 to N:
  A(i)  // Stage 0
  B(i)  // Stage 1
  C(i)  // Stage 2

流水线后 (maxStage = 3):
Prologue:
  A(0)
  A(1), B(0)
  A(2), B(1), C(0)

Kernel:
for i = 3 to N-3:
  A(i), B(i-1), C(i-2)

Epilogue:
  B(N-2), C(N-3)
  B(N-1), C(N-2)
  C(N-1)
```

**核心方法**:
```cpp
// 1. 初始化循环信息
bool initializeLoopInfo(ForOp op, const PipeliningOption &options);

// 2. 发射prologue
LogicalResult emitPrologue(RewriterBase &rewriter);

// 3. 分析跨阶段值
llvm::MapVector<Value, LiverangeInfo> analyzeCrossStageValues();

// 4. 创建kernel循环
scf::ForOp createKernelLoop(...);

// 5. 创建kernel主体
LogicalResult createKernel(...);

// 6. 发射epilogue
LogicalResult emitEpilogue(RewriterBase &rewriter);
```

**使用场景**:
- 指令级并行优化
- 隐藏内存延迟
- 提高循环吞吐量

**配置选项**:
```cpp
struct PipeliningOption {
  using ScheduleFn = std::function<void(ForOp, std::vector<std::pair<Operation*, unsigned>>&)>;

  ScheduleFn getScheduleFn;      // 获取调度函数
  bool peelEpilogue;             // 是否剥离epilogue
  bool supportDynamicLoops;      // 是否支持动态循环
  PredicateOpFn predicateFn;     // 谓词函数（用于动态循环）
  AnnotationlFnType annotateFn;  // 注释函数
};
```

---

### 2.2 LoopRangeFolding

**文件**: `LoopRangeFolding.cpp`

**作用**: 将归纳变量上的算术操作折叠到循环边界中。

**技术原理**:
- 仅当IV只有一个使用时执行
- 支持AddIOp和MulIOp操作
- 迭代到固定点

**算法**:
```cpp
// 检查IV只有一个使用
if (!iv.hasOneUse())
  return;

Operation *op = *iv.user_begin();

// 如果是 addi: 更新lb和ub
if (auto addi = dyn_cast<arith::AddIOp>(op)) {
  if (addi.getLhs() == iv) {
    newLb = lb + const_rhs;
    newUb = ub + const_rhs;
  }
}

// 如果是 muli: 更新lb, ub和step
if (auto muli = dyn_cast<arith::MulIOp>(op)) {
  if (muli.getLhs() == iv && isPositiveConstant(muli.getRhs())) {
    newStep = step * const_rhs;
    // 相应调整lb和ub
  }
}
```

**示例**:

转换前:
```mlir
scf.for %i = %lb to %ub step %step {
  %j = arith.addi %i, %c10 : index
  "use"(%j) : (index) -> ()
}
```

转换后:
```mlir
%new_lb = arith.addi %lb, %c10 : index
%new_ub = arith.addi %ub, %c10 : index
scf.for %i = %new_lb to %new_ub step %step {
  "use"(%i) : (index) -> ()
}
```

**使用场景**:
- 简化归纳变量的使用
- 为其他优化做准备
- 减少循环内的算术操作

---

### 2.3 LoopSpecialization

**文件**: `LoopSpecialization.cpp`

**作用**: 特化并行循环和for循环以便于展开和向量化。

**技术原理**:

#### 循环特化 (Specialization)
- 对于使用 `affine.min` 作为边界的循环
- 创建if检查，常量路径使用常量边界
- 变量路径保持原始边界

#### 循环剥离 (Peeling)
- 将循环分割为能被step整除的主循环+处理剩余迭代的if部分
- peelForLoop: 通用剥离
- peelForLoopFirstIteration: 仅剥离第一次迭代

**核心函数**:
```cpp
// 循环特化
static void specializeForLoopForUnrolling(ForOp op) {
  // 检查上界是否为affine.min
  auto minOp = ub.getDefiningOp<affine::AffineMinOp>();
  if (!minOp) return;

  // 创建if检查边界是否等于常量
  // then分支: 使用常量边界的循环
  // else分支: 原始循环
}

// 循环剥离
static LogicalResult peelForLoop(RewriterBase &rewriter, ForOp op,
                                 unsigned numIters,
                                 scf::ForOp &peeledLoop,
                                 scf::ForOp &remainderLoop) {
  // 计算新的上界: ub - (ub - lb) mod step
  // 创建主循环和部分迭代循环
}
```

**示例 - 特化**:

转换前:
```mlir
%ub = affine.min affine_map<(d0) -> (100, d0)>(%N)
scf.for %i = 0 to %ub step 4 {
  "body"(%i) : (index) -> ()
}
```

转换后:
```mlir
%cond = arith.cmpi eq %N, %c100 : index
scf.if %cond {
  scf.for %i = 0 to 100 step 4 {   // 常量边界，便于展开
    "body"(%i) : (index) -> ()
  }
} else {
  scf.for %i = 0 to %ub step 4 {   // 变量边界
    "body"(%i) : (index) -> ()
  }
}
```

**示例 - 剥离**:

转换前:
```mlir
scf.for %i = 0 to %N step 4 {
  "body"(%i) : (index) -> ()
}
```

转换后:
```mlir
%new_ub = %N - (%N - 0) mod 4
scf.for %i = 0 to %new_ub step 4 {   // 主循环，次数能被4整除
  "body"(%i) : (index) -> ()
}
%rem_start = new_ub
%rem_end = %N
scf.for %i = %rem_start to %rem_end {  // 剩余迭代
  "body"(%i) : (index) -> ()
}
```

**使用场景**:
- 为向量化做准备
- 循环展开优化
- 处理边界条件

---

### 2.4 RotateWhileLoop

**文件**: `RotateWhileLoop.cpp`

**作用**: 旋转scf.while循环，转换为do-while形式。

**技术原理**:
- 调用 `wrapWhileLoopInZeroTripCheck` 旋转while循环
- 将after块移到before块之前
- 避免重复条件检查
- 防止无限递归

**核心模式**:
```cpp
struct RotateWhileLoopPattern : OpRewritePattern<scf::WhileOp> {
  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<scf::WhileOp> result =
        scf::wrapWhileLoopInZeroTripCheck(whileOp, rewriter, false);
    return success(succeeded(result) && *result != whileOp);
  }
};
```

**使用场景**:
- 将while循环转换为do-while形式
- 消除重复的条件检查
- 与WrapInZeroTripCheck配合使用

---

### 2.5 WrapInZeroTripCheck

**文件**: `WrapInZeroTripCheck.cpp`

**作用**: 为while循环添加零行程检查，避免do-while形式在条件不满足时仍执行一次的问题。

**技术原理**:
- 克隆before块到循环前面（零行程检查）
- 创建if操作：then分支包含旋转后的while循环
- 旋转while循环：after块 → before块
- else分支返回预计算的值

**核心函数**:
```cpp
FailureOr<scf::WhileOp> wrapWhileLoopInZeroTripCheck(
    scf::WhileOp whileOp,
    RewriterBase &rewriter,
    bool forceCreateCheck) {

  // 1. 克隆before块作为零行程检查
  // 2. 创建旋转后的while循环
  // 3. 创建if op包装旋转后的循环
  // 4. else分支返回初始值
}
```

**示例**:

转换前:
```mlir
%0 = scf.while : (i32) -> i32 {
  scf.condition(...) %arg0 : i32
} do {
  ^bb0(%arg1: i32):
  %val = "compute"(%arg1) : (i32) -> i32
  scf.yield %val : i32
}
```

转换后:
```mlir
// 零行程检查
%condition = "eval_condition"()
scf.if %condition {
  // 旋转后的while循环
  %0 = scf.while : (i32) -> i32 {
    // 原始after块内容
    %val = "compute"(%arg1) : (i32) -> i32
    scf.condition(...) %val : i32
  } do {
    // 原始before块内容
    scf.yield %arg0 : i32
  }
} else {
  // 返回初始值
}
```

**使用场景**:
- 确保循环在条件不满足时不执行
- do-while循环优化
- 与RotateWhileLoop配合使用

---

## 3. 并行循环处理

### 3.1 ParallelLoopCollapsing

**文件**: `ParallelLoopCollapsing.cpp`

**作用**: 合并parallel循环的多个维度。

**技术原理**:
- 将多个循环维度映射为单个维度
- 计算新的上界和步长
- 重建原始的多维索引

**辅助函数**:
```cpp
FailureOr<ParallelOp> collapseParallelLoops(
    RewriterBase &rewriter,
    ParallelOp op,
    ArrayRef<std::vector<unsigned>> clonedLoops);
```

**示例**:

转换前:
```mlir
scf.parallel (%i, %j, %k) = (%lb0, %lb1, %lb2) to (%ub0, %ub1, %ub2)
    step (%step0, %step1, %step2) {
  "use"(%i, %j, %k) : (index, index, index) -> ()
}
```

转换后 (合并第0和第1维):
```mlir
%new_ub = (%ub0 - %lb0) * (%ub1 - %lb1)
scf.parallel (%combined, %k) = (0, %lb2) to (%new_ub, %ub2)
    step (%step0 * %step1, %step2) {
  %i = %lb0 + %combined / (%ub1 - %lb1)
  %j = %lb1 + %combined mod (%ub1 - %lb1)
  "use"(%i, %j, %k) : (index, index, index) -> ()
}
```

**使用场景**:
- 减少并行循环的维度
- 适应只支持较少维度的并行后端
- 优化负载均衡

---

### 3.2 ParallelLoopFusion

**文件**: `ParallelLoopFusion.cpp`

**作用**: 融合具有相同迭代空间的parallel循环。

**技术原理**:
- 检查两个parallel循环是否有相同的迭代空间
- 验证内存依赖（写后读模式）
- 将第一个循环体操作移动到第二个循环中
- 合并init值和reduce操作

**检查条件**:
```cpp
// 1. 无嵌套parallel op
bool hasNestedParallelOp(ParallelOp op);

// 2. 相同迭代空间
bool equalIterationSpaces(ParallelOp firstPloop, ParallelOp secondPloop);

// 3. 依赖关系合法
LogicalResult verifyDependencies(ParallelOp firstPloop, ParallelOp secondPloop);
```

**核心函数**:
```cpp
static void fuseIfLegal(ParallelOp firstPloop,
                        ParallelOp &secondPloop,
                        OpOperand *parallelOpOperand,
                        SmallVectorImpl<Operation *> &fusedOps) {
  // 验证融合条件
  // 创建新的parallel op并合并两个循环体
  // 合并reduce操作
}
```

**示例**:

转换前:
```mlir
scf.parallel (%i) = (0) to (100) step (1) {
  %A = "load_A"(%i) : (index) -> f32
  %B = "compute"(%A) : (f32) -> f32
  "store_B"(%i, %B) : (index, f32) -> ()
}

scf.parallel (%i) = (0) to (100) step (1) {
  %B = "load_B"(%i) : (index) -> f32
  %C = "compute2"(%B) : (f32) -> f32
  "store_C"(%i, %C) : (index, f32) -> ()
}
```

转换后:
```mlir
scf.parallel (%i) = (0) to (100) step (1) {
  %A = "load_A"(%i) : (index) -> f32
  %B = "compute"(%A) : (f32) -> f32
  "store_B"(%i, %B) : (index, f32) -> ()
  %B2 = "load_B"(%i) : (index) -> f32
  %C = "compute2"(%B2) : (f32) -> f32
  "store_C"(%i, %C) : (index, f32) -> ()
}
```

**使用场景**:
- 减少循环开销
- 提高数据局部性
- 减少同步操作

---

### 3.3 ParallelLoopTiling

**文件**: `ParallelLoopTiling.cpp`

**作用**: 对parallel循环进行平铺（tiling）优化。

**技术原理**:
- 创建外层循环（步长 *= tile size）
- 创建内层循环（上界 = min(步长, 原上界-外层IV)）
- 支持noMinMaxBounds模式（使用边界检查的if）
- 内层循环的IV替换为 外层IV + 内层IV

**核心函数**:
```cpp
std::pair<ParallelOp, ParallelOp> tileParallelLoop(
    ParallelOp op,
    ArrayRef<int64_t> tileSizes,
    bool noMinMaxBounds) {

  // 创建外层循环（调整步长）
  // 创建内层循环（调整上界，可能使用affine.min或if检查）
  // 替换IV使用为 outerIV + innerIV
}
```

**示例**:

转换前:
```mlir
scf.parallel (%i, %j) = (0, 0) to (1024, 1024) step (1, 1) {
  "use"(%i, %j) : (index, index) -> ()
}
```

转换后 (tileSizes = [64, 64]):
```mlir
scf.parallel (%outer_i, %outer_j) = (0, 0) to (1024, 1024) step (64, 64) {
  scf.parallel (%i, %j) = (0, 0)
      to (min(64, 1024 - %outer_i), min(64, 1024 - %outer_j))
      step (1, 1) {
    %real_i = %outer_i + %i
    %real_j = %outer_j + %j
    "use"(%real_i, %real_j) : (index, index) -> ()
  }
}
```

**使用场景**:
- 提高缓存利用率
- 向量化优化
- GPU kernel优化

---

## 4. 规范化和类型转换

### 4.1 LoopCanonicalization

**文件**: `LoopCanonicalization.cpp`

**作用**: 跨方言的循环规范化模式。

**技术原理**:

#### Dim操作折叠
- 将 `tensor.dim` 操作在循环携带参数上的使用折叠为在init参数上的使用
- 使用 `isShapePreserving` 分析验证类型不变性

#### Affine操作规范化
- 规范化循环上下文中的 `AffineMin`/`AffineMax` 操作

**核心模式**:
```cpp
// 检查形状保持不变
static bool isShapePreserving(ForOp forOp, int64_t arg) {
  BlockArgument iterArg = forOp.getRegionIterArgs()[arg];
  Type yieldedType = yieldOp->getOperand(arg).getType();
  return iterArg.getType() == yieldedType;
}

// 折叠iter_args的dim操作
template <typename OpTy>
struct DimOfIterArgFolder : public OpRewritePattern<OpTy> {
  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    // 如果dim操作的对象是iter_arg且形状保持不变
    // 将其替换为init_arg的dim操作
  }
};

// 规范化AffineMin/AffineMax
template <typename OpTy>
struct AffineOpSCFCanonicalizationPattern : public OpRewritePattern<OpTy> {
  // 简化循环上下文中的affine.min/max操作
};
```

**示例**:

转换前:
```mlir
%init = tensor.empty() : tensor<128xf32>
%result = scf.for %i = 0 to 10 iter_args(%arg = %init) -> (tensor<128xf32>) {
  %new = "update"(%arg) : (tensor<128xf32>) -> (tensor<128xf32>)
  scf.yield %new : tensor<128xf32>
}
%d = tensor.dim %result, %c0 : tensor<128xf32>
```

转换后:
```mlir
%init = tensor.empty() : tensor<128xf32>
%d = tensor.dim %init, %c0 : tensor<128xf32>  // 直接使用init
%result = scf.for %i = 0 to 10 iter_args(%arg = %init) -> (tensor<128xf32>) {
  %new = "update"(%arg) : (tensor<128xf32>) -> (tensor<128xf32>)
  scf.yield %new : tensor<128xf32>
}
```

**使用场景**:
- 循环不变代码外提
- 简化循环分析
- 为其他优化做准备

---

### 4.2 StructuralTypeConversions

**文件**: `StructuralTypeConversions.cpp`

**作用**: 实现SCF操作的结构化类型转换，支持1:N类型转换。

**技术原理**:
- 支持1:N类型转换（一个类型转换为多个类型）
- 转换操作结果类型并记录偏移量
- 内联原始region到新操作
- 处理yield/condition操作数的更新

**核心类**:
```cpp
// CRTP基类，处理1:N类型转换
template <typename SourceOp, typename ConcretePattern>
class Structural1ToNConversionPattern : public OpConversionPattern<SourceOp> {
  LogicalResult matchAndRewrite(SourceOp op,
                                OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 转换结果类型并记录偏移
    // 调用派生类的convertSourceOp
    // 打包返回值
  }
};

// ForOp类型转换
class ConvertForOpTypes : public Structural1ToNConversionPattern<ForOp, ...> {
  std::optional<ForOp> convertSourceOp(...) const override {
    // 转换region类型
    // 创建新ForOp并内联region
  }
};
```

**示例**:

转换前 (1:1类型转换):
```mlir
%result:2 = scf.for %i = 0 to 10 iter_args(%a = %v1, %b = %v2)
    -> (tensor<f32>, tensor<f32>) {
  scf.yield %a, %b : tensor<f32>, tensor<f32>
}
```

转换后 (1:2类型转换，每个tensor分解为data和validity):
```mlir
%result:4 = scf.for %i = 0 to 10
    iter_args(%a_data = %v1_data, %a_valid = %v1_valid,
              %b_data = %v2_data, %b_valid = %v2_valid)
    -> (tensor<f32>, i1, tensor<f32>, i1) {
  scf.yield %a_data, %a_valid, %b_data, %b_valid
}
```

**使用场景**:
- Bufferization过程中的类型转换
- 添加额外的类型信息（如validity标志）
- 复杂的数据布局转换

---

### 4.3 Buffer相关接口

**文件**:
- `BufferDeallocationOpInterfaceImpl.cpp`
- `BufferizableOpInterfaceImpl.cpp`

**作用**:
- BufferizableOpInterface: 将tensor类型转换为memref类型
- BufferDeallocationOpInterface: 管理缓冲区释放

#### BufferizableOpInterface

**支持的SCF操作**:
- ConditionOp
- ExecuteRegionOp
- IfOp
- ForOp
- WhileOp
- ForallOp
- YieldOp

**核心接口实现**:
```cpp
struct ForOpInterface : public BufferizableOpInterface::ExternalModel<...> {
  // 分析结果是否缓冲化
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const;

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const;

  // 获取缓冲区类型
  FailureOr<BaseMemRefType> getBufferType(...) const;

  // 执行缓冲化转换
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, ...) const {
    // 创建新的scf.for，使用memref类型
    auto newForOp = rewriter.create<scf.ForOp>(..., castedInitArgs);
    // 包装memref iter_args为ToTensorOp
    // 移动循环体
  }
};
```

**示例 - ForOp缓冲化**:

转换前:
```mlir
%init = tensor.empty() : tensor<128xf32>
%result = scf.for %i = 0 to 10 iter_args(%arg = %init) -> (tensor<128xf32>) {
  %new = tensor.insert %val into %arg[%i] : tensor<128xf32>
  scf.yield %new : tensor<128xf32>
}
```

转换后:
```mlir
%mem = memref.alloc() : memref<128xf32>
%result_mem = scf.for %i = 0 to 10 iter_args(%arg = %mem) -> (memref<128xf32>) {
  memref.store %val, %arg[%i] : memref<128xf32>
  scf.yield %arg : memref<128xf32>
}
%result = bufferization.to_tensor %result_mem : memref<128xf32>
```

#### BufferDeallocationOpInterface

**支持的操作**:
- `scf.forall.in_parallel`
- `scf.reduce.return`

**核心实现**:
```cpp
struct InParallelOpInterface : public BufferDeallocationOpInterface::ExternalModel<...> {
  FailureOr<Operation *> process(Operation *op,
                                  DeallocationState &state,
                                  ...) const {
    return deallocation_impl::insertDeallocOpForReturnLike(state, op, ...);
  }
};
```

**使用场景**:
- Tensor到Memref的转换
- 内存管理优化
- 与Bufferization pass集成

---

## 5. 通用接口平铺

### 5.1 TileUsingInterface

**文件**: `TileUsingInterface.cpp`

**作用**: 使用TilingInterface实现通用的循环平铺，支持多种操作类型。

**技术原理**:
- 使用 `TilingInterface` 获取迭代域
- 支持多种循环类型：`scf.for`、`scf.forall`、`scf.parallel`
- 计算tile sizes和num threads
- 处理interchange向量（循环置换）
- 支持reduction平铺策略

**配置选项**:
```cpp
class SCFTilingOptions {
public:
  enum class LoopType { ForOp, ForallOp, ParallelOp };

  LoopType loopType = LoopType::ForOp;

  // Tile size计算函数
  std::function<SmallVector<OpFoldResult>(OpBuilder &, Operation *)>
      tileSizeComputationFunction;

  // 交换向量（用于循环置换）
  SmallVector<int64_t> interchangeVector;
};
```

**核心函数**:
```cpp
// 执行平铺
FailureOr<TilingResult> tileUsingSCF(RewriterBase &rewriter,
                                      TilingInterface op,
                                      const SCFTilingOptions &options);

// 平铺消费者并融合生产者
FailureOr<TilingResult> tileConsumerAndFuseProducers(
    RewriterBase &rewriter,
    TilingInterface consumerOp,
    const SCFTilingOptions &options,
    std::optional<SCFTilingOptions> producerTileOption);
```

**平铺流程**:
```cpp
// 1. 获取迭代域
SmallVector<IterationDomain> iterationDomains = op.getIterationDomains(rewriter);

// 2. 计算tile sizes
SmallVector<OpFoldResult> tileSizes = options.tileSizeComputationFunction(...);

// 3. 应用interchange
if (!options.interchangeVector.empty()) {
  // 置换循环维度
}

// 4. 创建循环嵌套
switch (options.loopType) {
  case LoopType::ForOp:
    loopNest = generateTileLoopNest<scf::ForOp>(...);
    break;
  case LoopType::ForallOp:
    loopNest = generateTileLoopNest<scf::ForallOp>(...);
    break;
  case LoopType::ParallelOp:
    loopNest = generateTileLoopNest<scf::ParallelOp>(...);
    break;
}

// 5. 内联tile body
// 6. 替换原始操作
```

**示例 - 使用scf.for平铺**:

转换前:
```mlir
%result = linalg.matmul
    ins(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>)
    outs(%C: tensor<128x128xf32>) -> tensor<128x128xf32>
```

转换后 (tileSize = 32):
```mlir
%result = scf.for %i = 0 to 128 step 32 iter_args(%arg = %C)
    -> tensor<128x128xf32> {
  %inner = scf.for %j = 0 to 128 step 32 iter_args(%arg2 = %arg)
      -> tensor<128x128xf32> {
    %innermost = scf.for %k = 0 to 128 step 32 iter_args(%arg3 = %arg2)
        -> tensor<128x128xf32> {
      %tile = linalg.matmul
          ins(%A[%i, %i+32, %k, %k+32],
              %B[%k, %k+32, %j, %j+32])
          outs(%arg3[%i, %i+32, %j, %j+32])
          -> tensor<32x32xf32>
      %updated = tensor.insert_slice %tile into %arg3[...] [...]
      scf.yield %updated : tensor<128x128xf32>
    }
    scf.yield %innermost : tensor<128x128xf32>
  }
  scf.yield %inner : tensor<128x128xf32>
}
```

**示例 - 使用scf.forall平铺**:
```mlir
%tile_sizes = [32, 32]
%result = scf.forall (%i, %j) in ([0, 0]) to ([128, 128]) step (%tile_sizes[0], %tile_sizes[1]) {
  %tile = linalg.matmul
      ins(%A[%i, %i+32, :], %B[:, %j, %j+32])
      outs(%C[%i, %i+32, %j, %j+32])
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %tile into %result[...] [...]
  }
}
```

**高级特性**:

1. **Producer-Consumer融合**:
```cpp
tileConsumerAndFuseProducers(rewriter, consumerOp, options, producerOptions);
// 自动将生产者操作融合到消费者循环中
```

2. **循环交换 (Interchange)**:
```cpp
options.interchangeVector = {1, 0};  // 交换前两个维度
```

3. **Reduction平铺**:
```cpp
options.tileSizeComputationFunction = [](OpBuilder &b, Operation *op) {
  // 为reduction维度计算特殊的tile size
};
```

**使用场景**:
- 多态操作的平铺优化
- 自动向量化
- GPU kernel生成
- 缓存优化

---

## 总结

MLIR SCF方言的17个Transform涵盖了以下优化领域：

| 类别 | Transform | 主要用途 |
|------|-----------|----------|
| **循环转换** | ForToWhile, ForallToFor, ForallToParallel, UpliftWhileToFor | 循环表示形式转换 |
| **循环优化** | Pipelining, RangeFolding, Specialization, Rotation, ZeroTripCheck | 性能优化和边界处理 |
| **并行循环** | Collapsing, Fusion, Tiling | 并行化和数据局部性 |
| **类型转换** | StructuralTypeConversions, Bufferization接口 | 内存布局转换 |
| **通用优化** | Canonicalization, TileUsingInterface | 跨方言优化 |

这些变换可以组合使用，形成完整的优化流水线。例如：
```
原始代码 -> ForallToFor -> LoopSpecialization -> LoopTiling -> Vectorization -> CodeGen
```
