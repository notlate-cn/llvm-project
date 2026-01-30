# LLVM MLIR Bufferization方言Transform技术详解

本文档详细梳理LLVM MLIR中Bufferization方言的所有Transform Pass的作用、技术原理和应用场景。

**目录路径**: `mlir/lib/Dialect/Bufferization/Transforms/`

---

## 目录

1. [Bufferization核心概念](#Bufferization核心概念)
2. [核心分析与转换](#核心分析与转换)
3. [内存管理](#内存管理)
4. [优化Pass](#优化Pass)
5. [接口与工具](#接口与工具)
6. [完整Pipeline](#完整Pipeline)

---

## Bufferization核心概念

### 什么是Bufferization?

**Bufferization**是将基于value语义的**tensor类型**转换为基于引用语义的**memref类型**的过程。

```mlir
// 转换前: Tensor (value语义)
%result = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
                        outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>

// 转换后: Memref (引用语义)
linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x16xf32>)
              outs(%C : memref<4x16xf32>)
```

### 核心挑战

#### 1. In-place vs Out-of-place
- **In-place**: 直接修改现有buffer
- **Out-of-place**: 分配新buffer并拷贝

#### 2. Read-After-Write (RaW) 冲突
```mlir
// 冲突示例
%0 = tensor.extract_slice %t[0][4][1] : tensor<8xf32> to tensor<4xf32>
%1 = tensor.insert_slice %v into %t[4][4][1] : tensor<4xf32> into tensor<8xf32>
// %0和%1都使用%t, 存在RaW冲突
```

#### 3. Aliasing分析
确定哪些tensor/memref可能指向相同的内存。

---

## 核心分析与转换

### 1. One-Shot Analysis (单次分析)

**文件**: `OneShotAnalysis.cpp` (1376行)

#### 1.1 作用
**核心分析引擎**,一次遍历函数体,计算所有tensor操作的最优bufferization策略(in-place或out-of-place)。

#### 1.2 技术原理

##### AnalysisState结构

```cpp
class OneShotAnalysisState {
  // Alias分析: 追踪哪些值可能指向同一buffer
  AliasInfo aliasInfo;

  // Equivalence分析: 追踪哪些值必定指向同一buffer
  EquivalenceInfo equivalentInfo;

  // In-place决策映射
  DenseMap<OpOperand*, bool> inplaceBufferized;

  // 定义链缓存
  DenseMap<Value, SetVector<Value>> definitionCache;
};
```

##### RaW冲突检测算法

**核心函数**: `hasReadAfterWriteInterference()`

```cpp
// 伪代码
bool hasReadAfterWriteInterference(OpOperand* operand) {
  // 1. 收集可能被此operand写入的所有alias
  Set<Value> aliasesWritten = getAliasesWrittenBy(operand);

  // 2. 收集可能从这些alias读取的所有后续操作
  Set<OpOperand*> aliasesRead = getSubsequentReads(aliasesWritten);

  // 3. 使用支配性分析判断是否存在冲突
  for (OpOperand* read : aliasesRead) {
    if (!happensBefore(write, read, dominanceInfo)) {
      return true;  // 存在RaW冲突
    }
  }
  return false;
}
```

##### 四种冲突检测规则

1. **Simple Dominance**:
   ```mlir
   %0 = read(%t)   // 读
   %1 = write(%t)  // 写
   // 读支配写 → 无冲突
   ```

2. **Loops Prevent Analysis**:
   ```mlir
   scf.for %i {
     %0 = read(%t)
     %1 = write(%t)  // 循环中无法确定顺序 → 冲突
   }
   ```

3. **Mutually Exclusive Regions**:
   ```mlir
   scf.if %cond {
     %0 = read(%t)
   } else {
     %1 = write(%t)  // 互斥 → 无冲突
   }
   ```

4. **Element-wise Access**:
   ```mlir
   %0 = tensor.extract_slice %t[0][4][1]  // 读 [0:4)
   %1 = tensor.insert_slice %v into %t[4][4][1]  // 写 [4:8)
   // 不重叠 → 无冲突
   ```

#### 1.3 实例演示

**示例1: 安全的In-place**
```mlir
func.func @matmul(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>,
                  %C: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // 分析: %C只被写入,未被读取
  %result = linalg.matmul ins(%A, %B) outs(%C)
  // 决策: %C可以in-place bufferize
  return %result
}

// Bufferization结果
func.func @matmul(%A: memref<4x8xf32>, %B: memref<8x16xf32>,
                  %C: memref<4x16xf32>) {
  linalg.matmul ins(%A, %B) outs(%C)  // 直接写入%C
  return
}
```

**示例2: RaW冲突需要Copy**
```mlir
func.func @conflict(%t: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  %0 = tensor.extract_slice %t[0][4][1]  // 读 %t[0:4)
  %1 = linalg.fill ins(%cst) outs(%t)    // 写 整个%t
  // RaW冲突! %0读取的数据被%1覆盖
  return %0, %1
}

// 分析结果: 需要copy
// bufferization.alloc_tensor + copy操作会被插入
```

#### 1.4 分析选项

```cpp
OneShotBufferizationOptions {
  // 允许返回值分配
  bool allowReturnAllocs = true;

  // 允许未知操作
  bool allowUnknownOps = false;

  // 测试分析 (不实际bufferize)
  bool testAnalysisOnly = false;

  // 打印冲突信息
  bool printConflicts = false;
};
```

---

### 2. One-Shot Module Bufferize (模块级Bufferization)

**文件**: `OneShotModuleBufferize.cpp` (594行)

#### 2.1 作用
扩展One-Shot分析到整个模块,处理**函数边界**、**函数调用**和**递归**。

#### 2.2 技术原理

##### Call Graph分析

```cpp
// 1. 构建调用图
CallGraph callGraph = buildCallGraph(module);

// 2. 检测SCC (强连通分量 = 递归)
SmallVector<SmallVector<FuncOp>> sccs = callGraph.getSCCs();

// 3. 按逆拓扑序处理(被调用者先于调用者)
for (SCC scc : reverse(sccs)) {
  if (scc.size() > 1) {
    // 递归函数: 保守处理
    analyzeSCCConservatively(scc);
  } else {
    // 非递归函数: 精确分析
    analyzeFunction(scc[0]);
  }
}
```

##### FuncAnalysisState扩展

```cpp
class FuncAnalysisState {
  // 返回值是否alias参数?
  // returnVal -> {argIdx, argIdx, ...}
  DenseMap<OpResult, DenseSet<int64_t>> returnValAliasing;

  // 参数是否被读/写?
  // argIdx -> {read: bool, write: bool}
  DenseMap<BlockArgument, BufferRelation> bbArgReadWrite;
};
```

##### 函数边界ABI

**所有权语义**:
```mlir
// 1. 参数: 调用者拥有 (不释放)
// 2. 返回值: 被调用者拥有 (调用者负责释放)

func.func @foo(%arg: memref<4xf32>) -> memref<4xf32> {
  %result = memref.alloc() : memref<4xf32>
  // ... 计算
  return %result  // 调用者必须释放%result
}
```

#### 2.3 实例演示

**示例: 跨函数分析**
```mlir
// 被调用者
func.func @callee(%t: tensor<8xf32>) -> tensor<8xf32> {
  %c = arith.constant 0.0 : f32
  %result = linalg.fill ins(%c) outs(%t)
  // 分析: 写%t, 返回值alias %t
  return %result
}

// 调用者
func.func @caller(%input: tensor<8xf32>) -> tensor<8xf32> {
  %0 = func.call @callee(%input)  // 分析知道%0 aliases %input
  %1 = tensor.extract %0[%c0]     // 读%0
  // 决策: %input可以in-place传递给callee
  return %0
}

// Bufferization后
func.func @callee(%t: memref<8xf32>) {
  linalg.fill ins(%c) outs(%t)  // 直接修改%t
}

func.func @caller(%input: memref<8xf32>) -> memref<8xf32> {
  call @callee(%input)  // in-place调用
  %v = memref.load %input[%c0]
  return %input
}
```

**外部函数处理**:
```mlir
// 外部函数声明
func.func private @external(%t: tensor<?xf32>) -> tensor<?xf32>

// 保守假设:
// - 所有参数可能被读写
// - 返回值可能alias任意参数
// → 强制out-of-place
```

---

### 3. Tensor Copy Insertion (拷贝插入)

**文件**: `TensorCopyInsertion.cpp` (87行)

#### 3.1 作用
在tensor级别插入拷贝操作,解决分析发现的RaW冲突。

#### 3.2 技术原理

##### 两阶段处理

**阶段1**: 运行分析
```cpp
OneShotAnalysisState state;
analyzeOp(funcOp, state);  // 计算冲突
```

**阶段2**: 解决冲突
```cpp
funcOp.walk([&](Operation* op) {
  if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op)) {
    bufferizableOp.resolveConflicts(rewriter, state);
  }
});
```

##### 冲突解决策略

**默认实现**: 分配新tensor + 拷贝
```mlir
// 冲突操作
%result = some.op(%t)  // 需要out-of-place

// resolveConflicts插入:
%new = bufferization.alloc_tensor() : tensor<8xf32>
%copied = linalg.copy ins(%t) outs(%new)
%result = some.op(%copied)
```

#### 3.3 实例演示

**示例: 解决RaW冲突**
```mlir
// 原始IR (有冲突)
func.func @example(%t: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  %slice = tensor.extract_slice %t[0][4][1]  // 读%t
  %filled = linalg.fill ins(%c0) outs(%t)    // 写%t, 冲突!
  return %slice, %filled
}

// 拷贝插入后
func.func @example(%t: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  %slice = tensor.extract_slice %t[0][4][1]

  // 插入拷贝
  %new = bufferization.alloc_tensor() : tensor<8xf32>
  %t_copy = linalg.copy ins(%t) outs(%new)

  %filled = linalg.fill ins(%c0) outs(%t_copy)  // 使用拷贝
  return %slice, %filled
}

// Bufferization后
func.func @example(%t: memref<8xf32>) -> (memref<4xf32>, memref<8xf32>) {
  %slice = memref.subview %t[0][4][1]
  %new = memref.alloc() : memref<8xf32>
  memref.copy %t, %new
  linalg.fill ins(%c0) outs(%new)
  return %slice, %new
}
```

---

## 内存管理

### 4. Ownership-Based Buffer Deallocation (基于所有权的释放)

**文件**: `OwnershipBasedBufferDeallocation.cpp` (1104行)

#### 4.1 作用
**自动内存管理**,插入`bufferization.dealloc`操作,基于运行时所有权检查安全释放内存。

#### 4.2 技术原理

##### 所有权表示

```mlir
// 每个memref关联一个i1所有权标志
%buf : memref<4xf32>
%owns : i1  // true = 拥有, false = 不拥有
```

##### Dealloc操作语义

```mlir
// bufferization.dealloc (%memrefs) if (%conditions) retain (%retained)
bufferization.dealloc (%m1, %m2 : memref<4xf32>, memref<8xf32>)
                   if (%c1, %c2 : i1, i1)
               retain (%r1 : memref<4xf32>)
// 语义:
// - 如果 %m1与%r1不alias 且 %c1为true, 释放%m1
// - 如果 %m2与%r1不alias 且 %c2为true, 释放%m2
// - 返回%r1的所有权标志(OR of aliasing ownership)
```

##### 所有权传播算法

```cpp
// Block-level deallocation state
class DeallocationState {
  // memref -> ownership (i1 value)
  DenseMap<Value, Value> ownerships;

  // 更新所有权
  void updateOwnership(Value memref, Value ownership) {
    ownerships[memref] = ownership;
  }

  // 获取所有权
  Value getOwnership(Value memref) {
    return ownerships.lookup(memref);
  }
};
```

**处理不同操作类型**:

1. **Allocation**:
   ```mlir
   %buf = memref.alloc() : memref<4xf32>
   // ownership = true (新分配,拥有)
   ```

2. **Block Arguments**:
   ```mlir
   ^bb0(%arg: memref<4xf32>, %own: i1):
   // ownership = %own (传递进来的)
   ```

3. **View Operations**:
   ```mlir
   %view = memref.subview %base[...]
   // ownership(%view) = ownership(%base) (共享所有权)
   ```

4. **Clone**:
   ```mlir
   %clone = bufferization.clone %src
   // ownership(%clone) = true (新副本,拥有)
   ```

##### Block终止处理

在每个block的终止点:
```mlir
^bb0(%arg: memref<4xf32>, %own: i1):
  %buf1 = memref.alloc() : memref<4xf32>
  %buf2 = memref.subview %arg[...]
  br ^bb1(%buf1, %buf2 : memref<4xf32>, memref<4xf32>)

// 插入dealloc:
  bufferization.dealloc (%arg, %buf1 : memref<4xf32>, memref<4xf32>)
                     if (%own, %true : i1, i1)
                 retain (%buf1, %buf2 : memref<4xf32>, memref<4xf32>)
  // 传递所有权到后继block
  br ^bb1(%buf1, %buf2, %own1, %own2 : memref<4xf32>, memref<4xf32>, i1, i1)
```

#### 4.3 实例演示

**示例1: 简单分支**
```mlir
func.func @branch(%cond: i1, %t: memref<4xf32>) -> memref<4xf32> {
  cf.cond_br %cond, ^bb_true, ^bb_false

^bb_true:
  %alloc = memref.alloc() : memref<4xf32>
  cf.br ^bb_exit(%alloc : memref<4xf32>)

^bb_false:
  cf.br ^bb_exit(%t : memref<4xf32>)

^bb_exit(%result: memref<4xf32>):
  return %result : memref<4xf32>
}

// Deallocation插入后
func.func @branch(%cond: i1, %t: memref<4xf32>) -> memref<4xf32> {
  cf.cond_br %cond, ^bb_true, ^bb_false

^bb_true:
  %alloc = memref.alloc() : memref<4xf32>
  %true = arith.constant true
  // 传递所有权
  cf.br ^bb_exit(%alloc, %true : memref<4xf32>, i1)

^bb_false:
  %false = arith.constant false  // 不拥有%t
  cf.br ^bb_exit(%t, %false : memref<4xf32>, i1)

^bb_exit(%result: memref<4xf32>, %owns: i1):
  // 调用者负责根据%owns释放%result
  return %result, %owns : memref<4xf32>, i1
}
```

**示例2: 循环**
```mlir
func.func @loop(%lb: index, %ub: index) {
  %init = memref.alloc() : memref<4xf32>

  scf.for %i = %lb to %ub step %c1 {
    %buf = memref.alloc() : memref<4xf32>
    // 使用%buf
  }

  return
}

// Deallocation插入后
func.func @loop(%lb: index, %ub: index) {
  %init = memref.alloc() : memref<4xf32>

  scf.for %i = %lb to %ub step %c1 {
    %buf = memref.alloc() : memref<4xf32>
    // 使用%buf

    // Loop内自动释放
    %true = arith.constant true
    bufferization.dealloc (%buf : memref<4xf32>) if (%true : i1)
  }

  // 函数退出时释放
  %true = arith.constant true
  bufferization.dealloc (%init : memref<4xf32>) if (%true : i1)
  return
}
```

#### 4.4 接口驱动

**BufferDeallocationOpInterface**:
```cpp
// 操作自定义deallocation行为
class MyOp : BufferDeallocationOpInterface {
  LogicalResult process(DeallocationState& state) {
    // 自定义所有权更新逻辑
  }
};
```

---

### 5. Buffer Deallocation Simplification (释放简化)

**文件**: `BufferDeallocationSimplification.cpp` (498行)

#### 5.1 作用
优化`bufferization.dealloc`操作,使用别名分析减少运行时检查。

#### 5.2 关键优化模式

##### 模式1: 移除Must-Alias的Memref

**规则**: 如果to-dealloc memref与retained memref **must-alias**,则从dealloc列表移除。

```mlir
// 优化前
%0 = memref.alloc() : memref<4xf32>
%1 = memref.cast %0 : memref<4xf32> to memref<?xf32>
bufferization.dealloc (%0 : memref<4xf32>)
                   if (%true : i1)
               retain (%1 : memref<?xf32>)

// 分析: %0 must-alias %1

// 优化后
// dealloc操作完全移除
// 所有权传递给retained value
```

##### 模式2: 移除No-Alias的Retained

**规则**: 如果retained memref与所有to-dealloc memref **must-not-alias**,移除。

```mlir
// 优化前
%0 = memref.alloc() : memref<4xf32>
%1 = memref.alloc() : memref<8xf32>
%r = bufferization.dealloc (%0 : memref<4xf32>)
                         if (%c0 : i1)
                     retain (%1 : memref<8xf32>)

// 分析: %0和%1来自不同alloc, must-not-alias

// 优化后
bufferization.dealloc (%0 : memref<4xf32>) if (%c0 : i1)
%false = arith.constant false  // %1的所有权必为false
```

##### 模式3: 拆分独立Dealloc

**规则**: 将单个dealloc拆分为多个简单dealloc,减少运行时alias检查。

```mlir
// 优化前
bufferization.dealloc (%a, %b, %c : memref<4xf32>, memref<8xf32>, memref<16xf32>)
                   if (%ca, %cb, %cc : i1, i1, i1)
               retain (%r : memref<4xf32>)
// 如果%b和%c与%a, %r都不可能alias

// 优化后
bufferization.dealloc (%a : memref<4xf32>)
                   if (%ca : i1)
               retain (%r : memref<4xf32>)
bufferization.dealloc (%b : memref<8xf32>) if (%cb : i1)
bufferization.dealloc (%c : memref<16xf32>) if (%cc : i1)
```

#### 5.3 别名分析

**BufferOriginAnalysis**:
```cpp
// 追踪memref的来源
Value getBufferOrigin(Value memref) {
  // 向上追踪定义链
  while (isViewLike(memref)) {
    memref = getViewSource(memref);
  }
  // 返回allocation site
  return memref;  // memref.alloc, block arg, etc.
}

// Must-alias判断
bool mustAlias(Value v1, Value v2) {
  return getBufferOrigin(v1) == getBufferOrigin(v2);
}
```

---

## 优化Pass

### 6. Buffer Optimizations (Buffer优化)

**文件**: `BufferOptimizations.cpp` (479行)

#### 6.1 Buffer Hoisting (Buffer提升)

##### 作用
将allocation移出内层block到支配的外层block,减少重复分配。

##### 算法
```cpp
// 1. 遍历所有alloc
for (AllocOp alloc : allocs) {
  // 2. 找到最外层支配block
  Block* hoistTarget = findDominatingBlock(alloc);

  // 3. 检查是否越过loop边界
  if (crossesLoopBoundary(alloc.getBlock(), hoistTarget)) {
    continue;  // 不提升出循环
  }

  // 4. 移动alloc
  alloc->moveBefore(hoistTarget->getTerminator());
}
```

##### 示例
```mlir
// 优化前
func.func @compute(%n: index) {
  scf.if %cond {
    %buf = memref.alloc() : memref<4xf32>
    // use %buf
  } else {
    %buf2 = memref.alloc() : memref<4xf32>
    // use %buf2
  }
}

// 优化后
func.func @compute(%n: index) {
  %buf = memref.alloc() : memref<4xf32>  // 提升到外层
  scf.if %cond {
    // use %buf
  } else {
    // use %buf (复用同一alloc)
  }
}
```

#### 6.2 Buffer Loop Hoisting (循环提升)

##### 作用
将allocation移出循环,避免每次迭代都分配。

##### 关键检查
```cpp
bool canHoistOutOfLoop(AllocOp alloc, LoopLikeOp loop) {
  // 1. Buffer不逃逸循环
  for (User* user : alloc.getUsers()) {
    if (!loop.contains(user)) {
      return false;
  }

  // 2. 无别名逃逸
  for (Value alias : getAliases(alloc)) {
    if (escapesLoop(alias, loop)) {
      return false;
    }
  }

  return true;
}
```

##### 示例
```mlir
// 优化前
scf.for %i = %c0 to %c100 step %c1 {
  %buf = memref.alloc() : memref<4xf32>  // 每次迭代分配
  // compute using %buf
  memref.dealloc %buf
}

// 优化后
%buf = memref.alloc() : memref<4xf32>  // 循环外分配一次
scf.for %i = %c0 to %c100 step %c1 {
  // compute using %buf
}
memref.dealloc %buf
```

#### 6.3 Promote Buffers to Stack (栈提升)

##### 作用
将heap allocation (`memref.alloc`) 转换为stack allocation (`memref.alloca`)。

##### 条件
1. **大小限制**: `allocSize <= maxAllocSizeInBytes`
2. **作用域**: buffer不逃逸allocation scope
3. **无动态依赖**: 静态可知分配大小

##### 示例
```mlir
// 优化前
func.func @compute() {
  %buf = memref.alloc() : memref<16xf32>  // heap, 64 bytes
  // use %buf
  memref.dealloc %buf
}

// 优化后 (maxAllocSizeInBytes >= 64)
func.func @compute() {
  %buf = memref.alloca() : memref<16xf32>  // stack
  // use %buf
  // 自动释放
}
```

---

### 7. Empty Tensor Elimination (空张量消除)

**文件**: `EmptyTensorElimination.cpp` (235行)

#### 7.1 作用
消除`tensor.empty`操作,将其替换为目标tensor的subset extraction。

#### 7.2 优化原理

**Insight**: `tensor.empty`创建未定义内容的tensor,如果只使用其subset,可以直接从目标提取。

```mlir
// Pattern
%empty = tensor.empty() : tensor<10xf32>
// ... intermediate ops ...
%result = tensor.insert_slice %v into %dst[%offset][5][1]
          : tensor<5xf32> into tensor<10xf32>
// %empty只用于提供shape

// 替换为
%extracted = tensor.extract_slice %dst[%offset][5][1]
%result = tensor.insert_slice %v into %dst[%offset][5][1]
```

#### 7.3 算法

```cpp
// 1. 找到insert_slice等subset操作
for (SubsetOp op : subsetOps) {
  // 2. 反向追踪SSA使用链
  Value current = op.getSource();
  while (isEquivalent(current)) {
    if (auto empty = dyn_cast<EmptyOp>(current.getDefiningOp())) {
      // 3. 找到empty, 构造extract替换
      Value replacement = buildExtractOp(op.getDestination());
      empty.replaceAllUsesWith(replacement);
      break;
    }
    current = getNextEquivalent(current);
  }
}
```

#### 7.4 示例

```mlir
// 优化前
func.func @eliminate(%dst: tensor<100xf32>, %v: tensor<10xf32>,
                     %offset: index) -> tensor<100xf32> {
  %empty = tensor.empty() : tensor<10xf32>

  %filled = linalg.fill ins(%cst) outs(%empty) : tensor<10xf32>

  %result = tensor.insert_slice %filled into %dst[%offset][10][1]

  return %result
}

// 优化后
func.func @eliminate(%dst: tensor<100xf32>, %v: tensor<10xf32>,
                     %offset: index) -> tensor<100xf32> {
  // 直接从dst提取
  %extracted = tensor.extract_slice %dst[%offset][10][1]

  %filled = linalg.fill ins(%cst) outs(%extracted)

  %result = tensor.insert_slice %filled into %dst[%offset][10][1]

  return %result
}
```

---

### 8. Optimize Allocation Liveness (分配存活期优化)

**文件**: `OptimizeAllocationLiveness.cpp` (154行)

#### 8.1 作用
延迟deallocation placement,将释放操作移到最后一次使用之后。

#### 8.2 算法

```cpp
for (AllocOp alloc : allocs) {
  // 1. 找到所有使用点(包括alias)
  SmallVector<Operation*> users = getAllUsers(alloc);

  // 2. 找到最后一个使用
  Operation* lastUser = findLastUser(users);

  // 3. 找到对应的dealloc
  DeallocOp dealloc = findDealloc(alloc);

  // 4. 移动dealloc到lastUser之后
  dealloc->moveAfter(lastUser);
}
```

#### 8.3 示例

```mlir
// 优化前
func.func @compute() {
  %buf = memref.alloc() : memref<4xf32>
  memref.dealloc %buf  // 立即释放

  %v1 = memref.load %buf[%c0]  // 使用1
  // ... 很多操作 ...
  %v2 = memref.load %buf[%c1]  // 最后使用

  // ... 更多操作 ...
}

// 优化后
func.func @compute() {
  %buf = memref.alloc() : memref<4xf32>

  %v1 = memref.load %buf[%c0]
  // ... 很多操作 ...
  %v2 = memref.load %buf[%c1]

  memref.dealloc %buf  // 移到最后使用之后
  // ... 更多操作 ...
}
```

---

## 接口与工具

### 9. Func Bufferizable Op Interface (函数接口实现)

**文件**: `FuncBufferizableOpInterfaceImpl.cpp` (532行)

#### 9.1 作用
为`func.func`, `func.call`, `func.return`实现BufferizableOpInterface。

#### 9.2 FuncOp bufferization

```cpp
LogicalResult FuncOp::bufferize(RewriterBase& rewriter,
                                 const BufferizationOptions& options) {
  // 1. 转换函数签名
  FunctionType funcType = getFunctionType();
  SmallVector<Type> newArgTypes, newResultTypes;

  for (Type argType : funcType.getInputs()) {
    if (auto tensorType = dyn_cast<TensorType>(argType)) {
      newArgTypes.push_back(getMemRefType(tensorType));
    } else {
      newArgTypes.push_back(argType);
    }
  }

  for (Type resType : funcType.getResults()) {
    if (auto tensorType = dyn_cast<TensorType>(resType)) {
      newResultTypes.push_back(getMemRefType(tensorType));
    } else {
      newResultTypes.push_back(resType);
    }
  }

  // 2. 更新函数类型
  auto newFuncType = FunctionType::get(context, newArgTypes, newResultTypes);
  setFunctionType(newFuncType);

  // 3. 转换block arguments
  for (BlockArgument arg : getArguments()) {
    if (isa<TensorType>(arg.getType())) {
      arg.setType(getMemRefType(arg.getType()));
    }
  }
}
```

#### 9.3 CallOp bufferization

```cpp
LogicalResult CallOp::bufferize(RewriterBase& rewriter,
                                 const BufferizationOptions& options) {
  // 查询FuncAnalysisState
  FuncAnalysisState* funcState = options.getAnalysisState();

  // 检查哪些参数必须out-of-place
  for (OpOperand& operand : getOpOperands()) {
    if (funcState->isRead(callee, operand.getOperandNumber()) &&
        funcState->isWritten(callee, operand.getOperandNumber())) {
      // 读写参数: 需要copy
      insertCopy(operand);
    }
  }

  // 转换调用
  SmallVector<Value> newOperands = bufferizeOperands(getOperands());
  auto newCall = rewriter.create<func::CallOp>(
    getLoc(), getCallee(), newOperands
  );

  replaceOpWithBufferizedValues(rewriter, *this, newCall.getResults());
}
```

---

### 10. Buffer Results to Out Params (返回值转参数)

**文件**: `BufferResultsToOutParams.cpp` (247行)

#### 10.1 作用
将函数的memref返回值转换为out-parameter调用约定。

#### 10.2 转换策略

```mlir
// 转换前
func.func @compute(%input: memref<4xf32>) -> memref<4xf32> {
  %result = memref.alloc() : memref<4xf32>
  // ... 计算 ...
  return %result : memref<4xf32>
}

func.func @caller(%in: memref<4xf32>) {
  %out = func.call @compute(%in) : (memref<4xf32>) -> memref<4xf32>
  // use %out
}

// 转换后
func.func @compute(%input: memref<4xf32>, %output: memref<4xf32>) {
  // ... 计算直接写入%output ...
  return
}

func.func @caller(%in: memref<4xf32>) {
  %out = memref.alloc() : memref<4xf32>
  func.call @compute(%in, %out) : (memref<4xf32>, memref<4xf32>) -> ()
  // use %out
}
```

#### 10.3 支持的布局

- ✅ Static identity layout: `memref<4x8xf32>`
- ✅ Fully dynamic: `memref<?x?xf32>`
- ❌ Partial dynamic with strides: `memref<4x?xf32, strided<[?, 1]>>`

---

### 11. Drop Equivalent Buffer Results (丢弃等价返回)

**文件**: `DropEquivalentBufferResults.cpp` (134行)

#### 11.1 作用
移除返回值中与参数等价的memref。

#### 11.2 示例

```mlir
// 优化前
func.func @identity(%arg: memref<4xf32>) -> memref<4xf32> {
  return %arg : memref<4xf32>  // 返回值等于参数
}

func.func @caller(%m: memref<4xf32>) {
  %r = func.call @identity(%m)
  // use %r
}

// 优化后
func.func @identity(%arg: memref<4xf32>) {
  return  // 移除返回值
}

func.func @caller(%m: memref<4xf32>) {
  func.call @identity(%m)
  // use %m (直接使用参数)
}
```

---

### 12. 辅助工具

#### Buffer View Flow Analysis
**文件**: `BufferViewFlowAnalysis.cpp` (339行)

**作用**: 追踪buffer的数据流和别名关系。

```cpp
class BufferViewFlowAnalysis {
  // 获取所有可能的alias
  ValueSetT resolve(Value value);

  // 判断是否可能alias
  bool mayAlias(Value v1, Value v2);
};
```

#### Buffer Utils
**文件**: `BufferUtils.cpp` (182行)

**工具函数**:
- `getAllocDeallocPairs()`: 匹配alloc-dealloc对
- `getTopologicallySortedBlocks()`: 拓扑排序blocks
- `getMemRefTypeWithDynamicOffsets()`: 构造动态offset类型

---

## 完整Pipeline

### 推荐的Bufferization流程

```bash
# 完整bufferization pipeline

# === 阶段1: 预处理 ===

# 1. 空张量转换
--empty-tensor-to-alloc-tensor

# 2. 函数边界准备
# (注册BufferizableOpInterface实现)

# === 阶段2: 分析与拷贝插入 ===

# 3. One-Shot分析 + 拷贝插入
--one-shot-bufferize="
  allow-return-allocs=true
  allow-unknown-ops=false
  bufferize-function-boundaries=true
  function-boundary-type-conversion=identity-layout-map
"

# 或者分步:
# 3a. 仅分析(测试)
--one-shot-bufferize="test-analysis-only=true print-conflicts=true"

# 3b. 拷贝插入
--tensor-copy-insertion

# 3c. 实际bufferize
--one-shot-bufferize

# === 阶段3: 优化 ===

# 4. 空张量消除
--empty-tensor-elimination

# 5. Buffer优化
--buffer-hoisting
--buffer-loop-hoisting
--promote-buffers-to-stack="max-alloc-size-in-bytes=1024"

# === 阶段4: 内存管理 ===

# 6. 自动deallocation
--ownership-based-buffer-deallocation

# 7. Deallocation简化
--buffer-deallocation-simplification

# 8. 优化allocation存活期
--optimize-allocation-liveness

# === 阶段5: 函数约定转换(可选) ===

# 9. 移除等价返回
--drop-equivalent-buffer-results

# 10. 返回值转out-param
--buffer-results-to-out-params

# === 阶段6: Lowering ===

# 11. Lower deallocation到条件释放
--lower-deallocations

# 12. Canonicalize清理
--canonicalize

# 13. Lower到memref/scf/cf
--convert-bufferization-to-memref
--lower-affine
--convert-scf-to-cf

# === 阶段7: 最终lowering ===

# 14. Lower到LLVM
--convert-memref-to-llvm
--convert-func-to-llvm
--reconcile-unrealized-casts
```

### 关键选项说明

#### One-Shot Bufferize选项

```cpp
struct OneShotBufferizationOptions {
  // 允许函数返回新分配的buffer
  bool allowReturnAllocs = true;

  // 允许未注册BufferizableOpInterface的操作
  bool allowUnknownOps = false;

  // bufferize函数边界
  bool bufferizeFunctionBoundaries = true;

  // 函数边界类型转换策略
  // - "identity-layout-map": 使用identity layout
  // - "fully-dynamic-layout-map": 使用fully dynamic layout
  std::string functionBoundaryTypeConversion = "identity-layout-map";

  // 仅做分析,不实际bufferize(用于调试)
  bool testAnalysisOnly = false;

  // 打印RaW冲突信息
  bool printConflicts = false;

  // 创建deallocation操作
  bool createDeallocs = true;
};
```

---

## 核心概念总结

### 1. Equivalence vs Aliasing

**Equivalence** (等价):
- 两个值**必定**指向同一buffer
- 传递性关系
- 用于优化决策

**Aliasing** (别名):
- 两个值**可能**指向同一buffer
- 用于冲突检测
- 保守分析

```mlir
%0 = tensor.empty()
%1 = linalg.fill ins(%c) outs(%0)
%2 = tensor.extract_slice %1[0][4][1]
// %0 equivalent %1 (必定相同)
// %2 aliases %1 (可能部分重叠)
```

### 2. In-place决策

**In-place条件**:
1. ✅ 无RaW冲突
2. ✅ Buffer未逃逸
3. ✅ 写操作支配所有读操作
4. ✅ 操作语义允许修改

### 3. 所有权模型

**三种所有权状态**:
- **Static True**: 编译时已知拥有 (`memref.alloc`)
- **Static False**: 编译时已知不拥有 (函数参数)
- **Dynamic**: 运行时确定 (`i1`值, 控制流汇合点)

### 4. BufferizableOpInterface核心方法

```cpp
class BufferizableOpInterface {
  // 是否可以bufferize?
  bool bufferizesToMemoryRead(OpOperand& opOperand);
  bool bufferizesToMemoryWrite(OpOperand& opOperand);

  // Aliasing关系
  AliasingValueList getAliasingValues(OpOperand& opOperand);

  // 是否可以in-place?
  bool isWritable(Value value);

  // 冲突解决
  LogicalResult resolveConflicts(RewriterBase& rewriter,
                                 const AnalysisState& state);

  // 执行bufferization
  LogicalResult bufferize(RewriterBase& rewriter,
                         const BufferizationOptions& options);
};
```

---

## 性能影响

### 优化效果

| 优化 | 收益 | 场景 |
|------|------|------|
| In-place bufferize | 消除拷贝 | 无冲突操作 |
| Buffer hoisting | 减少分配次数 | 分支/循环 |
| Stack promotion | 减少heap开销 | 小buffer |
| Empty elimination | 避免初始化 | Subset操作 |
| Dealloc simplification | 减少运行时检查 | 复杂控制流 |

### Tradeoffs

**编译时间 vs 运行时性能**:
- 更激进的分析 → 更长编译时间
- 更多in-place → 更少拷贝 → 更快执行

**内存使用 vs 性能**:
- Buffer hoisting → 更长生命周期 → 更高内存峰值
- 但减少分配 → 更少碎片 → 更好局部性

---

## 调试技巧

### 1. 可视化冲突

```bash
mlir-opt input.mlir \
  --one-shot-bufferize="test-analysis-only=true print-conflicts=true" \
  2>&1 | grep "RaW conflict"
```

### 2. 检查bufferization决策

```bash
mlir-opt input.mlir \
  --one-shot-bufferize="test-analysis-only=true" \
  --mlir-print-ir-after-all
# 查看IR注解: {inplace = [true/false]}
```

### 3. 验证deallocation

```bash
# 启用ownership tracking可视化
mlir-opt input.mlir \
  --ownership-based-buffer-deallocation \
  --mlir-print-debuginfo
```

### 4. 性能profiling

```python
# 使用MLIR Python bindings
import mlir

module = mlir.ir.Module.parse(...)

# 统计拷贝次数
num_alloc_tensor = count_ops(module, "bufferization.alloc_tensor")
num_copy = count_ops(module, "linalg.copy")

print(f"Allocations: {num_alloc_tensor}, Copies: {num_copy}")
```

---

## 常见问题

### Q1: In-place失败的常见原因?
**A**:
1. RaW冲突
2. Buffer逃逸函数
3. 并行区域中的写操作
4. 动态形状不匹配

### Q2: 如何强制out-of-place?
**A**: 插入`bufferization.alloc_tensor` + copy:
```mlir
%new = bufferization.alloc_tensor() : tensor<4xf32>
%copied = linalg.copy ins(%t) outs(%new)
```

### Q3: Deallocation错误?
**A**: 检查:
- 所有权传递是否正确
- 是否有未处理的alias
- 是否有循环依赖

### Q4: 性能不如预期?
**A**:
1. 检查分析是否过于保守
2. 启用更多优化pass
3. 考虑手动标注hints

---

## 扩展阅读

### 学术论文
- **"Composable, Sound Transformations of Nested Recursion and Loops"** - MLIR bufferization设计
- **"One-Shot Bufferization"** - MLIR文档
- **"Buffer Assignment in MLIR"** - Early design notes

### 相关文档
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [BufferizableOpInterface](https://mlir.llvm.org/docs/Interfaces/)
- [Ownership-based Deallocation](https://mlir.llvm.org/docs/Dialects/Bufferization/)

### 相关Dialect
- **memref**: 内存引用操作
- **tensor**: 值语义张量
- **linalg**: 线性代数操作

---

**文档版本**: LLVM 主干分支 (2026-01)
**维护者**: MLIR Bufferization团队
**许可证**: Apache 2.0 with LLVM Exception
