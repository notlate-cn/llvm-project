# 【MLIR】Affine 方言深入研究

本文档基于[Claude Code + GLM4.7](https://www.cnblogs.com/notlate-cn/p/19452715) + [CodeReaderSkills](https://www.cnblogs.com/notlate-cn/p/19560365)完成。

## 1. 快速概览

### 1.1 代码统计

**目录结构:**
```
mlir/lib/Dialect/Affine/
├── IR/
│   ├── AffineOps.cpp (5523 行)          # 核心操作实现
│   ├── AffineValueMap.cpp              # Affine 值映射
│   └── ValueBoundsOpInterfaceImpl.cpp   # 值边界接口实现
├── Analysis/
│   ├── AffineAnalysis.cpp (729 行)      # 依赖分析、并行性检测
│   ├── AffineStructures.cpp             # 多面体结构
│   ├── LoopAnalysis.cpp                # 循环分析
│   └── Utils.cpp                       # 工具函数
├── Transforms/
│   ├── LoopTiling.cpp (222 行)         # 循环分块
│   ├── LoopFusion.cpp (1594 行)        # 循环融合
│   ├── LoopUnroll.cpp (155 行)         # 循环展开
│   ├── AffineDataCopyGeneration.cpp    # 数据拷贝优化
│   └── ...其他变换
└── Utils/
    ├── LoopUtils.cpp                  # 循环工具
    └── LoopFusionUtils.cpp            # 融合工具

mlir/include/mlir/Dialect/Affine/
├── IR/
│   ├── AffineOps.h (563 行)           # 操作定义头文件
│   ├── AffineOps.td (1268 行)         # TableGen 操作定义
│   └── AffineValueMap.h              # 值映射头文件
├── Analysis/
│   ├── AffineAnalysis.h             # 分析接口
│   └── AffineStructures.h           # 多面体结构接口
└── Transforms/
    └── Transforms.h                 # 变换接口
```

**总计:** 约 14,332 行 C++ 代码（不含注释和空行）

### 1.2 核心依赖

```
Affine Dialect
    ├── IR 基础设施
    │   ├── AffineMap/IntegerSet     # 仿射映射和整数集
    │   └── Value/Operation          # MLIR 核心概念
    ├── Arith Dialect                # 算术运算
    ├── MemRef Dialect               # 内存引用
    ├── Presburger 库                # 多面体分析
    │   ├── IntegerRelation          # 整数关系
    │   ├── Matrix                   # 矩阵运算
    │   └── Polyhedron               # 多面体
    └── 通用分析
        ├── DataFlow                 # 数据流分析
        └── Dominance                # 支配关系
```

### 1.3 设计目标

Affine 方言是 MLIR 中用于**多面体编译**的核心方言，主要目标包括：

1. **精确的依赖分析**: 通过仿射约束表示循环边界和内存访问
2. **程序变换**: 支持循环分块、融合、重排等高级优化
3. **并行化检测**: 自动识别可并行化的循环
4. **内存优化**: 数据局部性分析和缓存优化

---

## 2. 背景与动机

### 2.1 为什么 MLIR 需要 Affine 方言

**WHY 1: 传统循环的局限性**

```mlir
// SCF (Structured Control Flow) 方言 - 灵活但难以分析
scf.for %i = 0 to %n step 1 {
  %idx = arith.addi %i, %offset : index  // ← 复杂的索引计算
  %val = memref.load %A[%idx] : memref<?xf32>
  // 无法静态分析访问模式
}
```

**WHY 2: Affine 提供精确的数学表示**

```mlir
// Affine 方言 - 约束明确，易于分析
affine.for %i = 0 to %n {
  %idx = affine.apply affine_map<(d0) -> (d0 + 5)> (%i)
  %val = affine.load %A[%idx] : memref<100xf32>
  // 访问模式: A[i+5]，可以精确分析依赖关系
}
```

**WHY 3: 多面体编译的数学基础**

多面体模型将程序表示为：
- **迭代空间**: 由不等式定义的多面体
- **访问映射**: 从迭代空间到数据空间的仿射映射
- **依赖关系**: 可以通过整数规划精确计算

### 2.2 与其他方言的对比

| 特性 | Affine | SCF | Linalg |
|------|--------|-----|--------|
| 循环边界 | 仿射表达式 | 任意 SSA 值 | 任意 SSA 值 |
| 内存访问 | 仿射索引 | 任意索引 | 语义化操作 |
| 并行性分析 | 自动精确 | 需手动标注 | 部分自动 |
| 优化空间 | 多面体变换 | 通用优化 | 高级算子融合 |
| 适用场景 | 规则计算 | 通用控制流 | 张量计算 |

### 2.3 多面体编译理论基础

**整数集合和关系:**

```
迭代空间表示: {(i, j) | 0 <= i < N, 0 <= j < M}
访问映射:    (i, j) -> (i + j, i - j)
依赖检查:    ∃(i,j),(i',j') : src(i,j) = dst(i',j') AND (i,j) < (i',j')
```

**Presburger 算术:**
- 一阶逻辑扩展了整数加法和比较
- 可判定性: 所有公式都有算法可以验证真值
- MLIR 使用 Presburger 库进行约束求解

---

## 3. 核心概念

### 3.1 仿射映射 (Affine Maps)

**定义:**
```
affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 * 2 + s1)>
```

**WHY 分析:**
1. **WHY 分离维度和符号?**
   - 维度 (d0, d1): 循环归纳变量，随迭代变化
   - 符号 (s0, s1): 编译时常量或循环不变量
   - 分离后可以更精确地分析迭代空间

2. **WHY 限制为仿射表达式?**
   - 只允许: +, -, *, 常数 (无乘法中的两个变量)
   - 保证可逆性: 可以从访问地址反推迭代点
   - 保证可分析性: 依赖分析可以在多项式时间内完成

3. **WHY 支持多元仿射映射?**
   - 可以同时计算多个索引
   - 例如: `(i, j) -> (i+j, i-j)` 用于转置访问

**代码实现:**
```cpp
// AffineMap 是不可变的，在 Context 中唯一
class AffineMap {
  unsigned numDims;      // 维度数量
  unsigned numSymbols;   // 符号数量
  ArrayRef<AffineExpr> results;  // 结果表达式列表
};
```

### 3.2 整数集合 (Integer Sets)

**定义:**
```
affine_set<(d0, d1)[s0] : (d0 >= 0, d0 < s0, d1 = d0 + 1)>
```

**WHY 分析:**
1. **WHY 需要整数集合?**
   - 表示条件分支的执行条件
   - 表示循环的有效迭代空间
   - 表示数据依赖的约束

2. **WHY 使用等式和不等式?**
   - 等式 (=): 精确约束，如 `j = i + 1`
   - 不等式 (>=, <): 范围约束，如 `i >= 0`
   - 组合可以表示任意凸多面体

**affine.if 操作:**
```mlir
affine.if #set(%i, %j)[%N] {
  // then 分支: 当约束满足时执行
} else {
  // else 分支
}
```

### 3.3 Affine.for 操作

**语法:**
```
affine.for %i = max affine_map to min affine_map step constant {
  // 循环体
}
```

**WHY 分析:**
1. **WHY step 必须是正整数常量?**
   - 保证每次迭代前进固定的量
   - 简化依赖分析（无需考虑动态步长）
   - 便于计算精确的迭代次数

2. **WHY 支持多结果下界/上界?**
   - `max(a, b, c)`: 取多个下界的最大值
   - `min(x, y, z)`: 取多个上界的最小值
   - 可以表示复杂的边界条件

3. **WHY 需要 iter_args?**
   - 支持循环携带的归约变量
   - 可以返回最终值
   - 便于表示累加、最大值等操作

**代码分析 (AffineOps.td 第 121-337 行):**
```llvm
def AffineForOp : Affine_Op<"for", [...]> {
  let arguments = (ins
    Variadic<Index>:$lowerBoundOperands,
    Variadic<Index>:$upperBoundOperands,
    Variadic<AnyType>:$inits,
    AffineMapAttr:$lowerBoundMap,
    AffineMapAttr:$upperBoundMap,
    IndexAttr:$step
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
}
```

**关键方法:**

```cpp
class AffineForOp {
  BlockArgument getInductionVar();           // 获取循环变量
  AffineBound getLowerBound();               // 获取下界信息
  AffineBound getUpperBound();               // 获取上界信息
  int64_t getStepAsInt();                    // 获取步长
  bool hasConstantBounds();                  // 检查边界是否为常数
};
```

### 3.4 Affine.if 操作

**WHY 分析:**
1. **WHY 使用整数集合而不是布尔表达式?**
   - 整数集合可以编码多维约束
   - 便于与循环的多面体表示统一
   - 可以进行更精确的分析

2. **WHY 支持返回值?**
   - 可以在条件分支中计算值
   - 支持条件初始化
   - 便于表示边缘填充等操作

**示例:**
```llvm
#interior = affine_set<(i, j) : (i >= 1, j >= 1, 10 - i >= 0, 10 - j >= 0)>

%val = affine.if #interior (%i, %j) {
  %v = affine.load %A[%i - 1, %j - 1]
  affine.yield %v
} else {
  %v = arith.constant 0.0 : f32
  affine.yield %v
}
```

### 3.5 内存操作

**Affine.load:**
```mlir
%val = affine.load %A[%i + 3, %j * 2 + 1] : memref<100x100xf32>
```

**Affine.store:**
```mlir
affine.store %val, %A[%i, %j] : memref<100x100xf32>
```

**WHY 分析:**
1. **WHY 索引必须是仿射表达式?**
   - 保证可以静态计算访问模式
   - 依赖分析需要精确的访问函数
   - 便于应用多面体变换

2. **WHY 与标准 load/store 分离?**
   - Affine 版本可以进行更激进的优化
   - 可以证明内存访问的安全性
   - 便于向量化、分块等变换

**内存访问表示 (MemRefAccess 结构):**
```cpp
struct MemRefAccess {
  Value memref;                      // 被访问的内存
  Operation *opInst;                 // load/store 操作
  SmallVector<Value, 4> indices;     // 索引值

  // 获取访问关系: 迭代空间 -> 数据空间
  LogicalResult getAccessRelation(IntegerRelation &rel);
};
```

### 3.6 Affine.parallel 操作

**WHY 分析:**

1. **WHY 需要单独的 parallel 操作?**
   - 明确表示并行循环
   - 支持归约操作
   - 可以直接生成并行代码

2. **WHY 支持多维并行?**
   - 表示循环并行
   - 便于 GPU 等并行硬件映射
   - 可以优化线程块大小

**示例:**
```mlir
affine.parallel (%i, %j) = (0, 0) to (N, M) step (32, 32) {
  // 并行执行
} reduce ("addf", "mulf") -> (f32, f32)
```

### 3.7 Polyhedral 模型

**核心概念:**

1. **迭代空间 (Iteration Space)**
   
   ```
   循环: for i = 0 to N { for j = 0 to M { ... } }
   空间: {(i, j) ∈ Z² | 0 ≤ i < N, 0 ≤ j < M}
   ```
   
2. **访问映射 (Access Map)**
   ```
   访问: A[i+1][j*2]
   映射: (i, j) -> (i+1, j*2)
   ```

3. **依赖关系 (Dependence)**
   ```
   RAW 依赖: ∃(i,j),(i',j'): src(i,j) = dst(i',j')
   顺序约束: (i,j) < (i',j') (字典序)
   ```

**依赖分析算法 (AffineAnalysis.cpp 第 611-695 行):**

```cpp
DependenceResult checkMemrefAccessDependence(
    const MemRefAccess &srcAccess,
    const MemRefAccess &dstAccess,
    unsigned loopDepth) {

  // 1. 构建访问关系
  IntegerRelation srcRel, dstRel;
  srcAccess.getAccessRelation(srcRel);
  dstAccess.getAccessRelation(dstRel);

  // 2. 组合源访问和目标访问的逆
  dstRel.inverse();
  dstRel.mergeAndCompose(srcRel);

  // 3. 添加顺序约束
  addOrderingConstraints(srcDomain, dstDomain, loopDepth);

  // 4. 检查是否为空
  if (dependenceDomain.isEmpty())
    return NoDependence;

  return HasDependence;
}
```

---

## 4. 关键代码深度解析

### 4.1 维度和符号验证 (isValidDim/isValidSymbol)

**WHY 分析 - 为什么要区分维度和符号?**

维度 (Dimension) 和符号 (Symbol) 是 Affine 方言的两大核心概念：
- **维度 (d0, d1, ...)**: 循环归纳变量，随迭代变化
- **符号 (s0, s1, ...)**: 编译时常量或循环不变量

**WHY 分离?**
1. 依赖分析需要区分迭代相关和无关的值
2. 符号可以在分析时当作常量处理
3. 简化迭代空间的数学表示

---

#### 4.1.1 isValidDim - 验证维度有效性

**源码位置:** `mlir/lib/Dialect/Affine/IR/AffineOps.cpp:291-344`

```cpp
// 步骤 1: 无 region 版本调用有 region 版本
bool mlir::affine::isValidDim(Value value) {
  // 场景 1: 类型必须是 index
  if (!value.getType().isIndex())
    return false;
    // WHY 非 index 类型不能作为仿射表达式的一部分
    // Affine 运算仅针对索引类型（表示数组索引、循环计数器等）

  // 场景 2: 值由操作定义，获取其 AffineScope 进行验证
  if (auto *defOp = value.getDefiningOp())
    return isValidDim(value, getAffineScope(defOp));
    // WHY 需要获取 AffineScope
    // 有效性是相对于特定区域的，不同区域规则不同

  // 场景 3: 值是块参数（如函数参数、循环归纳变量）
  // 步骤 2: 检查是否是仿射循环的归纳变量
  if (isAffineInductionVar(value))
    return true;
    // WHY 循环归纳变量是有效维度
    // 它们代表迭代空间中的坐标点

  // 步骤 3: 检查父操作是否具有 AffineScope 特征
  auto *parentOp = llvm::cast<BlockArgument>(value).getOwner()->getParentOp();
  return parentOp && parentOp->hasTrait<OpTrait::AffineScope>();
  // WHY AffineScope 内定义的值是符号
  // 这些值在该区域的所有仿射表达式中都是常量
}

// 步骤 4: 有 region 版本的完整验证逻辑
bool mlir::affine::isValidDim(Value value, Region *region) {
  // 场景 1: 类型检查
  if (!value.getType().isIndex())
    return false;
    // WHY 非索引类型直接拒绝
    // 保证类型安全的仿射表达式

  // 场景 2: 所有有效符号也是有效维度
  // WHY 符号是维度的超集
  if (isValidSymbol(value, region))
    return true;
    // 此时：value 是常量或循环不变量
    // 可以在仿射表达式中作为"参数"使用

  // 场景 3: 值由操作定义，递归检查
  auto *op = value.getDefiningOp();
  if (!op) {
    // 场景 3.1: 没有定义操作，必须是块参数
    // 步骤 5: 检查是否是仿射归纳变量
    return isAffineInductionVar(value);
    // WHY 块参数中没有定义操作时，只可能是归纳变量
    // 其他块参数（如 scf.for iter_args）不是有效维度
  }

  // 场景 4: 值由 affine.apply 操作定义
  if (auto applyOp = dyn_cast<AffineApplyOp>(op))
    return applyOp.isValidDim(region);
    // WHY 递归检查操作数
    // 如果所有输入都是有效维度，则输出也是
    // 例如: %idx = affine.apply (d0) -> (d0 * 2 + 1) (%i)
    //       若 %i 是有效维度，则 %idx 也是

  // 场景 5: 值由索引变换操作定义
  if (isa<AffineDelinearizeIndexOp, AffineLinearizeIndexOp>(op))
    return llvm::all_of(op->getOperands(),
                        [&](Value arg) { return ::isValidDim(arg, region); });
    // WHY 检查所有操作数
    // delinearize/linearize 是特殊的索引重排列操作

  // 场景 6: 值由 dim 操作定义（获取动态大小）
  if (auto dimOp = dyn_cast<ShapedDimOpInterface>(op))
    return isTopLevelValue(dimOp.getShapedValue());
    // WHY 顶层值是有效的
    // 动态大小如果在循环外定义，则是循环不变的符号

  // 场景 7: 不认识的操作，拒绝
  return false;
    // WHY 保守策略
    // 无法证明安全的情况下，认为无效
}
```

**执行流程示例 - 追踪具体值:**

```mlir
// 示例代码
func.func @example(%N: index, %M: index) {
  // 场景: %N, %M 是函数参数（块参数）
  // 验证 isValidDim(%N):
  //   步骤 1: %N 类型是 index ✓
  //   步骤 2: %N 没有定义操作（是块参数）
  //   步骤 3: isAffineInductionVar(%N) = false（不是归纳变量）
  //   步骤 4: parentOp 是 func.func，没有 AffineScope
  //   结果: isValidDim(%N) = false
  //   解释: 函数参数不是维度，但可能是符号

  affine.for %i = 0 to %N {
    // 场景: %i 是循环归纳变量
    // 验证 isValidDim(%i):
    //   步骤 1: %i 类型是 index ✓
    //   步骤 2: %i 是块参数（循环的归纳变量参数）
    //   步骤 3: isAffineInductionVar(%i) = true
    //   结果: isValidDim(%i) = true ✓

    %j = affine.apply affine_map<(d0) -> (d0 * 2)> (%i)
    // 场景: %j 由 affine.apply 定义
    // 验证 isValidDim(%j):
    //   步骤 1: %j 类型是 index ✓
    //   步骤 2: %j 有定义操作 (affine.apply)
    //   步骤 3: 进入场景 4，检查 applyOp.isValidDim(region)
    //         → 递归检查操作数 %i
    //         → isValidDim(%i) = true
    //   结果: isValidDim(%j) = true ✓

    %idx = affine.apply affine_map<(d0, s0) -> (d0 + s0)> (%i, %N)
    // 场景: %idx 使用了维度 %i 和符号 %N
    // 验证 isValidDim(%idx):
    //   步骤 1: %idx 类型是 index ✓
    //   步骤 2: %idx 有定义操作
    //   步骤 3: applyOp.isValidDim(region)
    //         → 检查操作数 %i（维度）→ true
    //         → 检查操作数 %N（符号）→ isValidSymbol(%N) = true
    //   结果: isValidDim(%idx) = true ✓
  }
}
```

**易错点标注:**
1. ⚠️ **函数参数不是维度**: `%N` 是符号而非维度
2. ⚠️ **递归深度限制**: 嵌套过深的 `affine.apply` 链可能导致性能问题
3. ⚠️ **跨 region 使用**: 在一个 region 有效的维度在另一个 region 可能无效

---

#### 4.1.2 isValidSymbol - 验证符号有效性

**源码位置:** `mlir/lib/Dialect/Affine/IR/AffineOps.cpp:405-429`

```cpp
bool mlir::affine::isValidSymbol(Value value) {
  // 步骤 1: 空值检查
  if (!value)
    return false;
    // WHY 防御性编程
    // 处理空指针情况

  // 步骤 2: 类型检查
  if (!value.getType().isIndex())
    return false;
    // WHY 符号也必须是 index 类型
    // 保证仿射表达式类型一致性

  // 步骤 3: 检查是否是顶层值
  if (isTopLevelValue(value))
    return true;
    // WHY 顶层值总是有效符号
    // 顶层值 = 在 AffineScope 区域顶层定义的值
    // 例如: 函数参数、常量

  // 步骤 4: 值由操作定义，获取 region 进行验证
  if (auto *defOp = value.getDefiningOp())
    return isValidSymbol(value, getAffineScope(defOp));
    // WHY 需要确定验证的上下文

  // 步骤 5: 值是块参数但没有定义操作
  return false;
    // WHY 无法验证
    // 例如: scf.for 的 iter_args 不是有效仿射符号
}
```

**执行流程示例:**

```mlir
func.func @symbol_example(%N: index, %M: index) -> index {
  // 场景: 验证函数参数
  // isValidSymbol(%N):
  //   步骤 1: value = %N, 非空 ✓
  //   步骤 2: 类型是 index ✓
  //   步骤 3: isTopLevelValue(%N)
  //         → %N 是函数的块参数
  //         → parentRegion = 函数体
  //         → %N 在该 region 定义
  //         → 返回 true
  //   结果: isValidSymbol(%N) = true ✓

  %c42 = arith.constant 42 : index
  // 场景: 常量总是有效符号
  // isValidSymbol(%c42):
  //   步骤 3: isTopLevelValue(%c42) = true
  //         → %c42 由 arith.constant 定义
  //         → 定义在函数体顶层
  //         → 返回 true
  //   结果: isValidSymbol(%c42) = true ✓

  affine.for %i = 0 to %N {
    // 场景: 归纳变量不是符号
    // isValidSymbol(%i):
    //   步骤 3: isTopLevelValue(%i) = false
    //         → %i 在循环体内，不在顶层
    //   步骤 4: %i 是块参数，没有定义操作
    //   步骤 5: 返回 false
    //   结果: isValidSymbol(%i) = false ✓
    //   解释: 归纳变量是维度，不是符号

    %dim = memref.dim %alloc, %i : memref<?xf32>
    // 场景: 循环依赖的 dim 操作
    // isValidSymbol(%dim):
    //   步骤 3: isTopLevelValue(%dim) = false
    //         → %dim 定义在循环内
    //   步骤 4: 有定义操作 (memref.dim)
    //         → 调用 isValidSymbol(%dim, region)
    //         → 检查 dim 操作的特殊规则
    //         → %i 不是顶层值
    //   结果: isValidSymbol(%dim) = false
    //   解释: 依赖循环变量的值不是符号
  }

  return %c42
}
```

**WHY 区分维度和符号的关键点:**

| 值类型 | 是否有效维度 | 是否有效符号 | WHY |
|--------|-------------|-------------|-----|
| 循环归纳变量 `%i` | ✅ 是 | ❌ 否 | 随迭代变化，不是常量 |
| 函数参数 `%N` | ❌ 否 | ✅ 是 | 在整个函数内不变 |
| 常量 `%c42` | ❌ 否 | ✅ 是 | 编译时常量 |
| `affine.apply` 结果（输入是维度） | ✅ 是 | ❌ 否 | 继承输入的属性 |
| `affine.apply` 结果（输入是符号） | ❌ 否 | ✅ 是 | 继承输入的属性 |
| `memref.dim`（顶层） | ❌ 否 | ✅ 是 | 循环不变 |
| `memref.dim`（循环依赖） | ❌ 否 | ❌ 否 | 循环变化 |

---

### 4.2 依赖分析 - checkMemrefAccessDependence

**源码位置:** `mlir/lib/Dialect/Affine/Analysis/AffineAnalysis.cpp:611-695`

**WHY 依赖分析是 Affine 方言的核心优势?**

依赖分析回答：两个内存访问是否可能访问相同位置，以及访问的顺序关系。这是循环变换（重排、融合、并行化）的基础。

```cpp
DependenceResult mlir::affine::checkMemrefAccessDependence(
    const MemRefAccess &srcAccess,    // 源访问（先执行）
    const MemRefAccess &dstAccess,    // 目标访问（后执行）
    unsigned loopDepth,               // 检查深度
    FlatAffineValueConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents,
    bool allowRAR) {

  // ========== 阶段 1: 前置检查 ==========

  // 步骤 1: 检查是否访问相同 memref
  // 场景 1: 不同的 memref
  if (srcAccess.memref != dstAccess.memref)
    return DependenceResult::NoDependence;
    // WHY 不同 memref 不可能有依赖
    // 每个独立的 memref 有独立的地址空间
    // 例如: A[i] 和 B[j] 不会有冲突

  // 步骤 2: 检查是否有写操作
  // 场景 2: 两个都是读操作（RAR）
  if (!allowRAR && !isa<AffineWriteOpInterface>(srcAccess.opInst) &&
      !isa<AffineWriteOpInterface>(dstAccess.opInst))
    return DependenceResult::NoDependence;
    // WHY 读-读依赖不影响变换
    // 只需要关心 RAW, WAR, WAW

  // 步骤 3: 检查分析范围
  // 场景 3: 不同的 affine scope
  if (getAffineAnalysisScope(srcAccess.opInst) !=
      getAffineAnalysisScope(dstAccess.opInst))
    return DependenceResult::Failure;
    // WHY 无法跨 scope 分析
    // 不同 scope 的循环结构没有已知的关联

  // 步骤 4: 检查公共块
  if (!getCommonBlockInAffineScope(srcAccess.opInst, dstAccess.opInst))
    return DependenceResult::Failure;
    // WHY 需要在同一控制流
    // 否则无法确定执行顺序

  // ========== 阶段 2: 构建访问关系 ==========

  // 步骤 5: 创建访问关系
  PresburgerSpace space = PresburgerSpace::getRelationSpace();
  IntegerRelation srcRel(space), dstRel(space);
  // WHY 使用 Presburger 算术
  // 可以精确表示仿射映射和迭代空间

  // 场景 4: 构建 srcAccess 的访问关系
  // 例如: affine.load %A[i*2 + j, i - j]
  //       srcRel: (i, j) -> (i*2 + j, i - j)
  if (failed(srcAccess.getAccessRelation(srcRel)))
    return DependenceResult::Failure;
    // WHY 可能失败
    // 访问映射太复杂（非仿射）无法表示

  // 场景 5: 构建 dstAccess 的访问关系
  // 例如: affine.store %val, %A[i + 1, j*2]
  //       dstRel: (i', j') -> (i' + 1, j'*2)
  if (failed(dstAccess.getAccessRelation(dstRel)))
    return DependenceResult::Failure;

  // 步骤 6: 提取迭代空间约束
  FlatAffineValueConstraints srcDomain(srcRel.getDomainSet());
  FlatAffineValueConstraints dstDomain(dstRel.getDomainSet());
  // WHY 分离域和值
  // 域 = 迭代空间（循环边界）
  // 值 = 访问的内存位置

  // ========== 阶段 3: 顺序约束检查 ==========

  // 步骤 7: 检查字典序顺序
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  // 场景 6: loopDepth > numCommonLoops
  if (!allowRAR && loopDepth > numCommonLoops &&
      !srcAppearsBeforeDstInAncestralBlock(srcAccess, dstAccess)) {
    return DependenceResult::NoDependence;
    // WHY 检查源是否在目标之前
    // 如果源在目标之后，不可能有 src->dst 的依赖
  }

  // ========== 阶段 4: 构建依赖多面体 ==========

  // 步骤 8: 组合访问关系
  // 目标: 找到 (i,j,i',j') 使得 src(i,j) = dst(i',j')
  dstRel.inverse();
  // WHY 反转 dstRel
  // dstRel: (i',j') -> (x',y')
  // inverse: (x',y') -> (i',j')
  // 这样可以组合 srcRel 和 inverse(dstRel)

  // 场景 7: 组合关系
  // srcRel: (i,j) -> (i*2+j, i-j)
  // inverse(dstRel): (x,y) -> (i'-1, j'/2)
  // 组合后: (i,j) -> (i*2+j, i-j) -> (i', j')
  //         使得 i*2+j = i'+1 且 i-j = j'/2
  dstRel.mergeAndCompose(srcRel);

  // 步骤 9: 转换变量种类
  dstRel.convertVarKind(VarKind::Domain, 0, dstRel.getNumDomainVars(),
                        VarKind::Range, 0);
  IntegerPolyhedron dependenceDomain(dstRel);
  // WHY 域变为值
  // 组合后我们关心的是迭代对之间的约束

  // ========== 阶段 5: 添加顺序约束 ==========

  // 步骤 10: 添加 src < dst 的约束
  addOrderingConstraints(srcDomain, dstDomain, loopDepth, &dependenceDomain);
  // WHY 需要顺序约束
  // 即使访问相同位置，如果 src 在 dst 之后，也不是依赖
  // 字典序: (i,j) < (i',j') 当且仅当 i < i' 或 (i = i' 且 j < j')

  // ========== 阶段 6: 检查依赖是否存在 ==========

  // 步骤 11: 检查解空间是否为空
  if (dependenceDomain.isEmpty())
    return DependenceResult::NoDependence;
    // WHY 空集表示无依赖
    // 没有满足所有约束的 (i,j,i',j') 元组

  // 步骤 12: 计算方向向量
  if (dependenceComponents != nullptr)
    computeDirectionVector(srcDomain, dstDomain, loopDepth, &dependenceDomain,
                           dependenceComponents);
    // WHY 方向向量描述依赖类型
    // [0] = 同一次迭代
    // [1] = 相邻迭代
    // [>0] = 长距离依赖

  return DependenceResult::HasDependence;
}
```

**完整执行流程示例 - 具体数据追踪:**

```mlir
// 示例: 检查以下代码的依赖
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %v1 = affine.load %A[%i, %j]          // srcAccess: S1
    affine.store %v1, %A[%i + 1, %j]      // dstAccess: S2
  }
}
```

**分析步骤追踪:**

```
========== 输入 ==========
srcAccess: S1 = load %A[i, j]
dstAccess: S2 = store %A[i+1, j]
loopDepth = 2

问题: S1 读取的数据被 S2 写入覆盖了吗？(WAR 依赖)

========== 阶段 1: 前置检查 ==========
步骤 1: srcAccess.memref (%A) == dstAccess.memref (%A) ✓
步骤 2: S1 是读, S2 是写 → 检查 WAR 依赖 ✓
步骤 3: scope 相同 ✓
步骤 4: 在同一块 ✓

========== 阶段 2: 构建访问关系 ==========
步骤 5: 构建访问映射
  srcRel: (i, j) -> (i, j)      [S1 在迭代 (i,j) 读取 A[i,j]]
  dstRel: (i', j') -> (i'+1, j')  [S2 在迭代 (i',j') 写入 A[i'+1,j']]

========== 阶段 3: 组合关系 ==========
步骤 8: 反转并组合，找到访问相同位置的迭代对
  目标: 找到 (i,j,i',j') 使得 A[i,j] = A[i'+1,j']

  约束推导:
    i = i' + 1  AND  j = j'
    → i' = i - 1  AND  j' = j

  解释: S1 在 (i,j) 读取的位置，等于 S2 在 (i-1,j) 写入的位置

========== 阶段 4: 添加顺序约束 ==========
步骤 10: 检查是否存在 (i,j) < (i',j') 的解

  字典序定义: (i,j) < (i',j') 当且仅当
    i < i'  OR  (i = i' AND j < j')

  从访问关系: i' = i - 1

  检查顺序约束:
    (i, j) < (i-1, j)  成立吗?
    → i < i-1?  NO
    → 结论: 不成立!

  代入具体数值验证:
    当 (i=5, j=10) 时:
      S1: load %A[5, 10]    ← 读取 A[5][10]
      S2: store %A[6, 10]   ← 写入 A[6][10]

    当 (i=4, j=10) 时:
      S1: load %A[4, 10]    ← 读取 A[4][10]
      S2: store %A[5, 10]   ← 写入 A[5][10]

    检查 A[5][10] 的依赖:
      S2 在 (4, 10) 写入 A[5][10]
      S1 在 (5, 10) 读取 A[5][10]

      顺序: (4, 10) < (5, 10)?  YES (4 < 5)

  结论: S1→S2 方向 NoDependence
       但存在 S2→S1 的反向依赖! (WAR)

  解释: S2 → S1 表示 "S1 依赖于 S2"
        = S2 必须在 S1 之前执行
        = S1 读取的值来自 S2 的写入

  具体例子:
    S2 在 (4, 10) 写入 A[5][10]
    S1 在 (5, 10) 读取 A[5][10]
    → S1 需要等待 S2 完成

========== 更清晰的依赖分析示例 ==========

示例 1: 明显的 RAW 依赖

affine.for %i = 0 to 99 {
  %v1 = affine.load %A[%i]           // S1: 读取 A[i]
  affine.store %v1, %A[%i + 1]       // S2: 写入 A[i+1]
}

执行追踪:
  i=0: S1 读 A[0], S2 写 A[1]
  i=1: S1 读 A[1], S2 写 A[2]
  ...

依赖: S2 在 i=0 写入 A[1]，S1 在 i=1 读取 A[1]
检查: (0) < (1)? YES
方向向量: [1]


示例 2: 对角线依赖

affine.for %i = 0 to 99 {
  affine.for %j = 0 to 99 {
    %v1 = affine.load %A[%i, %j]          // S1
    affine.store %v1, %A[%i + 1, %j + 1]  // S2
  }
}

访问关系:
  S1: (i, j) -> (i, j)
  S2: (i', j') -> (i'+1, j'+1)

相等约束: i = i'+1, j = j'+1
顺序检查: (i,j) < (i-1, j-1)?  NO (i > i-1)

结论: S1→S2 无依赖，S2→S1 有依赖


示例 3: 真正的 S1→S2 依赖

affine.for %i = 0 to 99 {
  %v1 = affine.load %A[%i + 1]      // S1: 读取 A[i+1]
  affine.store %v1, %A[%i]          // S2: 写入 A[i]
}

访问关系:
  S1: (i) -> (i+1)
  S2: (i') -> (i')

相等约束: i+1 = i'  →  i' = i+1
顺序检查: (i) < (i+1)?  YES (i < i+1)

结论: S1→S2 有依赖，方向向量 [1]

========== 核心理解 ==========

依赖分析检查的是:
1. 相同位置: src 访问的位置 = dst 访问的位置
2. 执行顺序: src 的迭代点 < dst 的迭代点（字典序）

对于原始代码:
  S1 读取 A[i][j], S2 写入 A[i+1][j]

  S1(i,j) 和 S2(i-1,j) 访问相同位置
  但 (i,j) > (i-1,j)，所以是 S2→S1 依赖，不是 S1→S2
```

**易错点标注:**
1. ⚠️ **方向向量符号**: 正数表示正向依赖，负数表示反向
2. ⚠️ **边界条件**: 循环边界可能切断依赖
3. ⚠️ **复杂映射**: 非仿射访问（如 `A[i*j]`）会失败

---

### 4.3 循环分块 - LoopTiling

**源码位置:** `mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp:99-150`

**WHY 分块提高性能?**

```
原始循环（不命中缓存）:
  for i = 0 to 1024:
    for j = 0 to 1024:
      使用 A[i][j]  // 每次访问新缓存行

分块后（命中缓存）:
  for ii = 0 to 1024 step 32:
    for jj = 0 to 1024 step 32:
      for i = ii to ii+32:     // 32x32 块适合缓存
        for j = jj to jj+32:
          使用 A[i][j]         // 重用已加载的数据
```

```cpp
void LoopTiling::getTileSizes(ArrayRef<AffineForOp> band,
                              SmallVectorImpl<unsigned> *tileSizes) {
  // ========== 步骤 1: 检查命令行选项 ==========
  if (tileSize) {
    // 场景 1: 用户指定了固定 tile size
    tileSizes->assign(band.size(), tileSize);
    return;
    // WHY 使用用户指定的值
    // 专家可能比自动算法更了解硬件特性
  }

  // ========== 步骤 2: 检查预配置的 tile sizes ==========
  if (!this->tileSizes.empty()) {
    tileSizes->assign(this->tileSizes.begin(), this->tileSizes.end());
    tileSizes->resize(band.size(), kDefaultTileSize);
    return;
    // WHY 填充默认值
    // 如果配置少于循环数，剩余使用默认值
  }

  // ========== 步骤 3: 初始化结果向量 ==========
  tileSizes->resize(band.size());
  // 此时：tileSizes = [0, 0, ...]（band.size() 个零）

  // ========== 步骤 4: 处理零缓存大小 ==========
  if (cacheSizeInKiB == 0) {
    llvm::fill(*tileSizes, 1);
    return;
    // WHY 使用最小有效值
    // 没有缓存信息时，保守地不进行分块
  }

  // ========== 步骤 5: 计算内存足迹 ==========
  AffineForOp rootForOp = band[0];
  std::optional<int64_t> fp = getMemoryFootprintBytes(rootForOp, 0);

  // 场景 2: 无法计算内存足迹
  if (!fp) {
    llvm::fill(*tileSizes, LoopTiling::kDefaultTileSize);
    if (avoidMaxMinBounds)
      adjustToDivisorsOfTripCounts(band, tileSizes);
    // WHY 调整为行程数的约数
    // 避免 min/max 边界检查，提高效率
    return;
  }
  // 此时：fp = 循环体访问的总字节数

  // ========== 步骤 6: 计算过载因子 ==========
  uint64_t cacheSizeBytes = cacheSizeInKiB * 1024;
  uint64_t excessFactor = llvm::divideCeil(*fp, cacheSizeBytes);
  // WHY 向上取整
  // 确保分块后数据适合缓存

  // 示例计算:
  //   fp = 4,194,304 字节 (4MB)
  //   cache = 262,144 字节 (256KB L2 缓存)
  //   excessFactor = ceil(4194304 / 262144) = 16

  // 场景 3: 数据已经适合缓存
  if (excessFactor <= 1) {
    llvm::fill(*tileSizes, 1);
    return;
    // WHY 不需要分块
    // 数据已经在缓存容量内
  }

  // ========== 步骤 7: 计算 n 维分块大小 ==========
  // WHY 使用 n 次方根
  // 将过载因子均匀分配到各维度
  unsigned tSize = floorl(pow(excessFactor, 1.0 / band.size()));
  // 示例:
  //   band.size() = 2 (二维循环)
  //   excessFactor = 16
  //   tSize = floor(16^(1/2)) = floor(4) = 4

  // ========== 步骤 8: 填充分块大小 ==========
  llvm::fill(*tileSizes, tSize);

  // 此时：tileSizes = [4, 4]（对于二维）
  // 解释: 每个维度缩小 4 倍，总共缩小 16 倍

  // ========== 步骤 9: 调整为约数（可选）==========
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
    // WHY 避免边界检查
    // 如果 tile size 是 trip count 的约数，所有块大小相等
}
```

**adjustToDivisorsOfTripCounts 详解:**

```cpp
static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  // 步骤 1: 遍历每个循环
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];

    // 步骤 2: 尝试获取常量行程数
    std::optional<uint64_t> mayConst = getConstantTripCount(band[i]);
    if (!mayConst)
      continue;
      // WHY 跳过动态行程数
      // 无法调整未知大小

    // 步骤 3: 获取行程数
    uint64_t constTripCount = *mayConst;
    // 示例: constTripCount = 100

    // 步骤 4: 检查是否过大
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;
      // WHY 限制最大值
      // tile size 不应超过行程数的一半
      // tSizeAdjusted = min(50, 4) = 4

    // 步骤 5: 向下调整为约数
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
      // WHY 向下调整
      // 找到最大的约数（小于等于当前值）

    // 示例执行:
    //   constTripCount = 100, tSizeAdjusted = 4
    //   100 % 4 = 0 ✓ → 停止，使用 4
    //
    // 示例执行 2:
    //   constTripCount = 100, tSizeAdjusted = 7
    //   100 % 7 = 2 → 调整为 6
    //   100 % 6 = 4 → 调整为 5
    //   100 % 5 = 0 ✓ → 停止，使用 5
  }
}
```

**完整执行流程示例:**

```mlir
// 输入: 三重嵌套循环
affine.for %i = 0 to 128 {
  affine.for %j = 0 to 256 {
    affine.for %k = 0 to 512 {
      %a = affine.load %A[%i, %j, %k]
      %b = affine.load %B[%i, %k]
      %c = arith.mulf %a, %b
      affine.store %c, %C[%i, %j]
    }
  }
}
```

**分块算法执行追踪:**

```
========== 输入 ==========
band = [for_i, for_j, for_k]
cacheSizeInKiB = 32 (32KB L1 缓存)

========== 步骤 1: 计算内存足迹 ==========
假设:
  A: 128 x 256 x 512 x 4 字节 = 67,108,864 字节
  B: 128 x 512 x 4 字节 = 262,144 字节
  C: 128 x 256 x 4 字节 = 131,072 字节

getMemoryFootprintBytes 返回: ~67 MB

========== 步骤 2: 计算过载因子 ==========
cacheSizeBytes = 32 * 1024 = 32,768
excessFactor = ceil(67,108,864 / 32,768) = ceil(2048) = 2048

========== 步骤 3: 计算 tile size ==========
band.size() = 3 (三维)
tSize = floor(2048^(1/3)) = floor(12.7) = 12

tileSizes = [12, 12, 12]

========== 步骤 4: 调整为约数 ==========
for_i: tripCount = 128
  128 % 12 = 8 → 调整
  128 % 11 = 7 → 调整
  ...
  128 % 8 = 0 ✓ → 使用 8

for_j: tripCount = 256
  256 % 12 = 4 → 调整
  256 % 11 = 3 → 调整
  ...
  256 % 8 = 0 ✓ → 使用 8

for_k: tripCount = 512
  512 % 12 = 8 → 调整
  ...
  512 % 8 = 0 ✓ → 使用 8

最终 tileSizes = [8, 8, 8]

========== 步骤 5: 生成分块代码 ==========
原始:
  for i in [0, 128):
    for j in [0, 256):
      for k in [0, 512):
        body(i, j, k)

分块后:
  for ii in [0, 128) step 8:      // 块外循环
    for jj in [0, 256) step 8:
      for kk in [0, 512) step 8:
        for i in [ii, min(ii+8, 128)):   // 块内循环
          for j in [jj, min(jj+8, 256)):
            for k in [kk, min(kk+8, 512)):
              body(i, j, k)

块大小: 8 * 8 * 8 = 512 次迭代
内存: 512 * (A + B + C) ≈ 可以放入 L1 缓存
```

**易错点标注:**
1. ⚠️ **边界块**: `min(ii+8, ub)` 处理不均匀边界
2. ⚠️ **零行程数**: 空循环应跳过分块
3. ⚠️ **嵌套过深**: 超过 4-5 维的分块收益递减

---

## 5. 变换操作详解

### 5.1 循环分块 (Loop Tiling)

**WHY 分析 - 为什么分块能提高性能?**

1. **局部性原理**: 数据在缓存中被重用
2. **减少延迟**: 小块数据适合缓存
3. **并行化**: 每个块可以独立处理

**算法 (LoopTiling.cpp):**

```cpp
void LoopTiling::runOnOperation() {
  // WHY 1: 寻找可分块的循环带
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(getOperation(), &bands);

  // WHY 2: 确定分块大小
  SmallVector<unsigned, 6> tileSizes;
  getTileSizes(band, &tileSizes);

  // WHY 3: 执行分块
  // 原: for i { for j { for k { ... } } }
  // 分: for ii { for jj { for kk {
  //        for i=ii to min(ii+tile, ub) {
  //          for j=jj to min(jj+tile, ub) {
  //            for k=kk to min(kk+tile, ub) { ... }
  //          }
  //        }
  //      } } }
  tilePerfectlyNested(band, tileSizes, &tiledNest);

  // WHY 4: 分离完整和部分块
  // 完整块: 大小 = tile_size
  // 部分块: 处理边界情况
  if (separate)
    separateFullTiles(intraTileLoops);
}
```

**分块大小选择:**

```cpp
// LoopTiling.cpp 第 99-176 行
void LoopTiling::getTileSizes(ArrayRef<AffineForOp> band,
                              SmallVectorImpl<unsigned> *tileSizes) {
  // WHY 1: 使用内存足迹指导分块
  std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);

  // WHY 2: 计算过载因子
  uint64_t excessFactor = llvm::divideCeil(*fp, cacheSizeBytes);

  // WHY 3: 使用 n 次方根分配维度
  // 对于 n 维循环，每个维度缩小 n√excessFactor 倍
  unsigned tSize = floorl(pow(excessFactor, 1.0 / band.size()));

  // WHY 4: 调整为行程数的约数
  // 避免 min/max 边界
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}
```

### 5.2 循环融合 (Loop Fusion)

**WHY 分析 - 融合的好处和挑战?**

**好处:**
1. 减少内存访问
2. 提高缓存利用率
3. 降低延迟

**挑战:**
1. 需要保持依赖正确
2. 可能增加寄存器压力
3. 需要确定融合深度

**融合算法 (LoopFusion.cpp):**

```cpp
void LoopFusion::runOnOperation() {
  // WHY 1: 构建依赖图
  // 节点 = 循环/操作
  // 边 = 生产者-消费者关系
  MemRefDependenceGraph mdg;
  mdg.build(getOperation());

  // WHY 2: 寻找融合候选
  // 遍历所有消费者，寻找其生产者
  for (unsigned dstId : loopsToFuse) {
    SmallVector<unsigned> srcIdCandidates;
    getProducerCandidates(dstId, mdg, srcIdCandidates);

    // WHY 3: 对每个候选评估融合
    for (unsigned srcId : srcIdCandidates) {
      // 计算融合切片
      ComputationSliceState slice;
      if (failed(computeSliceUnion(srcId, dstId, mdg, slice)))
        continue;

      // WHY 4: 评估盈利性
      // 计算额外计算的比例
      double extraCompute = getAdditionalComputeFraction(...);
      if (extraCompute > threshold)
        continue;

      // WHY 5: 执行融合
      fuseLoops(srcId, dstId, slice, mdg);
    }
  }
}
```

**融合模式:**

1. **相邻融合**: 直接合并两个循环
   ```
   for i { A[i] = ... }  for i { ... = B[i] }
   ↓
   for i { A[i] = ...; ... = B[i] }
   ```

2. **切片融合**: 插入部分生产者
   ```
   for i { for j { A[i,j] = ... } }  for i { ... = A[i,i+1] }
   ↓
   for i {
     for j { A[i,j] = ... }
     ... = A[i,i+1]  // 只使用需要的切片
   }
   ```

### 5.3 循环展开 (Loop Unroll)

**WHY 分析 - 为什么要展开循环?**

1. **减少分支开销**: 减少循环控制指令
2. **指令级并行**: 暴露更多的独立操作
3. **寄存器重用**: 值保持在寄存器中

**展开策略:**

```cpp
// LoopUnroll.cpp
LogicalResult mlir::affine::unrollLoop(AffineForOp forOp,
                                        uint64_t unrollFactor) {
  // WHY 1: 检查展开有效性
  // - 行程数必须能被因子整除（对于完全展开）
  // - 不能有循环携带依赖（对于并行循环）

  // WHY 2: 生成展开的迭代
  // 原: for i = 0 to N { body(i) }
  // 展开: body(0); body(1); body(2); body(3); ...
  //       for i = 4 to N step 4 { body(i); body(i+1); body(i+2); body(i+3); }

  // WHY 3: 处理边界情况
  // 如果 N 不能被因子整除，需要剩余迭代
}
```

### 5.4 循环不变代码提升

**WHY 分析 - 为什么要提升不变代码?**

如果计算不依赖循环变量，应该在循环外计算一次。

```cpp
// AffineLoopInvariantCodeMotion.cpp
LogicalResult mlir::affine::hoistLoopInvariantCode(AffineForOp forOp) {
  // WHY 1: 分析操作的操作数
  for (Operation &op : forOp.getBody()->withoutTerminator()) {
    // WHY 2: 检查是否所有操作数都是循环不变的
    bool isInvariant = llvm::all_of(op.getOperands(), [&](Value operand) {
      return isDefinedOutsideOfLoop(operand, forOp);
    });

    // WHY 3: 检查是否有副作用
    // 只有无副作用的操作才能提升
    if (isInvariant && isMemoryEffectFree(&op)) {
      // WHY 4: 移动到循环前
      op.moveBefore(forOp);
    }
  }
}
```

### 5.5 数据拷贝生成

**WHY 分析 - 为什么要显式管理数据拷贝?**

1. **软件管理缓存**: 为快速内存空间预取数据
2. **DMA 优化**: 使用异步数据传输
3. **局部性**: 在计算前准备数据

**算法:**

```cpp
// AffineDataCopyGeneration.cpp
LogicalResult mlir::affine::affineDataCopyGenerate(...) {
  // WHY 1: 分析内存访问
  // 识别热点数据

  // WHY 2: 分配临时缓冲区
  // 在快速内存空间中创建副本

  // WHY 3: 插入 DMA 操作
  // affine.dma_start: 开始异步传输
  // affine.dma_wait: 等待传输完成

  // WHY 4: 更新访问
  // 将原始访问替换为缓冲区访问

  // WHY 5: 流水线传输
  // 重叠计算和数据传输
  if (dmaLLVM) {
    pipelineDataTransfer(forOp, copyNests);
  }
}
```

---

## 6. 测试用例分析

### 6.1 基本操作测试 (ops.mlir)

**测试维度和符号验证:**

```mlir
// test3: 显式使用 symbol 关键字
func.func @test3(%arg0 : index, %arg1 : index) {
  %0 = memref.alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // WHY: symbol() 明确标记符号操作数
      %1 = affine.load %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)]
    }
  }
}
```

**测试嵌套仿射表达式:**

```mlir
// test4: 复杂的仿射表达式
func.func @test4(%arg0 : index, %arg1 : index) {
  %0 = memref.alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // WHY: 支持嵌套表达式和整数除法
      %1 = affine.load %0[(%i0 + symbol(%arg0)) floordiv 3 + 11,
                          (%i1 + symbol(%arg1)) mod 4 + 7]
    }
  }
}
```

### 6.2 循环分块测试 (loop-tiling.mlir)

**基本分块:**

```mlir
// CHECK-LABEL: @loop_tiling()
func.func @loop_tiling() {
  // 原始: 三重循环
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k)
      }
    }
  }
}

// 分块后:
// affine.for %ii = 0 to 256 step 32 {    // 外块循环
//   affine.for %jj = 0 to 512 step 32 {
//     affine.for %kk = 0 to 1024 step 32 {
//       affine.for %i = %ii to %ii + 32 {   // 内块循环
//         affine.for %j = %jj to %jj + 32 {
//           affine.for %k = %kk to %kk + 32 {
//             "test.foo"(%i, %j, %k)
//           }
//         }
//       }
//     }
//   }
// }
```

**边界处理:**

```mlir
// WHY: 部分块处理边界
func.func @tile_using_symbolic_loop_upper_bounds(%arg0: memref<?x?xf32>, ...) {
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  affine.for %i0 = 0 to %0 {    // 动态边界
    affine.for %i1 = 0 to %0 {
      ...
    }
  }
}

// 分块后需要 min 表达式:
// affine.for %ii = 0 to %0 step 32 {
//   affine.for %i = %ii to min(%ii + 32, %0) {  // 处理边界
//     ...
//   }
// }
```

### 6.3 循环融合测试 (loop-fusion.mlir)

**生产者-消费者融合:**

```mlir
// 原始程序: 两个独立的循环
func.func @producer_consumer(%A: memref<100xf32>, %B: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    affine.store %val, %A[%i]    // 生产者
  }
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i]      // 消费者
    affine.store %v, %B[%i]
  }
}

// 融合后:
// affine.for %i = 0 to 100 {
//   affine.store %val, %A[%i]    // 生产
//   %v = affine.load %A[%i]      // 立即消费
//   affine.store %v, %B[%i]
// }
```

**切片融合:**

```mlir
// 只使用部分生产者输出
func.func @slice_fusion() {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      affine.store ..., %A[%i, %j]    // 生产 100x100
    }
  }
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i, %i+1]      // 只使用对角线
    affine.store %v, %B[%i]
  }
}

// 融合后: 只计算需要的部分
// affine.for %i = 0 to 100 {
//   affine.for %j = %i to %i+1 {      // 只计算对角线
//     affine.store ..., %A[%i, %j]
//   }
//   %v = affine.load %A[%i, %i+1]
//   affine.store %v, %B[%i]
// }
```

---

## 7. 执行流程示例

### 7.1 矩阵乘法的完整优化流程

**原始代码:**

```mlir
// C = A × B
// A: 128 × 64, B: 64 × 96, C: 128 × 96
func.func @matmul(%A: memref<128x64xf32>,
                  %B: memref<64x96xf32>,
                  %C: memref<128x96xf32>) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 96 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<128x64xf32>
        %b = affine.load %B[%k, %j] : memref<64x96xf32>
        %c = affine.load %C[%i, %j] : memref<128x96xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<128x96xf32>
      }
    }
  }
  return
}
```

**维度说明:**
- A[128, 64]: 128 行，64 列
- B[64, 96]: 64 行，96 列
- C[128, 96]: 128 行，96 列
- 矩阵乘法: $C[i,j] = \sum_{k=0}^{63} A[i,k] × B[k,j]$

**步骤 1: 循环分块**

```
输入: 三个嵌套循环 (128, 96, 64)
分块大小: (32, 32, 16)
输出:
  - 外块循环: (0..128 step 32, 0..96 step 32, 0..64 step 16)
  - 内块循环: (ii..ii+32, jj..jj+32, kk..kk+16)
```

```mlir
affine.for %ii = 0 to 128 step 32 {
  affine.for %jj = 0 to 96 step 32 {
    affine.for %kk = 0 to 64 step 16 {
      affine.for %i = %ii to min(%ii + 32, 128) {
        affine.for %j = %jj to min(%jj + 32, 96) {
          affine.for %k = %kk to min(%kk + 16, 64) {
            // 原始计算
          }
        }
      }
    }
  }
}
```

**WHY 分块大小不同:**
- i, j 维度较大 (128, 96)，分块 32
- k 维度较小 (64)，分块 16
- 块大小: 32 × 32 × 16 = 16,384 次迭代

**步骤 2: 寄存器分块**

```
目的: 在寄存器中累加，减少存储次数
方法: 引入临时变量，内层 k 循环后存储

伪代码:
  for ii, jj, kk:  // 外块循环
    for i in block_ii:
      for j in block_jj:
        acc = 0  // 寄存器中的累加器
        for k in block_kk:  // 内层 k 循环
          acc += A[i,k] * B[k,j]
        C[i,j] = acc  // 只在 k 循环结束后存储一次
```

**步骤 3: 循环交换**

```
原顺序: i -> j -> k
  - A[i,k]: 按行访问（i固定，k变化）→ 连续 ✓
  - B[k,j]: 按列访问（k,j都变化）→ 跳跃 ✗
  - C[i,j]: 按行访问（i固定，j变化）→ 连续 ✓

交换后: i -> k -> j
  - A[i,k]: 按行访问（i固定，k变化）→ 连续 ✓
  - B[k,j]: 按行访问（k固定，j变化）→ 连续 ✓
  - C[i,j]: 跳跃访问（k,j都变化）→ 可能跳跃

目的: 提高 B（右矩阵）的局部性
```

**详细分析:**

原始顺序 i→j→k:
```
for i:
  for j:
    for k:
      A[i,k]  // i固定，k递增 → 连续访问行i
      B[k,j]  // k递增，j固定 → 跳跃访问不同行的第j列
              // 例如: B[0,j], B[1,j], B[2,j], ... B[127,j]
```

交换后 i→k→j:
```
for i:
  for k:
    for j:
      A[i,k]  // i固定，k递增 → 连续访问行i
      B[k,j]  // k固定，j递增 → 连续访问行k
              // 例如: B[k,0], B[k,1], B[k,2], ... B[k,127]
```

**WHY 交换提高性能:**
- B 的访问从按列变为按行，与行主序存储匹配
- 每次内层 j 循环，B 的一整行被连续访问
- 缓存行被充分利用

**步骤 4: 向量化**

```
交换后内层是 j 循环，检查其向量化可行性:

对于固定的 i, k:
  A[i,k]: 常量（可广播到向量）
  B[k,j]: j = 0,1,2,...,95 → 连续内存访问 ✓
  C[i,j]: j = 0,1,2,...,95 → 连续内存访问 ✓

向量化策略（新维度 128×64×96）:
  affine.for %i = 0 to 128 {
    affine.for %k = 0 to 64 {
      affine.for %j = 0 to 96 step 4 {      // 向量宽度 = 4
        %a = affine.load %A[%i, %k]         // A[i,k]: 标量广播
        %b_vec = vector.load %B[%k, %j]     // B[k,j]: 连续向量加载
        %c_vec = vector.load %C[%i, %j]     // C[i,j]: 连续向量加载
        %a_vec = vector.broadcast %a        // 广播 %a 到 4 个元素
        %p_vec = vector.fma %a_vec, %b_vec, %c_vec  // %p = %a×%b + %c（向量化）
        vector.store %p_vec, %C[%i, %j]
      }
    }
  }
}

// 内存访问分析（k=32 时）:
//   B[32, 0:3], B[32, 4:7], B[32, 8:11], ... B[32, 92:95] → 连续！
//   C[64, 0:3], C[64, 4:7], C[64, 8:11], ... C[64, 92:95] → 连续！

// WHY 可以向量化
1. B[k, j] 在 j 维度连续 → 可以向量加载
2. C[i, j] 在 j 维度连续 → 可以向量存储
3. A[i, k] 在 j 循环中不变 → 可以广播
4. j 循环各次迭代无依赖 → 可以并行执行

// vector.store 工作原理:
%p_vec = vector<4xf32>  // 包含 4 个元素
vector.store %p_vec, %C[%i, %j] 的含义: 将 %p_vec 的 4 个元素连续存储到从 C[%i, %j] 开始的 4 个位置

示例（i=64, j=0）:
  vector.store %p_vec[0,1,2,3], C[64, 0]
  → C[64,0] = %p_vec[0]
  → C[64,1] = %p_vec[1]
  → C[64,2] = %p_vec[2]
  → C[64,3] = %p_vec[3]

示例（i=64, j=4）:
  vector.store %p_vec[0,1,2,3], C[64, 4]
  → C[64,4] = %p_vec[0]
  → C[64,5] = %p_vec[1]
  → C[64,6] = %p_vec[2]
  → C[64,7] = %p_vec[3]
```

关键: 向量化发生在 **j 循环内**，i 和 k 在此时是**常量索引**

**向量化代码结构:**

```mlir
affine.for %i = 0 to 128 {            // 外层循环
  affine.for %k = 0 to 64 {           // 中层循环
    affine.for %j = 0 to 96 step 4 {  // 内层循环（被向量化）

      // i=64, k=32 时，j=0,4,8 的各次迭代:

      // ============ 外层变量的角色 ============
      // %i: 固定为 64，用于选择 A 和 C 的行
      // %k: 固定为 32，用于选择 A 的列和 B 的行
      // 它们不参与向量化计算

      %a = affine.load %A[64, 32]    // A[行64, 列32] → 标量
      %b_vec = vector.load %B[32, 0:3]  // B[行32, 列0-3] → 向量4
      %c_vec = vector.load %C[64, 0:3]  // C[行64, 列0-3] → 向量4

      // ============ 向量化计算 ============
      %a_vec = vector.broadcast %a        // 标量 → 向量(复制4份)

      // 向量 FMA: p[m] = a[m] × b[m] + c[m]
      //   其中 m 是 j 循环内的向量索引 (0, 1, 2, 3)
      //   每次计算 4 个结果，对应 j=0,4,8,12 时的存储

      vector.store %p_vec, %C[64, 0]  // 存储 4 个元素到 C[行64]
    }  // j 循环结束
  }  // k 循环结束
}  // i 循环结束
```

**WHY i 和 k 不需要向量化:**

- 它们是外层循环的归纳变量
- 在 j 循环内保持**不变**（是常量）
- 用于**索引** A、B、C 的维度
- 向量化的是 j 维度的**元素级并行**

**向量化效果:**
- 原 j 循环：96 次迭代，每次处理 1 个元素
- 向量化后：24 次迭代（96/4），每次处理 4 个元素
- 理论加速：约 4x（假设向量宽度与硬件匹配）

---

**扩展：i-k 并行化的详细分析**

### 问题的本质：循环携带依赖

原始三重嵌套循环:
```
for i in [0, 128):
  for k in [0, 64):
    C[i,j] += A[i,k] * B[k,j]  // C[i,j] 需要上一次迭代的值
```

**WHY 有依赖:**
- C[i,j] 在内层 j 循环中被累加
- 每次迭代需要读取上一次写入的 C[i,j]
- 这是**跨迭代的依赖**（循环携带依赖）

### 方案 1: affine.parallel 的限制

```mlir
affine.parallel (%i, %k) = (0, 0) to (128, 64) {
  affine.for %j = 0 to 96 {
    // 问题: C[i,j] 的归约在哪里?
    // affine.parallel 支持 reductions，但:
    //  - 归约只在循环结束后执行一次
    //  - 无法在内层循环中使用部分归约结果
  }
}
```

**WHY 不够:**
- `affine.parallel` 的归约操作在**最外层循环结束后**执行
- 无法满足 `C[i,j] += A[i,k] * B[k,j]` 的模式（需要在内层循环中使用部分结果）

### 方案 2: 循环交换 + 归约重组（实际可行）

**第 1 步: 循环交换解除依赖**

原始顺序: i → j → k
```
for i in [0, 128):        // 外层循环
  for j in [0, 96):       // 中层循环
    for k in [0, 64):     // 内层循环
      C[i,j] += A[i,k] * B[k,j]  // 依赖上一次迭代的 C[i,j]
```

**WHY 原始有依赖:**
- C[i,j] 在内层 k 循环中累加
- 每次迭代需要读取上一次写入的 C[i,j]
- k 循环有**循环携带依赖**

交换后顺序: k → j → i
```
for k in [0, 64):        // 外层循环 (原内层)
  for j in [0, 96):       // 中层循环 (原中层)
    sum = 0               // 可以在 i 循环前初始化
    for i in [0, 128):    // 内层循环 (原外层)
      sum += A[i,k] * B[k,j]  // 不再有循环携带依赖!
    C[k,j] = sum           // 在 i 循环结束后存储
```

**WHY 交换后可以并行:**
- k 和 j 在外层和中层（可以并行）
- i 在内层，循环结束时产生**完整的** sum（不需要跨迭代传递）
- C[k,j] 在 i 循环结束后一次性写入（无循环携带依赖）

**第 2 步: 应用 affine.parallel**

```mlir
affine.parallel (%k, %j) = (0, 0) to (64, 96) {
  // k 和 j 现在可以并行执行
  // 因为它们的迭代是独立的

  // 内层 i 循环带归约
  %sum = affine.for %i = 0 to 128 iter_args(%acc = %zero) {
    %a = affine.load %A[%i, %k]
    %b = affine.load %B[%k, %j]
    %p = arith.mulf %a, %b
    %new_acc = arith.addf %acc, %p
    affine.yield %new_acc  // 传递给下一次迭代
  }

  // 在 k,j 并行循环结束后存储
  affine.store %sum, %C[%k, %j]
}
```

### 完整的变换序列

```
原始: i → j → k (串行，有循环携带依赖)
  ↓
交换: k → j → i (依赖解除)
  ↓
并行: affine.parallel(k, j) (k 和 j 可以并行)
  ↓
向量化: 内层 i 循环向量化 (可以与并行化同时使用！)
```

### 并行化 + 向量化同时使用

**重要: 并行化和向量化是互补的优化技术，可以同时应用！**

```mlir
// 完整的并行 + 向量化版本
affine.parallel (%k, %j) = (0, 0) to (64, 96) {
  // ============ 并行层级 ============
  // 不同的 (k, j) 对在不同线程/核心上执行

  // ============ 向量化层级 ============
  // 每个 (k, j) 对内部，i 循环使用 SIMD
  affine.for %i = 0 to 128 step 4 {      // 向量宽度 = 4
    %a_vec = vector.load %A[%i, %k]     // A[i:i+4, k] 向量加载
    %b = affine.load %B[%k, %j]         // B[k, j] 标量加载
    %c_vec = vector.load %C[%i, %j]     // C[i:i+4, j] 向量加载

    %b_vec = vector.broadcast %b, 4     // 标量广播为向量

    %p_vec = vector.fma %a_vec, %b_vec, %c_vec  // 向量 FMA
    vector.store %p_vec, %C[%i, %j]
  }
}
```

**硬件执行模型:**

```
┌─────────────────────────────────────────────────────────┐
│                    多核处理器 + SIMD                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 核心 0    │  │ 核心 1   │  │ 核心 2    │  │ 核心 3    │  │
│  │(k=0,j=0) │  │(k=0,j=1) │  │(k=0,j=2) │  │(k=0,j=3) │  │
│  │          │  │          │  │          │  │          │  │
│  │SIMD:4×f32│  │SIMD:4×f32│  │SIMD:4×f32│  │SIMD:4×f32│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────┘

并行化: 4 个核心同时处理不同的 (k, j)
向量化: 每个核心用 SIMD 处理 4 个 i 元素
```

**加速效果:**
- 并行: 4x (假设 4 个核心)
- 向量化: 4x (向量宽度 4)
- 总加速: 16x (理想情况，无其他瓶颈)

### 实际应用案例

**这种并行化 + 向量化方案在工业界广泛使用！**

#### 1. 高性能矩阵乘法库

| 库 | 应用 | 优化技术 |
|---|------|----------|
| **Intel oneDNN** | 深度学习推理 | OpenMP + AVX-512 |
| **OpenBLAS** | 科学计算 | pthread + AVX2 |
| **BLIS** | HPC | OpenMP + ARM SVE |

**实际性能：** 相比朴素实现可达 50-100x 加速

#### 2. 深度学习框架

```
PyTorch/TensorFlow 的矩阵乘法调用链:

torch.matmul / tf.matmul
  ↓
oneDNN / cuBLAS / Eigen
  ↓
多线程 + SIMD 内核 (类似上述方案)
  ↓
硬件执行
```

#### 3. MLIR 在工业界的应用

**使用 MLIR Affine 优化循环的项目：**

1. **TensorFlow/XLA**
   - 使用 Affine dialect 进行循环优化
   - 自动 lowering 到 GPU/CPU 代码

2. **IREE (Google)**
   - MLIR 作为核心 IR
   - Affine → GPU (Vulkan/SPIRV/CUDA)

3. **torch-mlir**
   - PyTorch 程序的 MLIR 编译器
   - 自动应用分块、并行化、向量化

#### 4. LLVM Polly 优化器

```bash
# Polly 是 LLVM 的多面体优化框架
# 自动应用本文档讨论的所有变换

clang -O3 -mllvm -polly \
       -mllvm -polly-vectorizer=stripmine \
       matmul.c

# Polly 自动执行:
# 1. 依赖分析 (checkMemrefAccessDependence)
# 2. 循环分块 (LoopTiling)
# 3. 循环交换 (LoopInterchange)
# 4. OpenMP 并行化
# 5. SIMD 向量化
```

#### 5. Intel oneDNN 的矩阵乘法内核 (简化)

```cpp
// 文件: src/cpu/x64/jit_gemm_s8u8s32_avx512_core.cpp

// 外层：OpenMP 并行
#pragma omp parallel for collapse(2)
for (int k = 0; k < K; k += k_block) {
  for (int j = 0; j < N; j += j_block) {
    // 中层：分块
    for (int i = 0; i < M; i += i_block) {
      // 内层：AVX-512 向量化
      __m512i sum = _mm512_setzero_si512();
      for (int k_inner = 0; k_inner < k_block; k_inner += 16) {
        // 向量加载 (16 × int8)
        __m512i a = _mm512_loadu_si512(...);
        __m512i b = _mm512_loadu_si512(...);
        // 向量 FMA (融合乘加)
        sum = _mm512_dpbusd_epi32(sum, a, b);  // AVX-512 VNNI
      }
      _mm512_storeu_si512(..., sum);
    }
  }
}
```



### 关键理解

**WHY 循环交换能解除依赖:**
- 原始：C[i,j] 需要上一次迭代的值（跨 k 依赖）
- 交换后：C[k,j] 在内层 i 循环结束后计算完成（无跨迭代依赖）

**WHY affine.parallel 需要交换:**
- 原始结构有循环携带依赖（串行性质）
- 交换后结构可以分解为独立的并行任务
- 需要先做**分块**和**循环交换**来解除依赖

**实际编译器的做法:**
1. 首先进行循环分块
2. 然后应用循环交换
3. 最后在最内层使用向量化
4. 如果有 `affine.parallel` 支持，直接在最内层使用

****

### 7.2 依赖分析示例

**代码:**

```mlir
affine.for %i = 1 to 99 {
  affine.for %j = 1 to 99 {
    %v1 = affine.load %A[%i, %j]        // S1
    %v2 = affine.load %A[%i+1, %j]      // S2
    affine.store %v1, %A[%i, %j+1]      // S3
  }
}
```

**依赖分析:**

```
访问函数:
  S1.read:  (i, j) -> (i, j)
  S2.read:  (i, j) -> (i+1, j)
  S3.write: (i, j) -> (i, j+1)

依赖检查:
  S1 -> S3:
    约束: (i1, j1) = (i2, j2+1)
    顺序: (i1, j1) < (i2, j2)
    解: j1 = j2 + 1 且 (i1 < i2 或 (i1 = i2 且 j1 < j2))
         = j1 = j2 + 1 且 j1 < j2
         无解!
         结论: 无依赖

  S2 -> S3:
    约束: (i+1, j) = (i', j'+1)
    顺序: (i, j) < (i', j')
    解: i+1 = i', j = j'+1
         且 (i < i' 或 (i = i' 且 j < j'))
         i < i+1: 恒成立
         方向向量: [1, -1] (i 跨 1 步, j 跨 -1 步)

并行性:
  外层 i 循环: 有依赖 (跨步)，不能并行
  内层 j 循环: 有依赖 (反向)，不能并行
```

### 7.3 Affine 执行流程

**编译阶段:**

```
1. 解析 (Parser)
   ├─ 识别 affine.for/if/parallel
   ├─ 解析仿射映射
   └─ 验证维度/符号约束

2. 验证 (Verifier)
   ├─ isValidDim: 检查维度有效性
   ├─ isValidSymbol: 检查符号有效性
   └─ 检查操作数与映射一致性

3. 分析 (Analysis)
   ├─ 构建依赖图
   ├─ 计算迭代空间
   └─ 检测并行性

4. 变换 (Transforms)
   ├─ 应用优化 Pass
   ├─ 保持语义正确性
   └─ 更新依赖信息

5. 降低 (Lowering)
   ├─ affine.for -> scf.for
   ├─ affine.if -> scf.if
   └─ affine.load/store -> 标准操作
```

**运行时阶段:**

```
Affine 本身不引入运行时开销
所有分析都在编译时完成
生成的代码与手写循环性能相当或更优
```

---

## 总结

### Affine 方言的核心价值

1. **精确的数学表示**: 通过仿射约束精确表示程序行为
2. **强大的分析能力**: 依赖分析、并行性检测自动完成
3. **丰富的优化空间**: 多面体变换提供广阔的优化空间
4. **渐进式降低**: 可以逐步降低到更底层的方言

### 关键设计原则

1. **WHY 约束仿射表达式?**
   - 保证可分析性
   - 支持精确的依赖分析
   - 可以在编译时完全求值

2. **WHY 分离维度和符号?**
   - 区分迭代相关和无关的值
   - 简化分析算法
   - 提高优化精度

3. **WHY 提供专门的循环结构?**
   - 编码更多优化信息
   - 自动验证优化合法性
   - 简化变换实现

### 未来发展

1. **更丰富的分析**: 跨函数、跨模块分析
2. **自动调优**: 基于性能模型的自动参数选择
3. **GPU 支持**: 更好的 GPU 映射和优化
4. **与其他方言协同**: 与 Linalg、Vector 等方言的深度集成

---

**参考文献:**
1. MLIR Documentation: https://mlir.llvm.org/docs/Dialects/Affine/
2. Polyhedral compilation: "Polyhedral Compilation" by Louis-Noel Pouchet
3. Affine transformations: "Affine Transformations" in LLVM
4. Presburger arithmetic: "Decision Methods for the Algebra of Theory of Real Fields" by Tarski
