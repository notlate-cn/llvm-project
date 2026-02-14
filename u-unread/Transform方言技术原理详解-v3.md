# MLIR Transform 方言技术原理详解 v3.0

> 基于 MLIR 官方文档 [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/#overview) 和源代码 `mlir/lib/Dialect/Transform/`、`mlir/lib/Dialect/Linalg/TransformOps/` 深度分析生成。

## 目录

0. [快速入门](#0-快速入门)
1. [快速概览](#1-快速概览)
2. [背景与动机](#2-背景与动机)
3. [核心概念](#3-核心概念)
4. [类型系统](#4-类型系统)
5. [核心操作详解](#5-核心操作详解)
6. [源码实现：TransformState](#6-源码实现transformstate)
7. [源码实现：TransformDialectExtension](#7-源码实现transformdialectextension)
8. [执行模型](#8-执行模型)
9. [扩展开发完整教程](#9-扩展开发完整教程)
10. [实战案例](#10-实战案例)
11. [调试与排错](#11-调试与排错)
12. [性能与最佳实践](#12-性能与最佳实践)
13. [常见问题FAQ](#13-常见问题faq)
14. [参考资料](#14-参考资料)

---

## 0. 快速入门

### 0.1 Hello World：最简单的 Transform 示例

**目标：** 使用 Transform方言打印一个操作的名称。

```mlir
// ============================================================
// Payload IR - 被转换的目标 IR
// ============================================================
module {
  func.func @hello_world() {
    %0 = arith.constant 42 : i32
    return
  }
}

// ============================================================
// Transform IR - 控制转换逻辑的 IR
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 找到所有函数
  %funcs = transform.loop.match "func.func" in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印找到的函数
  transform.print %funcs { name = "Found functions" }

  transform.yield
}
```

**预期输出：**
```
// Found functions
func.func @hello_world
```

### 0.2 从头到尾的执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Transform Hello World 执行流程                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  步骤 1: 解析阶段                                                        │
│    ├── 加载 Transform 方言                                               │
│    ├── 解析 Transform IR                                                 │
│    └── 验证类型正确性                                                    │
│                                                                         │
│  步骤 2: 状态初始化                                                      │
│    ├── 创建 TransformState                                              │
│    └── 映射 %root → 顶层模块操作                                         │
│                                                                         │
│  步骤 3: 执行 transform.loop.match                                       │
│    ├── 从 %root 获取 Payload 操作                                        │
│    ├── 遍历查找所有 func.func 操作                                       │
│    └── 创建 %funcs Handle，指向 [func.func@hello_world]                  │
│                                                                         │
│  步骤 4: 执行 transform.print                                           │
│    ├── 从 %funcs 获取 Payload 操作                                       │
│    ├── 打印操作到 stdout                                                 │
│    └── 输出: "Found functions: func.func @hello_world"                   │
│                                                                         │
│  步骤 5: transform.yield 结束                                            │
│    └── 返回成功状态                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 0.3 完整可运行的示例

```mlir
// ============================================================
// 完整示例：匹配并打印所有算术操作
// ============================================================
module {
  func.func @example(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %c0 = arith.constant 0.0 : f32
    %1 = arith.addf %arg0, %arg0 : tensor<10xf32>
    %2 = arith.mulf %1, %1 : tensor<10xf32>
    return %2 : tensor<10xf32>
  }
}

// ============================================================
// Transform 序列
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 匹配所有 arith.addf 操作
  %add_ops = transform.loop.match "arith.addf" in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印匹配结果
  transform.print %add_ops { name = "Add operations" }

  // 匹配所有 arith.mulf 操作
  %mul_ops = transform.loop.match "arith.mulf" in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印匹配结果
  transform.print %mul_ops { name = "Mul operations" }

  transform.yield
}
```

### 0.4 关键概念速览

| 概念 | 说明 | 示例 |
|------|------|------|
| **Payload IR** | 被转换的目标 IR | `func.func`, `arith.addf`, `scf.for` |
| **Transform IR** | 控制转换逻辑的 IR | `transform.sequence`, `transform.loop.match` |
| **Handle** | Transform IR 中指向 Payload IR 的引用 | `%funcs : !transform.any_op` |
| **TransformState** | 维护 Handle ↔ Payload 映射的运行时状态 | 内部对象，用户不可见 |

---

## 1. 快速概览

### 1.1 基本信息

**Transform 方言**是 MLIR 中用于**精细控制 IR 转换**的方言。

| 属性 | 说明 |
|-----|------|
| **方言名称** | `transform` |
| **C++ 命名空间** | `::mlir::transform` |
| **核心文件** | `mlir/lib/Dialect/Transform/IR/TransformDialect.cpp` |
| **ODS 定义** | `mlir/include/mlir/Dialect/Transform/IR/TransformOps.td` |
| **接口定义** | `mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.td` |

### 1.2 核心设计理念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Transform 方言架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐         ┌──────────────────┐                     │
│   │  Transform IR    │         │   Payload IR     │                     │
│   │  (控制转换逻辑)    │──────▶  │   (被转换的IR)    │                      │
│   └──────────────────┘         └──────────────────┘                     │
│          │                                                              │
│          │ 通过 Handle 关联                                               │
│          ▼                                                              │
│   ┌──────────────────┐                                                  │
│   │  Handle 类型系统  │                                                  │
│   │ • OperationHandle│                                                  │
│   │ • ValueHandle    │                                                  │
│   │ • ParamHandle    │                                                  │
│   └──────────────────┘                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 WHY 需要 Transform 方言？

**问题本质：** 传统编译器优化 Pass 缺乏精细控制能力

- **Pass 粒度太粗**：Pass 对所有匹配的 IR 操作应用相同的转换
- **Pass 组合困难**：需要特定顺序的转换时，Pass 组合会爆炸式增长
- **缺乏选择能力**：无法根据运行时信息选择不同的转换策略

**WHY 选择 Transform 方言：**

- **精细控制**：可以针对单个或一组操作应用特定转换
- **可编程组合**：使用 IR 本身表达转换序列，灵活组合
- **类型安全**：通过 Handle 类型系统确保转换前置条件
- **可扩展**：通过扩展机制添加方言特定的转换操作

---

## 2. 背景与动机

### 2.1 问题场景

考虑以下编译优化场景：

```mlir
// 假设有以下循环嵌套需要优化
scf.for %i = 0 to 1024 {
  scf.for %j = 0 to 1024 {
    scf.for %k = 0 to 1024 {
      // 一些计算
    }
  }
}
```

**传统 Pass 方法的局限：**

1. **无法区分**：Pass 对所有循环应用相同转换
2. **无法组合**：想要"先切分最内层循环，再切分次内层"需要编写新 Pass
3. **无法回退**：某种转换失败时，无法尝试备选方案

### 2.2 Transform 方言的解决方案

```mlir
// 使用 Transform 方言精细控制转换
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // 1. 找到所有循环
  %loops = transform.loop.structure %arg0 : (!transform.any_op) -> !transform.any_op

  // 2. 只对最内层循环应用切分
  %innermost = transform.loop.get_innermost %loops
  transform.loop.unroll %innermost { factor = 4 }

  // 3. 对次内层循环应用不同转换
  %middle = transform.loop.get_middle %loops
  transform.loop.tile %middle { tile_sizes = [8, 8] }
}
```

---

## 3. 核心概念

### 3.1 Payload IR 与 Transform IR

**概念：** Transform 方言引入了两个 IR 层级的分离

| IR 类型 | 作用 | 示例 |
|---------|------|------|
| **Payload IR** | 被转换的目标 IR | `linalg.matmul`, `scf.for`, `arith.addf` |
| **Transform IR** | 控制转换逻辑的 IR | `transform.sequence`, `transform.loop.tile` |

**WHY 这样分离：**

- **关注点分离**：转换逻辑与业务逻辑解耦
- **可重用性**：同一转换脚本可应用于不同的 Payload IR
- **类型安全**：Transform IR 的类型系统可以验证转换的正确性

### 3.2 Handle（句柄）

**概念：** Handle 是 Transform IR 中指向 Payload IR 对象的引用

```mlir
// Handle 示例
%0: !transform.any_op        // 指向任意操作的 Handle
%1: !transform.any_value     // 指向任意值的 Handle
%2: !transform.param<i32>     // 指向参数的 Handle
```

**Handle 的关键特性：**

1. **多对象关联**：一个 Handle 可以关联多个 Payload IR 对象
2. **类型约束**：Handle 类型编码了关联对象的属性
3. **批量执行**：大多数 Transform 操作对 Handle 关联的所有对象执行

### 3.3 Transform 类型接口详解

#### 3.3.1 TransformTypeInterfaceBase - 基础类型接口

```cpp
// TransformInterfaces.td
template <typename DerivedTy, typename PayloadTy>
class TransformTypeInterfaceBase : public TypeInterface<DerivedTy> {
public:
  virtual DiagnosedSilenceableFailure checkPayload(
      Location loc,
      ArrayRef<PayloadTy> payload) = 0;
};
```

#### 3.3.2 TransformHandleTypeInterface - 操作句柄接口

```cpp
// TransformInterfaces.td
def TransformHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformHandleTypeInterface",
                                 "::mlir::Operation *"> {
  let description = [{
    Types that can be used for the Transform dialect operation handle values.
  }];
}
```

**实现示例：**
```cpp
DiagnosedSilenceableFailure OperationType::checkPayload(
    Location loc, ArrayRef<Operation *> payload) {
  for (Operation *op : payload) {
    if (op->getName().getStringRef() != getOperationName()) {
      return emitSilenceableError(loc)
             << "expected '" << getOperationName() << "' operation, "
             << "but found '" << op->getName() << "'";
    }
  }
  return DiagnosedSilenceableFailure::success();
}
```

#### 3.3.3 TransformValueHandleTypeInterface - 值句柄接口

```cpp
// TransformInterfaces.td
def TransformValueHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformValueHandleTypeInterface",
                                 "::mlir::Value"> {
  let description = [{
    Types that can be used for the Transform dialect handle values pointing to
    Payload IR values.
  }];
}
```

#### 3.3.4 TransformParamTypeInterface - 参数句柄接口

```cpp
// TransformInterfaces.td
def TransformParamTypeInterface
    : TransformTypeInterfaceBase<"TransformParamTypeInterface",
                                 "::mlir::Attribute"> {
  let description = [{
    Types that can be used for the Transform dialect parameter values.
  }];
}
```

### 3.4 TransformOpInterface - 操作接口

**概念：** 所有 Transform 操作必须实现的核心接口。

#### 3.4.1 核心方法：apply

```cpp
virtual DiagnosedSilenceableFailure apply(
    TransformRewriter &rewriter,
    TransformResults &results,
    TransformState &state) = 0;
```

**参数说明：**

| 参数 | 类型 | 作用 |
|------|------|------|
| `rewriter` | `TransformRewriter&` | 用于修改 Payload IR 的重写器 |
| `results` | `TransformResults&` | 填充转换结果的容器 |
| `state` | `TransformState&` | 访问 Handle 映射的状态对象 |

**实现示例：**
```cpp
DiagnosedSilenceableFailure MyTransformOp::apply(
    TransformRewriter &rewriter,
    TransformResults &results,
    TransformState &state) {

  // 步骤 1: 获取目标操作
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  if (targets.empty()) {
    return emitSilenceableError() << "no operations to transform";
  }

  // 步骤 2: 对每个目标应用转换
  SmallVector<Operation *> transformedOps;
  for (Operation *target : targets) {
    FailureOr<Operation *> result = applyMyTransform(rewriter, target);
    if (failed(result)) {
      return emitDefaultSilenceableFailure(target);
    }
    transformedOps.push_back(*result);
  }

  // 步骤 3: 设置结果
  results.set(cast<OpResult>(getResult()), transformedOps);

  return DiagnosedSilenceableFailure::success();
}
```

---

## 4. 类型系统

### 4.1 类型层次结构

```
Transform 类型系统
├── TransformHandleTypeInterface (操作句柄)
│   ├── AnyOpType (!transform.any_op)
│   └── OperationType<!transform.op<"op_name">>
├── TransformValueHandleTypeInterface (值句柄)
│   └── AnyValueType (!transform.any_value)
└── TransformParamTypeInterface (参数句柄)
    ├── AnyParamType (!transform.any_param)
    ├── ParamType<!transform.param<Type>>
    ├── AffineMapParamType (!transform.affine_map)
    └── TypeParamType (!transform.type)
```

### 4.2 操作句柄类型 (Operation Handle Types)

#### 4.2.1 AnyOpType

**语法：** `!transform.any_op`

**定义：**
```tablegen
def Transform_AnyOpType : TypeDef<Transform_Dialect, "AnyOp",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let mnemonic = "any_op";
}
```

**WHY 设计 AnyOpType：**
- **灵活性**：可以指向任何操作
- **类型安全最小化**：不进行类型约束验证
- **适用场景**：操作类型未知或多样化

#### 4.2.2 OperationType

**语法：** `!transform.op<"operation_name">`

**定义：**
```tablegen
def Transform_OperationType : TypeDef<Transform_Dialect, "Operation",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let mnemonic = "op";
  let parameters = (ins
    StringRefParameter<"Name of the allowed payload operation">:$operation_name
  );
}
```

**WHY 设计 OperationType：**
- **类型安全**：编译时和运行时都验证操作类型
- **操作特化**：某些转换只适用于特定操作
- **文档作用**：清晰表达句柄的预期内容

### 4.3 值句柄类型 (Value Handle Types)

#### 4.3.1 AnyValueType

**语法：** `!transform.any_value`

**定义：**
```tablegen
def Transform_AnyValue : TypeDef<Transform_Dialect, "AnyValue",
    [DeclareTypeInterfaceMethods<TransformValueHandleTypeInterface>]> {
  let mnemonic = "any_value";
}
```

**WHY 需要值句柄：**

| 特性 | 操作句柄 | 值句柄 |
|------|---------|--------|
| 指向对象 | Operation | SSA Value |
| 用途 | 操作转换 | 值追踪/转换 |
| 示例 | `!transform.any_op` | `!transform.any_value` |

### 4.4 参数句柄类型 (Parameter Handle Types)

#### 4.4.1 ParamType

**语法：** `!transform.param<Type>`

**定义：**
```tablegen
def Transform_ParamType : TypeDef<Transform_Dialect, "Param",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let mnemonic = "param";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Underlying type of the parameter">:$type
  );
}
```

**WHY 需要参数句柄接口：**

| 需求 | 说明 | 示例 |
|------|------|------|
| **运行时参数** | 支持在 Transform 执行时传递参数 | 切分大小、向量宽度等 |
| **类型安全** | 确保参数类型正确 | 防止将字符串传给期望整数的转换 |
| **灵活性** | 允许动态配置转换行为 | 根据运行时信息选择参数 |

### 4.5 类型验证机制

每个 Transform 类型都实现了对应的接口，提供 `checkPayload` 方法进行运行时验证。

```cpp
// 接口定义
virtual DiagnosedSilenceableFailure checkPayload(
    Location loc,
    ArrayRef<Operation *> payload) = 0;
```

**WHY 需要运行时验证：**

| 验证时机 | 验证内容 | 原因 |
|---------|---------|------|
| 解析时 | 类型语法正确性 | MLIR 类型系统保证 |
| 执行时 | Payload 对象符合类型约束 | 运行时才知道 Payload 对象 |

### 4.6 类型系统设计决策

#### 4.6.1 WHY 使用参数化类型？

- **类型安全**：编译时就知道句柄指向的操作类型
- **优化机会**：编译器可以基于类型信息优化
- **文档作用**：类型本身就表达了约束

#### 4.6.2 WHY 分离三种 Handle 类型？

```
操作句柄 → 指向 Operation（可执行单元）
  ↓
值句柄 → 指向 Value（数据流）
  ↓
参数句柄 → 指向 Attribute（编译时常量）
```

**WHY 这样分离：**

1. **语义清晰**：操作转换、数据追踪、配置参数各司其职
2. **类型安全**：混用会导致类型混乱
3. **验证分离**：每种类型有不同的验证规则

---

## 5. 核心操作详解

### 5.1 transform.sequence - 转换序列

**语法：**
```mlir
transform.sequence failures(propagate|suppress) {
^bb0(%root: !transform.any_op):
  // 转换操作序列
  transform.yield %result : !transform.any_op
}
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `failure_propagation_mode` | 属性 | `propagate` 或 `suppress` |
| `root` | 可选 Handle | 顶层操作句柄 |

**返回值：** 变长的 Handle 列表（由 yield 产生）

**WHY 需要 sequence：**
- 组织多个转换操作
- 控制错误传播行为
- 提供作用域隔离

**使用示例：**
```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 匹配所有循环
  %loops = transform.loop.match "scf.for" in %root
      : (!transform.any_op) -> !transform.any_op

  // 应用分块
  %tiled = transform.loop.tile %loops tile_size = 32

  // 应用向量化
  transform.vectorize %tiled

  transform.yield %tiled : !transform.any_op
}
```

### 5.2 transform.match.ops / transform.structured.match - 操作匹配

**语法：**
```mlir
// 核心方言
%results = transform.match ops {["op_name1", "op_name2"]} in %target
    : (!transform.any_op) -> !transform.any_op

// Linalg 方言
%results = transform.structured.match ops {["linalg.matmul"]} in %target
    : (!transform.any_op) -> !transform.any_op
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | Handle | 在其中搜索的操作 |
| `ops` | 字符串数组 | 要匹配的操作名称 |
| `interface` | 可选属性 | 匹配特定接口 |
| `attributes` | 可选属性 | 匹配特定属性 |
| `filter_result_type` | 可选属性 | 过滤结果类型 |

**返回值：** 指向所有匹配操作的 Handle

**WHY 需要匹配：**
- 选择要转换的目标操作
- 支持复杂查询条件
- 类型安全的操作选择

**使用示例：**
```mlir
// 匹配所有 matmul 操作
%matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

// 匹配多种操作
%ops = transform.match ops {["scf.for", "affine.for"]} in %root
    : (!transform.any_op) -> !transform.any_op

// 使用接口匹配
%loops = transform.structured.match interface {LoopLikeInterface} in %root
    : (!transform.any_op) -> !transform.any_op
```

### 5.3 transform.structured.tile - 循环分块

**语法：**
```mlir
%tiled, %loops = transform.structured.tile %target [tile_sizes]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | Handle | 要分块的操作 |
| `tile_sizes` | 整数数组 | 每个维度的分块大小 |
| `interchange` | 可选数组 | 维度置换 |

**返回值：**
| 返回值 | 说明 |
|--------|------|
| `tiled` | 分块后的操作 |
| `loops` | 新生成的循环 |

**WHY 需要分块：**
- 提高缓存局部性
- 启用向量化
- 减少内存访问延迟

**使用示例：**
```mlir
// 矩阵乘法分块
%matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

// 分块：M=64, N=32, K=16
%tiled, %loops = transform.structured.tile %matmuls [64, 32, 16]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### 5.4 transform.structured.vectorize - 向量化

**语法：**
```mlir
// 自动推断向量大小
transform.structured.vectorize %target : !transform.any_op

// 指定向量大小
transform.structured.vectorize %target vector_sizes [4, 8]
    : (!transform.any_op) -> !transform.any_op
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | Handle | 要向量化的操作 |
| `vector_sizes` | 可选数组 | 向量形状 |
| `scalable_sizes` | 可选数组 | 可扩展向量标志 |

**返回值：** 无（消费目标 Handle）

**WHY 需要向量化：**
- 利用 SIMD 指令
- 提高并行度
- 减少指令数量

**使用示例：**
```mlir
// 先分块再向量化
%matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

%tiled, %loops = transform.structured.tile %matmuls [16, 16]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

// 向量化最内层循环
transform.structured.vectorize %tiled vector_sizes [4, 8]
    : (!transform.any_op) -> !transform.any_op
```

### 5.5 transform.print - 调试输出

**语法：**
```mlir
transform.print %target { name = "Debug output" }
    : !transform.any_op
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | 可选 Handle | 要打印的操作 |
| `name` | 可选字符串 | 打印前缀 |
| `assume_verified` | 可选属性 | 跳过验证 |
| `use_local_scope` | 可选属性 | 局部作用域打印 |
| `skip_regions` | 可选属性 | 跳过子区域 |

**返回值：** 无

**WHY 需要 print：**
- 调试转换序列
- 验证中间结果
- 理解转换流程

**使用示例：**
```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root

  // 打印匹配结果
  transform.print %ops { name = "Matched loops" }

  // 应用转换
  %tiled = transform.loop.tile %ops [32]

  // 打印转换结果
  transform.print %tiled { name = "After tiling" }

  transform.yield
}
```

### 5.6 transform.verify - 验证 IR

**语法：**
```mlir
transform.verify %target : !transform.any_op
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | Handle | 要验证的操作 |

**返回值：** 无

**WHY 需要 verify：**
- 确保 IR 合法性
- 类似断言的作用
- 捕获转换错误

**使用示例：**
```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root

  // 转换前验证
  transform.verify %ops { name = "Before transform" }

  %tiled = transform.loop.tile %ops [32]

  // 转换后验证
  transform.verify %tiled { name = "After transform" }

  transform.yield
}
```

### 5.7 transform.alternatives - 备选方案

**语法：**
```mlir
%result = transform.alternatives %scope : !transform.any_op {
^bb0(%arg0: !transform.any_op):
  // 备选方案 1
  %r1 = transform.try_vectorize %arg0
  transform.yield %r1 : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 备选方案 2
  %r2 = transform.scalar_optimize %arg0
  transform.yield %r2 : !transform.any_op
}
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `scope` | 可选 Handle | 备选方案的作用域 |
| `alternatives` | 区域列表 | 备选方案区域 |

**返回值：** 第一个成功方案的返回值

**WHY 需要 alternatives：**
- 尝试多种优化策略
- 提供回退机制
- 提高转换成功率

**使用示例：**
```mlir
// 尝试不同的向量化策略
%result = transform.alternatives %loops : !transform.any_op {
^bb0(%arg0: !transform.any_op):
  // 策略 1: 向量化 + 循环分发
  %v = transform.vectorize %arg0 vector_sizes [256]
  %d = transform.distribute %v
  transform.yield %d : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 策略 2: 分块 + 向量化
  %t = transform.tile %arg0 [32]
  %v = transform.vectorize %t vector_sizes [32]
  transform.yield %v : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 策略 3: 仅向量化（回退）
  %v = transform.vectorize %arg0 vector_sizes [128]
  transform.yield %v : !transform.any_op
}
```

---

## 6. 源码实现：TransformState

### 6.1 核心数据结构

TransformState 是 Transform 方言执行的核心状态管理类，负责维护 Transform IR 与 Payload IR 之间的映射关系。

```cpp
// TransformInterfaces.h (简化版)
class TransformState {
public:
  TransformState(Region *region, Operation *payloadRoot,
                 const RaggedArray<MappedValue> &extraMappings = {},
                 const TransformOptions &options = TransformOptions());

private:
  // Handle → Payload 操作的映射
  DenseMap<Region *, std::unique_ptr<Mappings>> mappings;

  // 区域栈：跟踪当前处理的区域
  SmallVector<RegionScope *> regionStack;

  // 扩展数据：支持用户自定义状态
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;

  // 选项：配置 Transform 执行行为
  TransformOptions options;
};
```

### 6.2 Mappings 结构

```cpp
// TransformInterfaces.cpp
struct TransformState::Mappings {
  // 正向映射：Transform Value → Payload Operation
  DenseMap<Value, SmallVector<Operation *>> direct;

  // 反向映射：Payload Operation → Transform Value
  DenseMap<Operation *, SmallVector<Value>> reverse;

  // 值映射：Transform Value → Payload Value
  DenseMap<Value, SmallVector<Value>> values;

  // 参数映射：Transform Value → Payload Attribute
  DenseMap<Value, SmallVector<Attribute>> params;
};
```

### 6.3 核心方法：setPayloadOps

```cpp
LogicalResult TransformState::setPayloadOps(Value value,
                                            ArrayRef<Operation *> targets) {
  // 步骤 1: 断言检查
  assert(value != kTopLevelValue && "cannot set payload ops for the top-level");

  // 步骤 2: 类型检查
  auto iface = llvm::cast<TransformHandleTypeInterface>(value.getType());

  // 步骤 3: 类型验证
  DiagnosedSilenceableFailure result =
      iface.checkPayload(value.getLoc(), targets);

  // 步骤 4: 建立映射
  if (failed(result.checkAndReport()))
    return failure();

  mappings[region]->direct[value] = targets;
  for (Operation *op : targets) {
    mappings[region]->reverse[op].push_back(value);
  }

  return success();
}
```

**WHY 需要双向映射：**
- 正向映射：从 Transform IR 查找 Payload 操作
- 反向映射：从 Payload 操作查找 Transform Handle（用于失效检测）

### 6.4 RegionScope - 区域作用域管理

```cpp
class TransformState::RegionScope {
public:
  RegionScope(TransformState &state, Region &region)
      : state(state), region(region) {
    // 进入区域时：压栈
    state.regionStack.push_back(this);
  }

  ~RegionScope() {
    // 退出区域时：弹栈
    assert(state.regionStack.back() == this);
    state.regionStack.pop_back();
  }

private:
  TransformState &state;
  Region &region;
};
```

**WHY 需要区域作用域：**
- 支持嵌套的 Transform IR 区域
- 自动管理 Handle 映射的生命周期
- 确保退出区域时清理状态

---

## 7. 源码实现：TransformDialectExtension

### 7.1 扩展机制背景与动机

**问题：** Transform 方言需要扩展？

- 不同方言需要特定的转换操作
- 核心方言不应依赖特定方言
- 需要延迟加载和类型安全

**设计目标：**

1. **延迟加载**：扩展只在需要时加载
2. **解耦合**：Transform 方言不依赖特定方言
3. **类型安全**：自动验证扩展操作的接口实现
4. **易用性**：简单的 API 注册操作和类型

### 7.2 扩展机制设计原理

#### 7.2.1 CRTP 模式

```cpp
// TransformDialect.h
template <typename DerivedTy>
class TransformDialectExtension {
public:
  // 初始化方法
  void init() {
    // 调用派生类的实现
    static_cast<DerivedTy *>(this)->apply();
  }

  // 注册操作
  template <typename... OpTys>
  void registerTransformOps() {
    dialect->addOperations<OpTys...>();
  }

  // 声明依赖方言
  template <typename DialectTy>
  void declareDependentDialect() {
    dialect->declareDependentDialect<DialectTy>();
  }

protected:
  TransformDialect *dialect;
};
```

**WHY 使用 CRTP：**
- 编译时多态，避免虚函数开销
- 类型安全的扩展注册
- 简洁的 API 设计

#### 7.2.2 初始化流程

```cpp
// TransformDialect.cpp
void TransformDialect::initialize() {
  // 步骤 1: 注册核心操作
  addOperations<
#define GET_OP_LIST
#include "TransformOps.cpp.inc"
  >();

  // 步骤 2: 注册扩展
  for (const ExtensionInitialization &entry : extensionsToInitialize) {
    entry.initialize(*this);
  }
}
```

### 7.3 核心组件详解

#### 7.3.1 registerTransformOps - 注册操作

```cpp
// 使用示例
void MyExtension::init() {
  registerTransformOps<
#define GET_OP_LIST
#include "MyTransformOps.cpp.inc"
  >();
}
```

#### 7.3.2 declareDependentDialect vs declareGeneratedDialect

```cpp
// 依赖方言：扩展操作使用的类型
declareDependentDialect<LinalgDialect>();

// 生成方言：转换可能产生的操作
declareGeneratedDialect<SCFDialect>();
declareGeneratedDialect<VectorDialect>();
```

**WHY 区分两者：**
- **依赖方言**：必须在加载扩展前加载
- **生成方言**：转换执行时需要加载

#### 7.3.3 registerTypes - 注册类型

```cpp
// 注册自定义类型
void registerTypes() {
  dialect->addTypes<
#define GET_TYPEDEF_LIST
#include "MyTransformTypes.cpp.inc"
  >();
}
```

### 7.4 完整扩展示例：LinalgTransformDialectExtension

```cpp
// LinalgTransformDialectExtension.h
class LinalgTransformDialectExtension
    : public ::mlir::transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgTransformDialectExtension)

  using Base::Base;

  void init() {
    // 声明依赖
    declareDependentDialect<LinalgDialect>();
    declareGeneratedDialect<SCFDialect>();

    // 注册操作
    registerTransformOps<
#define GET_OP_LIST
#include "LinalgTransformOps.cpp.inc"
    >();
  }
};
```

### 7.5 TransformDialectData - 扩展间通信机制

```cpp
// TransformState.h
class TransformDialectData {
public:
  template <typename T>
  T &get() {
    TypeID id = TypeID::get<T>();
    auto it = data.find(id);
    if (it == data.end()) {
      it = data.emplace(id, std::make_unique<T>()).first;
    }
    return static_cast<T &>(*it->second);
  }

private:
  DenseMap<TypeID, std::unique_ptr<Extension>> data;
};
```

**WHY 需要扩展间通信：**
- 共享转换状态
- 避免重复计算
- 支持协作式转换

### 7.6 扩展自动加载机制详解

#### 7.6.1 完整加载流程

```
┌─────────────────────────────────────────────────────────────┐
│                    扩展自动加载流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 应用启动                                                 │
│     ├── 注册所有扩展（不加载）                               │
│     └── 记录扩展 → 操作映射                                  │
│                                                             │
│  2. Pass 运行                                               │
│     ├── 解析 Transform IR                                    │
│     ├── 识别使用的操作                                       │
│     └── 触发扩展加载                                         │
│                                                             │
│  3. 扩展加载                                                │
│     ├── 加载依赖方言                                         │
│     ├── 注册操作和类型                                       │
│     └── 应用扩展到 Dialect                                  │
│                                                             │
│  4. 转换执行                                                │
│     └── 所有操作可用                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 7.6.2 扩展应用逻辑

```cpp
// TransformDialect.cpp
void TransformDialect::loadAvailableExtensions() {
  for (const auto &entry : extensionRegistry) {
    if (isExtensionRequired(entry)) {
      // 加载扩展
      entry.create(*this);
      loadedExtensions.insert(entry.typeid);
    }
  }
}

bool TransformDialect::isExtensionRequired(
    const ExtensionEntry &entry) {
  // 检查是否需要此扩展
  for (Operation &op : getOperations()) {
    if (entry.providedOps.contains(op.getName())) {
      return true;
    }
  }
  return false;
}
```

**WHY 自动加载：**
- 用户体验：无需手动加载
- 按需加载：只加载需要的扩展
- 避免循环依赖：延迟加载机制

---

## 8. 执行模型 (Execution Model)

### 8.1 执行流程概述

Transform 方言的执行是一个**逐步应用转换**的过程，每个 Transform 操作通过 `TransformOpInterface::apply()` 方法实现。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Transform 执行流程                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 解析与验证阶段                                                         │
│     ├── 解析 Transform IR                                                │
│     ├── 验证类型约束                                                      │
│     └── 检查操作定义                                                      │
│                                                                         │
│  2. 状态初始化阶段                                                         │
│     ├── 创建 TransformState                                              │
│     ├── 建立 Payload IR 根映射                                            │
│     └── 初始化 Handle → Payload 对象映射                                   │
│                                                                          │
│  3. 转换执行阶段                                                           │
│     ├── 调用 TransformOpInterface::apply()                               │
│     │   ├── 场景 A：成功 → 更新 Handle 映射                                │
│     │   ├── 场景 B：Silenceable Failure → 回滚，尝试备选                   │
│     │   └── 场景 C：Definite Failure → 立即停止                           │
│     └── 处理 Handle 失效                                                 │
│                                                                         │
│  4. 清理阶段                                                              │
│     ├── 移除 nullptr 操作                                                 │
│     ├── 压缩 Handle 映射                                                  │
│     └── 验证最终状态                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 详细执行步骤

#### 8.2.1 解析与验证 Transform IR

```cpp
// TransformInterpreterUtils.cpp
LogicalResult transform::applyTransformNamedSequence(
    RaggedArray<MappedValue> bindings, TransformOpInterface transformRoot,
    ModuleOp transformModule, const TransformOptions &options) {

  // 步骤 1: 创建 TransformState
  TransformState state(transformRoot->getRegion(), /*payloadRoot=*/nullptr,
                      bindings, options);

  // 步骤 2: 应用 Transform 操作
  DiagnosedSilenceableFailure result = state.applyTransform(transformRoot);

  // 步骤 3: 检查执行结果
  if (failed(result.checkAndReport())) {
    return failure();
  }

  return success();
}
```

#### 8.2.2 应用单个 Transform 操作

```cpp
// TransformInterfaces.cpp
DiagnosedSilenceableFailure TransformState::applyTransform(
    TransformOpInterface transform) {
  // 步骤 1: 创建 TransformRewriter
  TransformRewriter rewriter(transform->getContext());

  // 步骤 2: 设置 Rewriter 的监听器
  auto listener = createTrackingListener(rewriter);
  rewriter.setListener(listener.get());

  // 步骤 3: 创建 TransformResults 容器
  TransformResults results(transform->getNumResults());

  // 步骤 4: 调用操作的 apply 方法
  DiagnosedSilenceableFailure result =
      transform.apply(rewriter, results, *this);

  // 步骤 5: 处理执行结果
  if (succeeded(result.isSuccess())) {
    if (failed(updateStateFromResults(results, transform->getResults()))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    recordOpHandleInvalidations(transform);
  }

  return result;
}
```

### 8.3 失败处理机制

#### 8.3.1 Silenceable Failure（可恢复失败）

**定义：** 转换未能应用，但 Payload IR 未被修改，可以尝试其他转换。

**特征：**
- 转换未修改 Payload IR（原子性保证）
- 可以尝试备选转换
- 延迟报告错误

**使用场景：**
```mlir
// 场景 1: 尝试多种转换策略
transform.alternatives {
^bb0(%arg0: !transform.any_op):
  // 策略 A：尝试向量化
  %v = transform.try_vectorize %arg0
  transform.yield %v : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 策略 B：向量化失败，尝试标量优化
  %s = transform.scalar_optimize %arg0
  transform.yield %s : !transform.any_op
}
```

**WHY 这样设计：**
- **灵活性**：允许"尽力而为"的转换策略
- **容错性**：某个转换失败不影响整个流程
- **探索性**：尝试多种优化，选择最佳的

#### 8.3.2 Definite Failure（不可恢复失败）

**定义：** Payload IR 可能处于不一致状态，必须立即停止。

**特征：**
- Payload IR 可能已被部分修改
- 必须立即停止，不能继续执行
- 立即报告错误

**WHY 区分两种失败：**

| 特性 | Silenceable | Definite |
|------|-------------|----------|
| Payload IR 状态 | 未修改 | 可能不一致 |
| 后续操作 | 可以继续 | 必须停止 |
| 错误报告 | 可延迟 | 立即报告 |
| 典型场景 | 前置条件不满足 | 内部错误/约束违反 |

### 8.4 Handle 失效规则 (Handle Invalidation)

当 Transform 操作消费或修改 Payload 操作时，相关的 Handle 会自动失效。

#### 8.4.1 失效触发条件

```cpp
void TransformState::recordOpHandleInvalidations(
    TransformOpInterface transform) {
  // 步骤 1: 获取被消费的 Handle 操作数
  SmallVector<OpOperand *> consumedOperands =
      getConsumedHandleOpOperands(transform);

  // 步骤 2: 检查每个被消费的 Handle
  for (OpOperand *operand : consumedOperands) {
    Value handle = operand->get();
    ArrayRef<Operation *> payloadOps = getPayloadOpsView(handle);

    for (Operation *payloadOp : payloadOps) {
      if (payloadOp->isDead()) {
        invalidatedHandles.insert(handle);
      }
    }
  }
}
```

#### 8.4.2 失效规则图解

```
Handle 失效规则
├── 消费 OperationHandle
│   ├── ✓ 该 Handle 本身失效
│   ├── ✓ 指向嵌套操作的 Handle 失效
│   └── ✓ 指向操作结果的 ValueHandle 失效
│
└── 消费 ValueHandle
    ├── ✓ 产生该值的操作 Handle 失效
    ├── ✓ 指向嵌套操作的 Handle 失效
    └── ✓ 指向包含该值的块参数的 Handle 失效
```

**WHY 这样设计：**
- **安全性**：防止引用已删除/替换的操作
- **一致性**：确保 Handle 指向有效的 Payload IR
- **可预测性**：明确的失效规则，易于理解

### 8.5 TransformRewriter 与 TrackingListener

#### 8.5.1 TransformRewriter 的特殊功能

```cpp
class TransformRewriter : public PatternRewriter {
public:
  void replaceOp(Operation *op, ValueRange newValues) override {
    if (listener) {
      listener->notifyOperationReplaced(op, newValues);
    }
    PatternRewriter::replaceOp(op, newValues);
  }

  void eraseOp(Operation *op) override {
    if (listener) {
      listener->notifyOperationErased(op);
    }
    PatternRewriter::eraseOp(op);
  }
};
```

#### 8.5.2 TrackingListener 的映射更新逻辑

```cpp
class TrackingListener : public RewriterBase::Listener {
public:
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    SmallVector<Value> handles;
    (void)state.getHandlesForPayloadOp(op, handles);

    for (Value handle : handles) {
      if (!newValues.empty()) {
        Operation *newOp = newValues[0].getDefiningOp();
        if (newOp) {
          state.updateMapping(handle, op, newOp);
        }
      } else {
        state.invalidateHandle(handle);
      }
    }
  }

private:
  TransformState &state;
};
```

---

## 9. 扩展开发完整教程

### 9.1 扩展开发步骤概览

```
扩展开发流程
│
├── 步骤 1: 定义扩展类
│   └── 继承 TransformDialectExtension
│
├── 步骤 2: 使用 TableGen 定义操作
│   ├── 继承 TransformDialectOp
│   ├── 实现 TransformOpInterface
│   └── 定义操作参数和结果
│
├── 步骤 3: 实现 C++ 类
│   ├── 实现 apply 方法
│   ├── 实现 getEffects 方法
│   └── 处理错误情况
│
├── 步骤 4: 注册扩展
│   └── 通过 DialectRegistry 注册
│
└── 步骤 5: 测试扩展
    └── 编写单元测试和集成测试
```

### 9.2 步骤 1：定义扩展类

```cpp
// MyTransformOps.h
#pragma once

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace my {
namespace transform {

class MyTransformDialectExtension
    : public ::mlir::transform::TransformDialectExtension<
          MyTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MyTransformDialectExtension)

  using Base::Base;

  void init() {
    // 声明依赖方言
    declareDependentDialect<MyDialect>();

    // 声明生成方言
    declareGeneratedDialect<::mlir::scf::SCFDialect>();
    declareGeneratedDialect<::mlir::vector::VectorDialect>();

    // 注册 Transform 操作
    registerTransformOps<
#define GET_OP_LIST
#include "MyTransformOps.cpp.inc"
    >();
  }
};

} // namespace transform
} // namespace my
```

### 9.3 步骤 2：使用 TableGen 定义操作

```tablegen
// MyTransformOps.td
#ifndef MY_TRANSFORM_OPS
#define MY_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MyCustomTransformOp : TransformDialectOp<"my_custom",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Applies my custom transformation to target operations";

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    OptionalAttr<I64Attr>$param,
    UnitAttr:$verbose
  );

  let results = (outs
    TransformHandleTypeInterface:$result
  );

  let assemblyFormat = [{
    $target `(` $param^ `,` `verbose` $verbose^?`)` attr-dict
      `:` type($target) `->` type($result)
  }];

  let hasVerifier = 1;
}

#endif // MY_TRANSFORM_OPS
```

### 9.4 步骤 3：实现 C++ 类

```cpp
// MyTransformOps.cpp
#include "MyTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::transform;

namespace {

struct MyCustomTransformOp
    : public Op<MyCustomTransformOp,
               TransformOpInterface::Trait,
               MemoryEffectsOpInterface::Trait> {
  using Op::Op;

  DiagnosedSilenceableFailure apply(
      TransformRewriter &rewriter,
      TransformResults &results,
      TransformState &state) override {

    // 步骤 1: 获取目标操作
    ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

    if (targets.empty()) {
      return emitSilenceableError()
             << "no operations found to transform";
    }

    // 步骤 2: 获取可选参数
    int64_t param = 0;
    if (auto paramAttr = getParam()) {
      param = paramAttr.getInt();
    }

    bool verbose = getVerboseAttr().hasValue();

    // 步骤 3: 对每个目标应用转换
    SmallVector<Operation *> transformedOps;
    transformedOps.reserve(targets.size());

    for (Operation *target : targets) {
      if (!isValidTarget(target)) {
        return emitDefaultSilenceableFailure(target);
      }

      FailureOr<Operation *> result = applyMyTransform(
          rewriter, target, param, verbose);

      if (failed(result)) {
        if (result.error.isSilenceable()) {
          return result.error;
        } else {
          return emitDefiniteFailure()
                 << "internal error during transformation";
        }
      }

      transformedOps.push_back(*result);
    }

    // 步骤 4: 设置结果
    results.set(cast<OpResult>(getResult()), transformedOps);

    return DiagnosedSilenceableFailure::success();
  }
};

} // namespace
```

### 9.5 步骤 4：注册扩展

```cpp
// MyTransformDialectExtension.cpp
#include "MyTransformOps.h"

using namespace mlir;
using namespace my::transform;

// 注册扩展到 DialectRegistry
void registerMyTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<
      MyTransformDialectExtension
  >();
}
```

### 9.6 步骤 5：测试扩展

```cpp
// unittests/MyTransformOpsTest.cpp

class MyTransformTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<transform::TransformDialect>();
    context.loadDialect<MyDialect>();

    DialectRegistry registry;
    my::transform::registerMyTransformDialectExtension(registry);
    context.appendDialectRegistry(registry);
  }

  MLIRContext context;
};

TEST_F(MyTransformTest, BasicTransform) {
  // 构造测试 IR
  Builder builder(&context);
  auto moduleOp = builder.create<ModuleOp>(builder.getUnknownLoc());

  // 构造 Transform IR
  auto transformModule = parseTransformModule(R"(
    transform.sequence {
    ^bb0(%root: !transform.any_op):
      %ops = transform.my_custom %root { param = 64 : i64 }
      transform.yield %ops : !transform.any_op
    }
  )");

  // 应用 Transform
  TransformOptions options;
  auto result = applyTransformNamedSequence(
      moduleOp, entryPoint, transformModule, options);

  EXPECT_TRUE(succeeded(result));
}
```

### 9.7 最佳实践

#### 9.7.1 编写 Transform 序列

**DO（推荐做法）：**

```mlir
// 使用 named_sequence 组织代码
transform.named_sequence @optimize_op(%arg: !transform.any_op) {
  %1 = transform.tile %arg [32]
  %2 = transform.vectorize %1
  transform.yield %2
}

// 使用 include 复用
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"my.op"} in %root
  %result = transform.include @optimize_op(%ops)
  transform.yield %result
}

// 添加错误处理
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"my.op"} in %root
  %result = transform.apply_patterns to %ops { ... }
  transform.yield %result
}
```

**DON'T（不推荐做法）：**

```mlir
// 过长的内联序列
transform.sequence {
^bb0(%root: !transform.any_op):
  %1 = transform.step1 %root
  %2 = transform.step2 %1
  // ... 50 多行 ...
  %50 = transform.step50 %49
}

// 重复代码
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops1 = transform.match.ops{"op1"} in %root
  %tiled1 = transform.tile %ops1 [32]
  %vect1 = transform.vectorize %tiled1

  %ops2 = transform.match.ops{"op2"} in %root
  %tiled2 = transform.tile %ops2 [32]  // 重复
  %vect2 = transform.vectorize %tiled2  // 重复
}
```

#### 9.7.2 调试技巧

**技巧 1：使用 print 调试**

```mlir
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root
  transform.print %ops { name = "Matched loops" }

  %tiled = transform.tile %ops [32]
  transform.print %tiled { name = "After tiling" }
}
```

**技巧 2：使用 verify 确保正确性**

```mlir
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root

  // 转换前验证
  transform.verify %ops { name = "Before transform" }

  %tiled = transform.tile %ops [32]

  // 转换后验证
  transform.verify %tiled { name = "After tiling" }
}
```

#### 9.7.3 性能考虑

**考虑 1：减少 Handle 查找**

```mlir
// 不推荐：重复查找
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops1 = transform.match.ops{"scf.for"} in %root
  // 使用 %ops1
  %ops2 = transform.match.ops{"scf.for"} in %root  // 重复查找
}

// 推荐：复用 Handle
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root
  // 使用 %ops
}
```

**考虑 2：批量操作**

```mlir
// 推荐：一次处理所有操作
transform.sequence {
^bb0(%root: !transform.any_op):
  %all_ops = transform.match.ops{"scf.for"} in %root
  transform.tile %all_ops [32]
}
```

---

## 10. 实战案例

### 10.1 案例1：矩阵乘法优化（tile + vectorize）

**目标：** 优化矩阵乘法运算，提高缓存局部性和并行度。

```mlir
// ============================================================
// 初始 Payload IR
// ============================================================
module {
  func.func @matmul(%A: tensor<1024x1024xf32>,
                    %B: tensor<1024x1024xf32>,
                    %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = linalg.matmul ins(%A, %B: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                         outs(%C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}

// ============================================================
// Transform IR：完整的优化序列
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // --------------------------------------------------------
  // 步骤 1: 匹配所有 matmul 操作
  // --------------------------------------------------------
  %matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印找到的操作（调试）
  transform.print %matmuls { name = "Found matmuls" }

  // --------------------------------------------------------
  // 步骤 2: 应用多层级分块
  // WHY: 提高缓存局部性，减少内存访问延迟
  // --------------------------------------------------------
  // 第一层分块：较大块（L2 缓存友好）
  %tiled_l1, %loops_l1 = transform.structured.tile %matmuls [64, 64, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // 打印第一层分块结果
  transform.print %tiled_l1 { name = "After L1 tiling" }

  // 第二层分块：较小块（L1 缓存友好）
  %tiled_l2, %loops_l2 = transform.structured.tile %tiled_l1 [8, 8, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // --------------------------------------------------------
  // 步骤 3: 向量化最内层循环
  // WHY: 利用 SIMD 指令，提高并行度
  // --------------------------------------------------------
  transform.structured.vectorize %tiled_l2 vector_sizes [4, 8]
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 4: 应用公共子表达式消除
  // WHY: 减少冗余计算
  // --------------------------------------------------------
  transform.apply_cse %root : !transform.any_op

  // --------------------------------------------------------
  // 步骤 5: 验证最终 IR
  // --------------------------------------------------------
  transform.verify %root : !transform.any_op

  transform.yield
}
```

**优化效果：**
- **缓存局部性**：通过多层级分块，提高数据重用
- **向量化**：利用 SIMD 指令，提高计算吞吐量
- **代码简化**：CSE 消除冗余计算

### 10.2 案例2：循环嵌套优化（多层级tile）

**目标：** 优化深层嵌套循环结构。

```mlir
// ============================================================
// 初始 Payload IR：三层嵌套循环
// ============================================================
module {
  func.func @nested_loops(%arg0: tensor<1024x1024x1024xf32>)
      -> tensor<1024x1024x1024xf32> {
    %0 = tensor.empty() : tensor<1024x1024x1024xf32>
    scf.for %i = 0 to 1024 {
      scf.for %j = 0 to 1024 {
        scf.for %k = 0 to 1024 {
          %1 = tensor.extract %arg0[%i, %j, %k] : tensor<1024x1024x1024xf32>
          %2 = arith.addf %1, %1 : f32
          %3 = tensor.insert %2 into %0[%i, %j, %k] : tensor<1024x1024x1024xf32>
        }
      }
    }
    return %0 : tensor<1024x1024x1024xf32>
  }
}

// ============================================================
// Transform IR：多层级循环优化
// ============================================================
transform.named_sequence @optimize_nested_loops(%root: !transform.any_op) {
  // --------------------------------------------------------
  // 步骤 1: 匹配所有 scf.for 循环
  // --------------------------------------------------------
  %all_loops = transform.match ops {["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 2: 获取最外层循环
  // --------------------------------------------------------
  %outer_loops = transform.loop.get_outermost %all_loops
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 3: 应用多层级循环分块
  // WHY: 每一层对应不同的缓存层级
  // --------------------------------------------------------

  // L3 缓存层：大分块
  %tiled_l3, %loops_l3 = transform.loop.tile %outer_loops [256, 256, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // L2 缓存层：中等分块
  %tiled_l2, %loops_l2 = transform.loop.tile %tiled_l3 [64, 64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // L1 缓存层：小分块
  %tiled_l1, %loops_l1 = transform.loop.tile %tiled_l2 [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // --------------------------------------------------------
  // 步骤 4: 循环展开（最内层）
  // WHY: 减少循环控制开销
  // --------------------------------------------------------
  %innermost = transform.loop.get_innermost %loops_l1
      : (!transform.any_op) -> !transform.any_op

  transform.loop.unroll %innermost { factor = 4 }
      : !transform.any_op

  // --------------------------------------------------------
  // 步骤 5: 向量化
  // --------------------------------------------------------
  transform.loop.vectorize %tiled_l1 vector_sizes [4, 4]
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 6: 应用 LICM（循环不变代码外提）
  // --------------------------------------------------------
  transform.apply_licm %root : !transform.any_op

  transform.yield %root : !transform.any_op
}

// ============================================================
// 主序列：应用优化
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %result = transform.include @optimize_nested_loops(%root)
      : (!transform.any_op) -> !transform.any_op

  transform.yield %result : !transform.any_op
}
```

**优化策略说明：**

| 层级 | 分块大小 | 目标缓存 | WHY |
|------|---------|---------|-----|
| L1 | 256 | L3 缓存 | 最大化 L3 缓存利用率 |
| L2 | 64 | L2 缓存 | 适应 L2 缓存大小 |
| L3 | 16 | L1 缓存 | 适应 L1 缓存大小 |
| 展开 | 4 | 寄存器 | 减少分支开销 |

### 10.3 案例3：GPU映射案例

**目标：** 将计算映射到 GPU 执行。

```mlir
// ============================================================
// 初始 Payload IR：简单的并行计算
// ============================================================
module {
  func.func @vector_add(%a: tensor<1024x1024xf32>,
                        %b: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = scf.for %i = 0 to 1024 iter_args(%acc = %0) -> tensor<1024x1024xf32> {
      %2 = scf.for %j = 0 to 1024 iter_args(%inner_acc = %acc) -> tensor<1024x1024xf32> {
        %3 = tensor.extract %a[%i, %j] : tensor<1024x1024xf32>
        %4 = tensor.extract %b[%i, %j] : tensor<1024x1024xf32>
        %5 = arith.addf %3, %4 : f32
        %6 = tensor.insert %5 into %inner_acc[%i, %j] : tensor<1024x1024xf32>
        scf.yield %6 : tensor<1024x1024xf32>
      }
      scf.yield %2 : tensor<1024x1024xf32>
    }
    return %1 : tensor<1024x1024xf32>
  }
}

// ============================================================
// Transform IR：GPU 映射序列
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // --------------------------------------------------------
  // 步骤 1: 匹配目标函数
  // --------------------------------------------------------
  %funcs = transform.match ops {["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 2: 匹配循环操作
  // --------------------------------------------------------
  %loops = transform.match ops {["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 3: GPU 映射策略
  // WHY: 将循环映射到 GPU 线程层级
  // --------------------------------------------------------

  // 方案 A：使用 forall 并行构造
  %result = transform.alternatives %loops : !transform.any_op {
  ^bb0(%arg0: !transform.any_op):
    // 尝试转换为 forall + GPU 映射
    %forall = transform.loop.to_forall %arg0
        : (!transform.any_op) -> !transform.any_op

    %gpu = transform.gpu.map %forall
        { grid_dims = [32, 32], block_dims = [16, 16] }
        : (!transform.any_op) -> !transform.any_op

    transform.yield %gpu : !transform.any_op
  }, {
  ^bb0(%arg0: !transform.any_op):
    // 方案 B：直接 GPU 映射（回退）
    %gpu = transform.gpu.launch %root
        { blocks = [32, 32, 1], threads = [16, 16, 1] }
        : (!transform.any_op) -> !transform.any_op

    transform.yield %gpu : !transform.any_op
  }

  // --------------------------------------------------------
  // 步骤 4: 向量化（使用 GPU 向量宽度）
  // --------------------------------------------------------
  %vectors = transform.gpu.vectorize %result vector_size = 128
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 5: 内存优化
  // --------------------------------------------------------
  // 共享内存优化
  %shared = transform.gpu.use_shared_memory %vectors
      { buffer_size = 4096 }
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 6: 验证 GPU IR
  // --------------------------------------------------------
  transform.verify %root : !transform.any_op

  transform.print %root { name = "Final GPU IR" }

  transform.yield %root : !transform.any_op
}
```

**GPU 映射策略：**

| GPU 层级 | 映射目标 | 线程数 |
|---------|---------|--------|
| Grid | 整个计算域 | 32 x 32 blocks |
| Block | 单个 block 内 | 16 x 16 threads |
| Vector | SIMD 宽度 | 128 |

---

## 11. 调试与排错

### 11.1 常见错误类型及解决方案

#### 11.1.1 Handle 类型不匹配

**错误示例：**
```
error: 'transform.loop.tile' op operand type mismatch:
  expected '!transform.op<"scf.for">', got '!transform.any_op'
```

**原因：** 操作期望特定类型的 Handle，但提供了通用类型。

**解决方案：**
```mlir
// 不正确
%loops = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.any_op
transform.loop.tile %loops [32]  // 错误：类型不匹配

// 正确：先进行类型转换
%loops = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.any_op
%typed_loops = transform.cast %loops to !transform.op<"scf.for">
    : (!transform.any_op) -> !transform.op<"scf.for">
transform.loop.tile %typed_loops [32]
```

#### 11.1.2 Handle 被重复消费

**错误示例：**
```
error: handle has already been consumed
```

**原因：** Handle 被消费后仍被使用。

**解决方案：**
```mlir
// 不正确
%handle = transform.match ops {["scf.for"]} in %root
%tiled1 = transform.loop.tile %handle [32]
%tiled2 = transform.loop.tile %handle [32]  // 错误：已被消费

// 正确：克隆 Handle
%handle = transform.match ops {["scf.for"]} in %root
%handle1, %handle2 = transform.split_handle %handle
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
%tiled1 = transform.loop.tile %handle1 [32]
%tiled2 = transform.loop.tile %handle2 [32]
```

#### 11.1.3 Transform 操作失败

**错误示例：**
```
error: transform failed: silenceable failure at location
```

**原因：** 转换前置条件不满足。

**解决方案：**
```mlir
// 添加错误处理
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root

  // 尝试转换
  %result = transform.loop.tile %ops [32]
      : (!transform.any_op) -> !transform.any_op
      or {
        // 回退方案
        transform.yield %ops : !transform.any_op
      }

  transform.yield %result : !transform.any_op
}
```

### 11.2 调试工具使用

#### 11.2.1 使用 print 调试

```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 调试：打印每步中间结果
  %ops1 = transform.match ops {["scf.for"]} in %root
  transform.print %ops1 { name = "Step 1: Matched loops" }

  %ops2 = transform.loop.tile %ops1 [32]
  transform.print %ops2 { name = "Step 2: After tiling" }

  %ops3 = transform.loop.vectorize %ops2
  transform.print %ops3 { name = "Step 3: After vectorize" }

  transform.yield
}
```

#### 11.2.2 使用 verify 检查

```mlir
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 在关键点验证 IR
  %ops = transform.match ops {["scf.for"]} in %root

  // 转换前验证
  transform.verify %ops { name = "Before transformation" }

  // 应用转换
  %tiled = transform.loop.tile %ops [32]

  // 转换后验证
  transform.verify %tiled { name = "After transformation" }

  transform.yield
}
```

#### 11.2.3 使用 mlir-transform-opt 工具

```bash
# 应用 Transform 并打印结果
mlir-opt input.mlir \
  --transform-interpreter \
  --transform-spec-library=transform.mlir

# 调试模式
mlir-opt input.mlir \
  --transform-interpreter=debug \
  --transform-spec-library=transform.mlir

# 只验证 Transform IR（不执行）
mlir-opt transform.mlir --verify-diagnostics
```

### 11.3 问题诊断流程

```
┌─────────────────────────────────────────────────────────────┐
│                    问题诊断流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 确认问题类型                                            │
│     ├── 编译错误？→ 检查类型和语法                           │
│     ├── 运行时错误？→ 检查 Handle 映射                       │
│     └── 转换失败？→ 检查前置条件                             │
│                                                             │
│  2. 收集信息                                                │
│     ├── 添加 print 语句                                      │
│     ├── 添加 verify 检查                                     │
│     └── 启用调试输出                                        │
│                                                             │
│  3. 定位问题                                                │
│     ├── 哪个 Transform 操作失败？                            │
│     ├── 哪个 Payload 操作导致失败？                          │
│     └── Handle 指向正确的操作吗？                           │
│                                                             │
│  4. 解决问题                                                │
│     ├── 修复类型不匹配                                       │
│     ├── 添加条件检查                                         │
│     └── 使用 alternatives 提供回退                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. 性能与最佳实践

### 12.1 性能考虑

#### 12.1.1 减少不必要的 Handle 查找

```mlir
// 不推荐：重复查找
%ops = transform.match ops {["scf.for"]} in %root
%tiled1 = transform.loop.tile %ops [32]

%ops2 = transform.match ops {["scf.for"]} in %root  // 重复
%tiled2 = transform.loop.tile %ops2 [16]

// 推荐：复用 Handle
%ops = transform.match ops {["scf.for"]} in %root
%h1, %h2 = transform.split_handle %ops
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
%tiled1 = transform.loop.tile %h1 [32]
%tiled2 = transform.loop.tile %h2 [16]
```

#### 12.1.2 批量操作优于单个操作

```mlir
// 推荐：批量处理
%all_ops = transform.match ops {["linalg.matmul"]} in %root
transform.structured.tile %all_ops [64, 64, 16]
transform.structured.vectorize %all_ops
```

#### 12.1.3 使用 named_sequence 提高复用性

```mlir
// 定义可复用的优化序列
transform.named_sequence @optimize_linalg_op(%op: !transform.any_op) {
  %tiled = transform.structured.tile %op [32, 32]
  transform.structured.vectorize %tiled
  transform.apply_cse %tiled
  transform.yield
}

// 应用
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.structured.match ops {["linalg.matmul"]} in %root
  %result = transform.foreach %ops
      iter_args(%op: !transform.any_op)
      -> (!transform.any_op) {
  ^bb0(%op: !transform.any_op):
    %optimized = transform.include @optimize_linalg_op(%op)
        : (!transform.any_op) -> !transform.any_op
    transform.yield %optimized : !transform.any_op
  }
  transform.yield
}
```

### 12.2 最佳实践

#### 12.2.1 错误处理策略

```mlir
// 策略 1：使用 suppress 模式处理可选转换
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  // 尝试优化，失败时继续
  %try = transform.loop.unroll %ops { factor = 4 }
      or transform.yield %ops
}

// 策略 2：使用 alternatives 提供回退
%result = transform.alternatives %ops {
^bb0(%arg0):
  // 乐观策略
  %fast = transform.fast_path %arg0
  transform.yield %fast
}, {
^bb0(%arg0):
  // 保守回退
  %safe = transform.safe_path %arg0
  transform.yield %safe
}
```

#### 12.2.2 模块化 Transform 序列

```mlir
// 将复杂转换分解为小模块
module {
  // 模块 1：循环优化
  transform.named_sequence @optimize_loops(%root: !transform.any_op) {
    // ...
  }

  // 模块 2：向量化
  transform.named_sequence @vectorize(%root: !transform.any_op) {
    // ...
  }

  // 模块 3：内存优化
  transform.named_sequence @optimize_memory(%root: !transform.any_op) {
    // ...
  }

  // 主序列：组合模块
  transform.named_sequence @full_optimization(%root: !transform.any_op) {
    %r1 = transform.include @optimize_loops(%root)
    %r2 = transform.include @vectorize(%r1)
    %r3 = transform.include @optimize_memory(%r2)
    transform.yield %r3
  }
}
```

#### 12.2.3 文档和注释

```mlir
// ============================================================
// 优化序列：矩阵乘法
// 目标：提高缓存局部性，启用向量化
// ============================================================
transform.named_sequence @optimize_matmul(%op: !transform.any_op) {
  // 步骤 1: 分块 L2 缓存
  // WHY: 提高数据重用，减少内存访问
  %tiled_l2 = transform.structured.tile %op [64, 64, 16]

  // 步骤 2: 分块 L1 缓存
  %tiled_l1 = transform.structured.tile %tiled_l2 [8, 8, 4]

  // 步骤 3: 向量化最内层
  // WHY: 利用 SIMD 指令
  transform.structured.vectorize %tiled_l1 vector_sizes [4, 8]

  transform.yield
}
```

---

## 13. 常见问题FAQ

### Q1: Handle类型不匹配怎么办？

**问题：** 收到 "operand type mismatch" 错误。

**解决方案：**
```mlir
// 方案 1：使用 cast 转换类型
%any_handle = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.any_op
%typed_handle = transform.cast %any_handle to !transform.op<"scf.for">
    : (!transform.any_op) -> !transform.op<"scf.for">

// 方案 2：使用正确的匹配类型
%typed_handle = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.op<"scf.for">
```

### Q2: Transform操作执行失败如何调试？

**问题：** 转换失败但不知道原因。

**调试步骤：**
```mlir
// 1. 添加 print 调试
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root
  transform.print %ops { name = "Before transform" }

  %tiled = transform.loop.tile %ops [32]
  transform.print %tiled { name = "After transform" }
}

// 2. 添加 verify 检查
transform.verify %ops { name = "Verification failed" }

// 3. 使用 suppress 模式继续执行
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  // 即使某些转换失败，继续执行
}
```

### Q3: 如何选择Transform还是Pass？

**对比：**

| 特性 | Transform 方言 | 传统 Pass |
|------|---------------|----------|
| **粒度** | 精细控制 | 粗粒度 |
| **组合性** | 灵活组合 | 固定顺序 |
| **调试性** | 可观察 | 难以定位 |
| **性能** | 略有开销 | 高效 |
| **适用场景** | 实验性/研究 | 生产环境 |

**选择建议：**
- **使用 Transform**：需要精细控制、实验新优化、条件化转换
- **使用 Pass**：已知优化序列、性能关键、生产环境

### Q4: 扩展加载失败的常见原因

**问题：** 自定义 Transform 操作无法使用。

**常见原因：**
```cpp
// 1. 忘记注册扩展
// 错误：
// DialectRegistry registry;
// context.loadDialect<TransformDialect>();

// 正确：
DialectRegistry registry;
my::transform::registerMyTransformDialectExtension(registry);
context.appendDialectRegistry(registry);

// 2. 依赖方言未加载
// 确保声明所有依赖方言
void MyExtension::init() {
  declareDependentDialect<LinalgDialect>();
  declareDependentDialect<SCFDialect>();
  // ...
}

// 3. 操作名称拼写错误
// 检查 TableGen 定义与使用是否一致
def MyOp : TransformDialectOp<"my_op"> { ... }
// 使用: transform.my_op (不是 transform.myOp)
```

---

## 14. 参考资料

### 14.1 官方文档

- [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)

### 14.2 源代码位置

| 组件 | 路径 |
|------|------|
| 核心方言定义 | `mlir/include/mlir/Dialect/Transform/IR/` |
| 核心实现 | `mlir/lib/Dialect/Transform/IR/` |
| Linalg Transform | `mlir/lib/Dialect/Linalg/TransformOps/` |
| SCF Transform | `mlir/lib/Dialect/SCF/TransformOps/` |
| 接口定义 | `mlir/include/mlir/Dialect/Transform/Interfaces/` |

### 14.3 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| Payload IR | Payload IR | 被转换的目标 IR |
| Transform IR | Transform IR | 控制转换逻辑的 IR |
| Handle | Handle | Transform IR 中指向 Payload IR 对象的引用 |
| Silenceable Failure | Silenceable Failure | 可恢复失败 |
| Definite Failure | Definite Failure | 不可恢复失败 |

### 14.4 延伸阅读

- [MLIR 编写转换指南](https://mlir.llvm.org/docs/Transformations/)
- [Linalg 结构化操作](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [SCF 结构化控制流](https://mlir.llvm.org/docs/Dialects/SCF/)
