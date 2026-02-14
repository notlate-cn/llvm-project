# MLIR Transform 方言完全掌握指南

> 基于 MLIR 官方文档 [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/#overview) 和源代码 `mlir/lib/Dialect/Transform/`、`mlir/lib/Dialect/Linalg/TransformOps/` 深度分析生成。

---

## 理解验证状态

| 核心概念 | 自我解释 | 理解"为什么" | 应用迁移 | 状态 |
|---------|---------|-------------|---------|------|
| Payload IR 与 Transform IR 分离 | ✅ | ✅ | ✅ | 已理解 |
| Handle 类型系统 | ✅ | ✅ | ✅ | 已理解 |
| TransformOpInterface | ✅ | ✅ | ✅ | 已理解 |
| TransformState 映射机制 | ✅ | ✅ | ✅ | 已理解 |
| 扩展机制 (TransformDialectExtension) | ✅ | ✅ | ✅ | 已理解 |
| 失败处理模式 | ✅ | ✅ | ✅ | 已理解 |
| Handle 失效规则 | ✅ | ✅ | ✅ | 已理解 |

---

## 目录

1. [快速概览](#1-快速概览)
2. [背景与动机](#2-背景与动机)
3. [核心概念](#3-核心概念)
4. [类型系统](#4-类型系统)
5. [源码实现：TransformState](#5-源码实现transformstate)
6. [源码实现：TransformDialectExtension](#6-源码实现transformdialectextension)
7. [扩展开发完整教程](#7-扩展开发完整教程)
8. [核心操作详解](#8-核心操作详解)
9. [执行模型](#9-执行模型)
10. [Linalg Transform 操作](#10-linalg-transform-操作)
11. [实战案例](#11-实战案例)
12. [性能与调试](#12-性能与调试)
13. [参考资料](#13-参考资料)

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

- ✅ **精细控制**：可以针对单个或一组操作应用特定转换
- ✅ **可编程组合**：使用 IR 本身表达转换序列，灵活组合
- ✅ **类型安全**：通过 Handle 类型系统确保转换前置条件
- ✅ **可扩展**：通过扩展机制添加方言特定的转换操作

**WHY 不用其他方案：**

| 方案 | WHY 不选 |
|------|----------|
| Pass Pipeline | 粒度太粗，无法针对特定操作 |
| C++ 代码 | 难以组合，缺乏可重用性 |
| 脚本语言 | 与 MLIR IR 集成不紧密 |

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
   ```mlir
   %0 = transform.match.ops{"scf.for"} in %root
   // %0 可能关联 10 个不同的 scf.for 操作
   ```

2. **类型约束**：Handle 类型编码了关联对象的属性
   ```mlir
   %0: !transform.op<"scf.for">  // 只能指向 scf.for 操作
   ```

3. **批量执行**：大多数 Transform 操作对 Handle 关联的所有对象执行
   ```mlir
   transform.loop.unroll %0 { factor = 4 }
   // 切分 %0 关联的所有循环
   ```

### 3.3 Transform 类型接口详解

Transform 方言定义了多个类型接口，用于确保类型安全和运行时验证。

#### 3.3.1 TransformTypeInterfaceBase - 基础类型接口

这是所有 Transform 类型接口的基类，定义了通用的验证方法。

```cpp
// TransformInterfaces.td:120-144
template <typename DerivedTy, typename PayloadTy>
class TransformTypeInterfaceBase : public TypeInterface<DerivedTy> {
public:
  // 声明：核心验证方法
  // 作用：检查 Payload 对象是否满足类型约束
  virtual DiagnosedSilenceableFailure checkPayload(
      Location loc,
      ArrayRef<PayloadTy> payload) = 0;

  // 辅助方法：创建 silenceable 错误
  DiagnosedSilenceableFailure emitSilenceableError(Location loc) const {
    Diagnostic diag(loc, DiagnosticSeverity::Error);
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
};
```

**WHY 设计基类：**
- **代码复用**：所有类型接口共享验证逻辑
- **一致性**：统一的错误报告机制
- **扩展性**：新的类型接口可以继承基类功能

#### 3.3.2 TransformHandleTypeInterface - 操作句柄接口

**作用：** 指向 Payload IR 操作的 Handle 类型必须实现此接口。

```cpp
// TransformInterfaces.td:146-155
def TransformHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformHandleTypeInterface",
                                 "::mlir::Operation *"> {
  let description = [{
    Types that can be used for the Transform dialect operation handle values.
    Such types define the properties of Payload IR operations associated with
    the handle. A user of such a handle can assume that these properties have
    been verified for any Payload IR operation associated with it.
  }];
}
```

**方法签名：**
```cpp
// 对 Payload 操作进行检查
virtual DiagnosedSilenceableFailure checkPayload(
    Location loc,
    ArrayRef<Operation *> payload) = 0;
```

**实现示例（OperationType）：**
```cpp
// OperationType 的 checkPayload 实现
DiagnosedSilenceableFailure OperationType::checkPayload(
    Location loc,
    ArrayRef<Operation *> payload) {

  // 步骤 1: 检查每个操作
  for (Operation *op : payload) {
    // 步骤 1.1: 验证操作名称
    // 此时：getOperationName() = "scf.for" (类型参数中指定的)
    //       op->getName() = 实际的操作名称
    if (op->getName().getStringRef() != getOperationName()) {
      // 步骤 1.2: 操作名称不匹配
      // 示例：getOperationName() = "scf.for"
      //       op->getName() = "linalg.matmul"
      return emitSilenceableError(loc)
             << "expected '" << getOperationName() << "' operation, "
             << "but found '" << op->getName() << "'";
      // WHY 返回 silenceable：
      //   这是类型不匹配错误，不是致命错误
      //   调用者可以选择处理或跳过
    }
  }

  // 步骤 2: 所有操作都通过验证
  return DiagnosedSilenceableFailure::success();
}
```

**执行流示例：**

**场景 A：验证成功**
```cpp
// Transform IR
%loops: !transform.op<"scf.for"> = ...

// Payload 操作
ArrayRef<Operation *> payload = {scf_for_op1, scf_for_op2};

// 执行 checkPayload
for (Operation *op : payload) {
  // scf_for_op1->getName() == "scf.for" → 通过
  // scf_for_op2->getName() == "scf.for" → 通过
}
// 返回：success()
```

**场景 B：验证失败**
```cpp
// Transform IR
%loops: !transform.op<"scf.for"> = ...

// Payload 操作（包含错误类型）
ArrayRef<Operation *> payload = {scf_for_op, linalg_matmul_op};

// 执行 checkPayload
// scf_for_op→getName() == "scf.for" → 通过
// linalg_matmul_op->getName() == "linalg.matmul" → 失败！
// 返回：DiagnosedSilenceableFailure
// 错误："expected 'scf.for' operation, but found 'linalg.matmul'"
```

**WHY 设计这个接口：**

| 特性 | 说明 |
|------|------|
| **类型安全** | 确保 Handle 指向的操作满足特定约束 |
| **延迟验证** | 类型约束在 Transform 执行时验证，而非解析时 |
| **灵活性** | 允许更宽松的类型，运行时检查 |
| **扩展性** | 自定义类型可以实现自定义验证逻辑 |

#### 3.3.3 TransformValueHandleTypeInterface - 值句柄接口

**作用：** 指向 Payload IR 值（SSA Value）的 Handle 类型必须实现此接口。

```cpp
// TransformInterfaces.td:168-177
def TransformValueHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformValueHandleTypeInterface",
                                 "::mlir::Value"> {
  let description = [{
    Types that can be used for the Transform dialect handle values pointing to
    Payload IR values. Such types define the properties of Payload IR values
    associated with the handle. Users of such a handle can assume that these
    properties have been verified for any Payload IR value associated with it.
  }];
}
```

**方法签名：**
```cpp
// 对 Payload 值进行检查
virtual DiagnosedSilenceableFailure checkPayload(
    Location loc,
    ArrayRef<Value> payload) = 0;
```

**WHY 需要值句柄接口：**

| 需求 | 说明 | 示例 |
|------|------|------|
| **细粒度控制** | 某些转换需要操作特定的值 | buffer 化时需要追踪 tensor → memref 的转换 |
| **结果追踪** | 跟踪操作产生的新值 | 获取操作的返回值并进一步处理 |
| **类型约束** | 确保值的类型满足转换要求 | 确保值是 tensor 类型才能进行 buffer 化 |

**使用场景示例：**
```mlir
// 场景：buffer 化转换
%tensor_value = transform.get_result %op { index = 0 }
                 : (!transform.any_op) -> !transform.any_value
// 此时：%tensor_value 指向操作的第一个结果值

%memref_value = transform.bufferize %tensor_value
                 : (!transform.any_value) -> !transform.any_value
// buffer 化后，值从 tensor 变为 memref
```

#### 3.3.4 TransformParamTypeInterface - 参数句柄接口

**作用：** 指向编译时参数（Attribute）的 Handle 类型必须实现此接口。

```cpp
// TransformInterfaces.td:157-166
def TransformParamTypeInterface
    : TransformTypeInterfaceBase<"TransformParamTypeInterface",
                                 "::mlir::Attribute"> {
  let description = [{
    Types that can be used for the Transform dialect parameter values. Such types
    define the structure of the parameters associated with the value, e.g., their
    underlying type. A user of such a handle can assume that the parameter has
    been verified.
  }];
}
```

**方法签名：**
```cpp
// 对参数进行检查
virtual DiagnosedSilenceableFailure checkPayload(
    Location loc,
    ArrayRef<Attribute> payload) = 0;
```

**实现示例（ParamType）：**
```cpp
// ParamType<!transform.param<i32>> 的 checkPayload 实现
DiagnosedSilenceableFailure ParamType::checkPayload(
    Location loc,
    ArrayRef<Attribute> payload) {

  // 步骤 1: 检查参数类型
  Type expectedType = getType();  // 例如：i32

  for (Attribute attr : payload) {
    // 步骤 1.1: 参数必须是整数属性
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      // 步骤 1.2: 检查整数类型
      if (intAttr.getType() != expectedType) {
        return emitSilenceableError(loc)
               << "expected parameter of type " << expectedType
               << ", but got " << intAttr.getType();
      }
    } else {
      // 步骤 1.3: 不是整数属性
      return emitSilenceableError(loc)
             << "expected integer parameter, but got "
             << attr.getType();
    }
  }

  return DiagnosedSilenceableFailure::success();
}
```

**WHY 需要参数句柄接口：**

| 需求 | 说明 | 示例 |
|------|------|------|
| **运行时参数** | 支持在 Transform 执行时传递参数 | 切分大小、向量宽度等 |
| **类型安全** | 确保参数类型正确 | 防止将字符串传给期望整数的转换 |
| **灵活性** | 允许动态配置转换行为 | 根据运行时信息选择参数 |

**使用场景示例：**
```mlir
// 场景：动态配置切分大小
%tile_size = transform.param.constant 64 : i32
               -> !transform.param<i32>

transform.tile %op tile_sizes = [%tile_size]
// 切分大小由参数决定，而非硬编码
```

### 3.4 TransformOpInterface - 操作接口

**概念：** 所有 Transform 操作必须实现的核心接口。

```cpp
// TransformInterfaces.td:18-70
def TransformOpInterface : OpInterface<"TransformOpInterface"> {
  let description = [{
    This interface is to be implemented by operations that identify
    transformations to be performed on other operations.
  }];
}
```

#### 3.4.1 核心方法：apply

```cpp
// 应用转换
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
  // 此时：getTarget() 是操作的输入 Handle
  //       targets 是 Handle 关联的所有 Payload 操作
  //       例如：targets = [op1, op2, op3]

  if (targets.empty()) {
    // 步骤 1.1: 没有目标操作
    return emitSilenceableError() << "no operations to transform";
  }

  // 步骤 2: 对每个目标应用转换
  SmallVector<Operation *> transformedOps;
  for (Operation *target : targets) {
    // 步骤 2.1: 执行具体转换逻辑
    FailureOr<Operation *> result = applyMyTransform(rewriter, target);

    if (failed(result)) {
      // 步骤 2.2: 转换失败
      return emitDefaultSilenceableFailure(target);
      // WHY 使用默认失败：
      //   提供标准的错误消息和位置信息
    }

    transformedOps.push_back(*result);
    // 此时：transformedOps = [new_op1, new_op2, new_op3]
  }

  // 步骤 3: 设置结果
  results.set(cast<OpResult>(getResult()), transformedOps);
  // WHY 使用 results.set：
  //   自动建立结果 Handle 到 Payload 操作的映射
  //   调用者可以直接使用返回的 Handle

  return DiagnosedSilenceableFailure::success();
}
```

#### 3.4.2 辅助方法：allowsRepeatedHandleOperands

```cpp
// 指示是否允许操作数关联相同的 Payload 操作
virtual bool allowsRepeatedHandleOperands() const;
```

**WHY 需要这个方法：**

```mlir
// 场景 A：不允许重复关联（默认）
%0 = transform.match.ops{"scf.for"} in %root
%1 = transform.match.ops{"scf.for"} in %root
// 如果 %0 和 %1 关联到同一个 scf.for 操作
// 某些转换可能会失败或产生意外结果

// 场景 B：允许重复关联
// 操作声明 allowsRepeatedHandleOperands = true
// 即使 %0 和 %1 指向相同操作，也能正常处理
```

**默认实现：**
```cpp
// 默认返回 false
bool allowsRepeatedHandleOperands() const { return false; }
// WHY 默认不允许：
//   - 防止意外修改同一个操作多次
//   - 大多数转换期望操作数之间没有重叠
```

#### 3.4.3 错误报告辅助方法

```cpp
// 创建 silenceable 错误
DiagnosedSilenceableFailure emitSilenceableError(
    const ::llvm::Twine &message = {}) {
  return ::mlir::emitSilenceableFailure($_op, message);
  // $_op 是 TableGen 生成的特殊变量，指向当前操作
}

// 创建 definite 错误
DiagnosedDefiniteFailure emitDefiniteFailure(
    const ::llvm::Twine &message = {}) {
  return ::mlir::emitDefiniteFailure($_op, message);
}

// 创建默认失败消息
DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
    ::mlir::Operation *target) {
  auto diag = ::mlir::emitDefiniteFailure($_op, "failed to apply");
  diag.attachNote(target->getLoc()) << "attempted to apply to this op";
  return diag;
}
```

**使用示例：**
```cpp
DiagnosedSilenceableFailure MyTransformOp::apply(...) {
  // 场景 1: 使用 silenceable 错误
  if (targets.empty()) {
    return emitSilenceableError()
           << "no operations found";
    // WHY silenceable：
    //   空操作集是可预期的，不是致命错误
  }

  // 场景 2: 使用 definite 错误
  if (internalStateCorrupted) {
    return emitDefiniteFailure()
           << "internal state corrupted";
    // WHY definite：
    //   内部状态损坏是致命错误
    //   必须立即停止
  }

  // 场景 3: 使用默认错误
  for (Operation *target : targets) {
    if (failed(applyTransform(target))) {
      return emitDefaultDefiniteFailure(target);
      // WHY 使用默认：
      //   提供一致错误消息
      //   自动附加目标操作位置
    }
  }
}
```

**WHY 需要这个接口：**

| 特性 | 说明 |
|------|------|
| **统一执行模型** | 所有转换通过相同的 `apply` 方法执行 |
| **错误处理** | 返回 `DiagnosedSilenceableFailure` 支持可恢复失败 |
| **状态管理** | 通过 `TransformState` 访问 Payload IR 映射 |
| **类型安全** | 编译时确保操作实现了必要方法 |

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

操作句柄用于指向 Payload IR 中的操作。

#### 4.2.1 AnyOpType

**语法：** `!transform.any_op`

**定义：**
```tablegen
// TransformTypes.td:26-34
def Transform_AnyOpType : TypeDef<Transform_Dialect, "AnyOp",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let description = [{
    Transform IR handle that can be associated with a list of arbitrary
    Payload IR operations.
  }];
  let mnemonic = "any_op";
}
```

**用途：** 指向任意 Payload IR 操作的通用句柄

**使用场景：**
```mlir
// 匹配任意类型的操作
%all_ops = transform.match.ops{"scf.for","arith.addf"} in %root
           : (!transform.any_op) -> !transform.any_op

// 此时：%all_ops 可能关联多种不同类型的操作
// WHY 使用 any_op：当不需要限制操作类型时使用
```

**WHY 设计 AnyOpType：**
- **灵活性**：可以指向任何操作
- **类型安全最小化**：不进行类型约束验证
- **适用场景**：
  - 操作类型未知或多样化
  - 后续转换会进一步筛选
  - 不关心具体操作类型

#### 4.2.2 OperationType

**语法：** `!transform.op<"operation_name">`

**定义：**
```tablegen
// TransformTypes.td:45-56
def Transform_OperationType : TypeDef<Transform_Dialect, "Operation",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let description = [{
    Transform IR handle that can be associated with a list of Payload IR
    operations with the specified operation name.
  }];
  let mnemonic = "op";
  let parameters = (ins
    StringRefParameter<"Name of the allowed payload operation">:$operation_name
  );
  let assemblyFormat = "`<` $operation_name `>`";
}
```

**参数：**
| 参数 | 类型 | 描述 |
|------|------|------|
| `operation_name` | `StringRef` | 允许的 Payload 操作名称 |

**使用场景：**
```mlir
// 只指向 scf.for 操作的句柄
%loops = transform.match.ops{"scf.for"} in %root
         : (!transform.any_op) -> !transform.op<"scf.for">

// 此时：%loops 只能关联 scf.for 操作
// 如果尝试关联其他操作，checkPayload 会失败
```

**WHY 设计 OperationType：**
- **类型安全**：编译时和运行时都验证操作类型
- **操作特化**：某些转换只适用于特定操作
- **文档作用**：清晰表达句柄的预期内容

**执行流示例：**

**场景 A：类型匹配成功**
```cpp
// Transform IR
%loops: !transform.op<"scf.for"> = transform.match.ops{"scf.for"} in %root

// 执行 checkPayload
// Payload 操作：[scf.for %i, scf.for %j]
// 结果：success() → 所有操作都是 scf.for
```

**场景 B：类型不匹配**
```cpp
// Transform IR
%loops: !transform.op<"scf.for"> = transform.match.ops{"scf.for"} in %root

// 执行 checkPayload
// Payload 操作：[scf.for %i, linalg.matmul %matmul]
// 结果：failure() → linalg.matmul 不是 scf.for
// 错误消息："expected 'scf.for' operation, but found 'linalg.matmul'"
```

### 4.3 值句柄类型 (Value Handle Types)

值句柄用于指向 Payload IR 中的 SSA Value。

#### 4.3.1 AnyValueType

**语法：** `!transform.any_value`

**定义：**
```tablegen
// TransformTypes.td:36-43
def Transform_AnyValue : TypeDef<Transform_Dialect, "AnyValue",
    [DeclareTypeInterfaceMethods<TransformValueHandleTypeInterface>]> {
  let description = [{
    Transform IR value that can be associated with a list of Payload IR values.
  }];
  let mnemonic = "any_value";
}
```

**用途：** 指向任意 Payload IR 值（SSA Value）

**使用场景：**
```mlir
// 获取循环的迭代参数
%args = transform.get_iter_args %loop : (!transform.any_op) -> !transform.any_value

// 此时：%args 关联到循环的迭代参数（BlockArgument）
// WHY 使用值句柄：
//   - 某些转换需要操作特定的值（如 buffer、tensor）
//   - 追踪操作产生的新值
//   - 确保值的类型满足转换要求
```

**WHY 需要值句柄：**

| 特性 | 操作句柄 | 值句柄 |
|------|---------|--------|
| 指向对象 | Operation | SSA Value |
| 用途 | 操作转换 | 值追踪/转换 |
| 示例 | `!transform.any_op` | `!transform.any_value` |

**使用示例：**
```mlir
// 场景：追踪 buffer 化的结果
%bufferized = transform.bufferization.one_shot_bufferize %op
              : (!transform.any_op) -> !transform.any_value
// 此时：%bufferized 指向 buffer 化后的 memref 值
```

### 4.4 参数句柄类型 (Parameter Handle Types)

参数句柄用于指向编译时参数（Attribute），允许在 Transform 执行时传递参数。

#### 4.4.1 AnyParamType

**语法：** `!transform.any_param`

**定义：**
```tablegen
// TransformTypes.td:58-66
def Transform_AnyParamType : TypeDef<Transform_Dialect, "AnyParam",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR value that can be associated with a list of parameters
    of any type.
  }];
  let mnemonic = "any_param";
}
```

**用途：** 指向任意类型的参数

**使用场景：**
```mlir
// 从外部获取参数
%param = transform.get_param "tile_size" : !transform.any_param
```

#### 4.4.2 ParamType

**语法：** `!transform.param<Type>`

**定义：**
```tablegen
// TransformTypes.td:68-82
def Transform_ParamType : TypeDef<Transform_Dialect, "Param",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR value that can be associated with the list of parameters
    of the given type. Types are currently limited to integers, but may be
    extended in the future to other types values of which can be contained
    in attributes.
  }];
  let mnemonic = "param";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Underlying type of the parameter">:$type
  );
  let assemblyFormat = "`<` $type `>`";
}
```

**参数：**
| 参数 | 类型 | 描述 |
|------|------|------|
| `type` | `Type` | 参数的底层类型 |

**支持的类型：**
- 整数类型：`i32`, `i64`, `index`
- 未来可能扩展：浮点数、数组等

**使用场景：**
```mlir
// 创建 32 位整数参数
%tile_size = transform.param.constant 64 : i32
             -> !transform.param<i32>

// 使用参数
transform.tile %op tile_sizes = [%tile_size]
```

#### 4.4.3 AffineMapParamType

**语法：** `!transform.affine_map`

**定义：**
```tablegen
// TransformTypes.td:16-24
def Transform_AffineMapParamType : TypeDef<Transform_Dialect, "AffineMapParam",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR parameter value that can be associated with a list of affine
    map attributes.
  }];
  let mnemonic = "affine_map";
}
```

**用途：** 指向 AffineMap 参数

**使用场景：**
```mlir
// 创建 affine map 参数
%map = transform.affine_map.parse "(d0, d1) -> (d0 + d1)"
       : !transform.affine_map
```

#### 4.4.4 TypeParamType

**语法：** `!transform.type`

**定义：**
```tablegen
// TransformTypes.td:84-92
def Transform_TypeParamType : TypeDef<Transform_Dialect, "TypeParam",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Transform IR parameter value that can be associated with a list of type
    attributes.
  }];
  let mnemonic = "type";
}
```

**用途：** 指向 Type 参数

**使用场景：**
```mlir
// 获取元素的类型
%type = transform.get_type %value : !transform.type
```

### 4.5 类型验证机制

每个 Transform 类型都实现了对应的接口，提供 `checkPayload` 方法进行运行时验证。

#### 4.5.1 TransformHandleTypeInterface::checkPayload

```cpp
// 接口定义
virtual DiagnosedSilenceableFailure checkPayload(
    Location loc,
    ArrayRef<Operation *> payload) = 0;
```

**验证逻辑：**

```cpp
// OperationType 的实现示例
DiagnosedSilenceableFailure OperationType::checkPayload(
    Location loc, ArrayRef<Operation *> payload) {

  // 步骤 1: 检查每个操作是否匹配
  for (Operation *op : payload) {
    // 步骤 1.1: 操作名称匹配
    if (op->getName().getStringRef() != getOperationName()) {
      // 此时：getOperationName() = "scf.for"
      //       op->getName() = "linalg.matmul"
      return emitSilenceableError(loc)
             << "expected '" << getOperationName() << "' operation, "
             << "but found '" << op->getName() << "'";
    }
  }

  return DiagnosedSilenceableFailure::success();
}
```

**WHY 需要运行时验证：**

| 验证时机 | 验证内容 | 原因 |
|---------|---------|------|
| 解析时 | 类型语法正确性 | MLIR 类型系统保证 |
| 执行时 | Payload 对象符合类型约束 | 运行时才知道 Payload 对象 |

#### 4.5.2 验证时机

```cpp
// 什么时候调用 checkPayload？
LogicalResult TransformState::setPayloadOps(Value value,
                                            ArrayRef<Operation *> targets) {
  // 步骤 1: 建立映射时验证
  auto iface = llvm::cast<TransformHandleTypeInterface>(value.getType());
  DiagnosedSilenceableFailure result =
      iface.checkPayload(value.getLoc(), targets);

  if (failed(result.checkAndReport()))
    return failure();
  // WHY 在建立映射时验证：
  //   - 确保映射的 Handle 和 Payload 对象类型一致
  //   - 尽早发现错误，避免后续转换失败
}
```

**执行流示例：**

**场景 A：延迟验证的优势**
```mlir
// Transform IR 定义时
%loops: !transform.op<"scf.for"> = transform.match.ops{"scf.for"} in %root
// 此时：只检查类型语法正确，不检查 Payload

// 执行时
// 如果 Payload 中没有 scf.for，checkPayload 才会失败
// WHY 延迟验证：
//   - Transform IR 可以独立于 Payload 定义
//   - 同一个 Transform IR 可以应用于不同的 Payload
```

### 4.6 类型转换

Transform 方言支持在特定条件下进行类型转换。

#### 4.6.1 类型兼容性规则

**向上转换（Widening）：**
```mlir
// 从具体类型到通用类型
%specific: !transform.op<"scf.for"> = ...
%general: !transform.any_op = %specific  // 允许
// WHY 允许：any_op 可以指向任何操作
```

**向下转换（Narrowing）：**
```mlir
// 从通用类型到具体类型
%general: !transform.any_op = ...
// 需要运行时验证才能转换为 !transform.op<"scf.for">
// WHY 需要验证：不是所有操作都是 scf.for
```

#### 4.6.2 类型转换操作

```mlir
// 显式类型转换（如果提供）
%converted = transform.cast %handle : !transform.any_op to !transform.op<"scf.for">
// 场景：
//   %handle 关联的操作必须是 scf.for
//   否则转换失败
```

### 4.7 类型系统设计决策

#### 4.7.1 WHY 使用参数化类型？

**问题：** 为什么 OperationType 需要参数 `operation_name`？

**答案：**
- **类型安全**：编译时就知道句柄指向的操作类型
- **优化机会**：编译器可以基于类型信息优化
- **文档作用**：类型本身就表达了约束

**对比其他方案：**

| 方案 | 优点 | 缺点 | WHY 不选 |
|------|------|------|---------|
| 参数化类型 | 类型安全，自文档 | 需要类型定义 | - |
| 属性约束 | 灵活 | 无编译时检查 | 运行时才发现错误 |
| 纯运行时检查 | 简单 | 无类型安全 | 容易出错 |

#### 4.7.2 WHY 分离三种 Handle 类型？

**设计理由：**

```
操作句柄 → 指向 Operation（可执行单元）
  ↓
值句柄 → 指向 Value（数据流）
  ↓
参数句柄 → 指向 Attribute（编译时常量）
```

**WHY 这样分离：**

1. **语义清晰**：
   - 操作转换：操作句柄
   - 数据追踪：值句柄
   - 配置参数：参数句柄

2. **类型安全**：
   - 混用会导致类型混乱
   - 例如：不能把 Value 当作 Operation 处理

3. **验证分离**：
   - 每种类型有不同的验证规则
   - 操作检查名称，值检查类型，参数检查格式

**如果统一为一种类型：**
```mlir
// 假设只有 !transform.handle
%0 = transform.match.ops{"scf.for"} in %root   // 返回 !transform.handle
%1 = transform.get_result %0                    // 也返回 !transform.handle
// 问题：
// - 无法从类型知道 %0 指向操作还是 %1 指向值
// - 需要运行时检查才能使用
// - 错误信息不够明确
```

---

## 5. 源码实现：TransformState

### 5.1 核心数据结构

TransformState 是 Transform 方言执行的核心状态管理类，负责维护 Transform IR 与 Payload IR 之间的映射关系。

```cpp
// TransformInterfaces.h:173 (简化版)
class TransformState {
public:
  // 构造函数：正常创建 TransformState
  // 此时：region 指向 Transform IR 所在区域
  //       payloadRoot 指向被转换的 Payload IR 根操作
  TransformState(Region *region, Operation *payloadRoot,
                 const RaggedArray<MappedValue> &extraMappings = {},
                 const TransformOptions &options = TransformOptions());

private:
  // 成员变量：映射关系存储
  // WHY 使用 DenseMap：快速查找，O(1) 平均时间复杂度
  // WHY 每个区域独立映射：支持嵌套的 Transform IR 区域（如 transform.sequence 内部）
  DenseMap<Region *, std::unique_ptr<Mappings>> mappings;

  // 区域栈：跟踪当前处理的区域
  // WHY 需要栈：处理嵌套区域时可以回溯到父区域
  SmallVector<RegionScope *> regionStack;

  // 扩展数据：支持用户自定义状态
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;

  // 选项：配置 Transform 执行行为
  TransformOptions options;
};
```

### 5.2 Mappings 结构

```cpp
// TransformInterfaces.h:47-63 (简化版)
struct Mappings {
  // 成员变量：直接映射（Transform IR Value → Payload IR Operation 列表）
  // 说明：一个 Handle 可以关联多个 Payload 操作
  DenseMap<Value, SmallVector<Operation *, 2>> direct;

  // 成员变量：反向映射（Payload IR Operation → Transform IR Value 列表）
  // WHY 需要反向映射：快速查找哪些 Handle 指向某个 Payload 操作
  DenseMap<Operation *, SmallVector<Value>> reverse;

  // 成员变量：值映射（Transform IR Value → Payload IR Value 列表）
  DenseMap<Value, SmallVector<Value>> values;

  // 成员变量：值反向映射（Payload IR Value → Transform IR Value）
  DenseMap<Value, SmallVector<Value>> reverseValues;

  // 成员变量：参数映射（Transform IR Value → 参数列表）
  DenseMap<Value, SmallVector<Param>> params;
};
```

### 5.3 核心方法：setPayloadOps

#### 5.3.1 kTopLevelValue - 根操作标识符

在深入 `setPayloadOps` 之前，先理解 `kTopLevelValue` 这个特殊的标识符。

```cpp
// TransformInterfaces.h:485
static constexpr Value kTopLevelValue = Value();
```

**什么是 kTopLevelValue？**

`kTopLevelValue` 是一个**哨兵值**（sentinel value），用于标识 Payload IR 的**根操作**。

| 特性 | kTopLevelValue | 普通 Handle (如 %0) |
|------|----------------|---------------------|
| **类型** | `Value()` (null) | 有效的 SSA Value |
| **用途** | 标识根操作 | 指向特定操作集 |
| **可重置** | ❌ 不允许 | ✅ 允许 |
| **生命周期** | 整个 Transform 执行期间 | 可被失效/消费 |

**WHY 需要这个标识符：**

```
TransformState 映射结构：
┌─────────────────────────────────────────────────────────────┐
│ TransformState 成员:                                         │
│   topLevel → module_op  // 根操作存储在成员变量中               │
│                                                             │
│ mappings[region].direct:                                    │
│   (kTopLevelValue 不在这里，它只是哨兵)                         │
│   %0 → [op1, op2]       // 普通 Handle 映射                   │
│   %1 → [op3]            // 普通 Handle 映射                   │
└─────────────────────────────────────────────────────────────┘
```

**使用场景示例：**

```mlir
// Transform IR
transform.sequence {
^bb0(%arg0: !transform.any_op):  // %arg0 是根 Handle
  // %arg0 映射到 Payload IR 的根操作（如 ModuleOp）
  // 这是 Transform 执行的起点

  %1 = transform.match.ops{"scf.for"} in %arg0
  // 从根操作开始查找 scf.for
}
```

**WHY 不允许重置根操作：**

- 根操作是 Transform 执行的**起点**
- 所有 `in` 查找都依赖于根操作
- 重置会导致映射状态不一致
- 例如：如果根操作被改变，后续的 `in %arg0` 会查找错误的范围

#### 5.3.2 setPayloadOps 方法详解

```cpp
// TransformInterfaces.cpp:219-253 (简化并注释)
LogicalResult
transform::TransformState::setPayloadOps(Value value,
                                         ArrayRef<Operation *> targets) {
  // 步骤 1: 基本断言检查，不允许操作根节点
  assert(value != kTopLevelValue &&
         "attempting to reset the transformation root");

  assert(llvm::isa<TransformHandleTypeInterface>(value.getType()) &&
         "wrong handle type");
  // WHY 类型检查：只有实现了 TransformHandleTypeInterface 的类型才能映射到操作

  // 步骤 2: 空指针检查
  for (Operation *target : targets) {
    if (!target) {
      return emitError(value.getLoc())
             << "attempting to assign a null payload op";
    }
  }

  // 步骤 3: 类型约束验证
  auto iface = llvm::cast<TransformHandleTypeInterface>(value.getType());
  DiagnosedSilenceableFailure result =
      iface.checkPayload(value.getLoc(), targets);
  // 步骤 3.1: 如果类型是 !transform.op<"scf.for">
  //          targets 必须都是 scf.for 操作
  //          如果包含 linalg.matmul，checkPayload 返回失败
  if (failed(result.checkAndReport()))
    return failure();
  // WHY 验证类型：确保 Handle 指向的操作满足类型约束

  // 步骤 4: 建立双向映射
  SmallVector<Operation *> storedTargets(targets);
  Mappings &mappings = getMapping(value);

  // 步骤 4.1: 插入直接映射
  bool inserted =
      mappings.direct.insert({value, std::move(storedTargets)}).second;
  assert(inserted && "value is already associated with another list");
  (void)inserted;
  // WHY 断言：一个 Handle 只能关联一次操作列表

  // 步骤 4.2: 建立反向映射
  // 示例：value = %0, targets = [op1, op2, op3]
  //       执行后：
  //       direct[%0] = [op1, op2, op3]
  //       reverse[op1] = [%0]
  //       reverse[op2] = [%0]
  //       reverse[op3] = [%0]
  for (Operation *op : targets)
    mappings.reverse[op].push_back(value);

  return success();
}
```

**执行流示例（多场景分析）：**

**场景 A：成功建立映射**
```cpp
// 初始状态
Value handle = %0;                    // Transform IR Value
ArrayRef<Operation *> targets = {op1, op2};  // 两个 Payload 操作

// 执行路径
1. 断言检查：value != kTopLevelValue → 通过
   // 此时：%0 不是 kTopLevelValue，是普通 Handle
2. 类型检查：value.getType() 实现 TransformHandleTypeInterface → 通过
3. 空指针检查：op1 和 op2 都非 null → 通过
4. 类型验证：checkPayload 确认 op1, op2 都是预期类型 → 通过
5. 建立映射：
   direct[%0] = [op1, op2]
   reverse[op1] = [%0]
   reverse[op2] = [%0]

// 返回：success()
```

**场景 B：尝试重置根操作（错误）**
```cpp
// 初始状态
Value handle = kTopLevelValue;  // 特殊标识符
ArrayRef<Operation *> targets = {new_root_op};

// 执行路径
1. 断言检查：value != kTopLevelValue → 失败！
   // 触发断言："attempting to reset the transformation root"
   // 程序终止（Debug 模式）或行为未定义（Release 模式）

// WHY 这样处理：
// - 根操作是 Transform 执行的起点
// - 如果允许重置，后续所有查找操作都会失败
// - 使用断言是因为这是程序错误，不是可恢复的运行时错误
```

**场景 C：类型验证失败**
```cpp
// 初始状态
Value handle = %0;  // 类型：!transform.op<"scf.for">
ArrayRef<Operation *> targets = {scf_for_op, linalg_matmul_op};

// 执行路径
1-3. 检查通过
4. 类型验证：
   checkPayload 检测到 linalg_matmul_op 不是 scf.for
   → 返回 DiagnosedSilenceableFailure
   → "expected scf.for operation, but found linalg.matmul"
5. 返回：failure()
```

### 5.4 RegionScope - 区域作用域管理

```cpp
// TransformInterfaces.h:352-382 (简化版)
class RegionScope {
public:
  // 步骤 1: 析构时自动清理映射
  // WHY 使用 RAII：确保异常安全，自动清理资源
  ~RegionScope() {
    // 步骤 1.1: 区域结束时，清除该区域内的所有映射
    state.mappings.erase(region);
    // WHY 需要清除：区域内的 Value 在区域外不可访问

    // 步骤 1.2: 从区域栈中弹出
    state.regionStack.pop_back();
  }

private:
  // 步骤 2: 构造时创建新作用域
  RegionScope(TransformState &state, Region &region)
      : state(state), region(&region) {
    // 步骤 2.1: 为新区域创建空映射
    auto res = state.mappings.insert(
        std::make_pair(&region, std::make_unique<Mappings>()));

    // 步骤 2.2: 压入区域栈
    state.regionStack.push_back(this);
  }

  TransformState &state;
  Region *region;
};
```

---

## 6. 源码实现：TransformDialectExtension

### 6.1 扩展机制背景与动机

#### 6.1.1 问题：Transform 方言需要扩展？

**核心问题：** Transform 方言作为"元方言"，本身只提供基础的操作和类型，而实际的转换操作需要针对特定的方言（如 Linalg、GPU、SCF 等）。

**WHY 需要扩展机制：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Transform 方言的两层结构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Transform 方言核心层                                             │    │
│  │  • transform.sequence - 顶层容器                                  │    │
│  │  • transform.match.ops - 匹配操作                                 │    │
│  │  • transform.print - 调试输出                                     │    │
│  │  • !transform.any_op - 通用句柄类型                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  特定方言扩展层 (通过 TransformDialectExtension 注入)               │    │
│  │  • transform.structured.tile - Linalg 特定                       │    │
│  │  • transform.gpu.map_to_threads - GPU 特定                       │    │
│  │  • transform.loop.unroll - SCF 特定                              │    │
│  │  • !transform.op<"linalg.matmul"> - 特定句柄类型                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**WHY 不直接在 Transform 方言中实现所有操作：**

| 问题 | 如果直接实现 | WHY 不这样 |
|------|-------------|-----------|
| **依赖循环** | Transform 方言依赖 Linalg、GPU 等 | Transform 应该是独立的基础方言 |
| **编译时间** | 所有方言都链接到 Transform | 只加载需要的扩展 |
| **维护负担** | Transform 变得庞大 | 各方言维护自己的扩展 |
| **扩展性** | 第三方方言无法添加操作 | 每个方言可以独立扩展 |

#### 6.1.2 设计目标

TransformDialectExtension 机制的设计目标：

1. **延迟加载**：扩展只在需要时加载
2. **解耦合**：Transform 方言不依赖特定方言
3. **类型安全**：自动验证扩展操作的接口实现
4. **易用性**：简单的 API 注册操作和类型

### 6.2 扩展机制设计原理

#### 6.2.1 CRTP 模式

```cpp
// TransformDialect.h:116-118
template <typename DerivedTy, typename... ExtraDialects>
class TransformDialectExtension
    : public DialectExtension<DerivedTy, TransformDialect, ExtraDialects...> {
```

**WHY 使用 CRTP：**

```cpp
// 示例：CRTP 基本结构
template <typename DerivedTy>
class Base {
protected:
  void interface() {
    // 步骤 1: 调用派生类的方法
    static_cast<DerivedTy *>(this)->implementation();
    // WHY 使用 static_cast：
    //   - 编译时知道 DerivedTy 的确切类型
    //   - 避免虚函数调用开销
    //   - 支持静态多态
  }
};

class Derived : public Base<Derived> {  // CRTP: 将自己作为模板参数
protected:
  void implementation() {
    // 派生类的具体实现
  }
};
```

**WHY CRTP 适合扩展机制：**

| 特性 | 说明 | 优势 |
|------|------|------|
| **静态多态** | 编译时解析，无虚函数开销 | 性能优于虚函数 |
| **类型安全** | 编译时检查派生类类型 | 及早发现错误 |
| **代码生成** | 配合 TableGen 生成代码 | 自动化重复代码 |

#### 6.2.2 初始化流程

```cpp
// TransformDialect.h:122-138
void apply(MLIRContext *context, TransformDialect *transformDialect,
           ExtraDialects *...) const final {
  // 步骤 1: 加载依赖方言，依赖方言是扩展操作定义时需要的方言
  for (const DialectLoader &loader : dialectLoaders) {
    loader(context);
    // 例如：LinalgTransformExtend 声明依赖 LinalgDialect
    // 因为扩展操作的类型定义在 Linalg 方言中
  }

  // 步骤 2: 加载生成方言（仅在非 build-only 模式），生成方言是转换可能产生的方言
  if (!buildOnly) {
    for (const DialectLoader &loader : generatedDialectLoaders) {
      loader(context);
      // 例如：LinalgTransformExtend 可能生成 SCF 循环
      //       SCFDialect 被声明为生成方言
    }
  }

  // 步骤 3: 执行初始化回调
  for (const Initializer &init : initializers) {
    init(transformDialect);
    // 注册操作、类型、数据等
  }
}
```

**执行流图：**

```
扩展应用流程
│
├── 1. 构造扩展对象
│   场景 A: 正常模式
│     buildOnly = false
│     → 调用 init() 收集初始化器
│     → 扩展会加载所有方言
│
│   场景 B: build-only 模式
│     buildOnly = true
│     → 只加载依赖方言
│     → 跳过生成方言
│     WHY build-only 模式：
│       - 仅用于构造 Transform IR
│       - 不需要执行转换
│       - 避免加载不需要的方言
│
├── 2. 应用扩展 (apply)
│   ├── 加载依赖方言
│   │   └── 例如：LinalgDialect
│   ├── 加载生成方言 (如果 !buildOnly)
│   │   └── 例如：SCFDialect, VectorDialect
│   └── 执行初始化器
│       ├── 注册操作
│       ├── 注册类型
│       └── 初始化数据
│
└── 3. 扩展就绪
```

### 6.3 核心组件详解

#### 6.3.1 registerTransformOps - 注册操作

```cpp
// TransformDialect.h:191-196
template <typename... OpTys>
void registerTransformOps() {
  initializers.push_back([](TransformDialect *transformDialect) {
    transformDialect->addOperationsChecked<OpTys...>();
    // WHY 使用 Checked 版本：
    //   - 验证操作实现了 TransformOpInterface
    //   - 验证操作实现了 MemoryEffectsOpInterface
    //   - 防止重复注册
  });
}
```

**WHY 使用延迟初始化（initializers）：**

```cpp
// 示例：扩展构造时
void init() {
  registerTransformOps<MyOp1, MyOp2>();
  // 此时只是向 initializers 列表添加回调
  // 实际注册发生在 apply() 时
  // WHY 延迟注册：
  //   - 构造时 TransformDialect 可能还不存在
  //   - 需要在 apply() 时才有 TransformDialect 实例
}
```

**接口验证（Debug 模式）：**

```cpp
// TransformDialect.h:258-263
addOperations<OpTy>();
#ifndef NDEBUG
  StringRef name = OpTy::getOperationName();
  detail::checkImplementsTransformOpInterface(name, getContext());
  // WHY 动态检查：
  //   - TableGen 生成的代码可能没有正确实现接口
  //   - 接口可能在运行时注册
  //   - Debug 模式确保操作正确实现接口
#endif
```

#### 6.3.2 declareDependentDialect vs declareGeneratedDialect

**依赖方言 (Dependent Dialect)：**

```cpp
// TransformDialect.h:218-222
template <typename DialectTy>
void declareDependentDialect() {
  dialectLoaders.push_back(
      [](MLIRContext *context) { context->loadDialect<DialectTy>(); });
}
```

**WHY 需要依赖方言：**

- **操作定义**：扩展操作的 TableGen 定义使用了依赖方言的类型
- **规范化**：操作规范化时可能需要依赖方言的模式
- **常量/属性**：操作默认值使用了依赖方言的属性

**示例：**
```cpp
// Linalg 扩展
void init() {
  declareDependentDialect<linalg::LinalgDialect>();
  // WHY Linalg 是依赖方言：
  //   - LinalgTransformOps 使用了 linalg::LinalgOp 类型
  //   - 操作的约束条件使用 Linalg 方言的接口
}
```

**生成方言 (Generated Dialect)：**

```cpp
// TransformDialect.h:230-234
template <typename DialectTy>
void declareGeneratedDialect() {
  generatedDialectLoaders.push_back(
      [](MLIRContext *context) { context->loadDialect<DialectTy>(); });
}
```

**WHY 需要生成方言：**

- **转换产物**：转换可能产生新的方言操作
- **不总是需要**：build-only 模式下不需要加载

**示例：**
```cpp
// Linalg 扩展
void init() {
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<vector::VectorDialect>();
  // WHY 这些是生成方言：
  //   - tile 操作可能生成 scf.for 循环
  //   - vectorize 操作可能生成 vector 操作
  //   - 只在执行转换时需要，构造 IR 时不需要
}
```

**对比：**

| 特性 | 依赖方言 | 生成方言 |
|------|---------|---------|
| **加载时机** | 总是加载 | build-only 模式下跳过 |
| **用途** | 操作定义使用 | 转换产物使用 |
| **示例** | LinalgDialect | SCFDialect, VectorDialect |

#### 6.3.3 registerTypes - 注册类型

```cpp
// TransformDialect.h:203-208
template <typename... TypeTys>
void registerTypes() {
  initializers.push_back([](TransformDialect *transformDialect) {
    transformDialect->addTypesChecked<TypeTys...>();
  });
}
```

**类型验证：**

```cpp
// TransformDialect.h:292-295
#ifndef NDEBUG
  detail::checkImplementsTransformHandleTypeInterface(
      TypeID::get<Type>(), getContext());
  // WHY 验证类型接口：
  //   - 确保类型实现了相应的 Handle 接口
  //   - 例如：OperationType 必须实现 TransformHandleTypeInterface
#endif
```

### 6.4 完整扩展示例：LinalgTransformDialectExtension

```cpp
// DialectExtension.cpp (完整源码)
namespace {

class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  // 定义：类型 ID
  // WHY 需要类型 ID：
  //   - MLIR 运行时类型识别
  //   - 扩展管理系统需要
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgTransformDialectExtension)

  using Base::Base;

  // 初始化方法
  void init() {
    // 步骤 1: 声明依赖方言
    declareDependentDialect<linalg::LinalgDialect>();
    // WHY Linalg 是依赖方言：
    //   - LinalgTransformOps 操作使用了 LinalgOp 接口
    //   - 操作类型定义在 Linalg 方言中

    // 步骤 2: 声明生成方言
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<index::IndexDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    // WHY 这些是生成方言：
    //   - tile → scf.for (循环平铺)
    //   - vectorize → vector.* (向量化)
    //   - bufferize → memref.* (buffer 化)
    //   - map_to_gpu → gpu.* (GPU 映射)

    // 步骤 3: 注册 Linalg Transform 操作
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"
    >();
    // 步骤 4: 注册 Linalg Match 操作
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.cpp.inc"
    >();
  }
};

} // namespace

// 注册扩展到 DialectRegistry
void mlir::linalg::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
  // WHY 通过 DialectRegistry 注册：
  //   - 当 Transform 方言加载时自动应用扩展
  //   - 支持延迟加载
  //   - 多个扩展可以共存
}
```

**执行流分析：**

**场景 A：加载 Transform 方言**
```cpp
// 用户代码
DialectRegistry registry;
registry.addExtensions<LinalgTransformDialectExtension>();

MLIRContext context;
context.loadDialect<transform::TransformDialect>();

// 执行流程：
// 1. loadDialect<TransformDialect>()
// 2. TransformDialect 构造
// 3. DialectRegistry 应用所有扩展
// 4. LinalgTransformDialectExtension::apply() 被调用
//    4.1: 加载 LinalgDialect (依赖方言)
//    4.2: 加载 SCFDialect, VectorDialect 等 (生成方言)
//    4.3: 注册所有 Linalg Transform 操作
// 5. Transform 方言就绪，可以使用 Linalg 特定操作
```

**场景 B：build-only 模式**
```cpp
// 用户代码
DialectRegistry registry;
registry.addExtensions<BuildOnly<LinalgTransformDialectExtension>>();

MLIRContext context;
context.loadDialect<transform::TransformDialect>();

// 执行流程：
// 1. loadDialect<TransformDialect>()
// 2. LinalgTransformDialectExtension 构造 (buildOnly=true)
// 3. apply() 被调用
//    3.1: 加载 LinalgDialect (依赖方言)
//    3.2: 跳过 SCFDialect 等 (buildOnly=true)
//    3.3: 注册操作
// 4. Transform 方言就绪，但生成的方言未加载
// WHY 这样做：
//   - 只需要构造 Transform IR
//   - 不需要执行转换
//   - 减少加载时间
```

### 6.5 TransformDialectData - 扩展间通信机制

```cpp
// TransformDialect.h:50-67
template <typename DerivedTy>
class TransformDialectData : public detail::TransformDialectDataBase {
protected:
  TransformDialectData(MLIRContext *ctx)
      : TransformDialectDataBase(TypeID::get<DerivedTy>(), ctx) {}
};
```

**WHY 需要数据共享：**

```
扩展间通信需求
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌──────────────────────┐         ┌──────────────────────┐              │
│  │ Linalg Extension     │         │ GPU Extension        │              │
│  │                      │         │                      │              │
│  │ 需要共享：             │◄───────►│ 需要共享：            │              │
│  │ • Tile 配置           │         │ • GPU 线程配置        │              │
│  │ • 向量化策略           │         │ • Block 大小         │              │
│  └──────────────────────┘         └──────────────────────┘              │
│                    │                              │                     │
│                    └──────────┬───────────────────┘                     │
│                               ▼                                         │
│                    ┌──────────────────────┐                             │
│                    │ TransformDialectData │                             │
│                    │ • 共享配置数据         │                             │
│                    │ • 类型 ID 识别        │                             │
│                    └──────────────────────┘                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**使用示例：**

```cpp
// 步骤 1: 定义共享数据类型
class TileConfigData : public transform::TransformDialectData<TileConfigData> {
public:
  TileConfigData(MLIRContext *ctx) : TransformDialectData(ctx) {
    // 初始化默认配置
    defaultTileSize = 32;
  }

  int defaultTileSize;
  SmallVector<int64_t> userTileSizes;
};

// 步骤 2: 在扩展中初始化数据
class MyExtension : public transform::TransformDialectExtension<MyExtension> {
  void init() override {
    addDialectDataInitializer<TileConfigData>(
        [](TileConfigData &data) {
          // 设置配置
          data.defaultTileSize = 64;
        });
  }
};

// 步骤 3: 在转换操作中访问数据
DiagnosedSilenceableFailure MyTileOp::apply(...) {
  TileConfigData &config = state.getOrCreateExtraData<TileConfigData>();
  // 使用配置
  int tileSize = config.defaultTileSize;
}
```

### 6.6 BuildOnly 模式

```cpp
// TransformDialect.h:309-313
template <typename DerivedTy>
class BuildOnly : public DerivedTy {
public:
  BuildOnly() : DerivedTy(/*buildOnly=*/true) {}
};
```

**WHY 需要 BuildOnly 模式：**

| 场景 | 需要的操作 | 不需要的操作 |
|------|-----------|-------------|
| **构造 Transform IR** | 扩展操作 | 生成方言 |
| **执行 Transform** | 扩展操作 + 生成方言 | - |
| **序列化 IR** | 扩展操作 | 生成方言 |

**使用场景：**

```cpp
// 场景 A: 构造 Transform IR（不需要执行）
DialectRegistry registry;
registry.addExtensions<BuildOnly<LinalgTransformDialectExtension>>();

MLIRContext context;
context.loadDialect<transform::TransformDialect>();

// 此时：
// - Linalg Transform 操作已注册
// - LinalgDialect 已加载（依赖）
// - SCFDialect 未加载（生成方言，不需要）
// WHY 优化：
//   - 减少方言加载时间
//   - 减少内存占用
//   - 避免循环依赖

// 场景 B: 执行 Transform（需要完整加载）
DialectRegistry registry;
registry.addExtensions<LinalgTransformDialectExtension>();  // 不用 BuildOnly

// 此时：
// - 所有方言都会加载
// - 可以执行转换
```

### 6.7 扩展注册检查

#### 6.7.1 操作重复注册检查

```cpp
// TransformDialect.h:254-270
template <typename OpTy>
void TransformDialect::addOperationIfNotRegistered() {
  // 场景 1: 检查操作是否已注册
  std::optional<RegisteredOperationName> opName =
      RegisteredOperationName::lookup(TypeID::get<OpTy>(), getContext());

  if (!opName) {
    // 场景 1.1: 操作未注册，进行注册
    addOperations<OpTy>();
    #ifndef NDEBUG
    detail::checkImplementsTransformOpInterface(name, getContext());
    #endif
    return;
  }

  // 场景 2: 操作已注册，检查是否是同一类型
  if (LLVM_LIKELY(opName->getTypeID() == TypeID::get<OpTy>()))
    return;
    // WHY LIKELY：
    //   - 同一操作类型尝试多次注册是正常情况
    //   - 多个编译单元可能包含相同的操作定义
    //   - 使用 LIKELY 优化常见路径

  // 场景 3: 不同类型尝试注册相同操作名
  reportDuplicateOpRegistration(OpTy::getOperationName());
  // WHY 报告错误：
  //   - 两个不同的 C++ 类注册相同的操作名
  //   - 会导致 ODR 违规
  //   - 必须立即报告
}
```

**执行流示例：**

**场景 A：首次注册**
```cpp
// 第一次尝试注册 MyTransformOp
addOperationIfNotRegistered<MyTransformOp>();

// 执行流程：
// 1. lookup(TypeID::get<MyTransformOp>()) → std::nullopt
// 2. addOperations<MyTransformOp>()
// 3. 返回
// 结果：MyTransformOp 成功注册
```

**场景 B：重复注册（相同类型）**
```cpp
// 第二次尝试注册 MyTransformOp（相同类型）
addOperationIfNotRegistered<MyTransformOp>();

// 执行流程：
// 1. lookup(...) → 返回 RegisteredOperationName
// 2. opName->getTypeID() == TypeID::get<MyTransformOp>() → true
// 3. 返回（静默跳过）
// 结果：不重复注册，正常返回
```

**场景 C：重复注册（不同类型，错误）**
```cpp
// 两个不同的 C++ 类注册相同操作名
struct MyOpV1 { ... };  // 类型 ID: 0x1234
struct MyOpV2 { ... };  // 类型 ID: 0x5678

addOperationIfNotRegistered<MyOpV1>();
addOperationIfNotRegistered<MyOpV2>();

// 执行流程：
// 1. 第一次注册 MyOpV1 → 成功
// 2. 第二次注册 MyOpV2
//    lookup(...) → 返回 MyOpV1 的信息
//    MyOpV1::getTypeID() (0x1234) != MyOpV2::getTypeID() (0x5678)
//    → reportDuplicateOpRegistration("my.op")
// 结果：程序终止，报告重复注册错误
```

### 6.8 扩展自动加载机制详解

#### 6.8.1 完整加载流程

**使用示例（详见 7.5 节）：**

```cpp
// 步骤 1: 注册扩展到 DialectRegistry
registry.addExtensions<MyTransformDialectExtension>();

// 步骤 2: 在 Pass 中使用（无需手动加载）
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 说明：MLIRContext 会自动加载扩展
    // 因为扩展已注册到 DialectRegistry
  }
};
```

**WHY 无需手动加载扩展：**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Transform Dialect 扩展自动加载完整流程                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  应用代码        │    │ DialectRegistry │    │  MLIRContext    │          │
│  │                 │    │                 │    │                 │          │
│  │ registry        │───▶│ extensions      │───▶│ getOrLoadDialect│          │
│  │   .addExt<...>()│    │   .apply()      │    │                 │          │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│                                                         │                   │
│                                                         ▼                   │
│                                              ┌──────────────────┐           │
│                                              │ 加载 Transform   │            │
│                                              │ Dialect          │           │
│                                              └────────┬─────────┘           │
│                                                       │                     │
│                       ┌───────────────────────────────┘                     │
│                       │                                                     │
│                       ▼                                                     │
│         ┌───────────────────────────────┐                                   │
│         │  applyExtensions(dialect)     │                                   │
│         │  (MLIRContext.cpp:488)        │                                   │
│         └───────────────┬───────────────┘                                   │
│                         │                                                   │
│                         ▼                                                   │
│         ┌───────────────────────────────────────────────────────┐           │
│         │  遍历 Registry 中的所有 Extension                       │           │
│         └───────────────┬───────────────────────────────────────┘           │
│                         │                                                   │
│         ┌───────────────┴───────────────────────────────────────┐           │
│         ▼                                                       │           │
│  ┌─────────────────────┐                                        │           │
│  │ 检查 requiredDialects│                                        │           │
│  │ 是否包含当前 dialect  │                                        │           │
│  └─────────┬───────────┘                                        │           │
│            │                                                    │           │
│    ┌───────┴────────┐                                           │            │
│    ▼                ▼                                           │            │
│  [包含]         [不包含]                                          │            │
│    │                │                                           │            │
│    ▼                └──────────────────┐                        │            │
│  ┌─────────────────┐                   │                        │            │
│  │检查其他 required │                   │                        │            │
│  │ dialects 是否    │                  │                        │            │
│  │都已加载          │                   │                        │            │
│  └────────┬────────┘                   │                        │            │
│           │                            │                        │            │
│    ┌──────┴────────┐                   │                        │            │
│    ▼               ▼                   │                        │            │
│  [是]           [否]                    │                        │            │
│    │               │                   │                        │            │
│    ▼               └───────────────────┘                        │            │
│  ┌─────────────────────┐                                        │            │
│  │ extension.apply()   │                                        │            │
│  │ 调用扩展的 init()     │                                        │            │
│  │ 方法注册操作和类型     │                                        │            │
│  └─────────────────────┘                                        │            │
│                                                                 │            │
└─────────────────────────────────────────────────────────────────┘            │
```

#### 6.8.2 注册阶段（应用启动时）

```cpp
// 源码：DialectRegistry.h:221-225
template <typename... ExtensionsT>
void DialectRegistry::addExtensions() {
  // 步骤 1: 展开参数包，为每个扩展类型调用 addExtension
  (addExtension(TypeID::get<ExtensionsT>(),
                std::make_unique<ExtensionsT>()), ...);
  // WHY 使用折叠表达式：C++17 特性，按顺序展开参数包
}

// 源码：DialectRegistry.h:215-218
bool addExtension(TypeID extensionID,
                  std::unique_ptr<DialectExtensionBase> extension) {
  // 步骤 2: 将扩展存储到 MapVector 中
  // WHY 用 MapVector：保持插入顺序，同时提供 O(1) 查找
  return extensions.try_emplace(extensionID, std::move(extension)).second;
}
```

**执行追踪：**
```cpp
// 调用 registry.addExtensions<MyTransformDialectExtension>()
// 此时：registry.extensions = {
//   TypeID::get<MyTransformDialectExtension>() -> MyTransformDialectExtension 实例
// }
```

#### 6.8.3 Pass 运行时自动加载

```cpp
// 源码：MLIRContext.cpp:438-444
Dialect *MLIRContext::getOrLoadDialect(StringRef name) {
  // 步骤 1: 检查方言是否已加载
  Dialect *dialect = getLoadedDialect(name);
  // 此时：name = "transform"
  if (dialect)
    return dialect;  // 场景 1: 已加载，直接返回

  // 步骤 2: 从 Registry 获取分配器
  DialectAllocatorFunctionRef allocator =
      impl->dialectsRegistry.getDialectAllocator(name);
  // 此时：allocator 是 lambda 函数，调用 ctx->getOrLoadDialect<TransformDialect>()

  // 步骤 3: 调用分配器（如果存在）
  return allocator ? allocator(this) : nullptr;
  // 场景 2: 未注册，返回 nullptr
}
```

#### 6.8.4 Dialect 创建与扩展应用

```cpp
// 源码：MLIRContext.cpp:451-507（关键部分）
Dialect *MLIRContext::getOrLoadDialect(
    StringRef dialectNamespace, TypeID dialectID,
    function_ref<std::unique_ptr<Dialect>()> ctor) {

  auto &impl = getImpl();

  // 步骤 1: 尝试在 loadedDialects 中插入或查找
  auto dialectIt = impl.loadedDialects.try_emplace(dialectNamespace, nullptr);
  // WHY 使用 try_emplace：避免不必要的临时对象构造

  // 场景 1: 方言首次加载（second == true）
  if (dialectIt.second) {
    // 步骤 2: 调用构造函数创建方言实例
    std::unique_ptr<Dialect> &dialectOwned =
        impl.loadedDialects[dialectNamespace] = ctor();
    // 此时：ctor() 创建 TransformDialect 实例
    //       TransformDialect::initialize() 被调用

    Dialect *dialect = dialectOwned.get();
    // 此时：dialect 指向新创建的 TransformDialect 实例

    // 步骤 3: 【关键】应用所有等待的扩展！
    impl.dialectsRegistry.applyExtensions(dialect);
    // WHY 在这里调用：确保方言已完全初始化，扩展可以安全地注册操作

    return dialect;
  }

  // 场景 2: 方言已存在，验证类型 ID 一致性
  if (dialect->getTypeID() != dialectID)
    llvm::report_fatal_error("a dialect with namespace '" +
                             dialectNamespace + "' has already been registered");

  return dialect.get();
}
```

#### 6.8.5 扩展应用逻辑

```cpp
// 源码：Dialect.cpp:250-299（简化版）
void DialectRegistry::applyExtensions(Dialect *dialect) const {
  StringRef dialectName = dialect->getNamespace();
  // 此时：dialectName = "transform"

  // 步骤 1: 定义应用 lambda
  auto applyExtension = [&](const DialectExtensionBase &extension) {
    // 步骤 2: 获取扩展依赖的方言列表
    ArrayRef<StringRef> dialectNames = extension.getRequiredDialects();
    // 对于 MyTransformDialectExtension：
    // dialectNames = {"transform"}

    // 场景 1: 扩展没有指定依赖方言（罕见）
    if (dialectNames.empty()) {
      extension.apply(ctx, dialect);
      return;
    }

    // 场景 2: 扩展依赖单个方言（最常见）
    if (dialectNames.size() == 1) {
      if (dialectNames.front() == dialectName)
        extension.apply(ctx, dialect);
      // WHY 检查名称：确保扩展是为当前方言定义的
      return;
    }

    // 场景 3: 扩展依赖多个方言
    // 检查当前方言是否在依赖列表中
    const StringRef *nameIt = llvm::find(dialectNames, dialectName);
    if (nameIt == dialectNames.end())
      return;  // 不是为这个方言触发的

    // 步骤 3: 验证所有依赖方言都已加载
    SmallVector<Dialect *> requiredDialects;
    requiredDialects.reserve(dialectNames.size());

    for (auto it = dialectNames.begin(), e = dialectNames.end(); it != e; ++it) {
      // 场景 3.1: 当前方言（已知已加载）
      if (it == nameIt) {
        requiredDialects.push_back(dialect);
        continue;
      }

      // 场景 3.2: 其他依赖方言，检查是否已加载
      Dialect *loadedDialect = ctx->getLoadedDialect(*it);
      if (!loadedDialect)
        return;  // 依赖未满足，延迟应用
      requiredDialects.push_back(loadedDialect);
    }

    // 步骤 4: 所有依赖满足，应用扩展
    extension.apply(ctx, requiredDialects);
  };

  // 步骤 5: 遍历所有注册的扩展
  applyExtensionsFn(applyExtension, extensions);
}
```

#### 6.8.6 TransformDialectExtension 的 apply 实现

```cpp
// 源码：TransformDialect.h:186-210
template <typename DerivedT>
class TransformDialectExtension
    : public DialectExtension<TransformDialectExtension<DerivedT>,
                             transform::TransformDialect> {
public:
  // 基类 apply 的最终实现
  void apply(MLIRContext *context,
             transform::TransformDialect *dialect) const final {
    // 步骤 1: 检查是否处于 BuildOnly 模式
    if (LLVM_LIKELY(!dialect->isBuildOnly())) {
      // 场景 1: 正常模式，执行扩展初始化
      // 步骤 2: 调用派生类的 init() 方法
      const_cast<DerivedT *>(static_cast<const DerivedT *>(this))->init();
      // WHY const_cast：apply 是 const 方法，但 init() 可能需要修改状态
      // WHY static_cast：确保调用正确的派生类版本

    } else {
      // 场景 2: BuildOnly 模式，仅注册依赖关系
      const_cast<DerivedT *>(static_cast<const DerivedT *>(this))
          ->init(dialect->getDependentDialects());
      // WHY 传递 DependentDialects：允许扩展声明依赖而不注册操作
    }
  }
};
```

#### 6.8.7 完整执行示例

**场景：Pass 中使用 Transform 操作**

```cpp
// Pass 定义
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 说明：无需手动加载 Transform Dialect！
    // MLIRContext 会自动处理
  }
};
```

**执行流程追踪：**

```
时间轴：
┌─────────────────────────────────────────────────────────────────┐
│ T0: Pass Manager 创建 MLIRContext                                │
│     context = new MLIRContext(registry)                         │
│     此时：context 已包含 registry 的所有扩展                        │
├─────────────────────────────────────────────────────────────────┤
│ T1: Parser 解析 IR，遇到 transform.* 操作                         │
│     IR: module {                                                │
│            transform.sequence {                                 │
│              %0 = transform.get_parent_op %arg : !transform.any_op│
│            }                                                    │
│          }                                                      │
├─────────────────────────────────────────────────────────────────┤
│ T2: Parser 调用 getOrLoadDialect("transform")                    │
│     ┌─────────────────────────────────────────┐                 │
│     │ 场景 1: transform 未加载                  │                 │
│     ▼                                         │                 │
│     ┌─────────────────────────────────────────┐                 │
│     │ 从 registry 获取 TransformDialect        │                 │
│     │ 分配器并创建实例                           │                 │
│     ▼                                         │                 │
│     ┌─────────────────────────────────────────┐                 │
│     │ TransformDialect::initialize()          │                 │
│     │ - 注册核心 Transform 操作                 │                 │
│     ▼                                         │                 │
│     ┌─────────────────────────────────────────┐                 │
│     │ applyExtensions(dialect)  ◄── 关键！     │                 │
│     │ 遍历 registry 中的所有扩展                 │                 │
│     └─────────────────────────────────────────┘                 │
│         │                                                       │
│         ▼                                                       │
│     ┌─────────────────────────────────────────┐                 │
│     │ MyTransformDialectExtension::init()     │                 │
│     │ - 注册 Linalg Transform 操作             │                 │
│     │ - 声明依赖的方言                          │                 │
│     └─────────────────────────────────────────┘                 │
│         │                                                       │
│         ▼                                                       │
│     返回 TransformDialect*                                       │
└─────────────────────────────────────────────────────────────────┘

T3: Parser 继续解析，所有 transform.* 操作都可用
```

#### 6.8.8 关键设计点解析

**WHY 使用延迟加载？**

```cpp
// ❌ 立即加载模式（假设）
void main() {
  DialectRegistry registry;
  registry.addExtensions<MyTransformDialectExtension>();

  // 如果立即加载：
  // - 需要实例化所有方言
  // - 注册所有操作、类型、属性
  // - 内存占用大，启动慢

  MLIRContext context(registry);  // 启动慢！
}

// ✅ 延迟加载模式（实际）
void main() {
  DialectRegistry registry;
  registry.addExtensions<MyTransformDialectExtension>();

  MLIRContext context(registry);  // 快速启动！

  // 只在真正需要时加载
  parseIR(source, &context);  // 此时才加载 transform
}
```

**WHY 扩展机制需要依赖声明？**

```cpp
class MyTransformDialectExtension : public TransformDialectExtension<MyTransformDialectExtension> {
  void init() {
    // 声明依赖方言
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<affine::AffineDialect>();

    // 注册转换操作
    registerTransformOps<MyCustomTransformOp>();
  }
};
```

**WHY 需要 `declareDependentDialect`？**

1. **防止递归加载死锁**：
```cpp
// 场景：Transform Dialect 扩展依赖 Linalg
//      Linalg Dialect 扩展依赖 Transform（假设）
// 如果没有依赖声明：
//   Transform 加载 → 触发扩展 → 需要 Linalg
//   Linalg 加载 → 触发扩展 → 需要 Transform
//   → 死锁！

// 有了依赖声明：
//   Transform 加载前 → 预加载 Linalg
//   → 避免递归依赖
```

2. **优化加载顺序**：
```cpp
// 源码：MLIRContext.cpp:461-466
#ifndef NDEBUG
if (impl.multiThreadedExecutionContext != 0)
  llvm::report_fatal_error(
      "Loading a dialect (" + dialectNamespace +
      ") while in a multi-threaded execution context");
#endif
// WHY 这个检查：多线程环境加载方言不安全
// 依赖声明确保方言在 Pass 执行前加载
```

**WHY 使用 CRTP 模式？**

```cpp
template <typename DerivedT>
class TransformDialectExtension : public DialectExtension<...> {
  void apply(MLIRContext *ctx, TransformDialect *dialect) const final {
    // 调用派生类的 init()
    const_cast<DerivedT *>(static_cast<const DerivedT *>(this))->init();
  }
};
```

```cpp
// ❌ 虚函数方案（假设）
class TransformDialectExtension {
  virtual void init() = 0;  // 纯虚函数
};

// 问题：
// 1. 虚函数调用有运行时开销
// 2. 无法在基类中提供类型安全的接口
// 3. 每个扩展都需要虚函数表

// ✅ CRTP 方案（实际）
template <typename DerivedT>
class TransformDialectExtension {
  void apply(...) const {
    // 编译时静态分发，无虚函数开销
    static_cast<const DerivedT *>(this)->init();
  }
};
```

#### 6.8.9 边界条件与错误处理

**边界条件 1：扩展重复注册**

```cpp
// 场景：多次注册同一扩展
registry.addExtensions<MyTransformDialectExtension>();
registry.addExtensions<MyTransformDialectExtension>();

// 源码：DialectRegistry.h:215-218
bool addExtension(TypeID extensionID, ...) {
  // WHY 返回 bool：第二次调用返回 false
  return extensions.try_emplace(extensionID, ...).second;
  // try_emplace 行为：
  // - 第一次插入：成功，返回 true
  // - 键已存在：失败，返回 false，不覆盖
}
```

**边界条件 2：依赖方言未注册**

```cpp
// 场景：扩展依赖未注册的方言
class BadExtension : public TransformDialectExtension<BadExtension> {
  void init() {
    declareDependentDialect<UnregisteredDialect>();  // 未注册！
  }
};

// 执行流程：
// 1. Transform Dialect 加载
// 2. applyExtensions(BadExtension)
// 3. 检查依赖：UnregisteredDialect 未加载
// 4. 延迟应用（等待 UnregisteredDialect 加载）
// 5. UnregisteredDialect 永远不会加载
// 6. BadExtension 永远不会被应用

// 结果：扩展静默失败，操作不可用
```

**边界条件 3：多线程环境加载方言**

```cpp
// 源码：MLIRContext.cpp:461-466
#ifndef NDEBUG
if (impl.multiThreadedExecutionContext != 0)
  llvm::report_fatal_error(
      "Loading a dialect (" + dialectNamespace +
      ") while in a multi-threaded execution context");
#endif
// WHY 这个检查：
// - loadedDialects 不是线程安全的
// - 并发加载会导致数据竞争

// 解决方案：使用 dependentDialects 声明
struct MyPass : public PassWrapper<...> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // 确保 Pass 执行前加载
    registry.insert<linalg::LinalgDialect>();
  }
};
```

#### 6.8.10 总结表

| 机制 | 时机 | 触发条件 | 作用 |
|-----|------|---------|------|
| `addExtensions` | 应用启动 | 调用 registry 方法 | 将扩展添加到注册表 |
| `getOrLoadDialect` | Parser/Pass | 遇到 dialect 操作 | 加载方言实例 |
| `applyExtensions(dialect)` | 方言加载后 | 方言首次创建 | 应用等待的扩展 |
| `extension.init()` | 扩展应用时 | 依赖方言满足 | 注册操作和类型 |

**关键点：**
- ✅ **延迟加载**：只在需要时加载方言和扩展
- ✅ **自动应用**：方言加载时自动应用扩展
- ✅ **依赖管理**：确保依赖方言先于扩展加载
- ✅ **线程安全**：多线程环境需预声明依赖

---

## 7. 扩展开发完整教程

### 7.1 扩展开发步骤概览

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

### 7.2 步骤 1：定义扩展类

```cpp
// MyTransformOps.h
#pragma once

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace my {
namespace transform {

// 定义：扩展类
// WHY 继承 TransformDialectExtension：
//   - 自动获得扩展机制的所有功能
//   - 使用 CRTP 模式
class MyTransformDialectExtension
    : public ::mlir::transform::TransformDialectExtension<
          MyTransformDialectExtension> {
public:
  // 定义：类型 ID（必需）
  // WHY 需要类型 ID：
  //   - MLIR 运行时类型识别
  //   - 扩展管理系统需要
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MyTransformDialectExtension)

  using Base::Base;

  // 初始化方法
  // 重写 init() 来注册扩展内容
  void init() {
    // 步骤 1: 声明依赖方言
    // WHY 声明依赖：
    //   - 扩展操作使用了 MyDialect 的类型
    //   - 必须提前加载才能解析操作定义
    declareDependentDialect<MyDialect>();

    // 步骤 2: 声明生成方言
    // 转换可能产生这些方言的操作
    declareGeneratedDialect<::mlir::scf::SCFDialect>();
    declareGeneratedDialect<::mlir::vector::VectorDialect>();

    // 步骤 3: 注册 Transform 操作
    registerTransformOps<
#define GET_OP_LIST
#include "MyTransformOps.cpp.inc"
    >();
  }
};

} // namespace transform
} // namespace my
```

**WHY 这样组织代码：**

| 组件 | 作用 | WHY 这样设计 |
|------|------|-----------|
| **类型 ID 宏** | 运行时类型识别 | MLIR 类型系统需要 |
| **using Base::Base** | 导入基类构造函数 | 简化构造函数定义 |
| **init() 方法** | 注册扩展内容 | 延迟初始化模式 |
| **分离头文件** | 声明扩展类 | 支持 include |

### 7.3 步骤 2：使用 TableGen 定义操作

```tablegen
// MyTransformOps.td
#ifndef MY_TRANSFORM_OPS
#define MY_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 定义：自定义 Transform 操作
def MyCustomTransformOp : TransformDialectOp<"my_custom",
    // 声明：实现的接口
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Applies my custom transformation to target operations";
  let description = [{
    This operation demonstrates a custom transform that:
    1. Takes a handle to operations as input
    2. Applies a specific transformation
    3. Returns handles to the transformed operations

    The transformation is applied to all operations associated with
    the input handle. The operation consumes the handle and produces
    a new handle pointing to the transformed operations.
  }];

  // 定义：参数
  let arguments = (ins
    // 目标操作句柄
    TransformHandleTypeInterface:$target,
    // 可选：转换参数
    OptionalAttr<I64Attr>$param,
    // 可选：标志位
    UnitAttr:$verbose
  );

  // 定义：结果
  let results = (outs
    // 转换后的操作句柄
    TransformHandleTypeInterface:$result
  );

  // 定义：汇编格式
  let assemblyFormat = [{
    $target `(` $param^ `, `verbose` $verbose^)? `)` attr-dict
      `:` type($target) `->` type($result)
  }];

  // 定义：额外约束
  let hasVerifier = 1;
  // WHY 需要验证器：
  //   - 检查参数范围
  //   - 验证标志位组合
}
```

**WHY 使用 TableGen：**

| 特性 | 说明 | 优势 |
|------|------|------|
| **声明式** | 声明操作属性而非手写代码 | 减少重复代码 |
| **类型推导** | 自动生成类型检查代码 | 类型安全 |
| **文档生成** | 自动生成操作文档 | 文档与代码同步 |

### 7.4 步骤 3：实现 C++ 类

```cpp
// MyTransformOps.cpp
#include "MyTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::transform;

namespace {

// 结构定义：实现 apply 方法
struct MyCustomTransformOp
    : public Op<MyCustomTransformOp,
               TransformOpInterface::Trait,
               MemoryEffectsOpInterface::Trait> {
  using Op::Op;

  // 方法实现：TransformOpInterface::apply
  DiagnosedSilenceableFailure apply(
      TransformRewriter &rewriter,
      TransformResults &results,
      TransformState &state) override {

    // 步骤 1: 获取目标操作
    ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
    // 此时：getTarget() 是操作的 target 参数
    //       targets 是 Handle 关联的所有 Payload 操作
    //       例如：targets = [op1, op2, op3]

    if (targets.empty()) {
      // 场景 1: 没有目标操作
      return emitSilenceableError()
             << "no operations found to transform";
      // WHY 使用 silenceable：
      //   空操作集是可预期的，不是致命错误
    }

    // 步骤 2: 获取可选参数
    int64_t param = 0;
    if (auto paramAttr = getParam()) {
      param = paramAttr.getInt();
      // 此时：paramAttr = IntegerAttr(64)
      //       param = 64
    }

    bool verbose = getVerboseAttr().hasValue();
    // WHY 使用 hasValue() 检查：
    //   - UnitAttr 要么存在要么不存在
    //   - 不能直接访问，需要先检查

    // 步骤 3: 对每个目标应用转换
    SmallVector<Operation *> transformedOps;
    transformedOps.reserve(targets.size());

    for (Operation *target : targets) {
      // 场景 2: 检查前置条件
      if (!isValidTarget(target)) {
        return emitDefaultSilenceableFailure(target);
        // WHY 使用默认失败：
        //   提供标准错误消息
        //   附加目标操作位置
      }

      // 步骤 3.1: 应用具体转换逻辑
      FailureOr<Operation *> result = applyMyTransform(
          rewriter, target, param, verbose);

      if (failed(result)) {
        // 场景 3: 转换失败
        // 检查是否可以恢复
        if (result.error.isSilenceable()) {
          return result.error;
          // WHY 直接返回：
          //   SilenceableFailure 包含完整诊断信息
        } else {
          return emitDefiniteFailure()
                 << "internal error during transformation";
          // WHY 转换为 definite：
          //   内部错误应该是致命的
        }
      }

      transformedOps.push_back(*result);
      // 此时：transformedOps = [new_op1, new_op2, new_op3]
    }

    // 步骤 4: 设置结果
    results.set(cast<OpResult>(getResult()), transformedOps);
    // WHY 使用 results.set：
    //   自动建立结果 Handle 到 Payload 操作的映射
    //   调用者可以直接使用返回的 Handle

    return DiagnosedSilenceableFailure::success();
  }

  // 方法实现：MemoryEffectsOpInterface
  void getEffects(
      SmallVectorImpl<MemoryEffects::EffectInstance> &effects) override {
    // 步骤 1: 声明消费目标 Handle
    consumesHandle(getTargetMutable(), effects);
    // WHY 标记为消费：
    //   转换后原操作可能不存在
    //   Handle 不应再使用

    // 步骤 2: 声明产生结果 Handle
    producesHandle(getOperation()->getOpResults(), effects);
    // WHY 标记为生产：
    //   返回的 Handle 指向新操作

    // 步骤 3: 声明修改 Payload
    modifiesPayload(effects);
    // WHY 声明修改：
    //   转换会改变 Payload IR
    //   需要更新 Handle 映射
  }

  // 方法实现：验证器（如果 hasVerifier = 1）
  LogicalResult verify() override {
    // 步骤 1: 验证参数范围
    if (auto paramAttr = getParam()) {
      int64_t param = paramAttr.getInt();
      if (param < 0 || param > 1000) {
        return emitOpError()
               << "parameter must be in [0, 1000], got " << param;
        // WHY 返回逻辑错误：
        //   - 参数验证失败是编译时错误
        //   - 应该尽早发现
      }
    }

    return success();
  }

private:
  // 辅助方法：应用具体转换
  FailureOr<Operation *> applyMyTransform(
      TransformRewriter &rewriter,
      Operation *target,
      int64_t param,
      bool verbose) {
    // 步骤 1: 设置插入点
    rewriter.setInsertionPoint(target);
    // WHY 在目标操作之前插入：
    //   - 新操作应该插入到原操作附近
    //   - 保持 IR 结构清晰

    // 步骤 2: 创建新操作
    auto newOp = rewriter.create<MyOp>(
        target->getLoc(),
        target->getOperands(),
        target->getAttrs());
    // 此时：newOp 已创建，插入到 target 之前

    // 步骤 3: 替换原操作
    rewriter.replaceOp(target, newOp->getResults());
    // WHY 使用 replaceOp：
    //   - 删除旧操作
    //   - 更新使用关系
    //   - 通知 TrackingListener 更新映射

    if (verbose) {
      llvm::errs() << "Transformed " << target->getName()
                    << " to " << newOp->getName() << "\n";
    }

    return newOp;
  }

  // 辅助方法：验证目标
  bool isValidTarget(Operation *op) {
    // 步骤 1: 检查操作特征
    // 例如：必须实现特定接口
    return isa<MyOpInterface>(op);
  }
};

} // namespace
```

### 7.5 步骤 4：注册扩展

```cpp
// MyTransformOps.cpp
namespace my {
namespace transform {

void registerMyTransformDialectExtension(
    DialectRegistry &registry) {
  // 注册扩展到 DialectRegistry
  // WHY 使用 DialectRegistry：
  //   - 支持延迟加载
  //   - 多个扩展可以共存
  //   - 自动应用扩展
  registry.addExtensions<MyTransformDialectExtension>();
}

} // namespace transform
} // namespace my
```

**使用扩展：**

```cpp
// 在初始化代码中
void registerMyDialectExtensions(DialectRegistry &registry) {
  // 步骤 1: 注册扩展
  my::transform::registerMyTransformDialectExtension(registry);

  // 步骤 2: 确保依赖方言已注册
  registry.insert<MyDialect>();
}

// 在 Pass 中使用
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 说明：MLIRContext 会自动加载扩展
    // 因为扩展已注册到 DialectRegistry
  }
};
```

### 7.6 步骤 5：测试扩展

```cpp
// unittests/MyTransformOpsTest.cpp

namespace {

class MyTransformTest : public testing::Test {
protected:
  void SetUp() override {
    // 步骤 1: 初始化 MLIR Context
    context.loadDialect<transform::TransformDialect>();
    context.loadDialect<MyDialect>();

    // 步骤 2: 注册扩展
    DialectRegistry registry;
    my::transform::registerMyTransformDialectExtension(registry);
    context.appendDialectRegistry(registry);
  }

  MLIRContext context;
};

TEST_F(MyTransformTest, BasicTransform) {
  // 步骤 3: 构造测试 IR
  Builder builder(&context);
  auto moduleOp = builder.create<ModuleOp>(
      UnknownLoc::get(&context));

  builder.setInsertionPointToEnd(moduleOp.getBody());

  // 创建目标操作
  auto myOp = builder.create<MyOp>(
      builder.getUnknownLoc(),
      /*operands=*/{},
      /*attributes=*/{});

  // 步骤 4: 构造 Transform IR
  auto transformModule = ModuleOp::create(
      builder.getUnknownLoc(),
      builder.getStringAttr({
        // Transform IR 字符串
      }));

  parser::parseSourceString<ModuleOp>(transformModuleStr, &context);

  // 步骤 5: 应用 Transform
  TransformOptions options;
  options.enableExpensiveChecks(true);

  if (failed(applyTransformNamedSequence(
          moduleOp, entryPoint, transformModule, options))) {
    FAIL() << "Failed to apply transform";
  }

  // 步骤 6: 验证结果
  // 检查转换是否正确应用
  ASSERT_TRUE(isa<MyNewOp>(...));
}

} // namespace
```

---

---

## 8. 核心操作详解

### 8.1 transform.sequence

**作用：** 包含按顺序应用的转换序列

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %1 = transform.match.ops{"linalg.matmul"} in %arg0
  %2:2 = transform.structured.tile %1 [32, 32]
  transform.loop.unroll %2#1 { factor = 4 }
}
```

### 8.2 transform.structured.tile

**作用：** 对结构化操作进行切分

**WHY 切分：**

- **缓存优化**：提高数据局部性
- **并行化准备**：为并行执行做准备

---

## 9. 执行模型 (Execution Model)

### 9.1 执行流程概述

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

### 9.2 详细执行步骤

#### 步骤 1：解析与验证 Transform IR

```cpp
// TransformInterpreterUtils.cpp 中的执行入口
LogicalResult transform::applyTransformNamedSequence(
    RaggedArray<MappedValue> bindings, TransformOpInterface transformRoot,
    ModuleOp transformModule, const TransformOptions &options) {

  // 步骤 1: 创建 TransformState
  // 此时：transformRoot 指向 transform.sequence 或 transform.named_sequence
  //       bindings = [root_op] (Payload IR 的根操作)
  TransformState state(transformRoot->getRegion(), /*payloadRoot=*/nullptr,
                      bindings, options);

  // 步骤 2: 应用 Transform 操作
  // WHY 单独应用每个操作：支持细粒度错误处理和状态跟踪
  DiagnosedSilenceableFailure result = state.applyTransform(transformRoot);

  // 步骤 3: 检查执行结果
  if (failed(result.checkAndReport())) {
    return failure();
    // 场景 1: 执行失败
    // WHY 返回 failure 而非继续：Payload IR 可能处于不一致状态
  }

  return success();
}
```

**执行流示例：**

**场景 A：正常执行**
```cpp
// 初始状态
transformRoot = transform.sequence 操作
bindings = [root_func]  // Payload IR 中的函数

// 执行路径
1. TransformState 构造：
   - 创建顶层映射：mappings[top_level_region] = new Mappings()
   - 建立根映射：direct[kTopLevelValue] = [root_func]

2. applyTransform(transformRoot)：
   - 进入 sequence 的区域
   - 创建 RegionScope
   - 依次执行区域内的操作

3. 每个操作执行：
   - 调用 op->apply(...)
   - 成功则更新映射
   - 失败则根据类型处理

4. 返回 success()
```

**场景 B：Silenceable Failure**
```cpp
// 初始状态
某个 Transform 操作返回 SilenceableFailure

// 执行路径
1. apply() 返回 DiagnosedSilenceableFailure::silenceableFailure(...)

2. 检查 failure propagation mode：
   // 场景 1: failures(suppress)
   if (mode == FailurePropagationMode::Suppress) {
     // 忽略失败，继续执行下一个操作
     continue;
   }

   // 场景 2: failures(propagate)
   if (mode == FailurePropagationMode::Propagate) {
     // 立即传播失败
     return failure();
   }

// WHY 这样设计：
// - suppress：允许"尽力而为"的转换，尝试多个选项
// - propagate：确保关键转换必须成功
```

#### 步骤 2：TransformState 初始化

```cpp
// TransformInterfaces.cpp
TransformState::TransformState(Region *region, Operation *payloadRoot,
                               const RaggedArray<MappedValue> &extraMappings,
                               const TransformOptions &options)
    : options(options) {
  // 步骤 1: 为顶层区域创建映射
  // WHY 每个区域独立映射：支持嵌套区域（如 transform.sequence 内部）
  mappings[region] = std::make_unique<Mappings>();

  // 步骤 2: 设置根操作映射
  if (payloadRoot) {
    // 步骤 2.1: 映射顶层值到根操作
    mappings[region]->direct[kTopLevelValue] = {payloadRoot};
    // WHY 使用特殊值 kTopLevelValue：标识 Payload IR 的根
    //     允许转换从根开始查找操作
    mappings[region]->reverse[payloadRoot] = {kTopLevelValue};
  }

  // 步骤 3: 处理额外映射（用于嵌套调用）
  for (const MappedValue &mv : extraMappings) {
    // 场景 1: 操作映射
    if (auto op = llvm::dyn_cast_if_present<Operation *>(mv)) {
      // 映射到当前区域
    }
    // 场景 2: 参数映射
    else if (auto param = llvm::dyn_cast_if_present<Param>(mv)) {
      // 映射参数
    }
  }

  // 步骤 4: 压入顶层区域到栈
  regionStack.push_back(nullptr);  // 标记顶层
}
```

#### 步骤 3：应用单个 Transform 操作

```cpp
// TransformInterfaces.cpp
DiagnosedSilenceableFailure TransformState::applyTransform(
    TransformOpInterface transform) {
  // 步骤 1: 创建 TransformRewriter
  // WHY 使用专用 Rewriter：支持 TrackingListener，自动更新 Handle 映射
  TransformRewriter rewriter(transform->getContext());

  // 步骤 2: 设置 Rewriter 的监听器
  // WHY 需要 TrackingListener：自动追踪 Payload 操作的替换
  auto listener = createTrackingListener(rewriter);
  rewriter.setListener(listener.get());

  // 步骤 3: 创建 TransformResults 容器
  TransformResults results(transform->getNumResults());

  // 步骤 4: 调用操作的 apply 方法
  DiagnosedSilenceableFailure result =
      transform.apply(rewriter, results, *this);

  // 步骤 5: 处理执行结果
  // 场景 1: 成功
  if (succeeded(result.isSuccess())) {
    // 步骤 5.1: 更新 Handle 映射
    if (failed(updateStateFromResults(results, transform->getResults()))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }

    // 步骤 5.2: 处理 Handle 失效
    recordOpHandleInvalidations(transform);
    // WHY 需要显式失效：某些转换会删除/替换操作
    //     依赖这些操作的 Handle 必须被标记为无效
  }
  // 场景 2: 失败
  else {
    // Payload IR 未被修改（由转换保证）
    // 不需要回滚
  }

  return result;
}
```

**多场景执行流分析：**

**场景 A：简单转换成功**
```cpp
// Transform IR
%0 = transform.match.ops{"scf.for"} in %arg0
%1 = transform.loop.unroll %0 { factor = 4 }

// 执行流程
// 1. apply(match_ops)
//    → 查找所有 scf.for 操作
//    → results.set(result0, [for1, for2, for3])
//    → 映射：direct[%0] = [for1, for2, for3]

// 2. apply(loop.unroll)
//    → 获取 targets = state.getPayloadOps(%0) = [for1, for2, for3]
//    → 对每个循环应用 unroll
//    → for1 → for1_unrolled (替换)
//    → TrackingListener 通知：for1 被替换为 for1_unrolled
//    → 更新映射：direct[%0] = [for1_unrolled, for2_unrolled, for3_unrolled]

// 3. recordOpHandleInvalidations(loop.unroll)
//    → 检查 %0 是否被消费
//    → consumesHandle(%0) → true
//    → 失效 %0：从映射中移除
//    → WHY 需要移除：循环已被 unroll，原操作不存在
```

**场景 B：转换失败（Silenceable）**
```cpp
// Transform IR
%0 = transform.match.ops{"scf.for"} in %arg0
%1 = transform.loop.unroll %0 { factor = 4 }
// 假设某个循环的迭代次数不是 4 的倍数

// 执行流程
// 1. apply(match_ops) → success
//    direct[%0] = [for1, for2, for3]

// 2. apply(loop.unroll)
//    → 尝试 unroll for1
//    → 检测：for1 的迭代次数 = 10，不能被 4 整除
//    → 返回 emitSilenceableError("iteration count not divisible by 4")
//    → 场景 B.1: failures(suppress)
//        - 忽略错误，继续执行
//        - Payload IR 保持不变
//    → 场景 B.2: failures(propagate)
//        - 立即返回 failure
//        - 后续操作不执行

// WHY 不回滚 Payload IR：
// - 转换操作保证失败时不修改 IR
// - 使用 Rewriter 的 undo 机制
// - 或在修改前进行验证
```

### 9.3 失败处理机制

Transform 方言区分两种失败模式，每种有不同的语义和处理方式。

#### 9.3.1 Silenceable Failure（可恢复失败）

**定义：** 转换未能应用，但 Payload IR 未被修改，可以尝试其他转换。

**特征：**
- 转换未修改 Payload IR（原子性保证）
- 可以尝试备选转换
- 延迟报告错误（积累多个失败后统一报告）

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

// 执行流程：
// 1. 尝试策略 A
//    - 如果向量化失败（Silenceable Failure）
//    - 回滚策略 A 的修改
//    - 尝试策略 B
// 2. 如果策略 B 成功
//    - 返回策略 B 的结果
```

**WHY 这样设计：**

- **灵活性**：允许"尽力而为"的转换策略
- **容错性**：某个转换失败不影响整个流程
- **探索性**：尝试多种优化，选择最佳的

**代码示例：**

```cpp
// 产生 Silenceable Failure
DiagnosedSilenceableFailure MyTransformOp::apply(...) {
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // 场景 1: 没有目标操作
  if (targets.empty()) {
    return emitSilenceableError()
           << "no operations found to transform";
    // WHY 使用 silenceable：
    //   这是可预期的空结果，不是致命错误
    //   调用者可以决定如何处理（跳过、使用默认值等）
  }

  // 场景 2: 前置条件不满足
  for (Operation *target : targets) {
    if (!hasRequiredTrait(target)) {
      return emitDefaultSilenceableFailure(target);
      // WHY 使用默认失败：
      //   提供标准的错误消息和位置信息
      //   帮助用户快速定位问题
    }
  }

  // 场景 3: 转换成功
  // ... 执行转换 ...
  return DiagnosedSilenceableFailure::success();
}
```

#### 9.3.2 Definite Failure（不可恢复失败）

**定义：** Payload IR 可能处于不一致状态，必须立即停止。

**特征：**
- Payload IR 可能已被部分修改
- 必须立即停止，不能继续执行
- 立即报告错误

**使用场景：**

```cpp
// 产生 Definite Failure
DiagnosedSilenceableFailure MyTransformOp::apply(...) {
  // 场景 1: 内部错误
  if (internalStateCorrupted) {
    return emitDefiniteFailure()
           << "internal state corrupted";
    // WHY 使用 definite：
    //   系统处于不可预测状态
    //   继续执行会导致未定义行为
  }

  // 场景 2: 违反核心约束
  if (violatesCoreInvariant(payloadIR)) {
    auto diag = emitDefiniteFailure() << "core invariant violated";
    diag.attachNote(payloadIR.getLoc()) << "inconsistent IR state";
    return diag;
    // WHY 立即停止：
    //   约束被违反可能导致后续转换崩溃
    //   必须中止整个转换流程
  }

  return DiagnosedSilenceableFailure::success();
}
```

**WHY 区分两种失败：**

| 特性 | Silenceable | Definite |
|------|-------------|----------|
| Payload IR 状态 | 未修改 | 可能不一致 |
| 后续操作 | 可以继续 | 必须停止 |
| 错误报告 | 可延迟 | 立即报告 |
| 典型场景 | 前置条件不满足 | 内部错误/约束违反 |

### 9.4 Handle 失效规则 (Handle Invalidation)

当 Transform 操作消费或修改 Payload 操作时，相关的 Handle 会自动失效。

#### 9.4.1 失效触发条件

```cpp
// TransformInterfaces.cpp 中的失效逻辑
void TransformState::recordOpHandleInvalidations(
    TransformOpInterface transform) {
  // 步骤 1: 获取被消费的 Handle 操作数
  SmallVector<OpOperand *> consumedOperands =
      getConsumedHandleOpOperands(transform);

  // 步骤 2: 检查每个被消费的 Handle
  for (OpOperand *operand : consumedOperands) {
    Value handle = operand->get();

    // 步骤 2.1: 获取 Handle 关联的所有 Payload 操作
    ArrayRef<Operation *> payloadOps = getPayloadOpsView(handle);

    // 步骤 2.2: 检查每个 Payload 操作
    for (Operation *payloadOp : payloadOps) {
      // 场景 1: 操作已被替换/删除
      if (payloadOp->isDead()) {
        // Handle 必须失效
        invalidatedHandles.insert(handle);
      }

      // 步骤 2.2.2: 检查嵌套操作
      // 如果消费的操作包含嵌套操作，这些嵌套操作的 Handle 也失效
      for (Operation &nested : payloadOp->getRegions()) {
        recursivelyInvalidateHandles(nested);
      }
    }
  }

  // 步骤 3: 处理 Value Handle 失效
  // 当产生 Value 的操作被消费时，指向这些 Value 的 Handle 失效
  for (OpOperand *operand : consumedOperands) {
    Value handle = operand->get();
    for (Operation *payloadOp : getPayloadOpsView(handle)) {
      // 步骤 3.1: 操作的结果值
      for (Value result : payloadOp->getResults()) {
        // 步骤 3.1.1: 查找指向此结果的所有 Value Handle
        SmallVector<Value> valueHandles;
        (void)getHandlesForPayloadValue(result, valueHandles);

        // 步骤 3.1.2: 失效这些 Value Handle
        for (Value valueHandle : valueHandles) {
          invalidatedHandles.insert(valueHandle);
        }
      }

      // 步骤 3.2: 操作的块参数
      for (Region &region : payloadOp->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            // 类似地，查找并失效指向块参数的 Handle
          }
        }
      }
    }
  }
}
```

#### 9.4.2 失效规则图解

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

**示例：**

```mlir
// Transform IR
%0 = transform.match.ops{"scf.for"} in %root
%1 = transform.get_loop_body %0 : (!transform.any_op) -> !transform.any_value
%2 = transform.loop.unroll %0 { full }

// 此时：%0 被消费
// 失效规则：
// 1. %0 失效（被消费的 Handle）
// 2. %1 失效（%1 指向 %0 操作体的值）
// WHY %1 失效：unroll 后，原循环体结构被改变
```

**WHY 这样设计：**

- **安全性**：防止引用已删除/替换的操作
- **一致性**：确保 Handle 指向有效的 Payload IR
- **可预测性**：明确的失效规则，易于理解

#### 9.4.3 失效检测与错误报告

```cpp
// 尝试使用已失效的 Handle
DiagnosedSilenceableFailure MyTransformOp::apply(...) {
  Value handle = getTarget();

  // 场景 1: 检查 Handle 是否已失效
  if (state.isInvalidated(handle)) {
    return emitSilenceableError()
           << "handle has been invalidated by a previous transformation";
    // WHY 报告错误：
    //   使用已失效的 Handle 是程序错误
    //   但不是致命错误，可以 silenceable
  }

  // 场景 2: 正常执行
  ArrayRef<Operation *> targets = state.getPayloadOps(handle);
  // ...
}
```

**执行流示例：**

**场景 A：Handle 正常使用**

```mlir
// Transform IR
%0 = transform.match.ops{"scf.for"} in %root
%1 = transform.get_iter_args %0 : (!transform.any_op) -> !transform.any_value
transform.print %1

// 执行流程：
// 1. %0 → [for1, for2]
// 2. %1 → [for1.arg, for2.arg]  (迭代参数)
// 3. print(%1) → 成功打印
//    %1 和 %0 都未被失效
```

**场景 B：Handle 被失效**
```mlir
// Transform IR
%0 = transform.match.ops{"scf.for"} in %root
%1 = transform.get_iter_args %0 : (!transform.any_op) -> !transform.any_value
transform.loop.unroll %0 { full }  // 消费 %0
transform.print %1  // 错误：%1 已失效

// 执行流程：
// 1. %0 → [for1, for2]
// 2. %1 → [for1.arg, for2.arg]
// 3. loop.unroll(%0)
//    → 消费 %0
//    → %0 被标记为失效
//    → %1 被标记为失效（指向 %0 的结果）
// 4. print(%1)
//    → 检测到 %1 已失效
//    → 返回 Silenceable Failure
//    → "handle has been invalidated"
```

### 9.5 TransformRewriter 与 TrackingListener

#### 9.5.1 TransformRewriter 的特殊功能

```cpp
// TransformRewriter 是 PatternRewriter 的子类
class TransformRewriter : public PatternRewriter {
public:
  // 方法实现：替换操作时自动更新 Handle 映射
  void replaceOp(Operation *op, ValueRange newValues) override {
    // 步骤 1: 通知 TrackingListener
    if (listener) {
      listener->notifyOperationReplaced(op, newValues);
      // WHY 通知：让 TrackingListener 更新 Handle 映射
    }

    // 步骤 2: 执行实际替换
    PatternRewriter::replaceOp(op, newValues);
  }

  // 方法实现：删除操作时自动清理 Handle
  void eraseOp(Operation *op) override {
    if (listener) {
      listener->notifyOperationErased(op);
      // WHY 通知：移除指向被删除操作的 Handle
    }

    PatternRewriter::eraseOp(op);
  }

  // 方法实现：创建操作时自动跟踪
  void insert(Operation *op) override {
    if (listener) {
      listener->notifyOperationInserted(op);
      // WHY 通知：记录新创建的操作
      //     某些转换需要返回新操作的 Handle
    }

    PatternRewriter::insert(op);
  }
};
```

#### 9.5.2 TrackingListener 的映射更新逻辑

```cpp
// TrackingListener 实现
class TrackingListener : public RewriterBase::Listener {
public:
  // 方法实现：操作被替换
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    // 步骤 1: 查找所有指向旧操作的 Handle
    SmallVector<Value> handles;
    (void)state.getHandlesForPayloadOp(op, handles);

    // 步骤 2: 更新每个 Handle 的映射
    for (Value handle : handles) {
      // 场景 1: 新值是操作结果
      if (!newValues.empty()) {
        Operation *newOp = newValues[0].getDefiningOp();
        if (newOp) {
          // 步骤 2.1: 更新映射：旧操作 → 新操作
          state.updateMapping(handle, op, newOp);
          // WHY 更新而非删除：
          //   转换是"替换"而非"删除"
          //   Handle 应该指向新操作
        }
      }
      // 场景 2: 新值是常量/参数
      else {
        // Handle 失效（操作被完全移除）
        state.invalidateHandle(handle);
      }
    }
  }

  // 方法实现：操作被删除
  void notifyOperationErased(Operation *op) override {
    SmallVector<Value> handles;
    (void)state.getHandlesForPayloadOp(op, handles);

    for (Value handle : handles) {
      // 从映射中移除
      state.forgetMapping(handle);
      // WHY 而非 invalidate：
      //   操作被删除，无法追踪
      //   必须从映射中移除
    }
  }

private:
  TransformState &state;
};
```

**执行流示例：**

**场景：操作替换**
```cpp
// 初始状态
Handle %0 → [matmul_op]
Handle %1 → [matmul_op.result]  (ValueHandle)

// 执行：transform.apply_patterns 将 matmul 替换为 generic
rewriter.replaceOp(matmul_op, new_generic_op.getResults());

// TrackingListener 动作：
// 1. notifyOperationReplaced(matmul_op, new_results)
// 2. 查找 Handle：[%0, %1]
// 3. 更新 %0：matmul_op → new_generic_op
//    direct[%0] = [new_generic_op]
// 4. 更新 %1：matmul_op.result → new_generic_op.result
//    values[%1] = [new_generic_op.result]

// 最终状态：
Handle %0 → [new_generic_op]
Handle %1 → [new_generic_op.result]
```

---

## 10. 预期用途与集成 (Intended Use and Integrations)

### 10.1 Transform 方言的预期用途

Transform 方言旨在解决以下特定问题：

#### 10.1.1 问题域

**1. 精细转换控制**

传统 Pass Pipeline 的问题：
- Pass 粒度太粗，对所有匹配的操作应用相同转换
- 无法针对特定操作应用不同策略
- Pass 组合困难，需要编写新的 Pass

Transform 方言的解决方案：
```mlir
// 场景：只对特定大小的循环应用切分
transform.sequence {
^bb0(%arg0: !transform.any_op):
  %loops = transform.match.ops{"scf.for"} in %arg0

  // 获取循环的迭代次数
  %trip_counts = transform.get_trip_count %loops

  // 只对大循环切分
  %large_loops = transform.filter_by_size %loops, %trip_counts { min = 1000 }
  transform.tile %large_loops { sizes = [32, 32] }

  // 小循环保持不变
}
```

**2. 转换组合与重用**

传统方式的问题：
- Pass 组合需要编写新的 C++ 代码
- 难以在不同项目间共享转换策略

Transform 方言的解决方案：
```mlir
// 定义可复用的转换序列
transform.named_sequence @optimize_conv2d(%arg: !transform.any_op) {
  %padded = transform.pad %arg { pad_size = [1, 1] }
  %tiled:2 = transform.tile %padded [16, 16]
  %vectorized = transform.vectorize %tiled#1
  transform.yield %vectorized
}

// 在多处复用
transform.sequence {
^bb0(%root: !transform.any_op):
  %convs = transform.match.ops{"linalg.conv_2d"} in %root
  %optimized = transform.include @optimize_conv2d(%convs)
}
```

**3. 条件转换与回退**

传统方式的问题：
- 无法根据运行时信息选择转换策略
- 某个转换失败时，无法尝试备选方案

Transform 方言的解决方案：
```mlir
// 场景：根据操作大小选择策略
transform.sequence {
^bb0(%arg0: !transform.any_op):
  %ops = transform.match.ops{"linalg.matmul"} in %arg0

  // 尝试向量化
  %try_vector = transform.alternatives {
  ^bb0(%ops: !transform.any_op):
    // 策略 A：向量化
    %v = transform.vectorize %ops
    transform.yield %v
  }, {
  ^bb0(%ops: !transform.any_op):
    // 策略 B：向量化失败，使用标量优化
    %s = transform.scalarize %ops
    transform.yield %s
  }

  transform.yield %try_vector
}
```

#### 10.1.2 不是为了解决什么

**Transform 方言不适用于：**

| 问题 | WHY 不适用 | 替代方案 |
|------|-----------|---------|
| 全局分析（如数据流分析） | Transform 是局部操作导向 | 使用 Pass |
| 跨模块优化 | Transform 作用于单个 Module | 使用 IPO Pass |
| 需要复杂状态分析的优化 | Transform 状态有限 | 使用分析 Pass |
| 性能关键路径 | Transform IR 有开销 | 使用 C++ Pass |

### 10.2 与 MLIR 编译流程的集成

#### 10.2.1 典型集成点

```
MLIR 编译流程中的 Transform 方言
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Frontend → High-level IR → [Transform方言] → Low-level IR → Backend │
│                      ↑                                          │
│                   灵活的转换控制点                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**集成模式 1：Pass 中使用 Transform**

```cpp
// C++ Pass 实现
struct ApplyTransformPass
    : public PassWrapper<ApplyTransformPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 步骤 1: 加载 Transform 模块
    // WHY 从文件加载：允许用户自定义转换策略
    OwningOpRef<ModuleOp> transformModule;
    if (failed(parseTransformFromFile(transformModulePath, transformModule))) {
      signalPassFailure();
      return;
    }

    // 步骤 2: 查找入口点
    TransformOpInterface entryPoint =
        findTransformEntryPoint(module, *transformModule, "__transform_main");

    // 步骤 3: 应用 Transform
    TransformOptions options;
    options.strictMode = this->strictMode;
    // WHY 配置选项：允许控制执行行为

    if (failed(applyTransformNamedSequence(module, entryPoint,
                                          *transformModule, options))) {
      // 场景 1: 转换失败
      signalPassFailure();
    }
  }

private:
  Option<std::string> transformModulePath{
      *this, "transform-module",
      llvm::cl::desc("Path to transform module file")};

  Option<bool> strictMode{
      *this, "strict-mode",
      llvm::cl::desc("Enable strict mode"), llvm::cl::init(false)};
};
```

**集成模式 2：脚本驱动转换**

```python
# Python 脚本中使用 Transform
from mlir import ir
from mlir.dialects import transform, builtin, linalg

# 步骤 1: 创建 Context 并加载方言
ctx = ir.Context()
ctx.load_all_dialects()

# 步骤 2: 解析 Payload IR
payload_module = ctx.parse_op("""
module {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>)
      -> tensor<128x128xf32> {
    %C = linalg.matmul ins(%A, %B: tensor<128x128xf32>, tensor<128x128xf32>)
        outs(%init: tensor<128x128xf32>)
    return %C : tensor<128x128xf32>
  }
}
""")

# 步骤 3: 构建 Transform IR
with ir.InsertionContext(payload_module):
    transform_module = builtin.ModuleOp()
    with transform_module.body:
        # 定义优化序列
        @transform.named_sequence("@optimize")
        def optimize(root):
            matmuls = transform.structured.match(["linalg.matmul"], root)
            tiled, loops = transform.structured.tile_using_for(matmuls, [32, 32])
            transform.structured.vectorize(loops)
            return tiled

# 步骤 4: 应用 Transform
transform.apply_named_sequence(payload_module, transform_module)
```

#### 10.2.2 配置选项

```cpp
// TransformOptions 配置
struct TransformOptions {
  // 定义：严格模式
  // WHY 需要严格模式：
  //   - 非严格模式：允许部分 Handle 失效
  //   - 严格模式：任何失效都是错误
  bool strictMode = false;

  // 定义：调试模式
  // WHY 调试模式：
  //   - 打印详细执行信息
  //   - 显示每个操作的 Handle 映射变化
  bool debugMode = false;

  // 定义：超时设置
  // WHY 需要超时：
  //   - 防止无限循环
  //   - 限制资源使用
  std::optional<uint64_t> timeoutSeconds;

  // 定义：扩展选项
  // WHY 允许自定义选项：
  //   - 不同方言可能有特定配置
  //   - 扩展可以添加自己的选项
  llvm::StringMap<std::string> extensionOptions;
};
```

### 10.3 最佳实践

#### 10.3.1 编写 Transform 序列

**DO（推荐做法）：**

```mlir
// ✅ 使用 named_sequence 组织代码
transform.named_sequence @optimize_op(%arg: !transform.any_op) {
  %1 = transform.tile %arg [32]
  %2 = transform.vectorize %1
  transform.yield %2
}

// ✅ 使用 include 复用
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"my.op"} in %root
  %result = transform.include @optimize_op(%ops)
  transform.yield %result
}

// ✅ 添加错误处理
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"my.op"} in %root
  // 如果转换失败，立即停止
  %result = transform.apply_patterns to %ops { ... }
  transform.yield %result
}
```

**DON'T（不推荐做法）：**

```mlir
// ❌ 过长的内联序列
transform.sequence {
^bb0(%root: !transform.any_op):
  %1 = transform.step1 %root
  %2 = transform.step2 %1
  %3 = transform.step3 %2
  // ... 50 多行 ...
  %50 = transform.step50 %49
  // WHY 不推荐：
  //   - 难以阅读和维护
  //   - 无法复用
  //   - 错误难以定位
}

// ❌ 重复代码
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops1 = transform.match.ops{"op1"} in %root
  %tiled1 = transform.tile %ops1 [32]
  %vect1 = transform.vectorize %tiled1

  %ops2 = transform.match.ops{"op2"} in %root
  %tiled2 = transform.tile %ops2 [32]  // 重复
  %vect2 = transform.vectorize %tiled2  // 重复
  // 应该使用 named_sequence 复用
}
```

#### 10.3.2 调试技巧

**技巧 1：使用 print 调试**

```mlir
transform.sequence {
^bb0(%root: !transform.any_op):
  // 在每个关键步骤后打印
  %ops = transform.match.ops{"scf.for"} in %root
  transform.print %ops { name = "Matched loops" }

  %tiled = transform.tile %ops [32]
  transform.print %tiled { name = "After tiling" }

  %vect = transform.vectorize %tiled
  transform.print %vect { name = "After vectorization" }
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
  // 如果 IR 不合法，会立即失败
}
```

**技巧 3：使用 annotate 追踪操作**

```mlir
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root

  // 添加标记，方便调试
  transform.annotate %ops "matched_loop"

  %tiled = transform.tile %ops [32]

  // 给新操作添加标记
  transform.annotate %tiled "tiled_loop"
  // 在生成的 IR 中可以看到这些属性
}
```

#### 10.3.3 性能考虑

**考虑 1：减少 Handle 查找**

```mlir
// ❌ 不推荐：重复查找
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops1 = transform.match.ops{"scf.for"} in %root
  // 使用 %ops1

  %ops2 = transform.match.ops{"scf.for"} in %root  // 重复查找
  // 使用 %ops2
}

// ✅ 推荐：复用 Handle
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root
  // 使用 %ops
}
```

**考虑 2：批量操作**

```mlir
// ✅ 推荐：一次处理所有操作
transform.sequence {
^bb0(%root: !transform.any_op):
  %all_ops = transform.match.ops{"scf.for"} in %root
  // 一次性处理所有循环
  transform.tile %all_ops [32]
}

// ❌ 不推荐：逐个处理
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root
  // 假设 %ops = [for1, for2, for3]
  // 需要分别处理每个
}
```

---

## 10. Linalg Transform 操作

### 10.1 切分 (Tiling)

**操作：** `transform.structured.tile_using_for`

**WHY 切分：**

- **局部性**：提高数据重用
- **并行化**：为并行执行创造机会

### 10.2 向量化 (Vectorization)

**操作：** `transform.structured.vectorize`

**WHY 向量化：**

- **SIMD 利用**：充分利用硬件向量单元
- **循环消除**：减少循环开销

---

## 11. 实战案例

### 11.1 矩阵乘法优化流程

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @optimize_matmul(%arg0: !transform.any_op) {
    // 1. 匹配矩阵乘法
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg0

    // 2. 切分
    %tiled:2 = transform.structured.tile_using_for %matmuls tile_sizes [32, 32]

    // 3. 向量化
    transform.structured.vectorize %tiled#1 vector_sizes [4]

    transform.yield %tiled#0
  }
}
```

### 11.2 GPU 映射案例

```mlir
transform.sequence {
^bb0(%arg0: !transform.any_op):
  %bufferized = transform.bufferization.one_shot_bufferize %arg0
  %launch = transform.gpu.map_forall_to_blocks %bufferized grid_dims = [256, 256]
  transform.gpu.map_nested_forall_to_threads %launch block_dims = [32, 32]
}
```

---

## 12. 性能与调试

### 12.1 性能考虑

- **映射查找**：使用 DenseMap 实现 O(1) 平均查找
- **延迟验证**：类型约束在执行时验证
- **增量更新**：Handle 映射增量更新

### 12.2 调试技巧

```mlir
// 使用 transform.print 查看中间状态
transform.print %target { name = "After tiling" }

// 使用 transform.verify 确保正确性
transform.verify %target
```

---

## 13. 参考资料

### 13.1 官方文档

- [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)

### 13.2 源代码位置

| 组件 | 路径 |
|------|------|
| 核心方言定义 | `mlir/include/mlir/Dialect/Transform/IR/` |
| 核心实现 | `mlir/lib/Dialect/Transform/IR/` |
| Linalg Transform | `mlir/lib/Dialect/Linalg/TransformOps/` |
| 接口定义 | `mlir/include/mlir/Dialect/Transform/Interfaces/` |

---

## 附录：术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| Payload IR | Payload IR | 被转换的目标 IR |
| Transform IR | Transform IR | 控制转换逻辑的 IR |
| Handle | Handle | 指向 Payload IR 对象的引用 |
| Silenceable Failure | Silenceable Failure | 可恢复的失败 |
| Definite Failure | Definite Failure | 不可恢复的失败 |
