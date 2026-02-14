# MLIR Transform 方言技术原理详解 v2.0

> 基于 MLIR 官方文档 [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/#overview) 和源代码 `mlir/lib/Dialect/Transform/`、`mlir/lib/Dialect/Linalg/TransformOps/` 深度分析生成。

## 目录

1. [快速概览](#1-快速概览)
2. [背景与动机](#2-背景与动机)
3. [核心概念](#3-核心概念)
4. [类型系统](#4-类型系统)
5. [源码实现：TransformState](#5-源码实现transformstate)
6. [源码实现：TransformDialectExtension](#6-源码实现transformdialectextension)
7. [执行模型](#7-执行模型)
8. [扩展开发完整教程](#8-扩展开发完整教程)
9. [参考资料](#9-参考资料)

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

## 5. 源码实现：TransformState

### 5.1 核心数据结构

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

### 5.2 Mappings 结构

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

### 5.3 核心方法：setPayloadOps

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

### 5.4 RegionScope - 区域作用域管理

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

## 6. 源码实现：TransformDialectExtension

### 6.1 扩展机制背景与动机

**问题：** Transform 方言需要扩展？

- 不同方言需要特定的转换操作
- 核心方言不应依赖特定方言
- 需要延迟加载和类型安全

**设计目标：**

1. **延迟加载**：扩展只在需要时加载
2. **解耦合**：Transform 方言不依赖特定方言
3. **类型安全**：自动验证扩展操作的接口实现
4. **易用性**：简单的 API 注册操作和类型

### 6.2 扩展机制设计原理

#### 6.2.1 CRTP 模式

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

#### 6.2.2 初始化流程

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

### 6.3 核心组件详解

#### 6.3.1 registerTransformOps - 注册操作

```cpp
// 使用示例
void MyExtension::init() {
  registerTransformOps<
#define GET_OP_LIST
#include "MyTransformOps.cpp.inc"
  >();
}
```

#### 6.3.2 declareDependentDialect vs declareGeneratedDialect

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

#### 6.3.3 registerTypes - 注册类型

```cpp
// 注册自定义类型
void registerTypes() {
  dialect->addTypes<
#define GET_TYPEDEF_LIST
#include "MyTransformTypes.cpp.inc"
  >();
}
```

### 6.4 完整扩展示例：LinalgTransformDialectExtension

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

### 6.5 TransformDialectData - 扩展间通信机制

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

### 6.6 扩展自动加载机制详解

#### 6.6.1 完整加载流程

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

#### 6.6.2 扩展应用逻辑

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

## 7. 执行模型 (Execution Model)

### 7.1 执行流程概述

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

### 7.2 详细执行步骤

#### 7.2.1 解析与验证 Transform IR

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

#### 7.2.2 应用单个 Transform 操作

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

### 7.3 失败处理机制

#### 7.3.1 Silenceable Failure（可恢复失败）

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

#### 7.3.2 Definite Failure（不可恢复失败）

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

### 7.4 Handle 失效规则 (Handle Invalidation)

当 Transform 操作消费或修改 Payload 操作时，相关的 Handle 会自动失效。

#### 7.4.1 失效触发条件

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

#### 7.4.2 失效规则图解

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

### 7.5 TransformRewriter 与 TrackingListener

#### 7.5.1 TransformRewriter 的特殊功能

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

#### 7.5.2 TrackingListener 的映射更新逻辑

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

## 8. 扩展开发完整教程

### 8.1 扩展开发步骤概览

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

### 8.2 步骤 1：定义扩展类

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

### 8.3 步骤 2：使用 TableGen 定义操作

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

### 8.4 步骤 3：实现 C++ 类

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

### 8.5 步骤 4：注册扩展

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

### 8.6 步骤 5：测试扩展

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

### 8.7 最佳实践

#### 8.7.1 编写 Transform 序列

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

#### 8.7.2 调试技巧

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

#### 8.7.3 性能考虑

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

## 9. 参考资料

### 9.1 官方文档

- [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)

### 9.2 源代码位置

| 组件 | 路径 |
|------|------|
| 核心方言定义 | `mlir/include/mlir/Dialect/Transform/IR/` |
| 核心实现 | `mlir/lib/Dialect/Transform/IR/` |
| Linalg Transform | `mlir/lib/Dialect/Linalg/TransformOps/` |
| 接口定义 | `mlir/include/mlir/Dialect/Transform/Interfaces/` |

### 9.3 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| Payload IR | Payload IR | 被转换的目标 IR |
| Transform IR | Transform IR | 控制转换逻辑的 IR |
| Handle | Handle | Transform IR 中指向 Payload IR 对象的引用 |
| Silenceable Failure | Silenceable Failure | 可恢复失败 |
| Definite Failure | Definite Failure | 不可恢复失败 |
