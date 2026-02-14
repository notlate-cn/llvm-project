# MLIR Transform 方言技术原理详解

> 本文档基于 MLIR 官方文档 [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/#overview) 和源代码 `mlir/lib/Dialect/Transform/`、`mlir/lib/Dialect/Linalg/TransformOps/` 全面分析生成，旨在帮助初学者掌握 Transform 方言的核心概念和使用方法。

---

## 理解验证状态

| 核心概念 | 自我解释 | 理解"为什么" | 应用迁移 | 状态 |
|---------|---------|-------------|---------|------|
| Payload IR 与 Transform IR 分离 | ✅ | ✅ | ✅ | 已理解 |
| Handle 类型系统 | ✅ | ✅ | ⚠️ | 基本理解 |
| TransformOpInterface | ✅ | ✅ | ⚠️ | 基本理解 |
| 扩展机制 | ✅ | ✅ | ❌ | 需深入理解 |
| 失败处理模式 | ✅ | ✅ | ⚠️ | 基本理解 |
| Handle 失效规则 | ✅ | ✅ | ⚠️ | 基本理解 |

---

## 目录

1. [快速概览](#1-快速概览)
2. [背景与动机](#2-背景与动机)
3. [核心概念](#3-核心概念)
4. [类型系统](#4-类型系统)
5. [核心操作详解](#5-核心操作详解)
6. [扩展机制](#6-扩展机制)
7. [执行模型](#7-执行模型)
8. [Linalg Transform 操作](#8-linalg-transform-操作)
9. [实践示例](#9-实践示例)
10. [参考资料](#10-参考资料)

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

### 1.2 核心设计理念

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Transform 方言架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────┐         ┌──────────────────┐                 │
│   │  Transform IR    │         │   Payload IR     │                 │
│   │  (控制转换逻辑)    │──────▶  │   (被转换的IR)    │                  │
│   └──────────────────┘         └──────────────────┘                 │
│          │                                                          │
│          │ 通过 Handle 关联                                          │
│          ▼                                                          │
│   ┌──────────────────┐                                              │
│   │  Handle 类型系统  │                                              │
│   │ • OperationHandle│                                              │
│   │ • ValueHandle    │                                              │
│   │ • ParamHandle    │                                              │
│   └──────────────────┘                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
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
2. **无法组合**：想要"先平铺最内层循环，再平铺次内层"需要编写新 Pass
3. **无法回退**：某种转换失败时，无法尝试备选方案

### 2.2 Transform 方言的解决方案

```mlir
// 使用 Transform 方言精细控制转换
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // 1. 找到所有循环
  %loops = transform.loop.structure %arg0 : (!transform.any_op) -> !transform.any_op

  // 2. 只对最内层循环应用平铺
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
   // 平铺 %0 关联的所有循环
   ```

### 3.3 三种 Handle 类型接口

Transform 方言定义了三个核心类型接口：

#### 3.3.1 TransformHandleTypeInterface

**作用：** 指向 Payload IR 操作的 Handle

```cpp
// TransformDialect.h:146-155
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

**示例类型：**
- `!transform.any_op` - 任意操作
- `!transform.op<"linalg.matmul">` - 特定操作

#### 3.3.2 TransformValueHandleTypeInterface

**作用：** 指向 Payload IR 值（SSA Value）的 Handle

```cpp
// TransformDialect.h:168-177
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

**示例类型：**
- `!transform.any_value` - 任意值

#### 3.3.3 TransformParamTypeInterface

**作用：** 指向编译时参数（Attribute）的 Handle

```cpp
// TransformDialect.h:157-166
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

**示例类型：**
- `!transform.param<i32>` - 32位整数参数
- `!transform.param<index>` - 索引类型参数
- `!transform.any_param` - 任意类型参数

### 3.4 TransformOpInterface

**概念：** 所有 Transform 操作必须实现的核心接口

```cpp
// TransformInterfaces.td:18-70
def TransformOpInterface : OpInterface<"TransformOpInterface"> {
  let description = [{
    This interface is to be implemented by operations that identify
    transformations to be performed on other operations.
  }];

  let methods = [
    InterfaceMethod<
      /*returnType=*/"::mlir::DiagnosedSilenceableFailure",
      /*name=*/"apply",
      /*arguments=*/(ins
          "::mlir::transform::TransformRewriter &":$rewriter,
          "::mlir::transform::TransformResults &":$transformResults,
          "::mlir::transform::TransformState &":$state
      )
    >,
    // ...
  ];
}
```

**WHY 需要这个接口：**
- **统一执行模型**：所有转换通过相同的 `apply` 方法执行
- **错误处理**：返回 `DiagnosedSilenceableFailure` 支持可恢复失败
- **状态管理**：通过 `TransformState` 访问 Payload IR 映射

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

### 4.2 核心类型详解

#### 4.2.1 AnyOpType

**语法：** `!transform.any_op`

**用途：** 指向任意 Payload IR 操作的 Handle

**验证约束：** 无特殊约束

```mlir
// 示例
%all_ops = transform.match.ops{"scf.for","arith.addf"} in %root
           : (!transform.any_op) -> !transform.any_op
```

#### 4.2.2 OperationType

**语法：** `!transform.op<"operation_name">`

**用途：** 指向特定类型操作的 Handle

**参数：**
| 参数 | 类型 | 描述 |
|------|------|------|
| `operation_name` | `StringRef` | 允许的 Payload 操作名称 |

**WHY 使用 OperationType：**
- **类型安全**：编译时验证操作类型
- **操作特化**：某些转换只适用于特定操作

```mlir
// 示例：只指向 scf.for 操作的 Handle
%loops = transform.match.ops{"scf.for"} in %root
         : (!transform.any_op) -> !transform.op<"scf.for">
```

#### 4.2.3 ParamType

**语法：** `!transform.param<Type>`

**用途：** 指向特定类型参数的 Handle

**参数：**
| 参数 | 类型 | 描述 |
|------|------|------|
| `type` | `Type` | 参数的底层类型 |

```mlir
// 示例
%tile_size = transform.param.constant 64 : i32
             -> !transform.param<i32>
```

### 4.3 类型验证机制

每个类型接口都要求实现 `checkPayload` 方法：

```cpp
// TransformInterfaces.td:124-136
InterfaceMethod<
  /*desc=*/[{
    Checks if the given associated objects (Payload IR operations or attributes)
    satisfy the conditions defined by this type.
  }],
  /*returnType=*/"::mlir::DiagnosedSilenceableFailure",
  /*name=*/"checkPayload",
  /*arguments=*/(ins "::mlir::Location":$loc,
                     "::mlir::ArrayRef<" # cppObjectType # ">":$payload)
>
```

**WHY 这样设计：**
- **延迟验证**：类型约束在 Transform 执行时验证，而非解析时
- **灵活性与安全性平衡**：允许更宽松的类型，运行时检查

---

## 5. 核心操作详解

### 5.1 顶层操作（Top-Level Operations）

#### 5.1.1 transform.sequence

**作用：** 包含按顺序应用的转换序列

**语法：**
```mlir
transform.sequence failures(propagation_mode) (%root: T) -> (T1, ..., Tn) {
  // 转换操作序列
  transform.yield %result1, ..., %resultn : T1, ..., Tn
}
```

**参数：**
- `root`：可选的根操作 Handle
- `failure_propagation_mode`：失败传播模式
  - `propagate`：任何失败立即传播
  - `suppress`：忽略可恢复失败

**WHY 使用 sequence：**
- **组织转换流程**：将相关转换分组
- **错误处理**：统一管理转换失败

```mlir
// 示例
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %1 = transform.match.ops{"linalg.matmul"} in %arg0
  %2:2 = transform.structured.tile %1 [32, 32]
  transform.loop.unroll %2#1 { factor = 4 }
}
```

#### 5.1.2 transform.named_sequence

**作用：** 定义可复用的命名转换序列

**语法：**
```mlir
transform.named_sequence @my_sequence(%arg: !transform.any_op) -> !transform.any_op {
  // 转换操作
  transform.yield %result : !transform.any_op
}
```

**WHY 使用 named_sequence：**
- **代码复用**：通过 `transform.include` 调用
- **模块化设计**：将复杂转换分解为小函数

```mlir
// 定义
transform.named_sequence @tile_and_unroll(%arg: !transform.any_op) {
  %1 = transform.structured.tile %arg [32, 32]
  transform.loop.unroll %1 { factor = 4 }
  transform.yield %1 : !transform.any_op
}

// 调用
transform.sequence {
^bb0(%arg0: !transform.any_op):
  %1 = transform.include @tile_and_unroll(%arg0)
}
```

### 5.2 Handle 操作

#### 5.2.1 transform.match.structured

**作用：** 匹配结构化操作（Linalg 操作）并检查条件

**语法：**
```mlir
%result:2 = transform.match.structured @matcher
             failures(propagate)
             current: !transform.any_op
             -> (!transform.any_value, !transform.any_value) {
  // 匹配条件
  transform.match.structured.dim %current[0, 1] { parallel }
  transform.match.structured.rank %current { 2 }

  // 返回匹配的值
  transform.match.structured.input %current[0]
  transform.match.structured.init %current[0]
}
```

**WHY 需要结构化匹配：**
- **精确选择**：基于操作属性（维度、秩等）选择操作
- **类型安全**：确保后续转换适用

#### 5.2.2 transform.split_handle

**作用：** 将一个 Handle 分拆为多个 Handle

**语法：**
```mlir
%1:2 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

**WHY 使用 split_handle：**
- **访问元素**：从 Handle 集合中提取单个元素
- **对齐大小**：为需要相同大小 Handle 的操作准备数据

### 5.3 转换操作

#### 5.3.1 transform.structured.tile

**作用：** 对结构化操作进行平铺

**语法：**
```mlir
%tiled:2 = transform.structured.tile_using_for %target tile_sizes [32, 32]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

**参数：**
- `target`：要平铺的操作 Handle
- `tile_sizes`：平铺大小（静态或动态）

**返回：**
- 第一个结果：平铺后的操作 Handle
- 第二个结果：生成的循环 Handle

**WHY 平铺：**
- **缓存优化**：提高数据局部性
- **并行化准备**：为并行执行做准备

#### 5.3.2 transform.structured.vectorize

**作用：** 向量化结构化操作

**语法：**
```mlir
transform.structured.vectorize %target vector_sizes [4, 8]
    : !transform.any_op
```

**参数：**
- `target`：要向量化的操作 Handle
- `vector_sizes`：向量大小（可选）

**WHY 向量化：**
- **SIMD 利用**：利用硬件 SIMD 指令
- **性能提升**：减少循环开销

### 5.4 模式应用操作

#### 5.4.1 transform.apply_patterns

**作用：** 应用贪婪模式重写

**语法：**
```mlir
transform.apply_patterns to %target {
  ^bb0:
    // 模式描述符操作
    transform.apply_patterns.canonicalization
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
}
: !transform.any_op
```

**WHY 使用 apply_patterns：**
- **批量优化**：应用一系列简化模式
- **固定点收敛**：重复应用直到不再变化

### 5.5 调试操作

#### 5.5.1 transform.print

**作用：** 打印 Payload IR 操作

**语法：**
```mlir
transform.print %target { name = "After tiling" }
: !transform.any_op
```

**WHY 使用 print：**
- **调试转换流程**：检查中间状态
- **学习工具**：理解转换效果

#### 5.5.2 transform.verify

**作用：** 验证操作的正确性

**语法：**
```mlir
transform.verify %target : !transform.any_op
```

**WHY 使用 verify：**
- **断言机制**：确保 IR 不被破坏
- **调试辅助**：及早发现问题

---

## 6. 扩展机制

### 6.1 TransformDialectExtension

**概念：** Transform 方言的扩展机制允许注入自定义操作

**WHY 需要扩展：**
- **方言解耦**：避免 Transform 方言依赖特定方言的实现
- **模块化**：每个方言提供自己的转换操作

**扩展示例：**

```cpp
// 定义 Linalg Transform 扩展
class LinalgTransformDialectExtension
    : public TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  void init(Extension &extension) override {
    // 注册转换操作
    extension.registerOps<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"
    >();

    // 注册类型
    extension.registerTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Linalg/IR/LinalgTypes.cpp.inc"
    >();
  }
};
```

### 6.2 操作命名约定

**约定：** 扩展操作应使用前缀表示来源

| 前缀 | 来源 | 示例 |
|------|------|------|
| 无前缀 | Transform 方言核心 | `transform.sequence` |
| `affine.` | Affine 方言 | `transform.affine.simplify_bounded_affine_ops` |
| `gpu.` | GPU 方言 | `transform.gpu.map_nested_forall_to_threads` |
| `linalg.` | Linalg 方言（通过 structured） | `transform.structured.tile` |
| `structured.` | 结构化操作接口 | `transform.structured.vectorize` |

### 6.3 扩展操作要求

所有扩展操作必须：

1. **实现 TransformOpInterface**
   ```cpp
   struct MyTransformOp : public Op<MyTransformOp>,
                            public TransformOpInterface<MyTransformOp> {
     LogicalResult apply(TransformRewriter &rewriter,
                         TransformResults &results,
                         TransformState &state) {
       // 实现转换逻辑
     }
   };
   ```

2. **实现 MemoryEffectsOpInterface**
   ```cpp
   void getEffects(SmallVectorImpl<SideEffects::EffectInstance> &effects) {
     // 声明副作用
   }
   ```

---

## 7. 执行模型

### 7.1 执行流程

```
┌────────────────────────────────────────────────────────────────────┐
│                    Transform 执行流程                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. 解析 Transform IR                                               │
│     ├── 验证类型约束                                                 │
│     └── 检查操作定义                                                 │
│                                                                    │
│  2. 创建 TransformState                                             │
│     ├── 建立 Payload IR 根映射                                       │
│     └── 初始化 Handle 到 Payload 对象的映射                           │
│                                                                    │
│  3. 执行转换序列                                                     │
│     ├── 调用 TransformOpInterface::apply()                          │
│     ├── 处理 SilenceableFailure / DefiniteFailure                   │
│     └── 更新 Handle 映射                                            │
│                                                                    │
│  4. 处理 Handle 失效                                                │
│     └── 检查 Handle 是否被消费/失效                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 7.2 失败处理

Transform 方言支持两种失败模式：

#### 7.2.1 Silenceable Failure（可恢复失败）

**特征：**

- 转换未修改 Payload IR
- 可以尝试其他转换
- 延迟报告错误

**使用场景：**

```mlir
transform.sequence failures(suppress) {
  // 即使某些转换失败，继续执行其他转换
}
```

#### 7.2.2 Definite Failure（不可恢复失败）

**特征：**
- Payload IR 可能处于不一致状态
- 必须立即停止
- 立即报告错误

**使用场景：**
```mlir
transform.sequence failures(propagate) {
  // 任何失败立即传播
}
```

### 7.3 Handle 失效规则

**规则：** 当 Handle 被消费时，相关 Handle 自动失效

```
消费 OperationHandle:
├── 关联的操作 Handle 失效
├── 嵌套操作 Handle 失效
└── 操作结果的 Value Handle 失效

消费 ValueHandle:
├── 产生该值的操作 Handle 失效
├── 嵌套操作 Handle 失效
└── 包含该值的块参数 Handle 失效
```

**WHY 这样设计：**
- **安全性**：防止引用已删除的对象
- **一致性**：确保 Handle 指向有效的 Payload IR

---

## 8. Linalg Transform 操作

### 8.1 Linalg Transform 扩展

Linalg 方言通过 Transform 方言扩展提供结构化操作的转换。

**注册：**
```cpp
// LinalgTransformOps.cpp
void mlir::linalg::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
```

### 8.2 核心结构化转换

#### 8.2.1 平铺 (Tiling)

**操作：** `transform.structured.tile_using_for`

**效果：** 将大操作分解为小操作，提高缓存利用率

```mlir
// 平铺前
%0 = linalg.matmul ins(%A, %B: ...) outs(%C: ...)

// 平铺后（简化）
%0 = linalg.matmul ins(%A_tile, %B_tile: ...) outs(%C_tile: ...)
scf.for %i {
  scf.for %j {
    %sub_A = tensor.extract_slice %A[%i, %j]
    %sub_B = tensor.extract_slice %B[%j, %k]
    %sub_C = tensor.extract_slice %C[%i, %k]
    %partial = linalg.matmul ins(%sub_A, %sub_B) outs(%sub_C)
    %C = tensor.insert_slice %partial into %C[%i, %k]
  }
}
```

**WHY 平铺：**
- **局部性**：提高数据重用
- **并行化**：为并行执行创造机会

#### 8.2.2 交织 (Interchange)

**操作：** `transform.structured.interchange`

**效果：** 交换迭代维度

**WHY 交织：**
- **向量化准备**：对齐最内层维度
- **内存布局**：适应数据布局

#### 8.2.3 向量化 (Vectorization)

**操作：** `transform.structured.vectorize`

**效果：** 将操作转换为向量操作

```mlir
// 向量化前
%0 = linalg.generic {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = arith.addf %arg0, %arg1 : f32
    linalg.yield %1 : f32
}

// 向量化后
%0 = vector.addf %arg0, %arg1 : vector<4xf32>
```

**WHY 向量化：**
- **SIMD 利用**：充分利用硬件向量单元
- **循环消除**：减少循环开销

### 8.3 模式应用

Linalg 提供多种模式应用操作：

| 操作 | 作用 |
|------|------|
| `transform.apply_patterns.linalg.fold_unit_extent_dims` | 折叠单位维度 |
| `transform.apply_patterns.linalg.pad_vectorization` | Pad 向量化模式 |
| `transform.apply_patterns.linalg.tiling_canonicalization` | 平铺后规范化 |

---

## 9. 实践示例

### 9.1 完整优化流程示例

**目标：** 优化矩阵乘法

```mlir
module attributes {transform.with_named_sequence} {
  // 主优化序列
  transform.named_sequence @optimize_matmul(%arg0: !transform.any_op) {
    // 1. 匹配矩阵乘法
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg0
                 : (!transform.any_op) -> !transform.any_op

    // 2. 平铺
    %tiled:2 = transform.structured.tile_using_for %matmuls
                 tile_sizes [32, 32]
               : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. 向量化最内层循环
    transform.structured.vectorize %tiled#1 vector_sizes [4]

    // 4. 应用规范化模式
    transform.apply_patterns to %tiled#0 {
      ^bb0:
        transform.apply_patterns.linalg.tiling_canonicalization
    }

    transform.yield %tiled#0 : !transform.any_op
  }

  // 入口点
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.include @optimize_matmul(%arg0)
           failures(propagate)
    transform.yield %1 : !transform.any_op
  }
}
```

### 9.2 条件转换示例

**目标：** 根据操作大小选择不同策略

```mlir
transform.sequence {
^bb0(%arg0: !transform.any_op):
  // 获取操作的秩
  %rank = transform.match.structured.rank %arg0
           : (!transform.any_op) -> !transform.param<index>

  // 根据秩选择转换
  %result = transform.match.param.cmpi eq %rank, %2 {
    %small_ops = transform.sequence {
      ^bb1(%arg1: !transform.any_op):
        // 小矩阵策略
        transform.loop.unroll %arg1 { full }
    }

    %large_ops = transform.sequence {
      ^bb1(%arg1: !transform.any_op):
        // 大矩阵策略
        %1:2 = transform.structured.tile %arg1 [128, 128]
        transform.structured.vectorize %1#1
    }

    transform.alternatives {
      ^bb0(%arg2: !transform.any_op):
        // 尝试小矩阵策略
        transform.include @optimize_small(%arg2)
        transform.yield %small_ops
      }, {
      ^bb0(%arg2: !transform.any_op):
        // 回退到大矩阵策略
        transform.include @optimize_large(%arg2)
        transform.yield %large_ops
      }
    }
  }
}
```

### 9.3 GPU 映射示例

**目标：** 将计算映射到 GPU

```mlir
transform.sequence {
^bb0(%arg0: !transform.any_op):
  // 1. Bufferize
  %bufferized = transform.bufferization.one_shot_bufferize %arg0

  // 2. 映射到 GPU blocks
  %launch = transform.gpu.map_forall_to_blocks %bufferized
                grid_dims = [256, 256]

  // 3. 映射到 GPU threads
  transform.gpu.map_nested_forall_to_threads %launch
    block_dims = [32, 32]

  // 4. 应用 GPU 转换模式
  transform.apply_conversion_patterns to %launch {
    ^bb0:
      transform.apply_conversion_patterns.gpu.gpu_to_nvvm
      transform.apply_conversion_patterns.vector.vector_to_llvm
  }
}
```

---

## 10. 参考资料

### 10.1 官方文档

- [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
- [Linalg OpDSL](https://mlir.llvm.org/docs/Dialects/Linalg/)

### 10.2 源代码位置

| 组件 | 路径 |
|------|------|
| 核心方言定义 | `mlir/include/mlir/Dialect/Transform/IR/` |
| 核心实现 | `mlir/lib/Dialect/Transform/IR/` |
| Linalg Transform | `mlir/lib/Dialect/Linalg/TransformOps/` |
| 接口定义 | `mlir/include/mlir/Dialect/Transform/Interfaces/` |

### 10.3 相关接口

| 接口 | 用途 |
|------|------|
| `TransformOpInterface` | 实现转换操作 |
| `TransformHandleTypeInterface` | 实现操作 Handle 类型 |
| `PatternDescriptorOpInterface` | 实现模式选择操作 |
| `TilingInterface` | 支持平铺的操作接口 |

---

## 附录：术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| Payload IR | Payload IR | 被转换的目标 IR |
| Transform IR | Transform IR | 控制转换逻辑的 IR |
| Handle | Handle | 指向 Payload IR 对象的引用 |
| Silenceable Failure | Silenceable Failure | 可恢复的失败 |
| Definite Failure | Definite Failure | 不可恢复的失败 |
| Tiling | Tiling | 将大操作分解为小操作 |
| Vectorization | Vectorization | 将标量操作转换为向量操作 |
| Interchange | Interchange | 交换迭代维度 |
