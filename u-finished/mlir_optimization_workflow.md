# MLIR 优化流程详细解析

本文档详细展示三个 Pass 如何协同工作优化计算图。

---

## 原始 IR

```mlir
// 步骤 0: 未优化的原始代码
%cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

%transpose = linalg.transpose
    ins(%cst : tensor<2x4xf32>)
    outs(%init_4x2 : tensor<4x2xf32>)
    permutation = [1, 0]
// 结果: [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]

%reshape = tensor.expand_shape %transpose [[0, 1], [2]]
    output_shape [2, 2, 2]
    : tensor<4x2xf32> into tensor<2x2x2xf32>
// 4x2 -> 2x2x2

%result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reshape : tensor<2x2x2xf32>)
    outs(%out : tensor<2x2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mul = arith.mulf %in, %in : f32
    linalg.yield %mul : f32
} -> tensor<2x2x2xf32>
```

**问题**:
- 常量在运行时执行 transpose（浪费）
- reshape 阻碍了与 generic 的融合
- 多次内存布局转换

---

## 优化步骤 1：常量折叠 (Constant Folding)

**触发条件**: transpose 的输入是编译时常量

```mlir
// Pass: -canonicalize 或 -sccp

// 之前:
%cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
%transpose = linalg.transpose ins(%cst : ...) permutation = [1, 0]

// 之后: 直接折叠为转置后的常量
%cst_folded = arith.constant dense<[[1.0, 5.0],
                                     [2.0, 6.0],
                                     [3.0, 7.0],
                                     [4.0, 8.0]]> : tensor<4x2xf32>

// transpose 操作被消除！
%reshape = tensor.expand_shape %cst_folded [[0, 1], [2]] ...
%result = linalg.generic { ... }
```

**效果**:
- ✅ 消除运行时 transpose 计算
- ✅ 减少一次内存读写
- ✅ 暴露更多优化机会给后续 Pass

---

## 优化步骤 2：扩展融合 (Reshape by Expansion Fusion)

**触发条件**: expand_shape 的生产者是 generic op，且满足 `isFusableWithReshapeByDimExpansion`

```mlir
// Pass: -linalg-fuse-elementwise-ops

// 假设常量折叠后，我们有一个 generic 生产者：
%producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%cst_folded : tensor<4x2xf32>)
    outs(%init : tensor<4x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %in : f32
    linalg.yield %add : f32
} -> tensor<4x2xf32>

%reshape = tensor.expand_shape %producer [[0, 1], [2]]
    : tensor<4x2xf32> into tensor<2x2x2xf32>

// ========== 融合后 ==========

%fused = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]  // 维度扩展！
  } ins(%input_expanded : tensor<2x2x2xf32>)  // 输入也被 expand
    outs(%init_3d : tensor<2x2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %in : f32  // 计算逻辑不变
    linalg.yield %add : f32
} -> tensor<2x2x2xf32>

// reshape 操作被消除！
```

**关键机制**:
- `FoldReshapeWithGenericOpByExpansion` 模式匹配
- 通过 `fuseWithReshapeByExpansion` 将循环维度从 2D 扩展到 3D
- indexing map 从 `(d0, d1)` 变为 `(d0, d1, d2)`
- 原 reshape 的 `[[0, 1], [2]]` 映射到新的迭代空间

**效果**:
- ✅ 消除 reshape 操作
- ✅ 减少中间张量分配
- ✅ 计算直接在目标形状上进行

---

## 优化步骤 3：上浮 expand (Bubble Up Expand Shape)

**触发条件**: expand 的生产者是 collapse，且满足并行重关联条件

```mlir
// Pass: -test-tensor-transform-patterns=test-expand-shape-bubbling

// 假设有嵌套的 reshape：
%cst = arith.constant dense<...> : tensor<8x4xf32>

%collapse = tensor.collapse_shape %cst [[0, 1]]
    : tensor<8x4xf32> into tensor<32xf32>

%expand = tensor.expand_shape %collapse [[0, 1, 2]]
    output_shape [2, 4, 4]
    : tensor<32xf32> into tensor<2x4x4xf32>

%generic = linalg.generic { ... } ins(%expand : tensor<2x4x4xf32>) ...

// ========== 上浮后 ==========

%expand_first = tensor.expand_shape %cst [[0, 1, 2], [3]]
    output_shape [2, 2, 2, 4]
    : tensor<8x4xf32> into tensor<2x2x2x4xf32>

%collapse_after = tensor.collapse_shape %expand_first [[0, 1], [2], [3]]
    : tensor<2x2x2x4xf32> into tensor<4x2x4xf32>

// 现在 expand 在 collapse 之前！
// 为后续融合创造机会：
%generic = linalg.generic { ... } ins(%collapse_after : ...) ...
```

**更典型的场景**（expand 直接与 generic 相邻）:

```mlir
// 上浮前:
%collapsed = tensor.collapse_shape %input [[0], [1, 2], [3]]
    : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>

%expanded = tensor.expand_shape %collapsed [[0], [1], [2, 3]]
    : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>

%result = linalg.generic { ... } ins(%expanded : ...) ...

// 上浮后:
%expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4]]
    : tensor<?x?x?x?xf32> into tensor<?x?x?x?x?xf32>

%collapsed = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]]
    : tensor<?x?x?x?x?xf32> into tensor<?x?x?x?xf32>

%result = linalg.generic { ... } ins(%collapsed : ...) ...

// 此时可以触发步骤 2 的融合！
```

**效果**:
- ✅ 调整 reshape 顺序，暴露融合机会
- ✅ 有时可以消除冗余的 reshape 对
- ✅ 为步骤 2 的扩展融合创造前置条件

---

## 完整优化流程图

```
┌─────────────────────────────────────────────────────────┐
│ 原始 IR (未优化)                                         │
├─────────────────────────────────────────────────────────┤
│ Constant(2x4) → Transpose → Reshape(4x2→2x2x2) → Generic│
└────────────┬────────────────────────────────────────────┘
             │
             ▼
     ┌───────────────────┐
     │ Step 1: 常量折叠   │
     │ (Canonicalize)    │
     └────────┬──────────┘
              │ Transpose 被消除
              ▼
┌─────────────────────────────────────────────────┐
│ Constant(4x2) → Reshape(4x2→2x2x2) → Generic    │
└────────────┬────────────────────────────────────┘
             │
             ▼
     ┌───────────────────────────┐
     │ Step 3: 上浮 expand        │
     │ (Bubble Up Patterns)      │
     └────────┬──────────────────┘
              │ 调整 reshape 顺序
              ▼
┌─────────────────────────────────────────────────────┐
│ Constant(4x2) → Expand(4x2→2x2x2x1) → Collapse →    │
│                 Generic(operates on 2x2x2)          │
└────────────┬────────────────────────────────────────┘
             │
             ▼
     ┌───────────────────────────┐
     │ Step 2: 扩展融合           │
     │ (Reshape by Expansion)    │
     └────────┬──────────────────┘
              │ Generic 循环维度扩展，消除 reshape
              ▼
┌─────────────────────────────────────────────────┐
│ 最终优化 IR                                      │
├─────────────────────────────────────────────────┤
│ Constant(2x2x2) → Generic(直接在 3D 上计算)      │
│   - 无 transpose                                │
│   - 无 reshape                                  │
│   - 循环已融合                                  │
└─────────────────────────────────────────────────┘
```

---

## 实际代码对比

### 优化前 (原始)

```mlir
func.func @before(%out: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

  %transpose = linalg.transpose
      ins(%cst : tensor<2x4xf32>)
      outs(%t_init : tensor<4x2xf32>)
      permutation = [1, 0]

  %reshape = tensor.expand_shape %transpose [[0, 1], [2]]
      output_shape [2, 2, 2]
      : tensor<4x2xf32> into tensor<2x2x2xf32>

  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%reshape : tensor<2x2x2xf32>)
      outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
  } -> tensor<2x2x2xf32>

  return %result : tensor<2x2x2xf32>
}

// 内存操作: 3 次分配 (transpose结果, reshape结果, generic结果)
// 计算次数: transpose + reshape + generic
```

### 优化后 (全部 Pass 应用)

```mlir
func.func @after(%out: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  // 常量已经是最终形状（折叠 + 融合的结果）
  %cst = arith.constant dense<[[[1.0, 5.0],
                                 [2.0, 6.0]],
                                [[3.0, 7.0],
                                 [4.0, 8.0]]]> : tensor<2x2x2xf32>

  // 直接计算，无中间步骤
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%cst : tensor<2x2x2xf32>)
      outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
  } -> tensor<2x2x2xf32>

  return %result : tensor<2x2x2xf32>
}

// 内存操作: 1 次分配 (仅 generic 结果)
// 计算次数: 仅 generic (transpose和reshape被消除)
```

---

## 性能提升总结

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 操作数 | 3 (transpose + reshape + generic) | 1 (generic) | **67% 减少** |
| 内存分配 | 3 次 | 1 次 | **67% 减少** |
| 运行时开销 | transpose 计算 + 2 次拷贝 | 0 次额外操作 | **100% 消除** |
| 循环融合 | 否 | 是 | **缓存友好** |

---

## 关键要点

### 1. Pass 顺序很重要

- **常量折叠先行** → 暴露静态信息
- **上浮 expand** → 调整结构
- **扩展融合** → 最终合并

### 2. 协同效应

- 单个 Pass 可能只做局部优化
- 组合后能消除整个操作链

### 3. 前提条件

- 需要 `isProjectedPermutation` 保证简单映射
- 需要 `hasPureTensorSemantics` 保证无副作用
- 需要满足并行重关联条件

---

## 相关 Pass 命令

```bash
# 常量折叠
mlir-opt --canonicalize input.mlir

# 上浮 expand shape
mlir-opt --test-tensor-transform-patterns=test-expand-shape-bubbling input.mlir

# Linalg 元素级融合（包含扩展融合）
mlir-opt --linalg-fuse-elementwise-ops input.mlir

# 完整优化流程
mlir-opt --canonicalize \
         --test-tensor-transform-patterns=test-expand-shape-bubbling \
         --linalg-fuse-elementwise-ops \
         input.mlir
```

---

## 附录：关键数据结构

### isProjectedPermutation 判定条件

```cpp
bool AffineMap::isProjectedPermutation(bool allowZeroInResults) const {
  // 1. 无符号变量
  if (getNumSymbols() > 0)
    return false;

  // 2. 结果数 ≤ 输入数
  if (getNumResults() > getNumInputs())
    return false;

  // 3. 每个输入维度最多出现一次
  SmallVector<bool, 8> seen(getNumInputs(), false);
  for (auto expr : getResults()) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      if (seen[dim.getPosition()])
        return false;
      seen[dim.getPosition()] = true;
    } else {
      // 允许常量 0（当 allowZeroInResults=true 时）
      auto constExpr = dyn_cast<AffineConstantExpr>(expr);
      if (!allowZeroInResults || !constExpr || constExpr.getValue() != 0)
        return false;
    }
  }

  return true;
}
```

### 示例

```mlir
// ✓ 投影排列
(d0, d1, d2) -> (d1, d0)      // 选择 + 置换
(d0, d1, d2) -> (d2)          // 仅投影
(d0, d1) -> (d1, d0, 0)       // 带零（需 allowZeroInResults=true）

// ✗ 非投影排列
(d0, d1) -> (d0, d0)          // 重复
(d0, d1) -> (d0 + d1)         // 仿射表达式
(d0, d1) -> (d0, d1, d2)      // 结果维度多于输入
```

---

## 参考文件

- `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`
- `mlir/lib/Dialect/Tensor/Transforms/ReshapePatterns.cpp`
- `mlir/lib/IR/AffineMap.cpp`
- `mlir/test/Dialect/Tensor/bubble-reshapes.mlir`
- `mlir/test/Dialect/Linalg/data-layout-propagation.mlir`
