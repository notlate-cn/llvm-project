# affine-simplify-min-max Pass 解析

## 概述

`affine-simplify-min-max` pass用于简化affine表达式中的min/max操作。它通过分析变量的上下界，利用约束信息来简化min/max表达式，甚至完全消除它们。

该pass的主要功能：
- **完全简化**: 当可以确定min/max的结果时，直接替换为结果
- **部分简化**: 消除min/max中不可能的选项
- **约束传播**: 利用变量的上下界信息进行推理
- **支持多变量**: 支持多个变量的min/max表达式

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/simplify-min-max-ops.mlir`

## RUN命令

该测试文件包含以下RUN命令：

1. `mlir-opt -pass-pipeline="builtin.module(func.func(affine-simplify-min-max))" %s | FileCheck %s`

## 测试用例解析

### 用例 1: min_max_full_simplify

**原始代码:**

```mlir
func.func @min_max_full_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s0, 192, s1)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s0, 192, s1)>()[%0, %1]
  return %r0, %r1 : index, index
}
```

**说明:**

此用例测试可以完全简化的min/max表达式。

**变量约束:**
- `%0`: min=0, max=128
- `%1`: min=256, max=512

**分析:**
1. **min(s0, 192, s1)**:
   - s0的最大值是128
   - 192是常数
   - s1的最小值是256
   - 因此min(s0, 192, s1) = s0（因为s0 <= 128 < 192 < 256 <= s1）

2. **max(s0, 192, s1)**:
   - s0的最大值是128
   - 192是常数
   - s1的最小值是256
   - 因此max(s0, 192, s1) = s1（因为s0 <= 128 < 192 < 256 <= s1）

**优化后的行为:**

```mlir
func.func @min_max_full_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  return %0, %1 : index, index
}
```

**关键点:**

1. **完全消除**: min/max表达式被完全消除
2. **约束推理**: 利用变量的上下界进行推理
3. **确定结果**: 可以确定min/max的结果

---

### 用例 2: min_only_simplify

**原始代码:**

```mlir
func.func @min_only_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 512 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s0, 32, s1)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s0, 32, s1)>()[%0, %1]
  return %r0, %r1 : index, index
}
```

**说明:**

此用例测试只能部分简化的min/max表达式。

**变量约束:**
- `%0`: min=0, max=512
- `%1`: min=256, max=512

**分析:**
1. **min(s0, 32, s1)**:
   - s0的范围是[0, 512]
   - 32是常数
   - s1的最小值是256
   - 因此可以消除s1（因为32 < 256 <= s1）
   - 简化为min(s0, 32)

2. **max(s0, 32, s1)**:
   - s0的范围是[0, 512]
   - 32是常数
   - s1的范围是[256, 512]
   - 无法消除任何选项
   - 但可以重新排序为max(s1, s0)

**优化后的行为:**

```mlir
func.func @min_only_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 512 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %r0 = affine.min affine_map<()[s0] -> (32, s0)>()[%0]
  %r1 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%0, %1]
  return %r0, %r1 : index, index
}
```

**关键点:**

1. **部分简化**: 只能消除部分选项
2. **消除不可能选项**: s1在min中不可能被选中
3. **重新排序**: max表达式中的选项被重新排序

---

### 用例 3: max_only_simplify

**原始代码:**

```mlir
func.func @max_only_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 0 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s0, 256, s1)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s0, 256, s1)>()[%0, %1]
  return %r0, %r1 : index, index
}
```

**说明:**

此用例测试只能部分简化的min/max表达式。

**变量约束:**
- `%0`: min=0, max=128
- `%1`: min=0, max=512

**分析:**
1. **min(s0, 256, s1)**:
   - s0的最大值是128
   - 256是常数
   - s1的范围是[0, 512]
   - 无法消除任何选项
   - 但可以重新排序为min(s0, s1)

2. **max(s0, 256, s1)**:
   - s0的最大值是128
   - 256是常数
   - s1的范围是[0, 512]
   - 可以消除s0（因为s0 <= 128 < 256）
   - 简化为max(256, s1)

**优化后的行为:**

```mlir
func.func @max_only_simplify() -> (index, index) {
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 0 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s1, s0)>()[%0, %1]
  %r1 = affine.max affine_map<()[s1] -> (256, s1)>()[%1]
  return %r0, %r1 : index, index
}
```

**关键点:**

1. **部分简化**: 只能消除部分选项
2. **消除不可能选项**: s0在max中不可能被选中
3. **重新排序**: min表达式中的选项被重新排序

---

### 用例 4: overlapping_constraints

**原始代码:**

```mlir
func.func @overlapping_constraints() -> (index, index) {
  %0 = test.value_with_bounds {max = 192 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 384 : index, min = 128 : index}
  %2 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %r0 = affine.min affine_map<()[s0, s1, s2] -> (s0, s1, s2)>()[%0, %1, %2]
  %r1 = affine.max affine_map<()[s0, s1, s2] -> (s0, s1, s2)>()[%0, %1, %2]
  return %r0, %r1 : index, index
}
```

**说明:**

此用例测试有重叠约束的变量。

**变量约束:**
- `%0`: min=0, max=192
- `%1`: min=128, max=384
- `%2`: min=256, max=512

**分析:**
1. **min(s0, s1, s2)**:
   - s0的范围是[0, 192]
   - s1的范围是[128, 384]
   - s2的范围是[256, 512]
   - 无法消除任何选项
   - 但可以重新排序为min(s0, s1)

2. **max(s0, s1, s2)**:
   - s0的范围是[0, 192]
   - s1的范围是[128, 384]
   - s2的范围是[256, 512]
   - 无法消除任何选项
   - 但可以重新排序为max(s1, s2)

**优化后的行为:**

```mlir
func.func @overlapping_constraints() -> (index, index) {
  %0 = test.value_with_bounds {max = 192 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 384 : index, min = 128 : index}
  %2 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %r0 = affine.min affine_map<()[s0, s1] -> (s1, s0)>()[%0, %1]
  %r1 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%1, %2]
  return %r0, %r1 : index, index
}
```

**关键点:**

1. **重叠约束**: 变量的范围有重叠
2. **无法完全消除**: 只能重新排序
3. **保守策略**: 确保正确性

---

### 用例 5: nested_min_max

**原始代码:**

```mlir
func.func @nested_min_max() -> index {
  %0 = test.value_with_bounds {max = 128 : index, min = 0 : index}
  %1 = test.value_with_bounds {max = 512 : index, min = 256 : index}
  %2 = affine.min affine_map<()[s0, s1] -> (s0, 192, s1)>()[%0, %1]
  %3 = affine.max affine_map<()[s0] -> (s0, 128)>()[%2]
  return %3 : index
}
```

**说明:**

此用例测试嵌套的min/max表达式。

**分析:**
1. **min(s0, 192, s1)**:
   - 如用例1所述，结果为s0

2. **max(s0, 128)**:
   - s0的最大值是128
   - 因此max(s0, 128) = 128

**优化后的行为:**

```mlir
func.func @nested_min_max() -> index {
  %c128 = arith.constant 128 : index
  return %c128 : index
}
```

**关键点:**

1. **嵌套简化**: 先简化内层min，再简化外层max
2. **完全消除**: 最终结果为常数

---

### 用例 6: min_max_with_constants

**原始代码:**

```mlir
func.func @min_max_with_constants() -> (index, index) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %r0 = affine.min affine_map<() -> (100, 200)>()
  %r1 = affine.max affine_map<() -> (100, 200)>()
  return %r0, %r1 : index, index
}
```

**说明:**

此用例测试纯常数的min/max表达式。

**分析:**
1. **min(100, 200)** = 100
2. **max(100, 200)** = 200

**优化后的行为:**

```mlir
func.func @min_max_with_constants() -> (index, index) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  return %c100, %c200 : index, index
}
```

**关键点:**

1. **常数折叠**: 纯常数的min/max被直接计算
2. **完全消除**: min/max表达式被消除

---

## 总结

`affine-simplify-min-max` pass是一个强大的表达式简化pass，它可以：

1. **完全简化**: 当可以确定min/max的结果时，直接替换为结果
2. **部分简化**: 消除min/max中不可能的选项
3. **约束传播**: 利用变量的上下界信息进行推理
4. **支持多变量**: 支持多个变量的min/max表达式
5. **支持嵌套**: 支持嵌套的min/max表达式
6. **常数折叠**: 支持纯常数的min/max表达式

该pass在简化affine表达式时非常有用，特别是对于循环边界和约束条件。但需要注意：
- 需要变量的上下界信息
- 对于没有约束信息的变量，无法进行简化
- 采用保守策略，确保正确性
- 简化后的表达式可能仍然包含min/max

## 应用场景

1. **循环边界简化**: 简化循环的上下界表达式
2. **约束条件简化**: 简化affine约束条件
3. **代码生成**: 为后续代码生成提供更简单的表达式
4. **性能优化**: 减少运行时计算
