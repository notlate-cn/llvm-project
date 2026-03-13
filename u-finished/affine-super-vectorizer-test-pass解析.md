# affine-super-vectorizer-test Pass 解析

## 概述

`affine-super-vectorizer-test` pass是`affine-super-vectorize` pass的测试版本，用于测试向量化器的内部功能。它提供了一些测试选项，用于验证向量化的各个阶段和功能。

该pass的主要功能：
- **测试向量化决策**: 测试向量化器的决策逻辑
- **测试不支持的情况**: 验证向量化器对不支持情况的处理
- **测试内部工具**: 测试向量化器的内部工具函数
- **调试支持**: 提供调试信息，帮助理解向量化过程

## 测试文件来源

- 文件路径: 
  - `mlir/test/Dialect/Affine/SuperVectorize/vectorize_unsupported.mlir`
  - `mlir/test/Dialect/Affine/SuperVectorize/vector_utils.mlir`
  - `mlir/test/Dialect/Affine/SuperVectorize/compose_maps.mlir`
  - `mlir/test/Dialect/Affine/slicing-utils.mlir`

## RUN命令

该pass支持的主要选项：

1. `vectorize-affine-loop-nest`: 测试affine循环嵌套的向量化

## 测试用例解析

### 用例 1: unparallel_loop_reduction_unsupported

**原始代码:**

```mlir
func.func @unparallel_loop_reduction_unsupported(%in: memref<256x512xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
    %add = arith.addf %red_iter, %red_iter : f32
    affine.yield %add : f32
  }
  return
}
```

**说明:**

此用例测试不支持向量化的reduction循环。循环是一个reduction，但不是并行的，因为每次迭代都依赖于前一次迭代的结果。

**测试输出:**

```
Outermost loop cannot be parallel
```

**关键点:**

1. **非并行循环**: 循环不能并行化，因为存在迭代依赖
2. **无法向量化**: 由于非并行，无法进行向量化
3. **错误检测**: 向量化器正确检测到不支持的情况

---

### 用例 2: iv_mapped_to_multiple_indices_unsupported

**原始代码:**

```mlir
#map = affine_map<(d0)[s0] -> (d0 mod s0)>
#map1 = affine_map<(d0)[s0] -> (d0 floordiv s0)>

func.func @iv_mapped_to_multiple_indices_unsupported(%arg0: index) -> memref<2x2xf32> {
  %c2 = arith.constant 2 : index
  %cst = arith.constant 1.0 : f32
  %alloc = memref.alloc() : memref<2x2xf32>
    
  affine.for %i = 0 to 4 {
    %row = affine.apply #map1(%i)[%c2]  
    %col = affine.apply #map(%i)[%c2]  
    affine.store %cst, %alloc[%row, %col] : memref<2x2xf32>
  }
    
  return %alloc : memref<2x2xf32>
}
```

**说明:**

此用例测试循环变量被映射到多个索引的情况。循环变量`%i`通过affine.apply被映射到`%row`和`%col`，这种复杂的映射不支持向量化。

**测试输出:**

```
#[[$ATTR_0:.+]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
#[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>

func.func @iv_mapped_to_multiple_indices_unsupported(%arg0: index) -> memref<2x2xf32> {
  %c2 = arith.constant 2 : index
  affine.for %i = 0 to 4 {
    %row = affine.apply #[[$ATTR_0]](%i)[%c2]  
    %col = affine.apply #[[$ATTR_1]](%i)[%c2]  
    affine.store %cst, %alloc[%row, %col] : memref<2x2xf32>
  }
  return %alloc : memref<2x2xf32>
}
```

**关键点:**

1. **复杂索引映射**: 循环变量被映射到多个索引
2. **不支持向量化**: 这种复杂的映射不支持向量化
3. **保持原样**: 循环保持原样，不进行向量化

---

### 用例 3: slicing-utils

**原始代码:**

```mlir
// 测试循环切片工具
func.func @slicing_test() {
  // ... 测试代码
}
```

**说明:**

此测试文件测试向量化器的循环切片工具，用于分析循环的依赖关系和切片。

**关键点:**

1. **循环切片**: 分析循环的依赖关系
2. **切片工具**: 提供切片分析的工具函数

---

### 用例 4: vector_utils

**原始代码:**

```mlir
// 测试向量工具函数
func.func @vector_utils_test() {
  // ... 测试代码
}
```

**说明:**

此测试文件测试向量化器的向量工具函数，用于处理向量操作。

**关键点:**

1. **向量工具**: 提供向量操作的工具函数
2. **辅助向量化**: 辅助向量化过程

---

### 用例 5: compose_maps

**原始代码:**

```mlir
// 测试affine_map的组合
func.func @compose_maps_test() {
  // ... 测试代码
}
```

**说明:**

此测试文件测试affine_map的组合，用于优化向量索引计算。

**关键点:**

1. **Affine map组合**: 组合多个affine_map
2. **索引优化**: 优化向量索引计算

---

## 总结

`affine-super-vectorizer-test` pass是一个测试用的pass，主要用于：

1. **测试向量化决策**: 测试向量化器的决策逻辑
2. **测试不支持的情况**: 验证向量化器对不支持情况的处理
3. **测试内部工具**: 测试向量化器的内部工具函数
4. **调试支持**: 提供调试信息，帮助理解向量化过程

该pass主要用于开发和测试向量化器，不是用于生产环境。它帮助开发者：
- 理解向量化器的行为
- 调试向量化问题
- 验证向量化器的正确性
- 测试新的向量化功能

## 与affine-super-vectorize的区别

| 特性 | affine-super-vectorize | affine-super-vectorizer-test |
|------|----------------------|----------------------------|
| 用途 | 生产环境向量化 | 测试和调试 |
| 输出 | 向量化后的代码 | 测试信息和调试输出 |
| 选项 | 向量化选项 | 测试选项 |
| 性能 | 优化性能 | 不关注性能 |
| 稳定性 | 稳定 | 可能不稳定 |

## 应用场景

1. **开发向量化器**: 测试新的向量化功能
2. **调试向量化问题**: 理解向量化器的行为
3. **验证正确性**: 验证向量化器的正确性
4. **教学和研究**: 帮助理解向量化原理

## 注意事项

1. **仅用于测试**: 不应在生产环境使用
2. **可能不稳定**: 测试pass可能不稳定
3. **输出可能变化**: 测试输出可能随着实现变化
4. **需要专业知识**: 需要了解向量化器的内部实现
