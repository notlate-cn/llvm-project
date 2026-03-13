# affine-scalrep Pass 解析

## 概述

`affine-scalrep` pass用于标量替换（Scalar Replacement）。它将memref的store/load操作替换为标量变量，从而减少内存访问，提高性能。该pass通过依赖分析，识别可以安全替换的store/load对，并进行优化。

该pass的主要功能：
- **Store-Load Forwarding**: 将store的值直接forward到后续的load，消除load操作
- **Load CSE (Common Subexpression Elimination)**: 消除重复的load操作
- **Dead Store Elimination**: 消除冗余的store操作
- **支持向量操作**: 支持affine.vector_load/affine.vector_store的优化
- **支持并行循环**: 支持affine.parallel的优化

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/scalrep.mlir`

## RUN命令

该测试文件包含以下RUN命令：

1. `mlir-opt -allow-unregistered-dialect %s -affine-scalrep | FileCheck %s`

## 测试用例解析

### 用例 1: simple_store_load

**原始代码:**

```mlir
func.func @simple_store_load() {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
  }
  memref.dealloc %m : memref<10xf32>
  return
}
```

**说明:**

这是一个最简单的store-load forwarding例子。循环中先store一个值，然后立即load同一个位置。

**优化后的行为:**

```mlir
%c7 = arith.constant 7.000000e+00 : f32
affine.for %i0 = 0 to 10 {
  arith.addf %c7, %c7 : f32
}
return
```

**关键点:**

1. **Store-Load Forwarding**: store的值`%cf7`直接forward到load，消除load操作
2. **Memref消除**: 由于memref不再被使用，分配和释放操作也被消除
3. **性能提升**: 消除了内存访问，直接使用寄存器

---

### 用例 2: multi_store_load

**原始代码:**

```mlir
func.func @multi_store_load() {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %m = gpu.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
    affine.store %cf8, %m[%i0] : memref<10xf32>
    affine.store %cf9, %m[%i0] : memref<10xf32>
    %v2 = affine.load %m[%i0] : memref<10xf32>
    %v3 = affine.load %m[%i0] : memref<10xf32>
    %v4 = arith.mulf %v2, %v3 : f32
  }
  gpu.dealloc %m : memref<10xf32>
  return
}
```

**说明:**

此用例测试多次store和load的情况。有多个store操作，最后一个store会forward到后续的load。

**优化后的行为:**

```mlir
%c7 = arith.constant 7.000000e+00 : f32
arith.constant 8.000000e+00 : f32
%c9 = arith.constant 9.000000e+00 : f32
affine.for %i0 = 0 to 10 {
  arith.addf %c7, %c7 : f32
  arith.mulf %c9, %c9 : f32
}
return
```

**关键点:**

1. **最后一次store有效**: 只有最后一次store（`%cf9`）会forward到后续的load
2. **中间store被消除**: 中间的store（`%cf8`）因为没有后续的load而被消除
3. **Load CSE**: 两个相同的load被合并为一个

---

### 用例 3: store_load_affine_apply

**原始代码:**

```mlir
func.func @store_load_affine_apply() -> memref<10x10xf32> {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %t0 = affine.apply affine_map<(d0, d1) -> (d1 + 1)>(%i0, %i1)
      %t1 = affine.apply affine_map<(d0, d1) -> (d0)>(%i0, %i1)
      %idx0 = affine.apply affine_map<(d0, d1) -> (d1)> (%t0, %t1)
      %idx1 = affine.apply affine_map<(d0, d1) -> (d0 - 1)> (%t0, %t1)
      affine.store %cf7, %m[%idx0, %idx1] : memref<10x10xf32>
      %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return %m : memref<10x10xf32>
}
```

**说明:**

此用例测试通过affine.apply计算的索引。store和load使用不同的索引表达式，但通过依赖分析可以确定它们访问的是同一个位置。

**优化后的行为:**

```mlir
%c7 = arith.constant 7.000000e+00 : f32
%m = memref.alloc() : memref<10x10xf32>
affine.for %i0 = 0 to 10 {
  affine.for %i1 = 0 to 10 {
    %t0 = affine.apply affine_map<(d0, d1) -> (d1 + 1)>(%i0, %i1)
    %t1 = affine.apply affine_map<(d0, d1) -> (d0)>(%i0, %i1)
    %idx0 = affine.apply affine_map<(d0, d1) -> (d1)> (%t0, %t1)
    %idx1 = affine.apply affine_map<(d0, d1) -> (d0 - 1)> (%t0, %t1)
    affine.store %c7, %m[%idx0, %idx1] : memref<10x10xf32>
    %v1 = arith.addf %c7, %c7 : f32
  }
}
return %m : memref<10x10xf32>
```

**关键点:**

1. **Affine Apply支持**: 可以看穿affine.apply操作，理解索引计算
2. **依赖分析**: 通过依赖分析确定store和load访问同一位置
3. **Memref保留**: 由于memref被返回，store操作不能被消除

---

### 用例 4: store_load_nested

**原始代码:**

```mlir
func.func @store_load_nested(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return
}
```

**说明:**

此用例测试嵌套循环中的store-load forwarding。store在外层循环，load在内层循环。

**优化后的行为:**

```mlir
%c7 = arith.constant 7.000000e+00 : f32
affine.for %i0 = 0 to 10 {
  affine.for %i1 = 0 to %N {
    arith.addf %c7, %c7 : f32
  }
}
return
```

**关键点:**

1. **跨循环forwarding**: store在外层循环，load在内层循环，仍然可以forward
2. **Memref消除**: memref不再被使用，被消除

---

### 用例 5: multi_store_load_nested_no_fwd

**原始代码:**

```mlir
func.func @multi_store_load_nested_no_fwd(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      affine.store %cf8, %m[%i1] : memref<10xf32>
    }
    affine.for %i2 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return
}
```

**说明:**

此用例测试不能forward的情况。内层循环有store操作，可能覆盖外层循环的store，因此不能forward。

**优化后的行为:**

```mlir
affine.for %i0 = 0 to 10 {
  affine.store %cf7, %m[%i0] : memref<10xf32>
  affine.for %i1 = 0 to %N {
    affine.store %cf8, %m[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to %N {
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
  }
}
```

**关键点:**

1. **依赖冲突**: 内层循环的store可能覆盖外层循环的store
2. **不进行forwarding**: 由于不确定性，不进行forwarding
3. **保守策略**: 确保正确性优先

---

### 用例 6: multi_store_load_nested_fwd

**原始代码:**

```mlir
func.func @multi_store_load_nested_fwd(%N : index) {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  %cf9 = arith.constant 9.0 : f32
  %cf10 = arith.constant 10.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    affine.for %i1 = 0 to %N {
      affine.store %cf8, %m[%i1] : memref<10xf32>
    }
    affine.for %i2 = 0 to %N {
      affine.store %cf9, %m[%i2] : memref<10xf32>
    }
    affine.store %cf10, %m[%i0] : memref<10xf32>
    affine.for %i3 = 0 to %N {
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return
}
```

**说明:**

此用例测试可以forward的情况。虽然内层循环有store，但最后一次store postdominates所有其他store，可以forward。

**优化后的行为:**

```mlir
affine.for %i0 = 0 to 10 {
  affine.store %cf7, %m[%i0] : memref<10xf32>
  affine.for %i1 = 0 to %N {
    affine.store %cf8, %m[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to %N {
    affine.store %cf9, %m[%i2] : memref<10xf32>
  }
  affine.store %cf10, %m[%i0] : memref<10xf32>
  affine.for %i3 = 0 to %N {
    arith.addf %cf10, %cf10 : f32
  }
}
```

**关键点:**

1. **Postdominance**: 最后一次store postdominates所有其他store
2. **可以forwarding**: 确定最后一次store是唯一到达load的store
3. **精确分析**: 依赖分析确定可以安全forward

---

### 用例 7: vector_forwarding

**原始代码:**

```mlir
func.func @vector_forwarding(%in : memref<512xf32>, %out : memref<512xf32>) {
  %tmp = memref.alloc() : memref<512xf32>
  affine.for %i = 0 to 16 {
    %ld0 = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld0, %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    %ld1 = affine.vector_load %tmp[32*%i] : memref<512xf32>, vector<32xf32>
    affine.vector_store %ld1, %out[32*%i] : memref<512xf32>, vector<32xf32>
  }
  return
}
```

**说明:**

此用例测试向量操作的store-load forwarding。从%in加载，存储到%tmp，再从%tmp加载，存储到%out。

**优化后的行为:**

```mlir
affine.for %i = 0 to 16 {
  %ldval = affine.vector_load %in[32*%i] : memref<512xf32>, vector<32xf32>
  affine.vector_store %ldval, %out[32*%i] : memref<512xf32>, vector<32xf32>
}
```

**关键点:**

1. **向量支持**: 支持affine.vector_load/affine.vector_store的优化
2. **中间memref消除**: %tmp的store和load被消除
3. **直接传递**: 从%in直接传递到%out

---

### 用例 8: simple_three_loads

**原始代码:**

```mlir
func.func @simple_three_loads(%in : memref<10xf32>) {
  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %in[%i0] : memref<10xf32>
    %v1 = affine.load %in[%i0] : memref<10xf32>
    %v2 = arith.addf %v0, %v1 : f32
    %v3 = affine.load %in[%i0] : memref<10xf32>
    %v4 = arith.addf %v2, %v3 : f32
  }
  return
}
```

**说明:**

此用例测试Load CSE（Common Subexpression Elimination）。同一个位置被load了三次。

**优化后的行为:**

```mlir
affine.for %i0 = 0 to 10 {
  %v0 = affine.load %in[%i0] : memref<10xf32>
  %v2 = arith.addf %v0, %v0 : f32
  %v4 = arith.addf %v2, %v0 : f32
}
```

**关键点:**

1. **Load CSE**: 三个相同的load被合并为一个
2. **性能提升**: 减少了内存访问次数

---

### 用例 9: redundant_store_elim

**原始代码:**

```mlir
func.func @redundant_store_elim(%out : memref<512xf32>) {
  %cf1 = arith.constant 1.0 : f32
  %cf2 = arith.constant 2.0 : f32
  affine.for %i = 0 to 16 {
    affine.store %cf1, %out[32*%i] : memref<512xf32>
    affine.store %cf2, %out[32*%i] : memref<512xf32>
  }
  return
}
```

**说明:**

此用例测试冗余store消除。连续两个store到同一位置，第一个store是冗余的。

**优化后的行为:**

```mlir
affine.for %i = 0 to 16 {
  affine.store %cf2, %out[32*%i] : memref<512xf32>
}
```

**关键点:**

1. **Dead Store Elimination**: 第一个store被消除
2. **保留最后一个**: 只保留最后一个store

---

### 用例 10: parallel_store_load

**原始代码:**

```mlir
func.func @parallel_store_load() {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10xf32>
  affine.parallel (%i0) = (0) to (10) {
    affine.store %cf7, %m[%i0] : memref<10xf32>
    %v0 = affine.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
  }
  memref.dealloc %m : memref<10xf32>
  return
}
```

**说明:**

此用例测试并行循环中的store-load forwarding。

**优化后的行为:**

```mlir
%c7 = arith.constant 7.000000e+00 : f32
affine.parallel (%i0) = (0) to (10) {
  arith.addf %c7, %c7 : f32
}
return
```

**关键点:**

1. **并行循环支持**: 支持affine.parallel的优化
2. **每个迭代独立**: 每个并行迭代都有自己的store和load，可以独立优化

---

### 用例 11: zero_d_memrefs

**原始代码:**

```mlir
func.func @zero_d_memrefs() {
  %c0_i32 = arith.constant 0 : i32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<i32>
  affine.store %c0_i32, %alloc_0[] : memref<i32>
  affine.for %arg0 = 0 to 9 {
    %2 = affine.load %alloc_0[] : memref<i32>
    arith.addi %2, %2 : i32
  }
  return
}
```

**说明:**

此用例测试0维memref的优化。0维memref没有索引，只有一个元素。

**优化后的行为:**

```mlir
%c0 = arith.constant 0 : i32
affine.for %arg0 = 0 to 9 {
  arith.addi %c0, %c0 : i32
}
```

**关键点:**

1. **0维memref支持**: 支持0维memref的优化
2. **标量替换**: 0维memref本质上就是一个标量

---

## 总结

`affine-scalrep` pass是一个强大的标量替换优化pass，它可以：

1. **Store-Load Forwarding**: 将store的值直接forward到后续的load，消除load操作
2. **Load CSE**: 消除重复的load操作
3. **Dead Store Elimination**: 消除冗余的store操作
4. **支持复杂索引**: 可以处理affine.apply计算的索引
5. **支持嵌套循环**: 可以跨循环进行forwarding
6. **支持向量操作**: 支持affine.vector_load/affine.vector_store的优化
7. **支持并行循环**: 支持affine.parallel的优化
8. **支持0维memref**: 支持0维memref的优化

该pass在优化内存访问时非常有用，可以显著减少内存访问次数。但需要注意：
- 需要进行精确的依赖分析
- 不能forward有依赖冲突的情况
- 需要考虑postdominance关系
- 对于非affine区域或未知操作，采用保守策略
