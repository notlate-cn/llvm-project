# affine-parallelize Pass 解析

## 概述

`affine-parallelize` pass用于将affine循环转换为并行循环（affine.parallel）。它会分析循环的依赖关系，将可以并行执行的循环转换为并行循环，从而提高程序的并行性。

该pass支持以下主要选项：

- `max-nested`: 设置最大嵌套并行循环数（默认为无限制）
- `parallel-reductions`: 是否并行化reduction操作（默认为false）

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/parallelize.mlir`

## RUN命令

该测试文件包含以下RUN命令：

1. `mlir-opt %s -allow-unregistered-dialect -affine-parallelize | FileCheck %s`

2. `mlir-opt %s -allow-unregistered-dialect -affine-parallelize='max-nested=1' | FileCheck --check-prefix=MAX-NESTED %s`

3. `mlir-opt %s -allow-unregistered-dialect -affine-parallelize='parallel-reductions=1' | FileCheck --check-prefix=REDUCE %s`

## 测试用例解析

### 用例 1: reduce_window_max

**原始代码:**

```mlir
func.func @reduce_window_max() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<1x8x8x64xf32>
  %1 = memref.alloc() : memref<1x18x18x64xf32>
  affine.for %arg0 = 0 to 1 {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 64 {
          affine.store %cst, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
        }
      }
    }
  }
  affine.for %arg0 = 0 to 1 {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 64 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 3 {
              affine.for %arg6 = 0 to 3 {
                affine.for %arg7 = 0 to 1 {
                  %2 = affine.load %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
                  %3 = affine.load %1[%arg0 + %arg4, %arg1 * 2 + %arg5, %arg2 * 2 + %arg6, %arg3 + %arg7] : memref<1x18x18x64xf32>
                  %4 = arith.cmpf ogt, %2, %3 : f32
                  %5 = arith.select %4, %2, %3 : f32
                  affine.store %5, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
                }
              }
            }
          }
        }
      }
    }
  }
  return
}
```

**说明:**

这是一个复杂的窗口最大值归约操作。包含两个嵌套循环结构：
1. 第一个循环：初始化数组`%0`
2. 第二个循环：执行窗口最大值归约

**并行化后的行为:**

第一个循环的所有层都会被并行化：
```mlir
affine.parallel (%arg0) = (0) to (1) {
  affine.parallel (%arg1) = (0) to (8) {
    affine.parallel (%arg2) = (0) to (8) {
      affine.parallel (%arg3) = (0) to (64) {
        affine.store %cst, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
      }
    }
  }
}
```

第二个循环中，外层循环会被并行化，但内层循环（%arg5, %arg6）保持串行，因为存在依赖关系：
```mlir
affine.parallel (%a0) = (0) to (1) {
  affine.parallel (%a1) = (0) to (8) {
    affine.parallel (%a2) = (0) to (8) {
      affine.parallel (%a3) = (0) to (64) {
        affine.parallel (%a4) = (0) to (1) {
          affine.for %a5 = 0 to 3 {
            affine.for %a6 = 0 to 3 {
              affine.parallel (%a7) = (0) to (1) {
                %lhs = affine.load %0[%a0, %a1, %a2, %a3] : memref<1x8x8x64xf32>
                %rhs = affine.load %1[%a0 + %a4, %a1 * 2 + %a5, %a2 * 2 + %a6, %a3 + %a7] : memref<1x18x18x64xf32>
                %res = arith.cmpf ogt, %lhs, %rhs : f32
                %sel = arith.select %res, %lhs, %rhs : f32
                affine.store %sel, %0[%a0, %a1, %a2, %a3] : memref<1x8x8x64xf32>
              }
            }
          }
        }
      }
    }
  }
}
```

关键点：
1. 初始化循环的所有层都可以并行化，因为没有依赖关系
2. 归约循环中，外层循环可以并行化，但内层循环（%arg5, %arg6）保持串行
3. 最内层循环（%arg7）可以并行化

---

### 用例 2: loop_nest_3d_outer_two_parallel

**原始代码:**

```mlir
func.func @loop_nest_3d_outer_two_parallel(%N : index) {
  %0 = memref.alloc() : memref<1024 x 1024 x vector<64xf32>>
  %1 = memref.alloc() : memref<1024 x 1024 x vector<64xf32>>
  %2 = memref.alloc() : memref<1024 x 1024 x vector<64xf32>>
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %N {
      %7 = affine.load %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      affine.for %k = 0 to %N {
        %5 = affine.load %0[%i, %k] : memref<1024x1024xvector<64xf32>>
        %6 = affine.load %1[%k, %j] : memref<1024x1024xvector<64xf32>>
        %8 = arith.mulf %5, %6 : vector<64xf32>
        %9 = arith.addf %7, %8 : vector<64xf32>
        affine.store %9, %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      }
    }
  }
  return
}
```

**说明:**

这是一个矩阵乘法的例子。三层嵌套循环，外层两层（%i, %j）可以并行化，但内层循环（%k）必须保持串行，因为存在依赖关系（累加操作）。

**并行化后的行为:**

```mlir
affine.parallel (%arg1) = (0) to (symbol(%arg0)) {
  affine.parallel (%arg2) = (0) to (symbol(%arg0)) {
    affine.for %arg3 = 0 to %arg0 {
      %5 = affine.load %0[%arg1, %arg3] : memref<1024x1024xvector<64xf32>>
      %6 = affine.load %1[%arg3, %arg2] : memref<1024x1024xvector<64xf32>>
      %8 = arith.mulf %5, %6 : vector<64xf32>
      %9 = arith.addf %7, %8 : vector<64xf32>
      affine.store %9, %2[%arg1, %arg2] : memref<1024x1024xvector<64xf32>>
    }
  }
}
```

关键点：
1. 外层两层循环（%i, %j）被并行化
2. 内层循环（%k）保持串行，因为存在累加依赖
3. 符号边界`%N`被正确处理

---

### 用例 3: unknown_op_conservative

**原始代码:**

```mlir
func.func @unknown_op_conservative() {
  affine.for %i = 0 to 10 {
    "unknown"() : () -> ()
  }
  return
}
```

**说明:**

此用例测试包含未知操作的情况。未知操作会被保守处理，循环不会被并行化。

**并行化后的行为:**

```mlir
affine.for %arg1 = 0 to 10 {
  "unknown"() : () -> ()
}
```

关键点：
1. 未知操作会导致循环保持串行
2. 保守处理确保正确性

---

### 用例 4: non_affine_load

**原始代码:**

```mlir
func.func @non_affine_load() {
  %0 = memref.alloc() : memref<100 x f32>
  affine.for %i = 0 to 100 {
    memref.load %0[%i] : memref<100 x f32>
  }
  return
}
```

**说明:**

此用例测试使用非affine load（memref.load而不是affine.load）的情况。非affine操作会导致循环保持串行。

**并行化后的行为:**

```mlir
affine.for %arg1 = 0 to 100 {
  memref.load %0[%arg1] : memref<100 x f32>
}
```

关键点：
1. 非affine操作会导致循环保持串行
2. 只有affine操作才能被分析

---

### 用例 5: for_with_minmax

**原始代码:**

```mlir
func.func @for_with_minmax(%m: memref<?xf32>, %lb0: index, %lb1: index,
                      %ub0: index, %ub1: index) {
  affine.for %i = max affine_map<(d0, d1) -> (d0, d1)>(%lb0, %lb1)
          to min affine_map<(d0, d1) -> (d0, d1)>(%ub0, %ub1) {
    affine.load %m[%i] : memref<?xf32>
  }
  return
}
```

**说明:**

此用例测试使用min/max运算的循环边界。循环的下界是`max(%lb0, %lb1)`，上界是`min(%ub0, %ub1)`。

**并行化后的行为:**

```mlir
affine.parallel (%arg0) = (max(%arg1, %arg2)) to (min(%arg3, %arg4)) {
  affine.load %m[%arg0] : memref<?xf32>
}
```

关键点：
1. min/max边界被正确处理
2. 循环被并行化

---

### 用例 6: nested_for_with_minmax

**原始代码:**

```mlir
func.func @nested_for_with_minmax(%m: memref<?xf32>, %lb0: index,
                             %ub0: index, %ub1: index) {
  affine.for %j = 0 to 10 {
    affine.for %i = max affine_map<(d0, d1) -> (d0, d1)>(%lb0, %j)
            to min affine_map<(d0, d1) -> (d0, d1)>(%ub0, %ub1) {
      affine.load %m[%i] : memref<?xf32>
    }
  }
  return
}
```

**说明:**

此用例测试嵌套循环中使用min/max运算的循环边界。内层循环的下界依赖于外层循环变量。

**并行化后的行为:**

```mlir
affine.parallel (%arg0) = (0) to (10) {
  affine.parallel (%arg1) = (max(%arg2, %arg0)) to (min(%arg3, %arg4)) {
    affine.load %m[%arg1] : memref<?xf32>
  }
}
```

关键点：
1. 两层循环都被并行化
2. 内层循环的边界依赖于外层循环变量

---

### 用例 7: max_nested (MAX-NESTED模式)

**原始代码:**

```mlir
func.func @max_nested(%m: memref<?x?xf32>, %lb0: index, %lb1: index,
                 %ub0: index, %ub1: index) {
  affine.for %i = affine_map<(d0) -> (d0)>(%lb0) to affine_map<(d0) -> (d0)>(%ub0) {
    affine.for %j = affine_map<(d0) -> (d0)>(%lb1) to affine_map<(d0) -> (d0)>(%ub1) {
      affine.load %m[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
```

**说明:**

此用例测试`max-nested=1`选项。当设置最大嵌套并行循环数为1时，只有外层循环会被并行化。

**MAX-NESTED模式下的行为:**

```mlir
affine.parallel (%arg0) = (%arg1) to (%arg2) {
  affine.for %arg3 = (%arg4) to (%arg5) {
    affine.load %m[%arg0, %arg3] : memref<?x?xf32>
  }
}
```

关键点：
1. 只有外层循环被并行化
2. 内层循环保持串行

---

### 用例 8: max_nested_1 (MAX-NESTED模式)

**原始代码:**

```mlir
func.func @max_nested_1(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
  %0 = memref.alloc() : memref<4096x4096xf32>
  affine.for %arg3 = 0 to 4096 {
    affine.for %arg4 = 0 to 4096 {
      affine.for %arg5 = 0 to 4096 {
        %1 = affine.load %arg0[%arg3, %arg5] : memref<4096x4096xf32>
        %2 = affine.load %arg1[%arg5, %arg4] : memref<4096x4096xf32>
        %3 = affine.load %0[%arg3, %arg4] : memref<4096x4096xf32>
        %4 = arith.mulf %1, %2 : f32
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %0[%arg3, %arg4] : memref<4096x4096xf32>
      }
    }
  }
  return
}
```

**说明:**

此用例测试矩阵乘法在`max-nested=1`选项下的行为。这是一个典型的矩阵乘法，三层嵌套循环。

**MAX-NESTED模式下的行为:**

```mlir
affine.parallel (%arg3) = (0) to (4096) {
  affine.for %arg4 = 0 to 4096 {
    affine.for %arg5 = 0 to 4096 {
      %1 = affine.load %arg0[%arg3, %arg5] : memref<4096x4096xf32>
      %2 = affine.load %arg1[%arg5, %arg4] : memref<4096x4096xf32>
      %3 = affine.load %0[%arg3, %arg4] : memref<4096x4096xf32>
      %4 = arith.mulf %1, %2 : f32
      %5 = arith.addf %3, %4 : f32
      affine.store %5, %0[%arg3, %arg4] : memref<4096x4096xf32>
    }
  }
}
```

关键点：
1. 只有最外层循环被并行化
2. 内层两层循环保持串行

---

### 用例 9: iter_args (REDUCE模式)

**原始代码:**

```mlir
func.func @iter_args(%in: memref<10xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %final_red = affine.for %i = 0 to 10 iter_args(%red_iter = %cst) -> (f32) {
    %ld = affine.load %in[%i] : memref<10xf32>
    %add = arith.addf %red_iter, %ld : f32
    affine.yield %add : f32
  }
  return
}
```

**说明:**

此用例测试带iter_args的循环。这是一个典型的reduction操作（求和）。当设置`parallel-reductions=1`时，reduction循环会被并行化。

**默认模式下的行为:**

```mlir
%cst = arith.constant 0.000000e+00 : f32
%final_red = affine.for %i = 0 to 10 iter_args(%red_iter = %cst) -> (f32) {
  %ld = affine.load %in[%i] : memref<10xf32>
  %add = arith.addf %red_iter, %ld : f32
  affine.yield %add : f32
}
```

**REDUCE模式下的行为:**

```mlir
%cst = arith.constant 0.000000e+00 : f32
%reduced = affine.parallel (%i) = (0) to (10) reduce ("addf") {
  %red_value = affine.load %in[%i] : memref<10xf32>
  affine.yield %red_value : f32
}
%final_red = arith.addf %cst, %reduced : f32
```

关键点：
1. 默认情况下，带iter_args的循环不会被并行化
2. 当设置`parallel-reductions=1`时，reduction循环会被并行化
3. 并行化后，reduction操作被转换为`affine.parallel`的reduce属性
4. 初始值需要在并行循环外额外处理

---

### 用例 10: nested_iter_args (REDUCE模式)

**原始代码:**

```mlir
func.func @nested_iter_args(%in: memref<20x10xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 20 {
    %final_red = affine.for %j = 0 to 10 iter_args(%red_iter = %cst) -> (f32) {
      %ld = affine.load %in[%i, %j] : memref<20x10xf32>
      %add = arith.addf %red_iter, %ld : f32
      affine.yield %add : f32
    }
  }
  return
}
```

**说明:**

此用例测试嵌套循环中带iter_args的内层循环。外层循环没有依赖，内层循环是reduction操作。

**默认模式下的行为:**

```mlir
affine.parallel (%i) = (0) to (20) {
  %final_red = affine.for %j = 0 to 10 iter_args(%red_iter = %cst) -> (f32) {
    %ld = affine.load %in[%i, %j] : memref<20x10xf32>
    %add = arith.addf %red_iter, %ld : f32
    affine.yield %add : f32
  }
}
```

**REDUCE模式下的行为:**

```mlir
affine.parallel (%i) = (0) to (20) {
  %reduced = affine.parallel (%j) = (0) to (10) reduce ("addf") {
    %red_value = affine.load %in[%i, %j] : memref<20x10xf32>
    affine.yield %red_value : f32
  }
}
```

关键点：
1. 外层循环被并行化
2. 默认情况下，内层reduction循环保持串行
3. REDUCE模式下，内层reduction循环也被并行化

---

### 用例 11: strange_butterfly (REDUCE模式)

**原始代码:**

```mlir
func.func @strange_butterfly() {
  %cst1 = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32
  affine.for %i = 0 to 10 iter_args(%it1 = %cst1, %it2 = %cst2) -> (f32, f32) {
    %0 = arith.addf %it1, %it2 : f32
    affine.yield %0, %0 : f32, f32
  }
  return
}
```

**说明:**

此用例测试非标准reduction模式。两个iter_arg被yield相同的值，这不是一个简单的reduction。

**REDUCE模式下的行为:**

```mlir
%cst1 = arith.constant 0.0 : f32
%cst2 = arith.constant 1.0 : f32
affine.for %i = 0 to 10 iter_args(%it1 = %cst1, %it2 = %cst2) -> (f32, f32) {
  %0 = arith.addf %it1, %it2 : f32
  affine.yield %0, %0 : f32, f32
}
```

关键点：
1. 非标准reduction模式不会被并行化
2. 两个iter_arg被yield相同的值，不是简单的reduction

---

### 用例 12: repeated_use (REDUCE模式)

**原始代码:**

```mlir
func.func @repeated_use() {
  %cst1 = arith.constant 0.0 : f32
  affine.for %i = 0 to 10 iter_args(%it1 = %cst1) -> (f32) {
    %0 = arith.addf %it1, %it1 : f32
    affine.yield %0 : f32
  }
  return
}
```

**说明:**

此用例测试iter_arg被多次使用的情况。`%it1`在加法操作中被使用了两次，这不是一个简单的reduction。

**REDUCE模式下的行为:**

```mlir
%cst1 = arith.constant 0.0 : f32
affine.for %i = 0 to 10 iter_args(%it1 = %cst1) -> (f32) {
  %0 = arith.addf %it1, %it1 : f32
  affine.yield %0 : f32
}
```

关键点：
1. iter_arg被多次使用时，不会被并行化
2. 这不是简单的reduction模式

---

### 用例 13: use_in_backward_slice (REDUCE模式)

**原始代码:**

```mlir
func.func @use_in_backward_slice() {
  %cst1 = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32
  affine.for %i = 0 to 10 iter_args(%it1 = %cst1, %it2 = %cst2) -> (f32, f32) {
    %0 = "test.some_modification"(%it2) : (f32) -> f32
    %1 = arith.addf %it1, %0 : f32
    affine.yield %1, %1 : f32, f32
  }
  return
}
```

**说明:**

此用例测试iter_arg在定义被reduction值的操作链中被使用的情况。`%it2`被用于计算`%0`，然后`%0`被用于reduction操作，这不是简单的reduction。

**REDUCE模式下的行为:**

```mlir
%cst1 = arith.constant 0.0 : f32
%cst2 = arith.constant 1.0 : f32
affine.for %i = 0 to 10 iter_args(%it1 = %cst1, %it2 = %cst2) -> (f32, f32) {
  %0 = "test.some_modification"(%it2) : (f32) -> f32
  %1 = arith.addf %it1, %0 : f32
  affine.yield %1, %1 : f32, f32
}
```

关键点：
1. iter_arg在操作链中被使用时，不会被并行化
2. 这不是简单的reduction模式

---

### 用例 14: nested_min_max

**原始代码:**

```mlir
func.func @nested_min_max(%m: memref<?xf32>, %lb0: index,
                     %ub0: index, %ub1: index) {
  affine.for %j = 0 to 10 {
    affine.for %i = max affine_map<(d0, d1) -> (d0, d1)>(%lb0, %j)
            to min affine_map<(d0, d1) -> (d0, d1)>(%ub0, %ub1) {
      affine.load %m[%i] : memref<?xf32>
    }
  }
  return
}
```

**说明:**

此用例测试嵌套循环中使用min/max运算的循环边界。内层循环的下界依赖于外层循环变量。

**并行化后的行为:**

```mlir
affine.parallel (%j) = (0) to (10) {
  affine.parallel (%i) = (max(%lb0, %j)) to (min(%ub0, %ub1)) {
    affine.load %m[%i] : memref<?xf32>
  }
}
```

关键点：
1. 两层循环都被并行化
2. 内层循环的边界依赖于外层循环变量

---

### 用例 15: local_alloc

**原始代码:**

```mlir
func.func @local_alloc() {
  %cst = arith.constant 0.0 : f32
  affine.for %i = 0 to 100 {
    %m = memref.alloc() : memref<1xf32>
    %ma = memref.alloca() : memref<1xf32>
    affine.store %cst, %m[0] : memref<1xf32>
  }
  return
}
```

**说明:**

此用例测试循环内局部分配memref的情况。每个迭代都分配新的memref，这些memref只在当前迭代中使用。

**并行化后的行为:**

```mlir
affine.parallel (%i) = (0) to (100) {
  %m = memref.alloc() : memref<1xf32>
  %ma = memref.alloca() : memref<1xf32>
  affine.store %cst, %m[0] : memref<1xf32>
}
```

关键点：
1. 局部分配的memref不会阻止并行化
2. 每个并行迭代会有自己的局部memref

---

### 用例 16: local_alloc_cast

**原始代码:**

```mlir
func.func @local_alloc_cast() {
  %cst = arith.constant 0.0 : f32
  affine.for %i = 0 to 100 {
    %m = memref.alloc() : memref<128xf32>
    affine.for %j = 0 to 128 {
      affine.store %cst, %m[%j] : memref<128xf32>
    }
    affine.for %j = 0 to 128 {
      affine.store %cst, %m[0] : memref<128xf32>
    }
    %r = memref.reinterpret_cast %m to offset: [0], sizes: [8, 16],
           strides: [16, 1] : memref<128xf32> to memref<8x16xf32>
    affine.for %j = 0 to 8 {
      affine.store %cst, %r[%j, %j] : memref<8x16xf32>
    }
  }
  return
}
```

**说明:**

此用例测试循环内局部分配memref并进行类型转换的情况。包含多个内层循环和一个memref.reinterpret_cast操作。

**并行化后的行为:**

```mlir
affine.parallel (%i) = (0) to (100) {
  %m = memref.alloc() : memref<128xf32>
  affine.parallel (%j) = (0) to (128) {
    affine.store %cst, %m[%j] : memref<128xf32>
  }
  affine.for %j = 0 to 128 {
    affine.store %cst, %m[0] : memref<128xf32>
  }
  %r = memref.reinterpret_cast %m to offset: [0], sizes: [8, 16],
         strides: [16, 1] : memref<128xf32> to memref<8x16xf32>
  affine.parallel (%j) = (0) to (8) {
    affine.store %cst, %r[%j, %j] : memref<8x16xf32>
  }
}
```

关键点：
1. 外层循环被并行化
2. 第一个内层循环被并行化（没有依赖）
3. 第二个内层循环保持串行（所有迭代都写入同一位置）
4. 第三个内层循环被并行化

---

### 用例 17: iter_arg_memrefs

**原始代码:**

```mlir
func.func @iter_arg_memrefs(%in: memref<10xf32>) {
  %mi = memref.alloc() : memref<f32>
  %mo = affine.for %i = 0 to 10 iter_args(%m_arg = %mi) -> (memref<f32>) {
    affine.yield %m_arg : memref<f32>
  }
  return
}
```

**说明:**

此用例测试iter_arg是memref类型的情况。循环携带的memref会被视为串行化循环。

**并行化后的行为:**

```mlir
%mi = memref.alloc() : memref<f32>
%mo = affine.for %i = 0 to 10 iter_args(%m_arg = %mi) -> (memref<f32>) {
  affine.yield %m_arg : memref<f32>
}
```

关键点：
1. 循环携带的memref会导致循环保持串行
2. memref类型的iter_arg被视为串行化因素

---

### 用例 18: test_add_inv_or_terminal_symbol

**原始代码:**

```mlir
func.func @test_add_inv_or_terminal_symbol(%arg0: memref<9x9xi32>, %arg1: i1) {
  %idx0 = index.constant 1
  %29 = tensor.empty() : tensor<10xf16>
  memref.alloca_scope {
    %dim_30 = tensor.dim %29, %idx0 : tensor<10xf16>
    %alloc_31 = memref.alloc(%idx0, %idx0) {alignment = 64 : i64} : memref<?x?xf16>
    affine.for %arg3 = 0 to %dim_30 {
      %207 = affine.load %alloc_31[%idx0, %idx0] : memref<?x?xf16>
      affine.store %207, %alloc_31[%idx0, %idx0] : memref<?x?xf16>
    }
  }
  return
}
```

**说明:**

此用例测试复杂的操作组合，包括memref.alloca_scope、tensor操作等。确保affine分析机器不会崩溃。

**并行化后的行为:**

循环保持串行，因为所有迭代都访问相同的位置。

关键点：
1. 复杂的操作组合不会导致崩溃
2. 正确处理memref.alloca_scope和tensor操作

---

### 用例 19: explicit_parallel

**原始代码:**

```mlir
func.func @explicit_parallel(%arg0: memref<1x123x194xf64>, %arg5: memref<34x99x194xf64>) {
  affine.parallel (%arg7, %arg8) = (0, 0) to (85, 180) {
    affine.for %arg9 = 0 to 18 {
      %0 = affine.load %arg0[0, %arg7 + 19, %arg8 + 7] : memref<1x123x194xf64>
      %1 = affine.load %arg5[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64>
      %2 = arith.addf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %1, %arg0[0, %arg7 + 19, %arg8 + 7] : memref<1x123x194xf64>
    }
  }
  return
}
```

**说明:**

此用例测试已经包含显式并行循环的情况。确保在依赖分析时正确考虑外层并行循环。

**并行化后的行为:**

```mlir
affine.parallel (%arg7, %arg8) = (0, 0) to (85, 180) {
  affine.for %arg9 = 0 to 18 {
    %0 = affine.load %arg0[0, %arg7 + 19, %arg8 + 7] : memref<1x123x194xf64>
    %1 = affine.load %arg5[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64>
    %2 = arith.addf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %1, %arg0[0, %arg7 + 19, %arg8 + 7] : memref<1x123x194xf64>
  }
}
```

关键点：
1. 显式并行循环被正确处理
2. 内层循环保持串行，因为存在依赖（所有迭代都写入同一位置，该位置由外层并行循环索引）
3. 在计算循环深度时正确考虑外层并行循环

---

## 总结

`affine-parallelize` pass是一个强大的循环并行化pass，它可以：

1. **自动并行化循环**: 分析循环的依赖关系，将可以并行执行的循环转换为并行循环
2. **处理复杂边界**: 支持min/max运算的循环边界
3. **支持符号边界**: 可以处理符号变量的循环边界
4. **支持嵌套并行**: 可以并行化多层嵌套循环
5. **支持reduction**: 当设置`parallel-reductions=1`时，可以并行化reduction操作
6. **限制嵌套深度**: 可以通过`max-nested`选项限制并行循环的嵌套深度
7. **处理局部memref**: 可以正确处理循环内局部分配的memref

该pass在优化循环性能时非常有用，特别是对于可以并行执行的循环。但需要注意：
- 未知操作会导致循环保持串行
- 非affine操作会导致循环保持串行
- 复杂的reduction模式可能无法被并行化
- 循环携带的memref会导致循环保持串行
- 需要正确处理外层并行循环对内层循环的影响
