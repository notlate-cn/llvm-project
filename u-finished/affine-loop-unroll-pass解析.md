# affine-loop-unroll Pass 解析

## 概述

`affine-loop-unroll` pass用于将affine循环展开。它可以完全展开循环或按指定因子展开。该pass支持以下主要选项：

- `unroll-full`: 完全展开循环
- `unroll-factor`: 按指定因子展开
- `unroll-full-threshold`: 完全展开的阈值
- `cleanup-unroll`: 清理展开后的循环

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/unroll.mlir`

## RUN命令

该测试文件包含以下RUN命令：

1. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-full=true}))" | FileCheck %s --check-prefix UNROLL-FULL`

2. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-full=true unroll-full-threshold=2}))" | FileCheck %s --check-prefix SHORT`

3. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))" | FileCheck %s --check-prefix UNROLL-BY-4`

4. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=1}))" | FileCheck %s --check-prefix UNROLL-BY-1`

5. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=5 cleanup-unroll=true}))" | FileCheck %s --check-prefix UNROLL-CLEANUP-LOOP`

6. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(gpu.module(gpu.func(affine-loop-unroll{unroll-full=true})))" | FileCheck %s --check-prefix GPU-UNROLL-FULL`

## 测试用例解析

### 用例 1: loop_nest_simplest

**原始代码:**

```mlir
func.func @loop_nest_simplest() {
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = 0 to 4 {
      %x = arith.constant 1 : i32
    }
  }
  return
}
```

**说明:**

这是一个最简单的嵌套循环测试用例。外层循环从0到100，步长为2；内层循环从0到4，步长为1。

**UNROLL-FULL模式下的行为:**

当使用`unroll-full=true`时，内层循环会被完全展开。由于内层循环迭代4次（0到4），展开后会生成4个连续的操作，每个操作对应一次迭代。

展开后的结果类似：
```mlir
affine.for %arg0 = 0 to 100 step 2 {
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_0 = arith.constant 1 : i32
  %c1_i32_1 = arith.constant 1 : i32
  %c1_i32_2 = arith.constant 1 : i32
}
```

---

### 用例 2: loop_nest_simple_iv_use

**原始代码:**

```mlir
func.func @loop_nest_simple_iv_use() {
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = 0 to 4 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
    }
  }
  return
}
```

**说明:**

此用例测试循环变量在循环体内的使用。内层循环变量`%j`被用于加法操作。

**UNROLL-FULL模式下的行为:**

展开时，循环变量`%j`会被替换为具体的值（0, 1, 2, 3）。每次迭代中的`%j`会被替换为相应的常量，并通过`affine.apply`计算偏移。

展开后的结果类似：
```mlir
affine.for %arg0 = 0 to 100 step 2 {
  %0 = "addi32"(%c0, %c0) : (index, index) -> i32
  %1 = affine.apply #map(%c0)
  %2 = "addi32"(%1, %1) : (index, index) -> i32
  %3 = affine.apply #map1(%c0)
  %4 = "addi32"(%3, %3) : (index, index) -> i32
  %5 = affine.apply #map2(%c0)
  %6 = "addi32"(%5, %5) : (index, index) -> i32
}
```

---

### 用例 3: loop_nest_body_def_use

**原始代码:**

```mlir
func.func @loop_nest_body_def_use() {
  affine.for %i = 0 to 100 step 2 {
    %c0 = arith.constant 0 : index
    affine.for %j = 0 to 4 {
      %x = "affine.apply" (%j) { map = affine_map<(d0) -> (d0 + 1)> } :
        (index) -> (index)
      %y = "addi32"(%x, %c0) : (index, index) -> index
    }
  }
  return
}
```

**说明:**

此用例测试循环体内定义的值在循环体内的使用。`%x`是在循环体内定义的，并在后续操作`%y`中使用。

**UNROLL-FULL模式下的行为:**

展开时，不仅循环变量会被替换，循环体内定义的值也会被正确处理。每个展开的迭代都会有自己的`%x`和`%y`定义。

---

### 用例 4: loop_nest_strided

**原始代码:**

```mlir
func.func @loop_nest_strided() {
  affine.for %i = 0 to 100 {
    affine.for %j = 2 to 6 step 2 {
      %x = "affine.apply" (%j) { map = affine_map<(d0) -> (d0 + 1)> } :
        (index) -> (index)
      %y = "addi32"(%x, %x) : (index, index) -> index
    }
    affine.for %k = 2 to 7 step 2 {
      %z = "affine.apply" (%k) { map = affine_map<(d0) -> (d0 + 1)> } :
        (index) -> (index)
      %w = "addi32"(%z, %z) : (index, index) -> index
    }
  }
  return
}
```

**说明:**

此用例测试带步长的循环展开。有两个内层循环：
- `%j`循环：从2到6，步长为2（迭代2, 4）
- `%k`循环：从2到7，步长为2（迭代2, 4, 6）

**UNROLL-FULL模式下的行为:**

带步长的循环会被正确展开。循环变量会被替换为正确的值（考虑步长）。

---

### 用例 5: loop_nest_multiple_results

**原始代码:**

```mlir
func.func @loop_nest_multiple_results() {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 2 step 1 {
      %x = affine.apply affine_map<(d0, d1) -> (d0 + 1)> (%i, %j)
      %y = "addi32"(%x, %x) : (index, index) -> index
      %z = affine.apply affine_map<(d0, d1) -> (d0 + 3)> (%i, %j)
      %w:2 = "fma"(%z, %x, %x) : (index, index, index) -> (index, index)
    }
  }
  return
}
```

**说明:**

此用例测试返回多个结果的操作。`%w:2`表示操作返回两个结果。

**UNROLL-FULL模式下的行为:**

多结果操作会被正确处理，每个展开的迭代都会有相应的多结果操作。

---

### 用例 6: loop_nest_seq_imperfect

**原始代码:**

```mlir
func.func @loop_nest_seq_imperfect(%a : memref<128x128xf32>) {
  %c128 = arith.constant 128 : index
  affine.for %i = 0 to 100 {
    %ld = "vld"(%i) : (index) -> i32
    affine.for %j = 0 to 4 {
      %x = "affine.apply" (%j) { map = affine_map<(d0) -> (d0 + 1)> } :
        (index) -> (index)
       %y = "vmulf"(%j, %x) : (index, index) -> index
       %z = "vaddf"(%y, %y) : (index, index) -> index
    }
    %addr = "scale"(%c128, %i) : (index, index) -> index
    "vst"(%addr, %i) : (index, index) -> ()
  }
  return
}
```

**说明:**

这是一个非完美嵌套循环的例子。在内层循环前后都有操作：
- `%ld = "vld"(%i)`：在内层循环前
- `%addr = "scale"(%c128, %i)` 和 `"vst"(%addr, %i)`：在内层循环后

**UNROLL-FULL模式下的行为:**

展开内层循环后，非完美嵌套会变成完美嵌套。内层循环的操作会被展开，而循环前后的操作保持不变。

---

### 用例 7: loop_nest_seq_multiple

**原始代码:**

```mlir
func.func @loop_nest_seq_multiple() {
  affine.for %j = 0 to 4 {
    %x = "affine.apply" (%j) { map = affine_map<(d0) -> (d0 + 1)> } :
      (index) -> (index)
    "mul"(%x, %x) : (index, index) -> ()
  }

  %k = arith.constant 99 : index
  affine.for %m = 0 to 100 step 2 {
    affine.for %n = 0 to 4 {
      %y = "affine.apply" (%n) { map = affine_map<(d0) -> (d0 + 1)> } :
        (index) -> (index)
      %z = "affine.apply" (%n, %k) { map = affine_map<(d0) [s0] -> (d0 + s0 + 1)> } :
        (index, index) -> (index)
    }
  }
  return
}
```

**说明:**

此用例测试多个独立的循环。包含两个独立的循环结构：
1. 一个单层循环
2. 一个嵌套循环，其中内层循环使用了符号变量`%k`

**UNROLL-FULL模式下的行为:**

每个循环都会被独立展开。符号变量会被正确处理。

---

### 用例 8: loop_nest_unroll_full

**原始代码:**

```mlir
func.func @loop_nest_unroll_full() {
  affine.for %i = 0 to 1 {
    %x = "foo"() : () -> i32
    %y = "bar"() : () -> i32
  }
  return
}
```

**说明:**

这是一个只迭代一次的循环。从0到1，只迭代一次（i=0）。

**UNROLL-FULL模式下的行为:**

单次迭代的循环会被完全展开，循环结构被移除，只保留循环体内的操作：

```mlir
func.func @loop_nest_unroll_full() {
  %0 = "foo"() : () -> i32
  %1 = "bar"() : () -> i32
  return
}
```

---

### 用例 9: gpu_loop_nest_simplest (GPU模块)

**原始代码:**

```mlir
gpu.module @unroll_full {
  gpu.func @gpu_loop_nest_simplest() {
    affine.for %i = 0 to 100 step 2 {
      affine.for %j = 0 to 4 {
        %x = arith.constant 1 : i32
      }
    }
    gpu.return
  }
}
```

**说明:**

此用例测试在GPU模块中的循环展开。GPU函数使用`gpu.func`和`gpu.return`。

**GPU-UNROLL-FULL模式下的行为:**

GPU模块中的循环展开行为与普通函数类似，内层循环会被完全展开。

---

### 用例 10: loop_nest_outer_unroll (SHORT模式)

**原始代码:**

```mlir
func.func @loop_nest_outer_unroll() {
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 4 {
      %x = "affine.apply" (%j) { map = affine_map<(d0) -> (d0 + 1)> } :
        (index) -> (index)
      %y = "addi32"(%x, %x) : (index, index) -> index
    }
  }
  return
}
```

**说明:**

此用例测试外层循环的展开。当设置`unroll-full-threshold=2`时，外层循环（迭代2次）会被展开。

**SHORT模式下的行为:**

外层循环被展开为两个独立的内层循环：

```mlir
affine.for %arg0 = 0 to 4 {
  %0 = affine.apply #map(%arg0)
  %1 = "addi32"(%0, %0) : (index, index) -> index
}
affine.for %arg0 = 0 to 4 {
  %0 = affine.apply #map(%arg0)
  %1 = "addi32"(%0, %0) : (index, index) -> index
}
```

---

### 用例 11: loop_nest_seq_long

**原始代码:**

```mlir
func.func @loop_nest_seq_long() -> i32 {
  %A = memref.alloc() : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
  %B = memref.alloc() : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
  %C = memref.alloc() : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>

  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32

  %zero_idx = arith.constant 0 : index

  affine.for %n0 = 0 to 512 {
    affine.for %n1 = 0 to 8 {
      memref.store %one,  %A[%n0, %n1] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
      memref.store %two,  %B[%n0, %n1] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
      memref.store %zero, %C[%n0, %n1] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
    }
  }

  affine.for %x = 0 to 2 {
    affine.for %y = 0 to 2 {
      affine.for %arg2 = 0 to 8 {
        %b2 = "affine.apply" (%y, %arg2) {map = affine_map<(d0, d1) -> (16*d0 + d1)>} : (index, index) -> index
        %z = memref.load %B[%x, %b2] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
        "op1"(%z) : (i32) -> ()
      }
      affine.for %j1 = 0 to 8 {
        affine.for %j2 = 0 to 8 {
          %a2 = "affine.apply" (%y, %j2) {map = affine_map<(d0, d1) -> (16*d0 + d1)>} : (index, index) -> index
          %v203 = memref.load %A[%j1, %a2] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
          "op2"(%v203) : (i32) -> ()
        }
        affine.for %k2 = 0 to 8 {
          %s0 = "op3"() : () -> i32
          %c2 = "affine.apply" (%x, %k2) {map = affine_map<(d0, d1) -> (16*d0 + d1)>} : (index, index) -> index
          %s1 =  memref.load %C[%j1, %c2] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
          %s2 = "addi32"(%s0, %s1) : (i32, i32) -> i32
          memref.store %s2, %C[%j1, %c2] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
        }
      }
      "op4"() : () -> ()
    }
  }
  %ret = memref.load %C[%zero_idx, %zero_idx] : memref<512 x 512 x i32, affine_map<(d0, d1) -> (d0, d1)>, 2>
  return %ret : i32
}
```

**说明:**

这是一个复杂的嵌套循环结构，包含多个memref操作和多层嵌套循环。测试了在复杂场景下的循环展开行为。

**SHORT模式下的行为:**

由于设置了`unroll-full-threshold=2`，外层循环`%x`和`%y`（各迭代2次）会被展开，而内层循环保持不变。

---

### 用例 12: unroll_unit_stride_no_cleanup

**原始代码:**

```mlir
func.func @unroll_unit_stride_no_cleanup() {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 8 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
    affine.for %k = 0 to 8 {
    }
  }
  return
}
```

**说明:**

此用例测试单位步长循环的展开，且不需要清理循环。内层循环从0到8，迭代8次，正好是unroll-factor=4的倍数。

**UNROLL-BY-4模式下的行为:**

循环按因子4展开，步长变为4。由于8是4的倍数，不需要清理循环：

```mlir
affine.for %arg0 = 0 to 100 {
  affine.for %arg1 = 0 to 8 step 4 {
    %0 = "addi32"(%arg1, %arg1) : (index, index) -> i32
    %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    %2 = affine.apply #map(%arg1)
    %3 = "addi32"(%2, %2) : (index, index) -> i32
    %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // ... 重复4次
  }
  affine.for %arg1 = 0 to 8 {
  }
}
```

---

### 用例 13: unroll_unit_stride_cleanup

**原始代码:**

```mlir
func.func @unroll_unit_stride_cleanup() {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 10 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
  }
  return
}
```

**说明:**

此用例测试需要清理循环的情况。内层循环从0到10，迭代10次，不是unroll-factor=4的倍数。

**UNROLL-BY-4模式下的行为:**

循环按因子4展开，步长变为4。由于10不是4的倍数，会生成一个清理循环处理剩余的迭代（8到10）：

```mlir
affine.for %arg0 = 0 to 100 {
  affine.for %arg1 = 0 to 8 step 4 {
    // 展开的4次迭代
  }
  affine.for %arg1 = 8 to 10 {
    // 清理循环，处理剩余2次迭代
  }
}
```

---

### 用例 14: unroll_non_unit_stride_cleanup

**原始代码:**

```mlir
func.func @unroll_non_unit_stride_cleanup() {
  affine.for %i = 0 to 100 {
    affine.for %j = 2 to 48 step 5 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
  }
  return
}
```

**说明:**

此用例测试非单位步长循环的展开。内层循环从2到48，步长为5。

**UNROLL-BY-4模式下的行为:**

非单位步长的循环展开会更复杂。需要计算正确的边界和步长。

---

### 用例 15: loop_nest_single_iteration_after_unroll

**原始代码:**

```mlir
func.func @loop_nest_single_iteration_after_unroll(%N: index) {
  affine.for %i = 0 to %N {
    affine.for %j = 0 to 5 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
    }
  }
  return
}
```

**说明:**

此用例测试展开后变成单次迭代的循环。内层循环迭代5次，按因子4展开后，会剩余1次迭代。

**UNROLL-BY-4模式下的行为:**

展开后，清理循环只有1次迭代，会被提升（promote）为普通代码：

```mlir
affine.for %arg1 = 0 to %N {
  %0 = "addi32"(%c0, %c0) : (index, index) -> i32
  %1 = affine.apply #map(%c0)
  %2 = "addi32"(%1, %1) : (index, index) -> i32
  %3 = affine.apply #map1(%c0)
  %4 = "addi32"(%3, %3) : (index, index) -> i32
  %5 = affine.apply #map2(%c0)
  %6 = "addi32"(%5, %5) : (index, index) -> i32
  %7 = "addi32"(%c4, %c4) : (index, index) -> i32
}
```

---

### 用例 16: loop_nest_operand1

**原始代码:**

```mlir
func.func @loop_nest_operand1() {
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = 0 to affine_map<(d0) -> (d0 - d0 mod 4)> (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}
```

**说明:**

此用例测试循环边界操作数。内层循环的上界是一个affine_map，依赖于外层循环变量`%i`。

**UNROLL-BY-4模式下的行为:**

当循环边界是操作数时，展开会更复杂。如果边界可以被计算为4的倍数，则不需要清理循环。

---

### 用例 17: loop_nest_operand2

**原始代码:**

```mlir
func.func @loop_nest_operand2() {
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = affine_map<(d0) -> (d0)> (%i) to affine_map<(d0) -> (5*d0 + 4)> (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}
```

**说明:**

此用例测试循环上下界都是操作数的情况。

**UNROLL-BY-4模式下的行为:**

上下界都是affine_map时，展开需要计算正确的边界。

---

### 用例 18: floordiv_mod_ub

**原始代码:**

```mlir
func.func @floordiv_mod_ub(%M : index, %N : index) {
  affine.for %i = 0 to %N step 4 {
    affine.for %j = 0 to min affine_map<(d0)[s0] -> ((16 * d0) floordiv (4 * s0))>(%i)[%N] {
      "test.foo"() : () -> ()
    }
  }
  affine.for %i = 0 to %N step 4 {
    affine.for %j = 0 to min affine_map<(d0)[s0] -> ((16 * d0) mod (4 * s0))>(%i)[%N] {
      "test.foo"() : () -> ()
    }
  }
  return
}
```

**说明:**

此用例测试使用floordiv和mod运算的循环边界。

**UNROLL-BY-4模式下的行为:**

复杂的affine表达式会被正确处理。

---

### 用例 19: loop_nest_operand3

**原始代码:**

```mlir
func.func @loop_nest_operand3() {
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = affine_map<(d0) -> (d0)> (%i) to affine_map<(d0) -> (d0 + 9)> (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}
```

**说明:**

此用例测试循环边界差为常数但不是unroll因子倍数的情况。上下界差为9，不是4的倍数。

**UNROLL-BY-4模式下的行为:**

会生成清理循环，且清理循环只有1次迭代，会被提升。

---

### 用例 20: loop_nest_symbolic_bound

**原始代码:**

```mlir
func.func @loop_nest_symbolic_bound(%N : index) {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to %N {
      %x = "foo"() : () -> i32
    }
  }
  return
}
```

**说明:**

此用例测试符号边界。内层循环的上界是符号变量`%N`。

**UNROLL-BY-4模式下的行为:**

符号边界需要生成清理循环：

```mlir
affine.for %arg1 = 0 to 100 {
  affine.for %arg2 = 0 to #map()[%N] step 4 {
    // 展开的4次迭代
  }
  affine.for %arg2 = #map()[%N] to %N {
    // 清理循环
  }
}
```

---

### 用例 21: loop_nest_symbolic_bound_with_step

**原始代码:**

```mlir
func.func @loop_nest_symbolic_bound_with_step(%N : index) {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to %N step 3 {
      %x = "foo"() : () -> i32
    }
  }
  return
}
```

**说明:**

此用例测试符号边界和步长的组合。内层循环的上界是符号变量`%N`，步长为3。

**UNROLL-BY-4模式下的行为:**

需要计算正确的展开步长（LCM(3, 4) = 12）：

```mlir
affine.for %arg1 = 0 to 100 {
  affine.for %arg2 = 0 to #map()[%N] step 12 {
    // 展开的4次迭代
  }
  affine.for %arg2 = #map()[%N] to %N step 3 {
    // 清理循环
  }
}
```

---

### 用例 22: loop_nest_symbolic_and_min_upper_bound

**原始代码:**

```mlir
func.func @loop_nest_symbolic_and_min_upper_bound(%M : index, %N : index, %K : index) {
  affine.for %i = %M to min affine_map<()[s0, s1] -> (s0, s1, 1024)>()[%N, %K] {
    "test.foo"() : () -> ()
  }
  return
}
```

**说明:**

此用例测试使用min运算的循环上界。

**UNROLL-BY-4模式下的行为:**

使用min的复杂边界可能无法展开，因为无法确定精确的迭代次数。

---

### 用例 23: loop_nest_non_trivial_multiple_upper_bound

**原始代码:**

```mlir
func.func @loop_nest_non_trivial_multiple_upper_bound(%M : index, %N : index) {
  %T = affine.apply affine_map<(d0) -> (4*d0 + 1)>(%M)
  %K = affine.apply affine_map<(d0) -> (d0 - 1)> (%T)
  affine.for %i = 0 to min affine_map<(d0, d1) -> (4 * d0, d1, 1024)>(%N, %K) {
    "foo"() : () -> ()
  }
  return
}
```

**说明:**

此用例测试通过组合可以推断出是4的倍数的边界。

**UNROLL-BY-4模式下的行为:**

虽然边界看起来复杂，但通过组合可以推断出迭代次数是4的倍数，因此不需要清理循环。

---

### 用例 24: multi_upper_bound

**原始代码:**

```mlir
func.func @multi_upper_bound(%arg0: index) {
  affine.for %i = 0 to min affine_map<()[s0] -> (8 * s0, 12 * s0)>()[%arg0] {
    "test.foo"() : () -> ()
  }
  return
}
```

**说明:**

此用例测试多结果的上界。

**UNROLL-BY-4模式下的行为:**

多结果上界无法展开，因为无法确定精确的迭代次数。

---

### 用例 25: multi_lower_bound

**原始代码:**

```mlir
func.func @multi_lower_bound(%arg0: index) {
  affine.for %i = max affine_map<()[s0] -> (8 * s0, 12 * s0)>()[%arg0] to 100 {
    "test.foo"() : () -> ()
  }
  return
}
```

**说明:**

此用例测试使用max运算的循环下界。

**UNROLL-BY-4模式下的行为:**

多结果下界目前无法展开。

---

### 用例 26: loop_nest_non_trivial_multiple_upper_bound_alt

**原始代码:**

```mlir
func.func @loop_nest_non_trivial_multiple_upper_bound_alt(%M : index, %N : index) {
  %K = affine.apply affine_map<(d0) -> (4*d0)> (%M)
  affine.for %i = 0 to min affine_map<()[s0, s1] -> (4 * s0, s1, 1024)>()[%N, %K] {
    "foo"() : () -> ()
  }
  return
}
```

**说明:**

此用例测试可以推断出是4的倍数的边界。

**UNROLL-BY-4模式下的行为:**

可以正确展开，不需要清理循环。

---

### 用例 27: unroll_by_one_should_promote_single_iteration_loop

**原始代码:**

```mlir
func.func @unroll_by_one_should_promote_single_iteration_loop() {
  affine.for %i = 0 to 1 {
    %x = "foo"(%i) : (index) -> i32
  }
  return
}
```

**说明:**

此用例测试unroll-factor=1的情况。单次迭代的循环应该被提升。

**UNROLL-BY-1模式下的行为:**

循环被完全移除，只保留循环体内的操作：

```mlir
func.func @unroll_by_one_should_promote_single_iteration_loop() {
  %c0 = arith.constant 0 : index
  %0 = "foo"(%c0) : (index) -> i32
  return
}
```

---

### 用例 28: loop_unroll_with_iter_args_and_cleanup

**原始代码:**

```mlir
func.func @loop_unroll_with_iter_args_and_cleanup(%arg0 : f32, %arg1 : f32, %n : index) -> (f32,f32) {
  %cf1 = arith.constant 1.0 : f32
  %cf2 = arith.constant 2.0 : f32
  %sum:2 = affine.for %iv = 0 to 10 iter_args(%i0 = %arg0, %i1 = %arg1) -> (f32, f32) {
    %sum0 = arith.addf %i0, %cf1 : f32
    %sum1 = arith.addf %i1, %cf2 : f32
    affine.yield %sum0, %sum1 : f32, f32
  }
  return %sum#0, %sum#1 : f32, f32
}
```

**说明:**

此用例测试带iter_args的循环展开。循环有迭代参数，用于累加操作。

**UNROLL-BY-4模式下的行为:**

带iter_args的循环展开会更复杂，需要正确处理迭代参数的传递。会生成主循环和清理循环：

```mlir
%sum:2 = affine.for %iv = 0 to 8 step 4 iter_args(%i0 = %arg0, %i1 = %arg1) -> (f32, f32) {
  // 展开的4次迭代
  affine.yield %y1, %y2
}
%sum1:2 = affine.for %iv = 8 to 10 iter_args(%v1 = %sum#0, %v2 = %sum#1) -> (f32, f32) {
  // 清理循环
}
return %sum1#0, %sum1#1 : f32, f32
```

---

### 用例 29: unroll_with_iter_args_and_promotion

**原始代码:**

```mlir
func.func @unroll_with_iter_args_and_promotion(%arg0 : f32, %arg1 : f32) -> f32 {
  %from = arith.constant 0 : index
  %to = arith.constant 10 : index
  %step = arith.constant 1 : index
  %sum = affine.for %iv = 0 to 9 iter_args(%sum_iter = %arg0) -> (f32) {
    %next = arith.addf %sum_iter, %arg1 : f32
    affine.yield %next : f32
  }
  return %sum : f32
}
```

**说明:**

此用例测试带iter_args的循环展开，且清理循环是单次迭代。

**UNROLL-BY-4模式下的行为:**

清理循环只有1次迭代，会被提升：

```mlir
%sum = affine.for %iv = 0 to 8 step 4 iter_args(%v0 = %arg0) -> (f32) {
  // 展开的4次迭代
  affine.yield %v4
}
%res = arith.addf %sum, %arg1 : f32
return %res : f32
```

---

### 用例 30: unroll_zero_trip_count_case

**原始代码:**

```mlir
func.func @unroll_zero_trip_count_case() {
  affine.for %i = 0 to 0 {
  }
  return
}
```

**说明:**

此用例测试零迭代次数的循环。

**UNROLL-FULL模式下的行为:**

零迭代次数的循环保持不变：

```mlir
func.func @unroll_zero_trip_count_case() {
  affine.for %i = 0 to 0 {
  }
  return
}
```

---

### 用例 31: unroll_cleanup_loop_with_larger_unroll_factor

**原始代码:**

```mlir
func.func @unroll_cleanup_loop_with_larger_unroll_factor() {
  affine.for %i = 0 to 3 {
    %x = "foo"(%i) : (index) -> i32
  }
  return
}
```

**说明:**

此用例测试unroll-factor大于迭代次数的情况。循环迭代3次，但unroll-factor=5。

**UNROLL-CLEANUP-LOOP模式下的行为:**

由于unroll-factor大于迭代次数，循环会被完全展开，不需要清理循环：

```mlir
func.func @unroll_cleanup_loop_with_larger_unroll_factor() {
  %c0 = arith.constant 0 : index
  %0 = "foo"(%c0) : (index) -> i32
  %v1 = affine.apply #map()
  %1 = "foo"(%v1) : (index) -> i32
  %v2 = affine.apply #map()
  %2 = "foo"(%v2) : (index) -> i32
  return
}
```

---

### 用例 32: unroll_cleanup_loop_with_smaller_unroll_factor

**原始代码:**

```mlir
func.func @unroll_cleanup_loop_with_smaller_unroll_factor() {
  affine.for %i = 0 to 7 {
    %x = "foo"(%i) : (index) -> i32
  }
  return
}
```

**说明:**

此用例测试unroll-factor小于迭代次数的情况。循环迭代7次，unroll-factor=5。

**UNROLL-CLEANUP-LOOP模式下的行为:**

循环会被完全展开，因为设置了`cleanup-unroll=true`：

```mlir
func.func @unroll_cleanup_loop_with_smaller_unroll_factor() {
  %c0 = arith.constant 0 : index
  %0 = "foo"(%c0) : (index) -> i32
  // ... 7次迭代全部展开
  return
}
```

---

### 用例 33: unroll_cleanup_loop_with_identical_unroll_factor

**原始代码:**

```mlir
func.func @unroll_cleanup_loop_with_identical_unroll_factor() {
  affine.for %i = 0 to 5 {
    %x = "foo"(%i) : (index) -> i32
  }
  return
}
```

**说明:**

此用例测试unroll-factor等于迭代次数的情况。循环迭代5次，unroll-factor=5。

**UNROLL-CLEANUP-LOOP模式下的行为:**

循环会被完全展开：

```mlir
func.func @unroll_cleanup_loop_with_identical_unroll_factor() {
  %c0 = arith.constant 0 : index
  %0 = "foo"(%c0) : (index) -> i32
  // ... 5次迭代全部展开
  return
}
```

---

### 用例 34: known_multiple_ceildiv

**原始代码:**

```mlir
func.func @known_multiple_ceildiv(%N: index, %S: index) {
  %cst = arith.constant 0.0 : f32
  %m = memref.alloc(%S) : memref<?xf32>
  affine.for %i = 0 to affine_map<(d0) -> (32 * d0 + 64)>(%N) step 8 {
    affine.store %cst, %m[%i] : memref<?xf32>
  }
  affine.for %i = 0 to affine_map<(d0) -> ((32 * d0 + 64) floordiv 8)>(%N) {
    affine.store %cst, %m[%i] : memref<?xf32>
  }
  return
}
```

**说明:**

此用例测试使用ceildiv和floordiv的循环边界，且可以推断出是展开因子的倍数。

**UNROLL-BY-4模式下的行为:**

通过affine表达式的分析，可以推断出迭代次数是4的倍数，因此不需要清理循环。

---

## 总结

`affine-loop-unroll` pass是一个强大的循环优化pass，它可以：

1. **完全展开循环**: 当设置`unroll-full=true`时，会完全展开循环
2. **按因子展开**: 当设置`unroll-factor=N`时，会按指定因子展开循环
3. **生成清理循环**: 当迭代次数不是展开因子的倍数时，会生成清理循环处理剩余迭代
4. **提升单次迭代循环**: 当清理循环只有1次迭代时，会将其提升为普通代码
5. **处理复杂边界**: 可以处理符号边界、affine_map边界、min/max边界等
6. **支持iter_args**: 可以正确处理带迭代参数的循环
7. **支持GPU模块**: 可以在GPU模块中使用

该pass在优化循环性能时非常有用，特别是对于小循环或已知迭代次数的循环。
