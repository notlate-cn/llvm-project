# affine-loop-unroll-jam Pass 解析

## 概述

`affine-loop-unroll-jam` pass用于将嵌套循环进行展开和合并（unroll and jam）。这是一种循环优化技术，它将外层循环展开，然后将内层循环的多个迭代合并在一起执行。这种技术可以提高指令级并行性和缓存局部性。

该pass支持以下主要选项：

- `unroll-jam-factor`: 展开和合并的因子

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/unroll-jam.mlir`

## RUN命令

该测试文件包含以下RUN命令：

1. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))" | FileCheck %s`

2. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=4}))" | FileCheck --check-prefix=UJAM-FOUR %s`

3. `mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(gpu.module(gpu.func(affine-loop-unroll-jam{unroll-jam-factor=2})))" | FileCheck --check-prefix=GPU-HJAM %s`

## 测试用例解析

### 用例 1: unroll_jam_imperfect_nest

**原始代码:**

```mlir
func.func @unroll_jam_imperfect_nest() {
  affine.for %i = 0 to 101 {
    %x = "addi32"(%i, %i) : (index, index) -> i32
    affine.for %j = 0 to 17 {
      %y = "addi32"(%i, %i) : (index, index) -> i32
      %z = "addi32"(%y, %y) : (i32, i32) -> i32
    }
    %w = "foo"(%i, %x) : (index, i32) -> i32
  }
  return
}
```

**说明:**

这是一个非完美嵌套循环的例子。在外层循环内，内层循环前后都有操作：
- `%x = "addi32"(%i, %i)`: 在内层循环前
- `%w = "foo"(%i, %x)`: 在内层循环后

**unroll-jam-factor=2模式下的行为:**

外层循环按因子2展开，步长变为2。内层循环的两次迭代会被合并在一起执行：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  affine.for %arg1 = 0 to 17 {
    %res3 = "addi32"(%arg0, %arg0) : (index, index) -> i32
    "addi32"(%res3, %res3) : (i32, i32) -> i32
    %inc1 = affine.apply #map_plus_1(%arg0)
    %res4 = "addi32"(%inc1, %inc1) : (index, index) -> i32
    "addi32"(%res4, %res4) : (i32, i32) -> i32
  }
  "foo"(%arg0, %res1) : (index, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2) : (index, i32) -> i32
}

// 清理循环（单次迭代）
%res5 = "addi32"(%c100, %c100) : (index, index) -> i32
affine.for %arg0 = 0 to 17 {
  %res6 = "addi32"(%c100, %c100) : (index, index) -> i32
  "addi32"(%res6, %res6) : (i32, i32) -> i32
}
"foo"(%c100, %res5) : (index, i32) -> i32
return
```

关键点：
1. 外层循环步长变为2（从0到100）
2. 内层循环中，两次外层迭代的操作被合并在一起
3. 生成了清理循环处理剩余的迭代（100到101）
4. 清理循环只有1次迭代，但不会被提升，因为它包含内层循环

---

### 用例 2: gpu_loop_nest_simplest (GPU模块)

**原始代码:**

```mlir
gpu.module @unroll_jam {
  gpu.func @unroll_jam_imperfect_nest() {
    affine.for %i = 0 to 101 {
      %x = "addi32"(%i, %i) : (index, index) -> i32
      affine.for %j = 0 to 17 {
        %y = "addi32"(%i, %i) : (index, index) -> i32
        %z = "addi32"(%y, %y) : (i32, i32) -> i32
      }
      %w = "foo"(%i, %x) : (index, i32) -> i32
    }
    gpu.return
  }
}
```

**说明:**

此用例测试在GPU模块中的unroll-jam行为。与普通函数类似，但使用`gpu.func`和`gpu.return`。

**GPU-HJAM模式下的行为:**

GPU模块中的unroll-jam行为与普通函数类似。

---

### 用例 3: loop_nest_unknown_count_1

**原始代码:**

```mlir
func.func @loop_nest_unknown_count_1(%N : index) {
  affine.for %i = 1 to %N {
    affine.for %j = 1 to 100 {
      %x = "foo"() : () -> i32
    }
  }
  return
}
```

**说明:**

此用例测试外层循环边界为符号变量的情况。外层循环从1到`%N`，迭代次数未知。

**unroll-jam-factor=2模式下的行为:**

由于外层循环边界是符号变量，需要生成清理循环：

```mlir
// 主循环
affine.for %arg0 = 1 to #map_div_offset()[%N] step 2 {
  affine.for %arg1 = 1 to 100 {
    "foo"() : () -> i32
    "foo"() : () -> i32
  }
}

// 清理循环
affine.for %arg0 = #map_div_offset()[%N] to %N {
  affine.for %arg1 = 1 to 100 {
    "foo"() : () -> i32
  }
}
```

其中`#map_div_offset`计算清理循环的起始位置：`((s0 - 1) floordiv 2) * 2 + 1`

---

### 用例 4: loop_nest_unknown_count_2

**原始代码:**

```mlir
func.func @loop_nest_unknown_count_2(%N : index) {
  affine.for %i = %N to affine_map<()[s0] -> (s0+9)> ()[%N] {
    affine.for %j = 1 to 100 {
      "foo"(%i) : (index) -> ()
    }
  }
  return
}
```

**说明:**

此用例测试外层循环的下界和上界都是符号变量的情况。外层循环从`%N`到`%N+9`，迭代9次。

**UJAM-FOUR模式下的行为:**

外层循环按因子4展开。由于迭代9次，会生成清理循环处理剩余1次迭代：

```mlir
// 主循环
affine.for %arg0 = %N to #map_ub()[%N] step 4 {
  affine.for %arg1 = 1 to 100 {
    "foo"(%arg0) : (index) -> ()
    %iv_plus_1 = affine.apply #map_plus_1(%arg0)
    "foo"(%iv_plus_1) : (index) -> ()
    %iv_plus_2 = affine.apply #map_plus_2(%arg0)
    "foo"(%iv_plus_2) : (index) -> ()
    %iv_plus_3 = affine.apply #map_plus_3(%arg0)
    "foo"(%iv_plus_3) : (index) -> ()
  }
}

// 清理循环（单次迭代，被提升）
%res = affine.apply #map_ub()[%N]
affine.for %arg0 = 1 to 100 {
  "foo"(%res) : (index) -> ()
}
```

关键点：
1. 清理循环只有1次迭代，但因为它包含内层循环，所以不会被完全提升
2. 外层循环变量被替换为具体的值

---

### 用例 5: loop_nest_symbolic_and_min_upper_bound

**原始代码:**

```mlir
func.func @loop_nest_symbolic_and_min_upper_bound(%M : index, %N : index, %K : index) {
  affine.for %i = 0 to min affine_map<()[s0, s1] -> (s0, s1, 1024)>()[%M, %N] {
    affine.for %j = 0 to %K {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
```

**说明:**

此用例测试外层循环上界使用min运算的情况。外层循环从0到`min(%M, %N, 1024)`。

**unroll-jam-factor=2模式下的行为:**

由于外层循环上界使用min运算，无法确定精确的迭代次数，因此无法进行unroll-jam优化：

```mlir
affine.for %arg0 = 0 to min #map()[%M, %N] {
  affine.for %arg1 = 0 to %K {
    "test.foo"(%arg0, %arg1) : (index, index) -> ()
  }
}
```

---

### 用例 6: no_unroll_jam_dependent_ubound

**原始代码:**

```mlir
func.func @no_unroll_jam_dependent_ubound(%in0: memref<?xf32, 1>) {
  affine.for %i = 0 to 100 {
    affine.for %k = 0 to affine_map<(d0) -> (d0 + 1)>(%i) {
      %y = "addi32"(%k, %k) : (index, index) -> i32
    }
  }
  return
}
```

**说明:**

此用例测试内层循环的上界依赖于外层循环变量的情况。内层循环从0到`%i + 1`，每次外层迭代，内层循环的迭代次数都不同。

**unroll-jam-factor=2模式下的行为:**

由于内层循环的上界依赖于外层循环变量，无法进行unroll-jam优化：

```mlir
affine.for %arg0 = 0 to 100 {
  affine.for %arg1 = 0 to #map_plus_1(%arg0) {
    "addi32"(%arg1, %arg1) : (index, index) -> i32
  }
}
```

---

### 用例 7: unroll_jam_one_iter_arg

**原始代码:**

```mlir
func.func @unroll_jam_one_iter_arg() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %cst) -> (i32) {
      %y = "bar"(%i, %j, %acc) : (index, index, i32) -> i32
      affine.yield %y : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
```

**说明:**

此用例测试内层循环有1个iter_arg的情况。内层循环使用迭代参数`%acc`进行累加操作。

**unroll-jam-factor=2模式下的行为:**

当内层循环有iter_arg时，unroll-jam会复制iter_arg，使其数量与展开因子匹配：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %const1 = arith.constant 1 : i32
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %const2 = arith.constant 1 : i32
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  %res3:2 = affine.for %arg1 = 0 to 17 iter_args(%acc1 = %const1, %acc2 = %const2) -> (i32, i32) {
    %res4 = "bar"(%arg0, %arg1, %acc1) : (index, index, i32) -> i32
    %inc1 = affine.apply #map_plus_1(%arg0)
    %res5 = "bar"(%inc1, %arg1, %acc2) : (index, index, i32) -> i32
    affine.yield %res4, %res5 : i32, i32
  }
  "foo"(%arg0, %res1, %res3#0) : (index, i32, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2, %res3#1) : (index, i32, i32) -> i32
}

// 清理循环（单次迭代）
%const3 = arith.constant 1 : i32
%res6 = "addi32"(%c100, %c100) : (index, index) -> i32
%res7 = affine.for %arg0 = 0 to 17 iter_args(%acc = %const3) -> (i32) {
  %res8 = "bar"(%c100, %arg0, %acc) : (index, index, i32) -> i32
  affine.yield %res8 : i32
}
"foo"(%c100, %res6, %res7) : (index, i32, i32) -> i32
return
```

关键点：
1. 内层循环的iter_arg被复制为2个（%acc1和%acc2）
2. 内层循环返回2个结果（%res3:2）
3. 外层循环后的操作使用正确的迭代参数结果

---

### 用例 8: unroll_jam_iter_args

**原始代码:**

```mlir
func.func @unroll_jam_iter_args() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 0 : i32
    %cst1 = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red:2 = affine.for %j = 0 to 17 iter_args(%acc = %cst, %acc1 = %cst1) -> (i32, i32) {
      %y = "bar"(%i, %j, %acc) : (index, index, i32) -> i32
      %z = "bar1"(%i, %j, %acc1) : (index, index, i32) -> i32
      affine.yield %y, %z : i32, i32
    }
    %w = "foo"(%i, %x, %red#0, %red#1) : (index, i32, i32, i32) -> i32
  }
  return
}
```

**说明:**

此用例测试内层循环有多个iter_arg的情况。内层循环有2个迭代参数`%acc`和`%acc1`。

**unroll-jam-factor=2模式下的行为:**

当内层循环有多个iter_arg时，每个iter_arg都会被复制：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %const0 = arith.constant 0 : i32
  %const1 = arith.constant 1 : i32
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %const2 = arith.constant 0 : i32
  %const3 = arith.constant 1 : i32
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  %res3:4 = affine.for %arg1 = 0 to 17 iter_args(%acc0 = %const0, %acc1 = %const1, %acc2 = %const2, %acc3 = %const3) -> (i32, i32, i32, i32) {
    %res4 = "bar"(%arg0, %arg1, %acc0) : (index, index, i32) -> i32
    %res5 = "bar1"(%arg0, %arg1, %acc1) : (index, index, i32) -> i32
    %inc1 = affine.apply #map_plus_1(%arg0)
    %res6 = "bar"(%inc1, %arg1, %acc2) : (index, index, i32) -> i32
    %res7 = "bar1"(%inc1, %arg1, %acc3) : (index, index, i32) -> i32
    affine.yield %res4, %res5, %res6, %res7 : i32, i32, i32, i32
  }
  "foo"(%arg0, %res1, %res3#0, %res3#1) : (index, i32, i32, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2, %res3#2, %res3#3) : (index, i32, i32, i32) -> i32
}

// 清理循环（单次迭代）
%const4 = arith.constant 0 : i32
%const5 = arith.constant 1 : i32
%res8 = "addi32"(%c100, %c100) : (index, index) -> i32
%res9:2 = affine.for %arg0 = 0 to 17 iter_args(%acc = %const4, %acc1 = %const5) -> (i32, i32) {
  %res10 = "bar"(%c100, %arg0, %acc) : (index, index, i32) -> i32
  %res11 = "bar1"(%c100, %arg0, %acc1) : (index, index, i32) -> i32
  affine.yield %res10, %res11 : i32, i32
}
"foo"(%c100, %res8, %res9#0, %res9#1) : (index, i32, i32, i32) -> i32
return
```

关键点：
1. 原始的2个iter_arg被复制为4个（%acc0, %acc1, %acc2, %acc3）
2. 内层循环返回4个结果（%res3:4）
3. 外层循环后的操作使用正确的迭代参数结果

---

### 用例 9: unroll_jam_iter_args_func_arg

**原始代码:**

```mlir
func.func @unroll_jam_iter_args_func_arg(%in: i32) {
  affine.for %i = 0 to 101 {
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %in) -> (i32) {
      %y = "bar"(%i, %j, %acc) : (index, index, i32) -> i32
      affine.yield %y : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
```

**说明:**

此用例测试iter_arg的初始值是函数参数的情况。`%acc`的初始值是`%in`，它是函数的参数。

**unroll-jam-factor=2模式下的行为:**

当iter_arg的初始值是函数参数时，不会被替换：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  %res3:2 = affine.for %arg1 = 0 to 17 iter_args(%acc1 = %in, %acc2 = %in) -> (i32, i32) {
    %res4 = "bar"(%arg0, %arg1, %acc1) : (index, index, i32) -> i32
    %inc1 = affine.apply #map_plus_1(%arg0)
    %res5 = "bar"(%inc1, %arg1, %acc2) : (index, index, i32) -> i32
    affine.yield %res4, %res5 : i32, i32
  }
  "foo"(%arg0, %res1, %res3#0) : (index, i32, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2, %res3#1) : (index, i32, i32) -> i32
}

// 清理循环（单次迭代）
%res6 = "addi32"(%c100, %c100) : (index, index) -> i32
%res7 = affine.for %arg0 = 0 to 17 iter_args(%acc = %in) -> (i32) {
  %res8 = "bar"(%c100, %arg0, %acc) : (index, index, i32) -> i32
  affine.yield %res8 : i32
}
"foo"(%c100, %res6, %res7) : (index, i32, i32) -> i32
return
```

关键点：
1. iter_arg的初始值`%in`保持不变
2. 两个展开的迭代都使用相同的初始值`%in`

---

### 用例 10: unroll_jam_iter_args_nested

**原始代码:**

```mlir
func.func @unroll_jam_iter_args_nested() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %cst) -> (i32) {
      %red1 = affine.for %k = 0 to 35 iter_args(%acc1 = %acc) -> (i32) {
        %y = "bar"(%i, %j, %k, %acc1) : (index, index, index, i32) -> i32
        affine.yield %y : i32
      }
      affine.yield %red1 : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
```

**说明:**

此用例测试嵌套的内层循环，每个内层循环都有iter_arg。最内层循环的iter_arg使用外层循环的iter_arg作为初始值。

**unroll-jam-factor=2模式下的行为:**

嵌套的内层循环都会被正确处理：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %const1 = arith.constant 1 : i32
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %const2 = arith.constant 1 : i32
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  %res3:2 = affine.for %arg1 = 0 to 17 iter_args(%acc1 = %const1, %acc2 = %const2) -> (i32, i32) {
    %res4:2 = affine.for %arg2 = 0 to 35 iter_args(%acc3 = %acc1, %acc4 = %acc2) -> (i32, i32) {
      %res5 = "bar"(%arg0, %arg1, %arg2, %acc3) : (index, index, index, i32) -> i32
      %inc1 = affine.apply #map_plus_1(%arg0)
      %res6 = "bar"(%inc1, %arg1, %arg2, %acc4) : (index, index, index, i32) -> i32
      affine.yield %res5, %res6 : i32, i32
    }
    affine.yield %res4#0, %res4#1 : i32, i32
  }
  "foo"(%arg0, %res1, %res3#0) : (index, i32, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2, %res3#1) : (index, i32, i32) -> i32
}

// 清理循环（单次迭代）
%const3 = arith.constant 1 : i32
%res6 = "addi32"(%c100, %c100) : (index, index) -> i32
%res7 = affine.for %arg0 = 0 to 17 iter_args(%acc = %const3) -> (i32) {
  %res8 = affine.for %arg1 = 0 to 35 iter_args(%acc1 = %acc) -> (i32) {
    %res9 = "bar"(%c100, %arg0, %arg1, %acc1) : (index, index, index, i32) -> i32
    affine.yield %res9 : i32
  }
  affine.yield %res8 : i32
}
"foo"(%c100, %res6, %res7) : (index, i32, i32) -> i32
return
```

关键点：
1. 两层嵌套的内层循环都被正确处理
2. 最内层循环的iter_arg被复制为2个

---

### 用例 11: unroll_jam_iter_args_nested_affine_for_result

**原始代码:**

```mlir
func.func @unroll_jam_iter_args_nested_affine_for_result() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red = affine.for %j = 0 to 17 iter_args(%acc = %cst) -> (i32) {
      %red1 = affine.for %k = 0 to 35 iter_args(%acc1 = %acc) -> (i32) {
        %y = "bar"(%i, %j, %k, %acc1) : (index, index, index, i32) -> i32
        affine.yield %acc : i32
      }
      %red2 = affine.for %l = 0 to 36 iter_args(%acc2 = %red1) -> (i32) {
        %y = "bar"(%i, %j, %l, %acc2) : (index, index, index, i32) -> i32
        affine.yield %y : i32
      }
      affine.yield %red2 : i32
    }
    %w = "foo"(%i, %x, %red) : (index, i32, i32) -> i32
  }
  return
}
```

**说明:**

此用例测试嵌套的内层循环，其中一个循环使用其兄弟循环的结果作为iter_arg的初始值。`%red2`循环使用`%red1`的结果作为初始值。

**unroll-jam-factor=2模式下的行为:**

复杂的依赖关系会被正确处理：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %const1 = arith.constant 1 : i32
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %const2 = arith.constant 1 : i32
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  %res3:2 = affine.for %arg1 = 0 to 17 iter_args(%acc1 = %const1, %acc2 = %const2) -> (i32, i32) {
    %res4:2 = affine.for %arg2 = 0 to 35 iter_args(%acc3 = %acc1, %acc4 = %acc2) -> (i32, i32) {
      %res5 = "bar"(%arg0, %arg1, %arg2, %acc3) : (index, index, index, i32) -> i32
      %inc1 = affine.apply #map_plus_1(%arg0)
      %res6 = "bar"(%inc1, %arg1, %arg2, %acc4) : (index, index, index, i32) -> i32
      affine.yield %acc1, %acc2 : i32, i32
    }
    %res14:2 = affine.for %arg3 = 0 to 36 iter_args(%acc13 = %res4#0, %acc14 = %res4#1) -> (i32, i32) {
      %res15 = "bar"(%arg0, %arg1, %arg3, %acc13) : (index, index, index, i32) -> i32
      %inc1 = affine.apply #map_plus_1(%arg0)
      %res16 = "bar"(%inc1, %arg1, %arg3, %acc14) : (index, index, index, i32) -> i32
      affine.yield %res15, %res16 : i32, i32
    }
    affine.yield %res14#0, %res14#1 : i32, i32
  }
  "foo"(%arg0, %res1, %res3#0) : (index, i32, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2, %res3#1) : (index, i32, i32) -> i32
}

// 清理循环（单次迭代）
%const3 = arith.constant 1 : i32
%res6 = "addi32"(%c100, %c100) : (index, index) -> i32
%res7 = affine.for %arg0 = 0 to 17 iter_args(%acc = %const3) -> (i32) {
  %res8 = affine.for %arg1 = 0 to 35 iter_args(%acc1 = %acc) -> (i32) {
    %res9 = "bar"(%c100, %arg0, %arg1, %acc1) : (index, index, index, i32) -> i32
    affine.yield %acc : i32
  }
  %res17 = affine.for %arg2 = 0 to 36 iter_args(%acc2 = %res8) -> (i32) {
    %res18 = "bar"(%c100, %arg0, %arg2, %acc2) : (index, index, index, i32) -> i32
    affine.yield %res18 : i32
  }
  affine.yield %res17 : i32
}
"foo"(%c100, %res6, %res7) : (index, i32, i32) -> i32
return
```

关键点：
1. 兄弟循环之间的依赖关系被正确处理
2. `%red2`循环使用`%red1`的结果作为初始值

---

### 用例 12: unroll_jam_iter_args_nested_yield

**原始代码:**

```mlir
func.func @unroll_jam_iter_args_nested_yield() {
  affine.for %i = 0 to 101 {
    %cst = arith.constant 1 : i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    %red:3 = affine.for %j = 0 to 17 iter_args(%acc = %cst, %acc1 = %cst, %acc2 = %cst) -> (i32, i32, i32) {
      %red1 = affine.for %k = 0 to 35 iter_args(%acc3 = %acc) -> (i32) {
        %y = "bar"(%i, %j, %k, %acc3) : (index, index, index, i32) -> i32
        affine.yield %y : i32
      }
      %red2:2 = affine.for %l = 0 to 36 iter_args(%acc4 = %acc1, %acc5 = %acc2) -> (i32, i32) {
        %y = "bar1"(%i, %j, %l, %acc4, %acc5) : (index, index, index, i32, i32) -> i32
        affine.yield %y, %y : i32, i32
      }
      affine.yield %red1, %red1, %red2#1 : i32, i32, i32
    }
    %w = "foo"(%i, %x, %red#0, %red#2) : (index, i32, i32, i32) -> i32
  }
  return
}
```

**说明:**

此用例测试嵌套的内层循环，每个循环有多个iter_arg，且yield相同的值多次。最外层的内层循环有3个iter_arg，且yield相同的值`%red1`两次。

**unroll-jam-factor=2模式下的行为:**

复杂的yield模式会被正确处理：

```mlir
// 主循环
affine.for %arg0 = 0 to 100 step 2 {
  %const1 = arith.constant 1 : i32
  %res1 = "addi32"(%arg0, %arg0) : (index, index) -> i32
  %inc = affine.apply #map_plus_1(%arg0)
  %const2 = arith.constant 1 : i32
  %res2 = "addi32"(%inc, %inc) : (index, index) -> i32
  %res3:6 = affine.for %arg1 = 0 to 17 iter_args(%acc1 = %const1, %acc2 = %const1, %acc3 = %const1, %acc4 = %const2, %acc5 = %const2, %acc6 = %const2) -> (i32, i32, i32, i32, i32, i32) {
    %res4:2 = affine.for %arg2 = 0 to 35 iter_args(%acc7 = %acc1, %acc8 = %acc4) -> (i32, i32) {
      %res5 = "bar"(%arg0, %arg1, %arg2, %acc7) : (index, index, index, i32) -> i32
      %inc1 = affine.apply #map_plus_1(%arg0)
      %res6 = "bar"(%inc1, %arg1, %arg2, %acc8) : (index, index, index, i32) -> i32
      affine.yield %res5, %res6 : i32, i32
    }
    %res14:4 = affine.for %arg3 = 0 to 36 iter_args(%acc13 = %acc2, %acc14 = %acc3, %acc15 = %acc5, %acc16 = %acc6) -> (i32, i32, i32, i32) {
      %res15 = "bar1"(%arg0, %arg1, %arg3, %acc13, %acc14) : (index, index, index, i32, i32) -> i32
      %inc1 = affine.apply #map_plus_1(%arg0)
      %res16 = "bar1"(%inc1, %arg1, %arg3, %acc15, %acc16) : (index, index, index, i32, i32) -> i32
      affine.yield %res15, %res15, %res16, %res16 : i32, i32, i32, i32
    }
    affine.yield %res4#0, %res4#0, %res14#1, %res4#1, %res4#1, %res14#3 : i32, i32, i32, i32, i32, i32
  }
  "foo"(%arg0, %res1, %res3#0, %res3#2) : (index, i32, i32, i32) -> i32
  affine.apply #map_plus_1(%arg0)
  "foo"(%arg1, %res2, %res3#3, %res3#5) : (index, i32, i32, i32) -> i32
}

// 清理循环（单次迭代）
%const3 = arith.constant 1 : i32
%res6 = "addi32"(%c100, %c100) : (index, index) -> i32
%res7:3 = affine.for %arg0 = 0 to 17 iter_args(%acc = %const3, %acc1 = %const3, %acc2 = %const3) -> (i32, i32, i32) {
  %res8 = affine.for %arg1 = 0 to 35 iter_args(%acc3 = %acc) -> (i32) {
    %res9 = "bar"(%c100, %arg0, %arg1, %acc3) : (index, index, index, i32) -> i32
    affine.yield %res9 : i32
  }
  %res17:2 = affine.for %arg2 = 0 to 36 iter_args(%acc4 = %acc1, %acc5 = %acc2) -> (i32, i32) {
    %res18 = "bar1"(%c100, %arg0, %arg2, %acc4, %acc5) : (index, index, index, i32, i32) -> i32
    affine.yield %res18, %res18 : i32, i32
  }
  affine.yield %res8, %res8, %res17#1 : i32, i32, i32
}
"foo"(%c100, %res6, %res7#0, %res7#2) : (index, i32, i32, i32) -> i32
return
```

关键点：
1. 原始的3个iter_arg被复制为6个
2. yield相同的值多次会被正确处理

---

### 用例 13: unroll_jam_nested_iter_args_mulf

**原始代码:**

```mlir
func.func @unroll_jam_nested_iter_args_mulf(%arg0: memref<21x30xf32, 1>, %init : f32, %init1 : f32) {
  %0 = affine.for %arg3 = 0 to 21 iter_args(%arg4 = %init) -> (f32) {
    %1 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %init1) -> (f32) {
      %3 = affine.load %arg0[%arg3, %arg5] : memref<21x30xf32, 1>
      %4 = arith.addf %arg6, %3 : f32
      affine.yield %4 : f32
    }
    %2 = arith.mulf %arg4, %1 : f32
    affine.yield %2 : f32
  }
  return
}
```

**说明:**

此用例测试嵌套循环的unroll-jam，每个循环都有iter_arg，且使用乘法操作。外层循环迭代21次，内层循环迭代30次。

**unroll-jam-factor=2模式下的行为:**

外层循环按因子2展开，内层循环的两次迭代会被合并：

```mlir
%const0 = arith.constant 20 : index
%res:2 = affine.for %arg0 = 0 to 20 step 2 iter_args(%acc0 = %init, %acc1 = %init) -> (f32, f32) {
  %res1:2 = affine.for %arg1 = 0 to 30 iter_args(%acc2 = %init1, %acc3 = %init1) -> (f32, f32) {
    %load1 = affine.load %arg0[%arg0, %arg1] : memref<21x30xf32, 1>
    %add1 = arith.addf %acc2, %load1 : f32
    %inc1 = affine.apply #map_plus_1(%arg0)
    %load2 = affine.load %arg0[%inc1, %arg1] : memref<21x30xf32, 1>
    %add2 = arith.addf %acc3, %load2 : f32
    affine.yield %add1, %add2 : f32, f32
  }
  %mul1 = arith.mulf %acc0, %res1#0 : f32
  affine.apply #map_plus_1(%arg0)
  %mul2 = arith.mulf %acc1, %res1#1 : f32
  affine.yield %mul1, %mul2 : f32, f32
}

// Reduction op
%mul3 = arith.mulf %res#0, %res#1 : f32

// 清理循环（单次迭代）
%res2 = affine.for %arg0 = 0 to 30 iter_args(%acc4 = %init1) -> (f32) {
  %load3 = affine.load %arg0[%const0, %arg0] : memref<21x30xf32, 1>
  %add3 = arith.addf %acc4, %load3 : f32
  affine.yield %add3 : f32
}
%mul4 = arith.mulf %mul3, %res2 : f32
return
```

关键点：
1. 外层循环步长变为2（从0到20）
2. 内层循环的iter_arg被复制为2个
3. 外层循环后有一个reduction操作（`%mul3 = arith.mulf %res#0, %res#1`）
4. 清理循环只有1次迭代，处理第20次迭代

---

### 用例 14: unroll_jam_iter_args_addi

**原始代码:**

```mlir
func.func @unroll_jam_iter_args_addi(%arg0: memref<21xi32, 1>, %init : i32) {
  %0 = affine.for %arg3 = 0 to 21 iter_args(%arg4 = %init) -> (i32) {
    %1 = affine.load %arg0[%arg3] : memref<21xi32, 1>
    %2 = arith.addi %arg4, %1 : i32
    affine.yield %2 : i32
  }
  return
}
```

**说明:**

此用例测试单层循环的unroll-jam，循环有iter_arg。外层循环迭代21次。

**unroll-jam-factor=2模式下的行为:**

单层循环按因子2展开：

```mlir
%const0 = arith.constant 20 : index
%res:2 = affine.for %arg0 = 0 to 20 step 2 iter_args(%acc0 = %init, %acc1 = %init) -> (i32, i32) {
  %load1 = affine.load %arg0[%arg0] : memref<21xi32, 1>
  %add1 = arith.addi %acc0, %load1 : i32
  %inc1 = affine.apply #map_plus_1(%arg0)
  %load2 = affine.load %arg0[%inc1] : memref<21xi32, 1>
  %add2 = arith.addi %acc1, %load2 : i32
  affine.yield %add1, %add2 : i32, i32
}

// Reduction op
%add3 = arith.addi %res#0, %res#1 : i32

// 清理循环（单次迭代）
%load3 = affine.load %arg0[%const0] : memref<21xi32, 1>
%add4 = arith.addi %add3, %load3 : i32
return
```

关键点：
1. 单层循环也可以进行unroll-jam
2. iter_arg被复制为2个
3. 有一个reduction操作（`%add3 = arith.addi %res#0, %res#1`）
4. 清理循环只有1次迭代，被提升

---

## 总结

`affine-loop-unroll-jam` pass是一个强大的循环优化pass，它可以：

1. **展开和合并嵌套循环**: 将外层循环展开，内层循环的多次迭代合并在一起执行
2. **生成清理循环**: 当迭代次数不是展开因子的倍数时，会生成清理循环处理剩余迭代
3. **处理非完美嵌套**: 可以正确处理内层循环前后有操作的情况
4. **支持符号边界**: 可以处理符号变量的循环边界
5. **支持iter_args**: 可以正确处理带迭代参数的循环，包括多个iter_arg和嵌套的iter_arg
6. **支持GPU模块**: 可以在GPU模块中使用
7. **生成reduction操作**: 当展开后的iter_arg需要合并时，会生成相应的reduction操作

该pass在优化循环性能时非常有用，特别是对于嵌套循环，可以提高指令级并行性和缓存局部性。但需要注意：
- 内层循环的上界不能依赖于外层循环变量
- 循环边界使用min/max运算时可能无法展开
- 复杂的iter_arg依赖关系需要正确处理
