# affine-super-vectorize Pass 解析

## 概述

`affine-super-vectorize` pass用于自动向量化，将标量操作转换为向量操作。它分析affine循环，识别可以向量化操作，生成高效的向量代码。该pass支持多种向量化模式，包括1D、2D、3D向量化，reduction向量化，transpose向量化等。

该pass的主要功能：
- **自动向量化**: 自动将标量循环转换为向量循环
- **多维度支持**: 支持1D、2D、3D向量化
- **Reduction向量化**: 支持reduction操作的向量化
- **Transpose向量化**: 支持transpose操作的向量化
- **Uniform/Divergent分析**: 分析操作的uniform和divergent特性
- **内存访问优化**: 生成高效的向量内存访问

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/SuperVectorize/`
- 测试文件数量: 15个

## RUN命令

该pass支持的主要选项：

1. `virtual-vector-size`: 虚拟向量大小（如128）
2. `test-fastest-varying`: 测试最快变化的维度
3. `vectorize-reductions`: 是否向量化reduction操作

## 测试用例解析

### 用例 1: vec1d_1 (1D向量化)

**原始代码:**

```mlir
func.func @vec1d_1(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %P = memref.dim %B, %c2 : memref<?x?x?xf32>

  affine.for %i0 = 0 to %M {
    %a0 = affine.load %A[%c0, %c0] : memref<?x?xf32>
  }
  return
}
```

**说明:**

此用例测试最简单的1D向量化。循环内load一个标量值，虽然看起来没有意义，但可以演示向量化过程。

**向量化后的行为:**

```mlir
func.func @vec1d_1(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %P = memref.dim %B, %c2 : memref<?x?x?xf32>

  affine.for %i0 = 0 to %M step 128 {
    %idx0 = affine.apply affine_map<(d0) -> (d0)>(%c0)
    %idx1 = affine.apply affine_map<(d0) -> (d0)>(%c0)
    %poison = ub.poison : f32
    %a0 = vector.transfer_read %A[%idx0, %idx1], %poison {permutation_map = affine_map<(d0, d1) -> (0)>} : memref<?x?xf32>, vector<128xf32>
  }
  return
}
```

**关键点:**

1. **循环步长**: 循环步长变为128（向量大小）
2. **向量load**: `affine.load`被转换为`vector.transfer_read`
3. **Permutation map**: `permutation_map = affine_map<(d0, d1) -> (0)>`表示广播操作
4. **Poison值**: 使用`ub.poison`作为填充值

---

### 用例 2: vec1d_2 (1D向量化)

**原始代码:**

```mlir
func.func @vec1d_2(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %P = memref.dim %B, %c2 : memref<?x?x?xf32>

  affine.for %i3 = 0 to %M {
    %a3 = affine.load %A[%c0, %i3] : memref<?x?xf32>
  }
  return
}
```

**说明:**

此用例测试正常的1D向量化。循环变量`%i3`被用于load操作的索引。

**向量化后的行为:**

```mlir
func.func @vec1d_2(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %P = memref.dim %B, %c2 : memref<?x?x?xf32>

  affine.for %i3 = 0 to %M step 128 {
    %poison = ub.poison : f32
    %a3 = vector.transfer_read %A[%c0, %i3], %poison : memref<?x?xf32>, vector<128xf32>
  }
  return
}
```

**关键点:**

1. **连续访问**: load操作的索引是连续的，可以高效向量化
2. **向量传输**: 使用`vector.transfer_read`进行向量读取

---

### 用例 3: vecdim_reduction (Reduction向量化)

**原始代码:**

```mlir
func.func @vecdim_reduction(%in: memref<256x512xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 256 {
    %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
      %ld = affine.load %in[%i, %j] : memref<256x512xf32>
      %add = arith.addf %red_iter, %ld : f32
      affine.yield %add : f32
    }
    affine.store %final_red, %out[%i] : memref<256xf32>
  }
  return
}
```

**说明:**

此用例测试reduction操作的向量化。内层循环是一个求和reduction，可以被向量化。

**向量化后的行为:**

```mlir
func.func @vecdim_reduction(%in: memref<256x512xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 256 {
    %vzero = arith.constant dense<0.000000e+00> : vector<128xf32>
    %vred = affine.for %j = 0 to 512 step 128 iter_args(%red_iter = %vzero) -> (vector<128xf32>) {
      %ld = vector.transfer_read %in[%i, %j] : memref<256x512xf32>, vector<128xf32>
      %add = arith.addf %red_iter, %ld : vector<128xf32>
      affine.yield %add : vector<128xf32>
    }
    %final_sum = vector.reduction <add>, %vred : vector<128xf32> into f32
    affine.store %final_sum, %out[%i] : memref<256xf32>
  }
  return
}
```

**关键点:**

1. **向量reduction**: 标量reduction被转换为向量reduction
2. **向量初始值**: 初始值被转换为向量常量`dense<0.0>`
3. **向量操作**: 加法操作被转换为向量加法
4. **最终reduction**: 使用`vector.reduction`将向量归约为标量
5. **性能提升**: 向量化后，一次处理128个元素，显著提高性能

---

### 用例 4: vecdim_reduction_minf (最小值Reduction)

**原始代码:**

```mlir
func.func @vecdim_reduction_minf(%in: memref<256x512xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 0x7F800000 : f32  // +infinity
  affine.for %i = 0 to 256 {
    %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
      %ld = affine.load %in[%i, %j] : memref<256x512xf32>
      %min = arith.minimumf %red_iter, %ld : f32
      affine.yield %min : f32
    }
    affine.store %final_red, %out[%i] : memref<256xf32>
  }
  return
}
```

**说明:**

此用例测试最小值reduction的向量化。初始值为正无穷。

**向量化后的行为:**

```mlir
func.func @vecdim_reduction_minf(%in: memref<256x512xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 0x7F800000 : f32
  affine.for %i = 0 to 256 {
    %vmax = arith.constant dense<0x7F800000> : vector<128xf32>
    %vred = affine.for %j = 0 to 512 step 128 iter_args(%red_iter = %vmax) -> (vector<128xf32>) {
      %ld = vector.transfer_read %in[%i, %j] : memref<256x512xf32>, vector<128xf32>
      %min = arith.minimumf %red_iter, %ld : vector<128xf32>
      affine.yield %min : vector<128xf32>
    }
    %final_min = vector.reduction <minimumf>, %vred : vector<128xf32> into f32
    affine.store %final_min, %out[%i] : memref<256xf32>
  }
  return
}
```

**关键点:**

1. **最小值reduction**: 支持`minimumf`操作的向量化
2. **向量reduction**: 使用`vector.reduction <minimumf>`

---

### 用例 5: vecdim_reduction_maxf (最大值Reduction)

**原始代码:**

```mlir
func.func @vecdim_reduction_maxf(%in: memref<256x512xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 0xFF800000 : f32  // -infinity
  affine.for %i = 0 to 256 {
    %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
      %ld = affine.load %in[%i, %j] : memref<256x512xf32>
      %max = arith.maximumf %red_iter, %ld : f32
      affine.yield %max : f32
    }
    affine.store %final_red, %out[%i] : memref<256xf32>
  }
  return
}
```

**说明:**

此用例测试最大值reduction的向量化。初始值为负无穷。

**向量化后的行为:**

```mlir
func.func @vecdim_reduction_maxf(%in: memref<256xf32>, %out: memref<256xf32>) {
  %cst = arith.constant 0xFF800000 : f32
  affine.for %i = 0 to 256 {
    %vmin = arith.constant dense<0xFF800000> : vector<128xf32>
    %vred = affine.for %j = 0 to 512 step 128 iter_args(%red_iter = %vmin) -> (vector<128xf32>) {
      %ld = vector.transfer_read %in[%i, %j] : memref<256x512xf32>, vector<128xf32>
      %max = arith.maximumf %red_iter, %ld : vector<128xf32>
      affine.yield %max : vector<128xf32>
    }
    %final_max = vector.reduction <maximumf>, %vred : vector<128xf32> into f32
    affine.store %final_max, %out[%i] : memref<256xf32>
  }
  return
}
```

**关键点:**

1. **最大值reduction**: 支持`maximumf`操作的向量化
2. **向量reduction**: 使用`vector.reduction <maximumf>`

---

## 其他测试文件概览

### vectorize_2d.mlir
- 测试2D向量化
- 支持嵌套循环的向量化
- 处理2D内存访问模式

### vectorize_3d.mlir
- 测试3D向量化
- 支持三层嵌套循环的向量化
- 处理3D内存访问模式

### vectorize_transpose_2d.mlir
- 测试transpose操作的向量化
- 处理非连续内存访问
- 优化transpose性能

### uniform_divergent.mlir
- 测试uniform和divergent操作的分析
- Uniform操作：所有向量元素执行相同操作
- Divergent操作：不同向量元素执行不同操作

### vectorize_unsupported.mlir
- 测试不支持向量化的操作
- 确保正确性，不进行错误的向量化

### compose_maps.mlir
- 测试affine_map的组合
- 优化向量索引计算

### vectorize_affine_apply.mlir
- 测试affine.apply操作的向量化
- 处理复杂的索引计算

### vectorize_outer_loop_2d.mlir
- 测试外层循环的向量化
- 支持不同的向量化策略

### vectorize_reduction_2d.mlir
- 测试2D reduction的向量化
- 支持多层reduction

### vector_utils.mlir
- 测试向量工具函数
- 辅助向量化过程

---

## 总结

`affine-super-vectorize` pass是一个强大的自动向量化pass，它可以：

1. **自动向量化**: 自动将标量循环转换为向量循环
2. **多维度支持**: 支持1D、2D、3D向量化
3. **Reduction向量化**: 支持reduction操作的向量化（add、min、max等）
4. **Transpose向量化**: 支持transpose操作的向量化
5. **Uniform/Divergent分析**: 分析操作的uniform和divergent特性
6. **内存访问优化**: 生成高效的向量内存访问
7. **支持复杂索引**: 处理affine.apply等复杂索引计算

该pass在优化循环性能时非常有用，特别是对于计算密集型循环。但需要注意：
- 需要指定向量大小
- 需要分析内存访问模式
- 对于不支持的操作，不会进行向量化
- 需要考虑硬件的向量支持

## 应用场景

1. **科学计算**: 矩阵运算、向量运算
2. **图像处理**: 像素级操作
3. **机器学习**: 张量运算
4. **信号处理**: DSP算法
5. **数值模拟**: 数值计算

## 性能优化建议

1. **选择合适的向量大小**: 根据硬件选择合适的向量大小（如128、256、512）
2. **内存对齐**: 确保内存访问对齐，提高向量加载效率
3. **避免依赖**: 减少循环内的依赖，提高向量化效率
4. **Reduction优化**: 使用向量reduction代替标量reduction
5. **Transpose优化**: 对于transpose操作，使用专门的向量化策略
