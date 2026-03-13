# affine-pipeline-data-transfer Pass 解析

## 概述

`affine-pipeline-data-transfer` pass用于优化DMA（Direct Memory Access）数据传输。它通过双缓冲（double buffering）技术，将DMA传输与计算重叠，从而隐藏DMA延迟，提高整体性能。

该pass的主要功能：
- 自动识别可以进行流水线化的DMA操作
- 将单缓冲区转换为双缓冲区
- 生成prologue（前奏）、steady state（稳态）和epilogue（尾声）代码
- 处理嵌套的DMA操作
- 检测依赖关系，避免错误的优化

## 测试文件来源

- 文件路径: `mlir/test/Dialect/Affine/pipeline-data-transfer.mlir`

## RUN命令

该测试文件包含以下RUN命令：

1. `mlir-opt -allow-unregistered-dialect %s -split-input-file -affine-pipeline-data-transfer | FileCheck %s`

## 测试用例解析

### 用例 1: loop_nest_dma

**原始代码:**

```mlir
func.func @loop_nest_dma() {
  %A = memref.alloc() : memref<256 x f32, affine_map<(d0) -> (d0)>, 0>
  %Ah = memref.alloc() : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>

  %tag = memref.alloc() : memref<1 x f32>

  %zero = arith.constant 0 : index
  %num_elts = arith.constant 32 : index

  affine.for %i = 0 to 8 {
    affine.dma_start %A[%i], %Ah[%i], %tag[%zero], %num_elts : memref<256 x f32>, memref<32 x f32, 1>, memref<1 x f32>
    affine.dma_wait %tag[%zero], %num_elts : memref<1 x f32>
    %v = affine.load %Ah[%i] : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>
    %r = "compute"(%v) : (f32) -> (f32)
    affine.store %r, %Ah[%i] : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>
    affine.for %j = 0 to 32 {
      "do_more_compute"(%i, %j) : (index, index) -> ()
    }
  }
  memref.dealloc %tag : memref<1 x f32>
  memref.dealloc %Ah : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>
  return
}
```

**说明:**

这是一个典型的DMA流水线化例子。循环中包含：
1. DMA启动（`affine.dma_start`）：从慢速内存`%A`传输数据到快速内存`%Ah`
2. DMA等待（`affine.dma_wait`）：等待DMA传输完成
3. 计算：使用传输的数据进行计算
4. 嵌套循环：额外的计算

**流水线化后的行为:**

```mlir
%A = memref.alloc() : memref<256xf32>
%Ah = memref.alloc() : memref<2x32xf32, 1>  // 双缓冲区
%tag = memref.alloc() : memref<2x1xf32>     // 双缓冲tag

// Prologue: 启动第一次DMA
affine.dma_start %A[%i], %Ah[%i mod 2, %i], %tag[%i mod 2, 0], %num_elts : ...

// Steady state: 循环体
affine.for %i = 1 to 8 {
  // 启动下一次DMA
  affine.dma_start %A[%i], %Ah[%i mod 2, %i], %tag[%i mod 2, 0], %num_elts : ...
  
  // 等待上一次DMA完成
  affine.dma_wait %tag[(%i-1) mod 2, 0], %num_elts : ...
  
  // 使用上一次传输的数据进行计算
  %v = affine.load %Ah[(%i-1) mod 2, %i-1] : ...
  %r = "compute"(%v) : (f32) -> (f32)
  affine.store %r, %Ah[(%i-1) mod 2, %i-1] : ...
  
  affine.for %j = 0 to 32 {
    "do_more_compute"(%i-1, %j) : (index, index) -> ()
  }
}

// Epilogue: 处理最后一次DMA
affine.dma_wait %tag[7 mod 2, 0], %num_elts : ...
%v = affine.load %Ah[7 mod 2, 7] : ...
%r = "compute"(%v) : (f32) -> (f32)
affine.store %r, %Ah[7 mod 2, 7] : ...
affine.for %j = 0 to 32 {
  "do_more_compute"(7, %j) : (index, index) -> ()
}
```

**关键点:**

1. **双缓冲区**: 原始的单缓冲区`%Ah`被转换为双缓冲区`memref<2x32xf32, 1>`，大小翻倍
2. **双缓冲tag**: tag也被转换为双缓冲`memref<2x1xf32>`
3. **Prologue**: 循环前启动第一次DMA
4. **Steady state**: 循环中同时进行DMA和计算，使用`mod 2`操作切换缓冲区
5. **Epilogue**: 循环后处理最后一次DMA
6. **重叠执行**: DMA传输与计算重叠，隐藏DMA延迟

---

### 用例 2: loop_step

**原始代码:**

```mlir
func.func @loop_step(%arg0: memref<512xf32>, %arg1: memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  affine.for %i0 = 0 to 512 step 4 {
    %1 = memref.alloc() : memref<4xf32, 1>
    %2 = memref.alloc() : memref<1xi32>
    affine.dma_start %arg0[%i0], %1[%c0], %2[%c0], %c4,
              : memref<512xf32>, memref<4xf32, 1>, memref<1xi32>
    affine.dma_wait %2[%c0], %c4 : memref<1xi32>
    "compute"(%i0) : (index) -> ()
    memref.dealloc %2 : memref<1xi32>
    memref.dealloc %1 : memref<4xf32, 1>
  }
  return
}
```

**说明:**

此用例测试带步长的循环。循环步长为4，每次迭代传输4个元素。

**流水线化后的行为:**

```mlir
%buf = memref.alloc() : memref<2x4xf32, 1>  // 双缓冲区
%tag = memref.alloc() : memref<2x1xi32>     // 双缓冲tag

// Prologue: 启动第一次DMA
affine.dma_start %arg0[0], %buf[(0 floordiv 4) mod 2, 0], %tag[(0 floordiv 4) mod 2, 0], %c4 : ...

// Steady state: 循环体
affine.for %i0 = 4 to 512 step 4 {
  // 启动下一次DMA
  affine.dma_start %arg0[%i0], %buf[(%i0 floordiv 4) mod 2, 0], %tag[(%i0 floordiv 4) mod 2, 0], %c4 : ...
  
  // 等待上一次DMA完成
  affine.dma_wait %tag[((%i0-4) floordiv 4) mod 2, 0], %c4 : ...
  
  // 使用上一次传输的数据进行计算
  "compute"(%i0-4) : (index) -> ()
}

// Epilogue: 处理最后一次DMA
affine.dma_wait %tag[((512-4) floordiv 4) mod 2, 0], %c4 : ...
"compute"(512-4) : (index) -> ()

memref.dealloc %tag : memref<2x1xi32>
memref.dealloc %buf : memref<2x4xf32, 1>
```

**关键点:**

1. **步长处理**: 使用`floordiv`和`mod`计算缓冲区索引
2. **缓冲区索引**: `(i0 floordiv 4) mod 2`用于计算双缓冲区索引
3. **步长为4**: 每次迭代处理4个元素

---

### 用例 3: loop_dma_nested

**原始代码:**

```mlir
func.func @loop_dma_nested(%arg0: memref<512x32xvector<8xf32>>, %arg1: memref<512x32xvector<8xf32>>, %arg2: memref<512x32xvector<8xf32>>) {
  %num_elts = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<64x4xvector<8xf32>, 2>
  %1 = memref.alloc() : memref<64x4xvector<8xf32>, 2>
  %2 = memref.alloc() : memref<64x4xvector<8xf32>, 2>
  %3 = memref.alloc() : memref<2xi32>
  %4 = memref.alloc() : memref<2xi32>
  %5 = memref.alloc() : memref<2xi32>
  
  affine.for %i0 = 0 to 8 {
    %6 = affine.apply #map2(%i0)
    affine.dma_start %arg2[%6, %c0], %2[%c0, %c0], %5[%c0], %num_elts : ...
    affine.dma_wait %5[%c0], %num_elts : ...
    
    affine.for %i1 = 0 to 8 {
      %7 = affine.apply #map1(%i0, %i1)
      %8 = affine.apply #map2(%i1)
      affine.dma_start %arg0[%7, %c0], %0[%c0, %c0], %3[%c0], %num_elts : ...
      affine.dma_start %arg1[%8, %c0], %1[%c0, %c0], %4[%c0], %num_elts : ...
      affine.dma_wait %3[%c0], %num_elts : ...
      affine.dma_wait %4[%c0], %num_elts : ...
      
      affine.for %i2 = 0 to 4 {
        "foo"() : () -> ()
      }
    }
  }
  ...
}
```

**说明:**

此用例测试嵌套的DMA操作。外层循环有一个DMA（%arg2），内层循环有两个DMA（%arg0和%arg1）。

**流水线化后的行为:**

```mlir
// 外层DMA的双缓冲区
%buf_arg2 = memref.alloc() : memref<2x64x4xvector<8xf32>, 2>
%tag_arg2 = memref.alloc() : memref<2x2xi32>

// Prologue for 外层DMA
affine.dma_start %arg2[...], %buf_arg2[0, ...], %tag_arg2[0, 0], %num_elts : ...

affine.for %i0 = 1 to 8 {
  // 启动外层DMA的下一次传输
  affine.dma_start %arg2[...], %buf_arg2[%i0 mod 2, ...], %tag_arg2[%i0 mod 2, 0], %num_elts : ...
  
  // 等待外层DMA的上一次传输
  affine.dma_wait %tag_arg2[(%i0-1) mod 2, 0], %num_elts : ...
  
  // 内层DMA的双缓冲区
  %buf_arg0 = memref.alloc() : memref<2x64x4xvector<8xf32>, 2>
  %buf_arg1 = memref.alloc() : memref<2x64x4xvector<8xf32>, 2>
  %tag_arg0 = memref.alloc() : memref<2x2xi32>
  %tag_arg1 = memref.alloc() : memref<2x2xi32>
  
  // Prologue for 内层DMA
  affine.dma_start %arg0[...], %buf_arg0[0, ...], %tag_arg0[0, 0], %num_elts : ...
  affine.dma_start %arg1[...], %buf_arg1[0, ...], %tag_arg1[0, 0], %num_elts : ...
  
  affine.for %i1 = 1 to 8 {
    // 启动内层DMA的下一次传输
    affine.dma_start %arg0[...], %buf_arg0[%i1 mod 2, ...], %tag_arg0[%i1 mod 2, 0], %num_elts : ...
    affine.dma_start %arg1[...], %buf_arg1[%i1 mod 2, ...], %tag_arg1[%i1 mod 2, 0], %num_elts : ...
    
    // 等待内层DMA的上一次传输
    affine.dma_wait %tag_arg0[(%i1-1) mod 2, 0], %num_elts : ...
    affine.dma_wait %tag_arg1[(%i1-1) mod 2, 0], %num_elts : ...
    
    // 计算
    affine.for %i2 = 0 to 4 {
      "foo"() : () -> ()
    }
  }
  
  // Epilogue for 内层DMA
  affine.dma_wait %tag_arg0[7 mod 2, 0], %num_elts : ...
  affine.dma_wait %tag_arg1[7 mod 2, 0], %num_elts : ...
  
  memref.dealloc %tag_arg1 : memref<2x2xi32>
  memref.dealloc %tag_arg0 : memref<2x2xi32>
  memref.dealloc %buf_arg1 : memref<2x64x4xvector<8xf32>, 2>
  memref.dealloc %buf_arg0 : memref<2x64x4xvector<8xf32>, 2>
}

// Epilogue for 外层DMA
affine.dma_wait %tag_arg2[7 mod 2, 0], %num_elts : ...

// 在外层DMA的epilogue中，还有内层DMA的嵌套
%buf_arg0_nested = memref.alloc() : memref<2x64x4xvector<8xf32>, 2>
%buf_arg1_nested = memref.alloc() : memref<2x64x4xvector<8xf32>, 2>
%tag_arg0_nested = memref.alloc() : memref<2x2xi32>
%tag_arg1_nested = memref.alloc() : memref<2x2xi32>

// 内层DMA的prologue和steady state
...

memref.dealloc %tag_arg1_nested : memref<2x2xi32>
memref.dealloc %tag_arg0_nested : memref<2x2xi32>
memref.dealloc %buf_arg1_nested : memref<2x64x4xvector<8xf32>, 2>
memref.dealloc %buf_arg0_nested : memref<2x64x4xvector<8xf32>, 2>

memref.dealloc %tag_arg2 : memref<2x2xi32>
memref.dealloc %buf_arg2 : memref<2x64x4xvector<8xf32>, 2>
```

**关键点:**

1. **嵌套DMA**: 外层和内层DMA都被流水线化
2. **多层双缓冲**: 每层DMA都有自己的双缓冲区
3. **嵌套prologue/epilogue**: 每层DMA都有自己的prologue和epilogue
4. **内存管理**: 双缓冲区在循环内分配和释放

---

### 用例 4: loop_dma_dependent

**原始代码:**

```mlir
func.func @loop_dma_dependent(%arg2: memref<512x32xvector<8xf32>>) {
  %num_elts = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<64x4xvector<8xf32>, 2>
  %1 = memref.alloc() : memref<64x4xvector<8xf32>, 2>
  %2 = memref.alloc() : memref<64x4xvector<8xf32>, 2>
  %3 = memref.alloc() : memref<2xi32>
  %4 = memref.alloc() : memref<2xi32>
  %5 = memref.alloc() : memref<2xi32>

  affine.for %i0 = 0 to 8 {
    %6 = affine.apply #map2(%i0)
    affine.dma_start %arg2[%6, %c0], %2[%c0, %c0], %5[%c0], %num_elts : ...
    affine.dma_wait %5[%c0], %num_elts : ...

    affine.dma_start %2[%c0, %c0], %arg2[%6, %c0], %5[%c0], %num_elts : ...
    affine.dma_wait %5[%c0], %num_elts : ...
  }
  ...
}
```

**说明:**

此用例测试有依赖关系的DMA操作。同一个循环迭代中，有一个incoming DMA（从%arg2到%2）和一个outgoing DMA（从%2到%arg2），它们操作同一个memref。

**流水线化后的行为:**

```mlir
// 不进行流水线化
affine.for %i0 = 0 to 8 {
  %6 = affine.apply #map2(%i0)
  affine.dma_start %arg2[%6, %c0], %2[%c0, %c0], %5[%c0], %num_elts : ...
  affine.dma_wait %5[%c0], %num_elts : ...

  affine.dma_start %2[%c0, %c0], %arg2[%6, %c0], %5[%c0], %num_elts : ...
  affine.dma_wait %5[%c0], %num_elts : ...
}
```

**关键点:**

1. **依赖检测**: pass检测到同一迭代中有incoming和outgoing DMA操作同一个memref
2. **不进行优化**: 由于依赖关系，不进行流水线化
3. **正确性优先**: 确保优化不会改变程序语义

---

### 用例 5: escaping_use

**原始代码:**

```mlir
func.func @escaping_use(%arg0: memref<512 x 32 x f32>) {
  %c32 = arith.constant 32 : index
  %num_elt = arith.constant 512 : index
  %zero = arith.constant 0 : index
  %Av = memref.alloc() : memref<32 x 32 x f32, 2>
  %tag = memref.alloc() : memref<1 x i32>

  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
    affine.dma_wait %tag[%zero], %num_elt : ...
    // escaping use
    "foo"(%Av) : (memref<32 x 32 x f32, 2>) -> ()
  }
  ...
}
```

**说明:**

此用例测试缓冲区有escaping use的情况。`%Av`被传递给外部操作`"foo"`，这是一个escaping use。

**流水线化后的行为:**

```mlir
// 不进行流水线化
affine.for %kTT = 0 to 16 {
  affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
  affine.dma_wait %tag[%zero], %num_elt : ...
  "foo"(%Av) : (memref<32 x 32 x f32, 2>) -> ()
}
```

**关键点:**

1. **Escaping use检测**: pass检测到缓冲区被外部操作使用
2. **不进行优化**: 由于escaping use，不进行流水线化
3. **安全性**: 确保优化不会破坏外部依赖

---

### 用例 6: escaping_tag

**原始代码:**

```mlir
func.func @escaping_tag(%arg0: memref<512 x 32 x f32>) {
  %c32 = arith.constant 32 : index
  %num_elt = arith.constant 512 : index
  %zero = arith.constant 0 : index
  %Av = memref.alloc() : memref<32 x 32 x f32, 2>
  %tag = memref.alloc() : memref<1 x i32>

  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
    affine.dma_wait %tag[%zero], %num_elt : ...
    // escaping use
    "foo"(%tag) : (memref<1 x i32>) -> ()
  }
  ...
}
```

**说明:**

此用例测试tag有escaping use的情况。`%tag`被传递给外部操作`"foo"`。

**流水线化后的行为:**

```mlir
// 不进行流水线化
affine.for %kTT = 0 to 16 {
  affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
  affine.dma_wait %tag[%zero], %num_elt : ...
  "foo"(%tag) : (memref<1 x i32>) -> ()
}
```

**关键点:**

1. **Tag escaping use**: tag被外部操作使用
2. **不进行优化**: 由于tag的escaping use，不进行流水线化

---

### 用例 7: live_out_use

**原始代码:**

```mlir
func.func @live_out_use(%arg0: memref<512 x 32 x f32>) -> f32 {
  %c32 = arith.constant 32 : index
  %num_elt = arith.constant 512 : index
  %zero = arith.constant 0 : index
  %Av = memref.alloc() : memref<32 x 32 x f32, 2>
  %tag = memref.alloc() : memref<1 x i32>

  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
    affine.dma_wait %tag[%zero], %num_elt : ...
  }
  // Use live out of 'affine.for' op
  %v = affine.load %Av[%zero, %zero] : memref<32 x 32 x f32, 2>
  ...
  return %v : f32
}
```

**说明:**

此用例测试缓冲区在循环外被使用的情况。`%Av`在循环后被load。

**流水线化后的行为:**

```mlir
// 不进行流水线化
affine.for %kTT = 0 to 16 {
  affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
  affine.dma_wait %tag[%zero], %num_elt : ...
}
%v = affine.load %Av[%zero, %zero] : memref<32 x 32 x f32, 2>
```

**关键点:**

1. **Live out检测**: 缓冲区在循环外被使用
2. **不进行优化**: 由于live out use，不进行流水线化

---

### 用例 8: dynamic_shape_dma_buffer

**原始代码:**

```mlir
func.func @dynamic_shape_dma_buffer(%arg0: memref<512 x 32 x f32>, %Av: memref<? x ? x f32, 2>) {
  %num_elt = arith.constant 512 : index
  %zero = arith.constant 0 : index
  %tag = memref.alloc() : memref<1 x i32>

  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt : ...
    affine.dma_wait %tag[%zero], %num_elt : ...
  }
  return
}
```

**说明:**

此用例测试动态形状的DMA缓冲区。`%Av`的形状是动态的（`memref<? x ? x f32, 2>`）。

**流水线化后的行为:**

```mlir
// 获取动态形状的维度
%dim0 = memref.dim %Av, %c0 : memref<?x?xf32, 2>
%c1 = arith.constant 1 : index
%dim1 = memref.dim %Av, %c1 : memref<?x?xf32, 2>

// 分配双缓冲区
%buf = memref.alloc(%dim0, %dim1) : memref<2x?x?xf32, 2>
%tag = memref.alloc() : memref<2x1xi32>

// Prologue
affine.dma_start %arg0[%zero, %zero], %buf[0, 0, 0], %tag[0, 0], %num_elt : ...

// Steady state
affine.for %kTT = 1 to 16 {
  affine.dma_start %arg0[%zero, %zero], %buf[%kTT mod 2, 0, 0], %tag[%kTT mod 2, 0], %num_elt : ...
  affine.dma_wait %tag[(%kTT-1) mod 2, 0], %num_elt : ...
}

// Epilogue
affine.dma_wait %tag[15 mod 2, 0], %num_elt : ...
```

**关键点:**

1. **动态形状支持**: 支持动态形状的缓冲区
2. **维度获取**: 使用`memref.dim`获取动态维度
3. **双缓冲区分配**: 根据动态维度分配双缓冲区

---

### 用例 9: escaping_and_indexed_use_mix

**原始代码:**

```mlir
func.func @escaping_and_indexed_use_mix() {
  %A = memref.alloc() : memref<256 x f32, affine_map<(d0) -> (d0)>, 0>
  %Ah = memref.alloc() : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>
  %tag = memref.alloc() : memref<1 x f32>
  %zero = arith.constant 0 : index
  %num_elts = arith.constant 32 : index

  affine.for %i = 0 to 8 {
    affine.dma_start %A[%i], %Ah[%i], %tag[%zero], %num_elts : ...
    affine.dma_wait %tag[%zero], %num_elts : ...
    // escaping use
    "compute"(%Ah) : (memref<32 x f32, 1>) -> ()
    // indexed use
    %v = affine.load %Ah[%i] : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>
    "foo"(%v) : (f32) -> ()
  }
  ...
}
```

**说明:**

此用例测试混合使用escaping use和indexed use的情况。`%Ah`既被传递给外部操作（escaping use），又被load（indexed use）。

**流水线化后的行为:**

```mlir
// 不进行替换
affine.for %i = 0 to 8 {
  affine.dma_start %A[%i], %Ah[%i], %tag[%zero], %num_elts : ...
  affine.dma_wait %tag[%zero], %num_elts : ...
  "compute"(%Ah) : (memref<32 x f32, 1>) -> ()
  %v = affine.load %Ah[%i] : memref<32 x f32, affine_map<(d0) -> (d0)>, 1>
  "foo"(%v) : (f32) -> ()
}
```

**关键点:**

1. **混合使用检测**: 检测到escaping use和indexed use混合
2. **不进行替换**: 由于escaping use，不进行memref替换
3. **安全性**: 确保优化不会破坏程序语义

---

## 总结

`affine-pipeline-data-transfer` pass是一个专门用于DMA优化的pass，它可以：

1. **自动流水线化DMA**: 将DMA传输与计算重叠，隐藏DMA延迟
2. **双缓冲技术**: 使用双缓冲区实现流水线化
3. **生成prologue/epilogue**: 自动生成前奏和尾声代码
4. **处理嵌套DMA**: 支持多层嵌套的DMA操作
5. **支持动态形状**: 支持动态形状的缓冲区
6. **安全检测**: 检测依赖关系、escaping use、live out use等，避免错误优化

该pass在有DMA的系统中非常有用，可以显著提高性能。但需要注意：
- DMA操作必须有明确的start和wait配对
- 缓冲区和tag不能有escaping use
- 缓冲区不能在循环外被使用（live out）
- 同一迭代中不能有incoming和outgoing DMA操作同一memref
- 动态形状的缓冲区需要额外处理
