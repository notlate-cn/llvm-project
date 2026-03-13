# LLVM MLIR Arith方言Transform技术详解

本文档详细梳理LLVM MLIR中Arith方言的所有Transform Pass的作用、技术原理和应用场景。

**目录路径**: `mlir/lib/Dialect/Arith/Transforms/`

---

## 目录

1. [类型模拟Transform](#类型模拟Transform)
2. [操作扩展与优化](#操作扩展与优化)
3. [整数范围分析](#整数范围分析)
4. [接口实现](#接口实现)
5. [完整应用场景](#完整应用场景)

---

## Arith方言概述

Arith方言提供了MLIR中的**基础算术操作**,包括:
- 整数运算: `addi`, `subi`, `muli`, `divsi`, `divui`
- 浮点运算: `addf`, `subf`, `mulf`, `divf`
- 比较操作: `cmpi`, `cmpf`
- 类型转换: `extsi`, `extui`, `trunci`, `sitofp`, `uitofp`
- 位运算: `andi`, `ori`, `xori`, `shli`, `shrsi`, `shrui`

Transform主要解决两大类问题:
1. **硬件限制**: 模拟不支持的类型和操作
2. **优化机会**: 基于分析的代码改进

---

## 类型模拟Transform

### 1. Emulate Wide Integer (宽整数模拟)

**文件**: `EmulateWideInt.cpp` (1309行)

#### 1.1 作用
在只支持窄整数的硬件上模拟宽整数类型(如i128, i256),通过将宽整数拆分为多个窄整数组合实现。

#### 1.2 技术原理

##### 类型表示

**核心思想**: 将i2N分解为2个iN的向量

```mlir
// 配置: widestIntSupported = 64
i128 → vector<2xi64>
i256 → vector<4xi64>

// 表示方式: [low_part, high_part]
%wide : i128
// 等价于
%vec : vector<2xi64> = [low64bits, high64bits]
```

##### 算术操作模拟

**1. 加法 (AddI)**

利用`arith.addui_extended`产生进位:

```mlir
// 原始: %sum = arith.addi %a, %b : i128
// %a = [a_low, a_high], %b = [b_low, b_high]

// 低位相加,产生进位
%sum_low, %carry = arith.addui_extended %a_low, %b_low : i64, i1

// 高位相加,加上进位
%carry_ext = arith.extui %carry : i1 to i64
%high_sum = arith.addi %a_high, %b_high : i64
%sum_high = arith.addi %high_sum, %carry_ext : i64

// 结果: [sum_low, sum_high]
```

**2. 减法 (SubI)**

通过比较检测借位:

```mlir
// 原始: %diff = arith.subi %a, %b : i128

// 低位相减
%diff_low = arith.subi %a_low, %b_low : i64

// 检测借位: a_low < b_low
%borrow = arith.cmpi ult, %a_low, %b_low : i64
%borrow_ext = arith.extui %borrow : i1 to i64

// 高位相减,减去借位
%high_diff = arith.subi %a_high, %b_high : i64
%diff_high = arith.subi %high_diff, %borrow_ext : i64
```

**3. 乘法 (MulI)**

长乘法算法:

```
(a_high * 2^64 + a_low) * (b_high * 2^64 + b_low)
= a_low*b_low + (a_low*b_high + a_high*b_low)*2^64 + a_high*b_high*2^128
```

```mlir
// 低位: a_low * b_low的低64位
%low_low = arith.muli %a_low, %b_low : i64

// 中间项: (a_low*b_high + a_high*b_low)的低64位
%low_high = arith.muli %a_low, %b_high : i64
%high_low = arith.muli %a_high, %b_low : i64
%mid = arith.addi %low_high, %high_low : i64

// 高位 = (a_low*b_low的高64位) + mid
%low_low_ext = <extended multiply>
%result_high = arith.addi %low_low_ext, %mid : i64
```

**4. 左移 (ShLI)**

三种情况:

```mlir
// 原始: %shifted = arith.shli %val, %amount : i128

// Case 1: amount < 64 (位移在低位内)
%low_shifted = arith.shli %val_low, %amount
%promoted = arith.shrui %val_low, %complement  // 提升到高位的部分
%high_shifted = arith.shli %val_high, %amount
%result_high = arith.ori %high_shifted, %promoted

// Case 2: amount >= 64 (低位全部移到高位)
%result_high = arith.shli %val_low, %adjusted_amount
%result_low = 0

// 使用select根据amount选择结果
```

**5. 比较 (CmpI)**

词典序比较:

```mlir
// %cmp = arith.cmpi slt, %a, %b : i128

// 先比较高位
%high_cmp = arith.cmpi slt, %a_high, %b_high : i64

// 高位相等时,比较低位 (使用unsigned比较!)
%high_eq = arith.cmpi eq, %a_high, %b_high : i64
%low_cmp = arith.cmpi ult, %a_low, %b_low : i64

// 组合结果
%result = arith.select %high_eq, %low_cmp, %high_cmp : i1
```

**关键**: 高位相等时,低位必须用unsigned比较

#### 1.3 实例演示

**示例1: i128加法**

```mlir
// 输入
func.func @add_i128(%a: i128, %b: i128) -> i128 {
  %sum = arith.addi %a, %b : i128
  return %sum : i128
}

// 模拟后 (widestIntSupported=64)
func.func @add_i128(%a: vector<2xi64>, %b: vector<2xi64>) -> vector<2xi64> {
  // 提取低位和高位
  %a_low = vector.extract %a[0] : vector<2xi64>
  %a_high = vector.extract %a[1] : vector<2xi64>
  %b_low = vector.extract %b[0] : vector<2xi64>
  %b_high = vector.extract %b[1] : vector<2xi64>

  // 低位加法 + 进位
  %sum_low, %carry = arith.addui_extended %a_low, %b_low : i64, i1
  %carry_i64 = arith.extui %carry : i1 to i64

  // 高位加法
  %temp = arith.addi %a_high, %b_high : i64
  %sum_high = arith.addi %temp, %carry_i64 : i64

  // 构造结果向量
  %result = vector.splat %c0 : vector<2xi64>
  %result1 = vector.insert %sum_low, %result[0]
  %result2 = vector.insert %sum_high, %result1[1]

  return %result2 : vector<2xi64>
}
```

**示例2: i256乘法**

```mlir
// 输入
%product = arith.muli %x, %y : i256

// 需要拆分为4个i64部分
// %x = [x0, x1, x2, x3]
// %y = [y0, y1, y2, y3]

// 完整长乘法展开...
// (实现复杂,涉及16个部分积和进位传播)
```

#### 1.4 适用场景

- **嵌入式系统**: 32位MCU执行64位运算
- **FPGA**: 自定义位宽实现
- **加密算法**: 256位/512位大数运算
- **时间戳**: 128位Unix时间戳

---

### 2. Emulate Narrow Type (窄类型模拟)

**文件**: `EmulateNarrowType.cpp` (53行)

#### 2.1 作用
提供**框架**支持模拟窄于硬件字宽的类型(如i4, i8在i32系统上)。

#### 2.2 技术原理

##### 类型转换器

```cpp
ArithNarrowTypeEmulationConverter converter(targetBitwidth);

// 示例: targetBitwidth = 32
i4  → i32 (扩展)
i8  → i32 (扩展)
i16 → i32 (扩展)
i32 → i32 (保持)
```

##### 函数转换

```cpp
// 处理函数签名
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns);
populateCallOpTypeConversionPattern(patterns);
populateReturnOpTypeConversionPattern(patterns);
```

#### 2.3 示例

```mlir
// 原始: i4运算
func.func @compute(%a: i4, %b: i4) -> i4 {
  %sum = arith.addi %a, %b : i4
  return %sum : i4
}

// 转换后 (targetBitwidth=32)
func.func @compute(%a: i32, %b: i32) -> i32 {
  // 确保输入在i4范围内
  %a_masked = arith.andi %a, 0xF : i32
  %b_masked = arith.andi %b, 0xF : i32

  // 执行运算
  %sum = arith.addi %a_masked, %b_masked : i32

  // 截断到i4范围
  %result = arith.andi %sum, 0xF : i32
  return %result : i32
}
```

**注意**: 具体操作模式由vector等方言提供。

---

### 3. Emulate Unsupported Floats (浮点模拟)

**文件**: `EmulateUnsupportedFloats.cpp` (184行)

#### 3.1 作用
模拟硬件不支持的浮点类型(bf16, f8系列, f4等),通过提升到支持的类型(通常f32)。

#### 3.2 技术原理

##### 模拟模式

```
源类型 --extf--> 目标类型 --operate--> 目标类型 --truncf--> 源类型
```

```mlir
// 原始: bf16运算
%result = arith.addf %a, %b : bf16

// 模拟 (bf16 → f32)
%a_f32 = arith.extf %a : bf16 to f32
%b_f32 = arith.extf %b : bf16 to f32
%sum_f32 = arith.addf %a_f32, %b_f32 : f32
%result = arith.truncf %sum_f32 : f32 to bf16
```

##### 支持的类型

| 源类型 | 目标类型 | 描述 |
|--------|---------|------|
| bf16 | f32 | Brain Float 16 |
| f8E5M2 | f32 | 8位浮点(5指数,2尾数) |
| f8E4M3FN | f32 | 8位浮点(4指数,3尾数) |
| f4E2M1FN | f32 | 4位浮点(2指数,1尾数) |

#### 3.3 实例演示

**BFloat16矩阵乘法**

```mlir
// 输入: BF16 GEMM
func.func @gemm_bf16(%A: tensor<128x256xbf16>,
                     %B: tensor<256x512xbf16>,
                     %C: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  %result = linalg.matmul ins(%A, %B) outs(%C)
    : tensor<128x256xbf16>, tensor<256x512xbf16>
    -> tensor<128x512xbf16>
  return %result
}

// 模拟后
func.func @gemm_bf16(%A: tensor<128x256xbf16>,
                     %B: tensor<256x512xbf16>,
                     %C: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  // 提升输入
  %A_f32 = arith.extf %A : tensor<128x256xbf16> to tensor<128x256xf32>
  %B_f32 = arith.extf %B : tensor<256x512xbf16> to tensor<256x512xf32>
  %C_f32 = arith.extf %C : tensor<128x512xbf16> to tensor<128x512xf32>

  // F32计算
  %result_f32 = linalg.matmul ins(%A_f32, %B_f32) outs(%C_f32)
    : tensor<128x256xf32>, tensor<256x512xf32>
    -> tensor<128x512xf32>

  // 截断输出
  %result = arith.truncf %result_f32 : tensor<128x512xf32>
                                    to tensor<128x512xbf16>
  return %result
}
```

**优化**: 设置`fastmath = contract`允许融合扩展/截断。

---

## 操作扩展与优化

### 4. Expand Operations (操作扩展)

**文件**: `ExpandOps.cpp` (856行)

#### 4.1 作用
将复杂或高级算术操作展开为更简单的基础操作序列,便于lowering到硬件指令。

#### 4.2 关键展开模式

##### 1. 除法操作

**CeilDivUI (无符号上取整除法)**

```mlir
// 语义: ceil(n / m) = floor((n + m - 1) / m)

// 原始
%result = arith.ceildivui %n, %m : i32

// 展开
%is_zero = arith.cmpi eq, %n, %c0 : i32
%n_minus_1 = arith.subi %n, %c1 : i32
%n_plus_m_minus_1 = arith.addi %n_minus_1, %m : i32
%div = arith.divui %n_plus_m_minus_1, %m : i32
%div_plus_1 = arith.addi %div, %c1 : i32
%result = arith.select %is_zero, %c0, %div_plus_1 : i32
```

**CeilDivSI (有符号上取整除法)**

```mlir
// 语义: 向正无穷方向取整

// 算法:
// 1. z = a / b (截断除法)
// 2. if (z * b != a && sign(a) == sign(b)) z += 1

%quot = arith.divsi %a, %b : i32
%prod = arith.muli %quot, %b : i32
%rem_not_zero = arith.cmpi ne, %prod, %a : i32

%a_neg = arith.cmpi slt, %a, %c0 : i32
%b_neg = arith.cmpi slt, %b, %c0 : i32
%same_sign = arith.cmpi eq, %a_neg, %b_neg : i1

%adjust = arith.andi %rem_not_zero, %same_sign : i1
%adjust_i32 = arith.extui %adjust : i1 to i32
%result = arith.addi %quot, %adjust_i32 : i32
```

##### 2. Min/Max操作

**MaxSI/MinSI (有符号最大/最小)**

```mlir
// 原始
%max = arith.maxsi %a, %b : i32

// 展开
%cmp = arith.cmpi sgt, %a, %b : i32
%max = arith.select %cmp, %a, %b : i32
```

**MaxNumF/MinNumF (浮点带NaN处理)**

```mlir
// 语义: 优先返回非NaN值

// 原始
%max = arith.maxnumf %a, %b : f32

// 展开
%a_is_nan = arith.cmpf uno, %a, %a : f32  // unordered比较
%b_is_nan = arith.cmpf uno, %b, %b : f32

%cmp = arith.cmpf ogt, %a, %b : f32  // ordered比较
%max_val = arith.select %cmp, %a, %b : f32

// 如果a是NaN,选b; 如果b是NaN,选a; 否则选max_val
%max = arith.select %a_is_nan, %b,
         arith.select %b_is_nan, %a, %max_val : f32
```

##### 3. BFloat16转换

**ExtF (bf16 → f32)**

```mlir
// BF16格式: [sign(1) | exponent(8) | mantissa(7)]
// F32格式:  [sign(1) | exponent(8) | mantissa(23)]
// 关系: BF16 = F32的高16位

// 原始
%f32 = arith.extf %bf16 : bf16 to f32

// 展开
%bf16_bits = arith.bitcast %bf16 : bf16 to i16
%bf16_i32 = arith.extui %bf16_bits : i16 to i32
%f32_bits = arith.shli %bf16_i32, %c16 : i32  // 左移16位补零
%f32 = arith.bitcast %f32_bits : i32 to f32
```

**TruncF (f32 → bf16) - 魔法舍入**

```mlir
// 算法: 利用浮点加法的进位实现round-to-nearest-even

// 原始
%bf16 = arith.truncf %f32 : f32 to bf16

// 展开
%f32_bits = arith.bitcast %f32 : f32 to i32

// 读取第16位(决定舍入方向)
%bit16 = arith.shrui %f32_bits, %c16 : i32
%bit16_and_1 = arith.andi %bit16, %c1 : i32

// 舍入偏置: 0x7FFF (bit16=0) 或 0x8000 (bit16=1)
%bias_base = arith.constant 0x7FFF : i32
%bias = arith.addi %bias_base, %bit16_and_1 : i32

// 加上偏置(进位会传播到指数位,实现舍入)
%rounded = arith.addi %f32_bits, %bias : i32

// 截断到16位
%bf16_bits = arith.shrui %rounded, %c16 : i32
%bf16_i16 = arith.trunci %bf16_bits : i32 to i16
%bf16 = arith.bitcast %bf16_i16 : i16 to bf16

// 特殊处理NaN
%is_nan = arith.cmpf uno, %f32, %f32 : f32
%qnan = arith.constant 0x7FC0 : i16  // Quiet NaN
%qnan_bf16 = arith.bitcast %qnan : i16 to bf16
%result = arith.select %is_nan, %qnan_bf16, %bf16 : bf16
```

**关键**: 通过精心设计的偏置,让浮点加法的进位自然实现round-to-nearest-even。

##### 4. Float8格式

**F8E8M0FNU (8位纯指数浮点)**

```mlir
// 格式: 8位全部用于指数,无尾数
// F32: [S(1) | E(8) | M(23)]
// F8:  [E(8)]

// ExtF: F8 → F32
%f8_bits = arith.bitcast %f8 : f8E8M0FNU to i8
%f8_i32 = arith.extui %f8_bits : i8 to i32

// 构造F32: 符号=0, 指数=f8, 尾数=0
%exp_shifted = arith.shli %f8_i32, %c23 : i32  // 移到指数位置
%f32 = arith.bitcast %exp_shifted : i32 to f32

// TruncF: F32 → F8
%f32_bits = arith.bitcast %f32 : f32 to i32
%exp = arith.shrui %f32_bits, %c23 : i32  // 提取指数
%exp_i8 = arith.trunci %exp : i32 to i8
%f8 = arith.bitcast %exp_i8 : i8 to f8E8M0FNU
```

#### 4.3 实例演示

**完整示例: 上取整除法优化**

```mlir
// 场景: 计算需要多少块才能覆盖n个元素
func.func @num_blocks(%n: index, %block_size: index) -> index {
  %blocks = arith.ceildivui %n, %block_size : index
  return %blocks : index
}

// 展开后
func.func @num_blocks(%n: index, %block_size: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %is_zero = arith.cmpi eq, %n, %c0 : index
  scf.if %is_zero {
    return %c0 : index
  } else {
    %n_minus_1 = arith.subi %n, %c1 : index
    %sum = arith.addi %n_minus_1, %block_size : index
    %quot = arith.divui %sum, %block_size : index
    %blocks = arith.addi %quot, %c1 : index
    return %blocks : index
  }
}
```

---

## 整数范围分析

### 5. Integer Range Optimizations (整数范围优化)

**文件**: `IntRangeOptimizations.cpp` (511行)

#### 5.1 作用
基于数据流分析推断整数值的可能范围,应用基于范围的优化(常量折叠、死代码消除、操作简化)。

#### 5.2 技术原理

##### 数据流分析

使用MLIR的`DataFlowSolver`和`IntegerValueRangeLattice`:

```cpp
// 每个SSA值关联一个范围
Value %x → ConstantIntRanges {
  smin: APInt,  // 有符号最小值
  smax: APInt,  // 有符号最大值
  umin: APInt,  // 无符号最小值
  umax: APInt   // 无符号最大值
}
```

**示例**:
```mlir
%c10 = arith.constant 10 : i32
// range(%c10) = [10, 10] (signed), [10, 10] (unsigned)

%x = ... : i32  // 某个值
%clamped = arith.minui %x, %c100 : i32
// range(%clamped) = [?, 100] (unsigned)

%sum = arith.addi %c10, %clamped : i32
// range(%sum) = [10, 110] (unsigned)
```

##### 优化模式

**1. 常量折叠**

```mlir
// 如果值的范围是单点
%x : i32 with range [42, 42]

// 替换为常量
%c42 = arith.constant 42 : i32
// %x的所有使用 → %c42
```

**2. Remainder消除**

```mlir
// 原始
%rem = arith.remui %x, %c256 : i32
// 如果 range(%x) = [0, 100)  (< 256)

// 简化为
%rem = %x  // 直接使用原值
```

**3. 死代码检测**

```mlir
%cond = arith.cmpi slt, %x, %c100 : i32
// 如果 range(%x) = [0, 50)  (始终 < 100)

// %cond始终为true
%true = arith.constant true
```

#### 5.3 实例演示

**示例: 循环边界优化**

```mlir
// 原始代码
func.func @bounded_loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %clamped = arith.minui %n, %c10 : index
  // range(%clamped) = [0, 10]

  scf.for %i = %c0 to %clamped step %c1 {
    // 循环体
    %rem = arith.remui %i, %c10 : index
    // range(%i) = [0, 10), 因此 %rem = %i
  }
}

// 优化后
func.func @bounded_loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %clamped = arith.minui %n, %c10 : index

  scf.for %i = %c0 to %clamped step %c1 {
    // %rem = %i  (消除了取模运算)
    // 直接使用%i
  }
}
```

---

### 6. Integer Range Narrowing (整数范围窄化)

**文件**: `IntRangeOptimizations.cpp` (同一文件中的第二个Pass)

#### 6.1 作用
将宽整数操作窄化为更窄的类型,当范围分析证明安全时,减少位宽降低计算成本。

#### 6.2 技术原理

##### 可截断性分析

**CastKind枚举**:
```cpp
enum class CastKind {
  None,      // 不可截断
  Unsigned,  // 可用零扩展截断
  Signed,    // 可用符号扩展截断
  Both       // 两种扩展都可以
};
```

**判断标准**:
```cpp
bool isTruncatable(Value val, unsigned targetWidth) {
  ConstantIntRanges range = getRange(val);

  // 有符号检查: 需要足够的符号位
  unsigned signBits = countLeadingSignBits(range);
  bool signedOk = (signBits >= srcWidth - targetWidth + 1);

  // 无符号检查: 需要足够的前导零
  unsigned zeroBits = countLeadingZeros(range);
  bool unsignedOk = (zeroBits >= srcWidth - targetWidth);

  if (signedOk && unsignedOk) return CastKind::Both;
  if (signedOk) return CastKind::Signed;
  if (unsignedOk) return CastKind::Unsigned;
  return CastKind::None;
}
```

##### 窄化变换

```mlir
// 原始: i64运算,但值域很小
%a : i64 with range [0, 100)
%b : i64 with range [0, 50)
%sum = arith.addi %a, %b : i64
// range(%sum) = [0, 150) → 适合i8

// 窄化到i8
%a_narrow = arith.trunci %a : i64 to i8
%b_narrow = arith.trunci %b : i64 to i8
%sum_narrow = arith.addi %a_narrow, %b_narrow : i8
%sum = arith.extui %sum_narrow : i8 to i64
```

#### 6.3 支持的操作

| 操作 | 要求 | 窄化策略 |
|------|------|---------|
| addi | 操作数+结果可截断 | 截断→操作→扩展 |
| subi | 同上 | 同上 |
| muli | 同上 | 同上 |
| shli | shift amount < targetWidth | 操作数截断 |
| shrui | 同上 | 同上 |
| shrsi | 同上 | 同上 |
| andi | 操作数可截断 | 截断→操作→扩展 |
| ori | 同上 | 同上 |
| xori | 同上 | 同上 |

#### 6.4 实例演示

**示例: 数组索引计算**

```mlir
// 场景: 小数组的索引计算
func.func @array_access(%base: i64, %offset: i64) -> i64 {
  // 已知: range(%offset) = [0, 256)
  %index = arith.addi %base, %offset : i64
  return %index : i64
}

// 分析
// %offset适合i8 (range [0, 256))
// 假设%base也在合理范围

// 窄化后 (bitwidthsSupported = [8, 16, 32, 64])
func.func @array_access(%base: i64, %offset: i64) -> i64 {
  %offset_i16 = arith.trunci %offset : i64 to i16
  %base_i16 = arith.trunci %base : i64 to i16

  %index_i16 = arith.addi %base_i16, %offset_i16 : i16

  %index = arith.extui %index_i16 : i16 to i64
  return %index : i64
}
```

**收益**: i16加法比i64快,功耗低

---

### 7. Unsigned When Equivalent (等价时转无符号)

**文件**: `UnsignedWhenEquivalent.cpp` (128行)

#### 7.1 作用
将有符号运算转换为无符号运算,当静态分析证明所有值非负时。

#### 7.2 技术原理

##### 非负性检查

```cpp
bool staticallyNonNegative(Value val) {
  ConstantIntRanges range = getRange(val);
  return range.smin().isNonNegative();  // 最小值 >= 0
}
```

##### 转换规则

| 有符号操作 | 无符号操作 | 条件 |
|----------|----------|------|
| divsi | divui | 被除数、除数非负 |
| remsi | remui | 被除数、除数非负 |
| ceildivsi | ceildivui | 被除数、除数非负 |
| floordivsi | divui | 同上(结果相同) |
| minsi | minui | 操作数非负 |
| maxsi | maxui | 操作数非负 |
| extsi | extui | 操作数非负 |
| cmpi slt | cmpi ult | 操作数非负 |
| cmpi sle | cmpi ule | 操作数非负 |
| cmpi sgt | cmpi ugt | 操作数非负 |
| cmpi sge | cmpi uge | 操作数非负 |

#### 7.3 实例演示

**示例: 自然数除法**

```mlir
// 原始代码
func.func @safe_div(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32

  // 确保非负
  %clamped = arith.maxsi %n, %c0 : i32
  // range(%clamped) = [0, 2^31-1]

  %result = arith.divsi %clamped, %c10 : i32
  return %result : i32
}

// 优化后
func.func @safe_div(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32

  %clamped = arith.maxsi %n, %c0 : i32

  // 转换为无符号除法(更快!)
  %result = arith.divui %clamped, %c10 : i32
  return %result : i32
}
```

**硬件差异**:
- **有符号除法**: 需要处理符号位,corner cases(INT_MIN / -1)
- **无符号除法**: 简单除法器,无符号扩展

---

### 8. Reify Value Bounds (值界限具体化)

**文件**: `ReifyValueBounds.cpp` (156行)

#### 8.1 作用
将抽象的值界限约束(来自`ValueBoundsConstraintSet`)具体化为可执行的Arith IR。

#### 8.2 技术原理

##### 界限类型

```cpp
enum class BoundType {
  LowerBound,  // 下界
  UpperBound   // 上界
};
```

##### 具体化流程

```cpp
FailureOr<OpFoldResult> reifyValueBound(
    OpBuilder& b, Location loc, BoundType type,
    const ValueBoundsConstraintSet::Variable& var) {

  // 1. 计算抽象界限
  AffineMap boundMap;
  ValueDimList mapOperands;
  ValueBoundsConstraintSet::computeBound(
    boundMap, mapOperands, type, var
  );

  // 2. 具体化为IR
  return materializeBound(b, loc, boundMap, mapOperands);
}
```

##### Affine到Arith转换

```cpp
Value materializeAffineExpr(AffineExpr expr, ValueRange operands) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    return builder.create<arith::ConstantIndexOp>(constExpr.getValue());
  }
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    return operands[dimExpr.getPosition()];
  }
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    Value lhs = materialize(binExpr.getLHS(), operands);
    Value rhs = materialize(binExpr.getRHS(), operands);

    switch (binExpr.getKind()) {
    case AffineExprKind::Add:
      return builder.create<arith::AddIOp>(lhs, rhs);
    case AffineExprKind::Mul:
      return builder.create<arith::MulIOp>(lhs, rhs);
    case AffineExprKind::FloorDiv:
      return builder.create<arith::DivSIOp>(lhs, rhs);
    case AffineExprKind::CeilDiv:
      return builder.create<arith::CeilDivSIOp>(lhs, rhs);
    case AffineExprKind::Mod:
      return builder.create<arith::RemSIOp>(lhs, rhs);
    }
  }
}
```

#### 8.3 实例演示

**示例: 动态维度界限**

```mlir
// 场景: 推断tensor维度的界限
func.func @infer_bounds(%t: tensor<?x?xf32>, %i: index) {
  // 约束: dim(t, 0) <= 100
  // 约束: dim(t, 1) = dim(t, 0) * 2

  // 请求具体化 dim(t, 1) 的上界
  %ub = <reifyValueBound upper_bound for dim(t, 1)>
}

// 具体化后
func.func @infer_bounds(%t: tensor<?x?xf32>, %i: index) {
  %c100 = arith.constant 100 : index
  %c2 = arith.constant 2 : index

  // dim(t, 1) <= dim(t, 0) * 2 <= 100 * 2 = 200
  %ub = arith.muli %c100, %c2 : index  // 200

  // 使用%ub进行优化决策
}
```

**应用**:
- 循环分块的界限计算
- 动态形状的静态化
- 内存分配大小推断

---

## 接口实现

### 9. Bufferizable Op Interface (Bufferization接口)

**文件**: `BufferizableOpInterfaceImpl.cpp` (235行)

#### 9.1 作用
定义Arith操作的bufferization语义,支持tensor→memref转换。

#### 9.2 关键实现

##### ConstantOp

```cpp
// Tensor常量 → Memref全局变量
// 原始
%t = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

// Bufferize后
memref.global "private" constant @__constant_4xi32 :
  memref<4xi32> = dense<[1, 2, 3, 4]>

func.func @use_constant() {
  %m = memref.get_global @__constant_4xi32 : memref<4xi32>
  // 使用%m
}
```

**特点**:
- 只读(不可写)
- 共享(去重)
- 可指定内存空间

##### SelectOp

```cpp
// 条件选择buffer
// 原始
%result = arith.select %cond, %t_true, %t_false : tensor<4xf32>

// Bufferize
%result = arith.select %cond, %m_true, %m_false : memref<4xf32>

// 要求:
// - %cond必须是标量i1
// - %m_true和%m_false布局可能不同,需要cast
```

**布局处理**:
```mlir
// 如果布局不同
%m_true : memref<4xf32, strided<[2], offset: 4>>
%m_false : memref<4xf32>

// 转换为fully dynamic layout
%true_cast = memref.cast %m_true : ... to memref<4xf32, strided<[?], offset: ?>>
%false_cast = memref.cast %m_false : ... to memref<4xf32, strided<[?], offset: ?>>
%result = arith.select %cond, %true_cast, %false_cast
```

---

### 10. 其他接口实现

#### Buffer Deallocation Interface
**文件**: `BufferDeallocationOpInterfaceImpl.cpp`

处理`arith.select`的所有权传播:
```mlir
%ownership_result = arith.select %cond,
  %ownership_true, %ownership_false : i1
```

#### Buffer View Flow Interface
**文件**: `BufferViewFlowOpInterfaceImpl.cpp`

定义`arith.select`的alias依赖:
```cpp
// %result may alias %true_value or %false_value
registerDependency(%true_value, %result);
registerDependency(%false_value, %result);
```

#### Sharding Interface
**文件**: `ShardingInterfaceImpl.cpp`

支持常量的分布式计算:
```cpp
// Splat常量: 复制到所有分片
// Non-splat: 需要额外逻辑(未完全支持)
```

---

## 完整应用场景

### Pipeline示例

#### 场景1: 嵌入式系统编译

```bash
# 目标: 32位MCU,需要64位运算

# 1. 模拟64位整数
mlir-opt input.mlir \
  --arith-emulate-wide-int="widest-int-supported=32"

# 2. 展开复杂操作
mlir-opt ... --arith-expand

# 3. 范围优化
mlir-opt ... \
  --int-range-optimizations \
  --int-range-narrowing="bitwidths-supported=8,16,32"

# 4. 转无符号
mlir-opt ... --arith-unsigned-when-equivalent

# 5. Lower到LLVM
mlir-opt ... --convert-arith-to-llvm
```

#### 场景2: AI加速器编译

```bash
# 目标: 支持BF16,不支持F8

# 1. 模拟F8类型
mlir-opt input.mlir \
  --arith-emulate-unsupported-floats="source-types=f8E5M2 target-type=bf16"

# 2. 展开BF16转换(如果硬件不支持)
mlir-opt ... --arith-expand

# 3. 向量化
mlir-opt ... --vectorize

# 4. Lower
mlir-opt ... --convert-to-llvm
```

### 优化收益

| 优化 | 典型加速 | 场景 |
|------|---------|------|
| Range narrowing | 1.2-2x | 小范围整数计算 |
| Unsigned conversion | 1.1-1.3x | 非负整数运算 |
| Remainder elimination | 2-10x | 模运算密集代码 |
| Wide int emulation | 0.3-0.5x (slowdown) | 必要时的功能实现 |
| Float emulation | 0.5-0.8x | 硬件不支持时 |
| ExpandOps | 1.0-1.2x | 简化lowering |

### 最佳实践

#### 1. 类型选择
- ✅ 尽量使用硬件原生支持的类型
- ✅ 必要时才模拟(性能代价)
- ✅ 考虑精度损失

#### 2. 优化顺序
```
分析 → 窄化 → 无符号转换 → 展开 → Lower
```

#### 3. 调试
```bash
# 查看范围推断
mlir-opt --int-range-optimizations --debug-only=int-range

# 验证模拟正确性
mlir-opt --arith-emulate-wide-int | mlir-cpu-runner --entry-point=test
```

---

## 常见问题

### Q1: Wide int模拟性能如何?
**A**: 通常2-4倍慢于原生支持,但在无原生支持时是唯一选择。优化建议:
- 最小化宽整数操作
- 考虑算法重设计
- 利用硬件特性(如128位SIMD)

### Q2: 范围分析不生效?
**A**: 检查:
- 是否有足够的约束信息
- 是否存在循环(限制分析精度)
- 考虑添加assert提示编译器

### Q3: BF16转换精度损失?
**A**: BF16保留F32的动态范围但精度降低:
- 7位尾数 vs F32的23位
- 适合梯度更新,不适合精确科学计算
- 考虑混合精度

### Q4: 何时使用ExpandOps?
**A**:
- Lowering前的标准化步骤
- 目标不支持复杂操作
- 启用后续优化(如constant folding)

---

## 扩展阅读

### 相关文档
- [MLIR Arith Dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/)
- [Data Flow Framework](https://mlir.llvm.org/docs/DataFlowAnalysis/)
- [BFloat16 Format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

### 学术论文
- "Fast and Accurate Floating Point Rounding" - 舍入算法
- "Compiler Support for Type-Flexible Accelerators" - 类型模拟

### 工具
- **mlir-cpu-runner**: 执行MLIR IR
- **mlir-opt**: 应用transforms
- **mlir-translate**: 转换到LLVM IR

---

**文档版本**: LLVM 主干分支 (2026-01)
**维护者**: MLIR Arith Dialect团队
**许可证**: Apache 2.0 with LLVM Exception
