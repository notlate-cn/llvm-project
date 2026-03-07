让我先读一下官方文档，了解 TPosition 的完整映射关系：搜到了非常关键的信息，特别是 LLVM 2024 DevMtg 上已经有人在做 MLIR → AscendC Dialect 的工作，并且展示了 `!ascendc.GlobalTensor`、`!ascendc.LocalTensor`、`!ascendc.TQue` 这些类型，以及 `ascendc.free_tensor` 等 op 的 MLIR 表达。结合 TPosition 的完整映射关系，现在可以给出系统性的 Bufferization 建议。

------

## Bufferization 阶段建议

### 一、TPosition 与 transform 循环层次的映射关系

首先建立从你的三级循环结构到 AscendC TPosition 的完整映射：

| 循环层次               | 操作的数据           | TPosition          | 物理存储            | AscendC 类型                    |
| ---------------------- | -------------------- | ------------------ | ------------------- | ------------------------------- |
| func 参数（循环外）    | A/B/bias/output 全量 | `GLOBAL`           | Global Memory (GM)  | `GlobalTensor<T>`               |
| `for_TB_N` 内切片      | A[TB_M×K], B[K×TB_N] | `A1` / `B1`        | L1 Buffer           | `LocalTensor<T>` @ A1/B1        |
| `for_Tb_N` 内切片      | A[Tb_M×K], B[K×Tb_N] | `A2` / `B2`        | L0A / L0B           | `LocalTensor<T>` @ A2/B2        |
| `for_K` 内 matmul 输出 | C[Tb_M×Tb_N] 累加    | `CO1`              | L0C                 | `LocalTensor<T>` @ CO1          |
| add/max 的输入输出     | C after matmul, bias | `VECIN` / `VECOUT` | UB (Unified Buffer) | `LocalTensor<T>` @ VECIN/VECOUT |
| add/max 的中间结果     | 临时 tensor          | `VECCALC`          | UB                  | `LocalTensor<T>` @ VECCALC      |

------

### 二、Bufferization 的完整流水线

TPosition 的枚举值中，`VECIN`/`VECCALC`/`VECOUT` 用于向量编程（对应 add/max），`A1`/`A2`/`B1`/`B2`/`CO1`/`CO2` 用于矩阵编程（对应 matmul）。 基于此，Bufferization 分为四个阶段：

#### 阶段1：One-Shot Bufferize（标准 MLIR）

将 tensor 语义转为 memref 语义，这是后续所有阶段的基础：

```
// 推荐 pass pipeline:
mlir-opt output_step3.mlir \
  --one-shot-bufferize="
    allow-return-allocs-from-loops=true
    bufferize-function-boundaries=true
    unknown-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-linalg-to-loops \
  -o output_bufferized.mlir
```

关键选项说明：

- `bufferize-function-boundaries=true`：将 func 的 tensor 参数转为 `memref`（对应 `GlobalTensor`）
- `allow-return-allocs-from-loops=true`：允许循环内的 alloc，对应每层 LocalTensor 的分配

#### 阶段2：memref 分层标注（自定义 pass）

五组 annotation，覆盖所有需要区分的情况：

| Annotation                    | 打在哪里                         | 语义                                      |
| ----------------------------- | -------------------------------- | ----------------------------------------- |
| **A** TB层matmul              | `%matmul_TB_split#0`             | lhs=A1, rhs=B1, out=CO1 → 指导 GM→L1 搬运 |
| **B** TB层add/max             | `%add_fused_TB`, `%tiled_max_TB` | ins=VECIN, out=VECOUT → 指导 GM→UB 搬运   |
| **C** Tb层matmul（最终执行）  | `%matmul_in_for_K`               | lhs=A2, rhs=B2, out=CO1 → 指导 L1→L0 搬运 |
| **D** Tb层add/max（最终执行） | `%add_fused_Tb`, `%tiled_max_Tb` | ins=VECIN, out=VECOUT                     |
| **E** 循环级搬运方向          | `%for_TB_N`, `%for_Tb_N`         | 指导 DataCopy + EnQue/DeQue 插入位置      |

#### 阶段3：DataCopy 插入（GM → Local 搬运）

AscendC 使用 `DataCopy` 接口将 `GlobalTensor` 数据拷贝到 `LocalTensor`，计算 API 基于本地内存数据进行计算，所以数据需要先从全局内存搬运到本地内存，再使用计算接口完成计算，最后从本地内存搬出至全局内存。

对应到 MLIR，`memref.copy` 或 `linalg.copy` 需要被替换为带 TPosition 语义的搬运 op：

```
tensor.extract_slice (GM → A1)   → ascendc.DataCopy GlobalTensor → LocalTensor<A1>
memref.subview      (A1 → A2)    → ascendc.DataCopy LocalTensor<A1> → LocalTensor<A2>
tensor.insert_slice (CO1 → GM)   → ascendc.DataCopy LocalTensor<CO1> → GlobalTensor
```

#### 阶段4：TQue 插入（同步与流水）

Ascend C 使用 `TQue` 队列完成任务之间的数据通信和同步，提供 `EnQue`、`DeQue` 等基础 API；TQue 队列管理不同层级的物理内存时，用一种抽象的逻辑位置 TPosition 来表达各级别的存储，代替了片上物理存储的概念。

对应到 MLIR pass 的工作：在每个 DataCopy 后插入 `EnQue`，在计算 op 前插入 `DeQue`：

```mlir
// CopyIn 阶段:
ascendc.DataCopy(%local_A1, %global_A, size)
ascendc.EnQue(%inQueueA, %local_A1)   // 通知 compute 阶段数据就绪

// Compute 阶段:
%local_A1_ready = ascendc.DeQue(%inQueueA)
// linalg.matmul → ascendc.Mmad(CO1 ← A2×B2)

// CopyOut 阶段:
ascendc.EnQue(%outQueue, %local_CO2)
%local_out = ascendc.DeQue(%outQueue)
ascendc.DataCopy(%global_C, %local_out, size)
```

------

### 三、与 fc_relu 计算图的具体对应

针对你的 matmul + add + max 计算图，完整的数据流映射如下：

```
GlobalTensor A (GM)
    ↓ DataCopy [for_TB: TB_M×K]
LocalTensor A_L1 (A1)
    ↓ DataCopy [for_Tb_N: Tb_M×t_K]
LocalTensor A_L0 (A2)
    ↓ Mmad ──────────────────────────────┐
GlobalTensor B (GM)                      ↓
    ↓ DataCopy [for_TB: K×TB_N]    LocalTensor C_L0 (CO1)  ← K轴累加
LocalTensor B_L1 (B1)                    ↓ FixpointConv / Cast (CO1→CO2)
    ↓ DataCopy [for_Tb_N: t_K×Tb_N] LocalTensor C_UB (VECIN)
LocalTensor B_L0 (B2)                    ↓ Add(C_UB, bias_UB) → (VECCALC)
                                         ↓ Max(result, zero)  → (VECOUT)
GlobalTensor bias (GM)                   ↓ DataCopy
    ↓ DataCopy [for_TB: TB_M×TB_N]  GlobalTensor output (GM)
LocalTensor bias_UB (VECIN)
```

------

### 四、实施路径建议

最实用的实施方式有两条路，根据你的工程情况选择：

**路径 A（推荐，如果有 AscendC Dialect）**：基于 LLVM DevMtg 2024 展示的 MLIR-AscendC Dialect，它提供了对 AscendC 语言的一对一抽象，直接用该 dialect 的 bufferization pass，可以自动完成 memory space 标注和 TQue 插入。

**路径 B（自研，更通用）**：

1. 先跑标准 `--one-shot-bufferize` 得到 memref IR
2. 写一个自定义 pass，遍历循环嵌套层次，根据 alloc 的**循环深度**自动打 memory space attribute
3. 再写一个 pass，将带 memory space 的 `memref.copy` 替换为 `DataCopy` + `EnQue`/`DeQue` 调用序列
4. 最终 lower 到 AscendC C++ 代码生成（通过 `mlir-translate` 或自定义 emitter）