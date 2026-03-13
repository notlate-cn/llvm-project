## 面向存储层级感知的AI编译器 - 端到端Demo

## 核心技术

#### 多面体调度分析

利用多面体方法分析循环迭代空间，生成合法循环调度，确保依赖关系正确

#### Tile-based kernel 映射

将计算按 tile 分块映射到 UB / L2 / HBM 等存储层级，实现硬件感知的数据复用与高效访问

#### 容量约束与阶段延迟优化

构建 cost model，对存储容量约束进行求解，同时优化各阶段（stage）的延迟，实现整体性能平衡



## 端到端流程

原始计算图：`Matmul` + `Add` + `Relu`

计算逻辑是：

```python
Output = Relu( Matmul(A, B) + Bias )
```

#### 1. 基于Linalg方言表达的原始计算图

```llvm
// 文件: fc_add_relu.mlir
// 函数: fc_relu
// 功能: 全连接层 + 偏置 + ReLU 激活函数

func.func @fc_relu(%lhs: tensor<?x?xf32>,
                   %rhs: tensor<?x?xf32>,
                   %bias: tensor<?x?xf32>,
                   %output: tensor<?x?xf32>)
                   -> tensor<?x?xf32> {
  // 操作 1: 矩阵乘法
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 操作 2: 加偏置 (逐元素加法)
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul, %bias : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 操作 3: ReLU (逐元素 max(x, 0))
  %c0f = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %biased, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %biased, %c1 : tensor<?x?xf32>
  %zero_tensor = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %filled_zero = linalg.fill ins(%c0f : f32) outs(%zero_tensor : tensor<?x?xf32>) -> tensor<?x?xf32>
  %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %filled_zero : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  return %relued : tensor<?x?xf32>
}
```

#### 2. TileAndFuse，确定搬运时机，确定计算单元

本阶段完全基于Transform Dialect方言实现（复用已有能力），只需要自动生成Transform脚本即可。

```llvm
// 文件: transform_tile_and_fuse_3level.mlir (Transform Dialect 脚本)
// 函数: __transform_main
// 功能: 3级循环分块 + 融合 + 确定搬运时机 + 确定计算单元
// ============================================================
// 设计原则:
//   transform 脚本只携带调度策略，硬件拓扑知识集中在后续的自定义Pass(AscendCBufferPlacementPass)里。
//
//   携带的信息:
//     1. 循环结构: tile sizes (TB_M/TB_N/Tb_M/Tb_N/t_K) + 嵌套层次
//     2. 分核标注: ascendc.parallel = true
//                  标注 for_TB_M / for_TB_N 为分核循环
//                  后端 lowering 将其映射到多 AiCore 并行
//     3. 循环级搬运时机:
//          ascendc.prologue: 循环入口（或 AiCore 入口）执行的搬运
//          ascendc.epilogue: 循环出口（或 AiCore 出口）执行的搬运
//          格式: "角色:路径,角色:路径,..."（逗号分隔，支持多目标）
//          时机: 挂在哪层 for 就在那层入口/出口执行
//          粒度: 由该循环的 tile size 自然决定
//     4. Op执行单元
//          格式：ascendc.unit = "计算单元"
//          单元枚举：
//          AiCore.Cube: 矩阵乘法
//          AiCore.Vector: 向量加法/ReLU/累加
//          AiCpu: AiCpu执行单元
//
// 分核与搬运的关系:
//   for_TB_M × for_TB_N 共同构成分核空间（ascendc.parallel = true）
//   每个 AiCore 负责输出矩阵的一个 [TB_M × TB_N] 块，
//   对应唯一的 A[TB_M × K] 和 B[K × TB_N] 子块，互不重叠。
//   因此 GM->A1/B1 在分核后、Tb 循环前执行（挂在 for_TB_N.prologue），
//   每个 AiCore 独立搬运自己负责的数据，无重复搬运。
//
// 最终循环结构及 annotation 分布:
//
//   scf.for %TB_M {ascendc.parallel = true}
//     scf.for %TB_N {ascendc.parallel = true,
//                    prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
//                    epilogue = "result:VECOUT->GM"}
//       scf.for %Tb_M
//         scf.for %Tb_N
//           scf.for %K {prologue = "lhs:A1->A2,rhs:B1->B2",
//                        epilogue = "acc:CO1->VECIN"}
//             linalg.matmul          // unit 由 pass 推导: AiCore.Cube
//           linalg.elementwise add   // unit 由 pass 推导: AiCore.Vector
//           linalg.elementwise max   // unit 由 pass 推导: AiCore.Vector
//
// 搬运时机说明:
//   for_TB_N.prologue (GM->A1/B1/VECIN):
//     分核后每个 AiCore 的入口，搬运 TB_M×K / K×TB_N / TB_M×TB_N 粒度数据
//   for_TB_N.epilogue (VECOUT->GM):
//     每个 AiCore 完成全部 Tb 计算后写回，TB_M×TB_N 粒度
//   for_K.prologue (A1->A2 / B1->B2):
//     每次 K 迭代搬运 [Tb_M×t_K] / [t_K×Tb_N] 粒度的子块进 L0A/L0B
//   for_K.epilogue (CO1->VECIN):
//     K 轴所有迭代完成，CO1 是完整的 Tb_M×Tb_N 累加结果，
//     立刻 FixpipeOp 搬入 UB，供后续 add/max 消费
//
// AscendCBufferPlacementPass 推导规则:
//   Step A: 解析 prologue/epilogue，建立各层搬运任务表
//   Step B: 识别 ascendc.parallel = true 的循环为分核边界
//           prologue/epilogue 对应每个 AiCore 的入口/出口搬运
//   Step C: 按 op 类型推导执行单元
//             linalg.matmul      → AiCore.Cube  → lhs:A2, rhs:B2, out:CO1
//             linalg.elementwise → AiCore.Vector
//               ins 来自 CO1 epilogue 目标 → VECIN
//               ins 来自 GM prologue 目标  → VECIN
//               中间结果（有后续 Vector 消费者）→ VECCALC
//               最终输出（无后续 Vector 消费者）→ VECOUT
//   Step D: 插入显式搬运 op
//             GM->A1/B1:   ascendc.copy(GlobalTensor → LocalTensor<A1/B1>)
//             GM->VECIN:   ascendc.copy(GlobalTensor → LocalTensor<VECIN>)
//             A1->A2:      ascendc.copy(LocalTensor<A1> → LocalTensor<A2>)
//             B1->B2:      ascendc.copy(LocalTensor<B1> → LocalTensor<B2>)
//             CO1->VECIN:  ascendc.fixpipe(LocalTensor<CO1> → LocalTensor<VECIN>)
//             VECOUT->GM:  ascendc.copy(LocalTensor<VECOUT> → GlobalTensor)
// ============================================================

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}
  ) {

    // ----------------------------------------------------------------
    // Step 1: 匹配 func.func
    // ----------------------------------------------------------------
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op

    // ----------------------------------------------------------------
    // Step 2: add_index_args — 追加5个 index 参数
    //   顺序: TB_M, TB_N, Tb_M, Tb_N, t_K
    // ----------------------------------------------------------------
    %func_new, %TB_M, %TB_N, %Tb_M, %Tb_N, %t_K =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ----------------------------------------------------------------
    // Step 3: 匹配原始 linalg ops（在任何 tiling 之前）
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // IR 顺序: #0=add, #1=max
    %add, %max = transform.split_handle %elementwise
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // 预定义 param 常量
    // ----------------------------------------------------------------

    // 分核标注
    %p_true = transform.param.constant true -> !transform.any_param

    // for_TB_N 搬运时机（分核后每个 AiCore 的入口/出口）:
    //   prologue: 搬运该 AiCore 负责的 A/B/bias 子块到 L1/UB
    //   epilogue: 将计算结果从 UB 写回 GM
    %p_TB_prologue = transform.param.constant
        "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
        -> !transform.any_param
    %p_TB_epilogue = transform.param.constant
        "result:VECOUT->GM"
        -> !transform.any_param

    // for_K 搬运时机:
    //   prologue: 每次 K 迭代将 t_K 宽度的 A/B 子块从 L1 搬入 L0A/L0B
    //   epilogue: K 轴所有迭代完成，FixpipeOp 将完整累加结果从 L0C 搬入 UB
    %p_K_prologue = transform.param.constant
        "lhs:A1->A2,rhs:B1->B2"
        -> !transform.any_param
    %p_K_epilogue = transform.param.constant
        "acc:CO1->VECIN"
        -> !transform.any_param

    // Op 执行单元
    %p_cube   = transform.param.constant "AiCore.Cube"   -> !transform.any_param
    %p_vector = transform.param.constant "AiCore.Vector" -> !transform.any_param

    // ================================================================
    // 第一轮 TileAndFuse：TB 层
    // ================================================================

    // ----------------------------------------------------------------
    // Step 4: tile max [TB_M, TB_N] → for_TB_M / for_TB_N
    // ----------------------------------------------------------------
    %tiled_max_TB, %for_TB_M, %for_TB_N =
        transform.structured.tile_using_for %max
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse add into for_TB_N
    // ----------------------------------------------------------------
    %add_fused_TB, %loop_add_TB =
        transform.structured.fuse_into_containing_op %add into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 6: fuse matmul into for_TB_N
    // ----------------------------------------------------------------
    %matmul_fused_TB, %loop_matmul_TB =
        transform.structured.fuse_into_containing_op %matmul into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_TB_split:3 = transform.split_handle %matmul_fused_TB
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ★ 分核标注: for_TB_M / for_TB_N 共同构成分核空间
    //   AscendCBufferPlacementPass 识别这两层为分核边界
    //   后端 lowering 将其映射到多 AiCore 并行执行
    transform.annotate %for_TB_M "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param

    // ★ 搬运时机: for_TB_N（分核后每个 AiCore 的入口/出口）
    transform.annotate %for_TB_N "ascendc.prologue"
        = %p_TB_prologue : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.epilogue"
        = %p_TB_epilogue : !transform.any_op, !transform.any_param

    // ================================================================
    // 第二轮 TileAndFuse：Tb 层
    // ================================================================

    // ----------------------------------------------------------------
    // Step 7: tile tiled_max_TB [Tb_M, Tb_N] → for_Tb_M / for_Tb_N
    // ----------------------------------------------------------------
    %tiled_max_Tb, %for_Tb_M, %for_Tb_N =
        transform.structured.tile_using_for %tiled_max_TB
            tile_sizes [%Tb_M, %Tb_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 8: fuse add_fused_TB into for_Tb_N
    // ----------------------------------------------------------------
    %add_fused_Tb, %loop_add_Tb =
        transform.structured.fuse_into_containing_op %add_fused_TB into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 9: fuse matmul into for_Tb_N
    // ----------------------------------------------------------------
    %matmul_fused_Tb, %loop_matmul_Tb =
        transform.structured.fuse_into_containing_op %matmul_TB_split#0 into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_Tb_split:3 = transform.split_handle %matmul_fused_Tb
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 10: tile matmul [0, 0, t_K] → for_K
    // ----------------------------------------------------------------
    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_Tb_split#0
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_final = transform.structured.match ops{["linalg.matmul"]} in %for_K
        : (!transform.any_op) -> !transform.any_op
    transform.annotate %matmul_final "ascendc.unit"
        = %p_cube : !transform.any_op, !transform.any_param

    transform.annotate %add_fused_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param

    // ★ 搬运时机: for_K
    //   prologue: 每次 K 迭代将 [Tb_M×t_K]/[t_K×Tb_N] 子块搬入 L0A/L0B
    //   epilogue: K 轴完成后 FixpipeOp CO1→VECIN，供 add/max 消费
    transform.annotate %for_K "ascendc.prologue"
        = %p_K_prologue : !transform.any_op, !transform.any_param
    transform.annotate %for_K "ascendc.epilogue"
        = %p_K_epilogue : !transform.any_op, !transform.any_param

    // ----------------------------------------------------------------
    // Step 11: hoist_loop_invariant_subsets
    //   由内向外提升循环不变切片:
    //   for_Tb_N: 提升不依赖 iv_Tb_N 的切片到 for_Tb_M 内
    //   for_Tb_M: 继续提升不依赖 iv_Tb_M 的切片到 for_TB_N 内
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
```

执行命令如下：

```shell
# afir-opt工具的构建路径
export PATH=$PATH:/home/niu/code/Ascend-MLIR/build/bin

# 转换命令，命令结果同时打屏和落盘到 output_step1_tile_and_fuse_3level.mlir 文件
afir-opt fc_add_relu.mlir --transform-preload-library=transform-library-paths=transform_tile_and_fuse_3level.mlir --transform-interpreter=entry-point=__transform_main --canonicalize --cse | tee output_step1_tile_and_fuse_3level.mlir
```

结果经过注释和可读性重命名如下：

```llvm
#tile_guard = affine_map<(d0)[upper, tile] -> (-d0 + upper, tile)>

// ============================================================================
//  Fully Connected + Bias + ReLU
//
//  ================= 核心技术 =================
//
//  1. 多面体调度分析
//     利用多面体模型描述迭代域 D(i,j,k)
//     通过合法 schedule θ(i,j,k) 保证数据依赖正确
//     affine.min 实现 tile 边界合法化
//
//  2. Tile-based Kernel 映射
//     GM → Outer Tile (L2)
//         → Inner Tile (UB)
//             → K Tile (Register / Cube pipeline)
//     显式表达分层数据复用
//
//  3. 容量约束与阶段延迟优化
//     tileM_inner * tileK   <= UB_A_capacity
//     tileK * tileN_inner   <= UB_B_capacity
//     tileM_inner * tileN_inner <= UB_C_capacity
//     平衡 Cube 计算阶段与 Vector Epilogue 延迟
// ============================================================================

module {
  func.func @fc_relu(
      %A: tensor<?x?xf32>,        // 输入 A (M x K)
      %B: tensor<?x?xf32>,        // 输入 B (K x N)
      %Bias: tensor<?x?xf32>,     // Bias (M x N)
      %OutInit: tensor<?x?xf32>,  // 初始输出 (M x N)
      %tileM_outer_i64: i64,
      %tileN_outer_i64: i64,
      %tileM_inner_i64: i64,
      %tileN_inner_i64: i64,
      %tileK_i64: i64
  ) -> tensor<?x?xf32> {

    // ============================================================
    // 基础常量
    // ============================================================
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_f32 = arith.constant 0.000000e+00 : f32

    // ============================================================
    // tile 参数转换
    // ============================================================
    %tileK = arith.index_cast %tileK_i64 : i64 to index
    %tileN_inner = arith.index_cast %tileN_inner_i64 : i64 to index
    %tileM_inner = arith.index_cast %tileM_inner_i64 : i64 to index
    %tileN_outer = arith.index_cast %tileN_outer_i64 : i64 to index
    %tileM_outer = arith.index_cast %tileM_outer_i64 : i64 to index

    // ============================================================
    // 输出矩阵尺寸
    // ============================================================
    %M = tensor.dim %OutInit, %c0 : tensor<?x?xf32>
    %N = tensor.dim %OutInit, %c1 : tensor<?x?xf32>

    // 构造 ReLU 需要的 zero tensor
    %ZeroTensor = tensor.empty(%M, %N) : tensor<?x?xf32>
    %ZeroFilled = linalg.fill
        ins(%zero_f32 : f32)
        outs(%ZeroTensor : tensor<?x?xf32>) -> tensor<?x?xf32>

    // ============================================================
    // ===================== Outer Tile (L2 Blocking) =====================
    // 多面体调度第一层
    // ============================================================

    %OutAfterOuter =
    scf.for %m_outer = %c0 to %M step %tileM_outer
        iter_args(%Out_acc_outer = %OutInit)
        -> (tensor<?x?xf32>) {

      %OutAfterNOuter =
      scf.for %n_outer = %c0 to %N step %tileN_outer
          iter_args(%Out_acc_tile = %Out_acc_outer)
          -> (tensor<?x?xf32>) {

        // tile 边界保护
        %M_outer_size = affine.min #tile_guard(%m_outer)[%M, %tileM_outer]
        %N_outer_size = affine.min #tile_guard(%n_outer)[%N, %tileN_outer]

        %K = tensor.dim %A, %c1 : tensor<?x?xf32>

        // ---------------- GM → L2 tile ----------------

        %A_outer = tensor.extract_slice %A[%m_outer, 0][%M_outer_size, %K][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %B_outer = tensor.extract_slice %B[0, %n_outer][%K, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %Out_outer = tensor.extract_slice %Out_acc_tile[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %Bias_outer = tensor.extract_slice %Bias[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %Zero_outer = tensor.extract_slice %ZeroFilled[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

        // ============================================================
        // ===================== Inner Tile (UB Blocking) =================
        // ============================================================

        %OutAfterInner =
        scf.for %m_inner = %c0 to %M_outer_size step %tileM_inner iter_args(%Out_acc_inner = %Out_outer) -> (tensor<?x?xf32>) {

          %OutAfterNInner =
          scf.for %n_inner = %c0 to %N_outer_size step %tileN_inner iter_args(%Out_acc_block = %Out_acc_inner) -> (tensor<?x?xf32>) {

            %M_inner_size = affine.min #tile_guard(%m_inner)[%M_outer_size, %tileM_inner]
            %N_inner_size = affine.min #tile_guard(%n_inner)[%N_outer_size, %tileN_inner]

            %A_inner = tensor.extract_slice %A_outer[%m_inner, 0][%M_inner_size, %K][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %B_inner = tensor.extract_slice %B_outer[0, %n_inner][%K, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %Out_inner = tensor.extract_slice %Out_acc_block[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

            // ============================================================
            // ===================== K Blocking (Register/Cube) ============
            // ============================================================

            %OutAfterK =
            scf.for %k = %c0 to %K step %tileK iter_args(%Out_acc_k = %Out_inner) -> (tensor<?x?xf32>) {

              %K_tile_size = affine.min #tile_guard(%k)[%K, %tileK]

              %A_k = tensor.extract_slice %A_inner[0, %k][%M_inner_size, %K_tile_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %B_k = tensor.extract_slice %B_inner[%k, 0][%K_tile_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %Out_tile = tensor.extract_slice %Out_acc_k[0, 0][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

              // ---------------- Stage 1: Cube GEMM ----------------

              %MatmulTile = linalg.matmul {ascendc.unit = "AiCore.Cube"}
                ins(%A_k, %B_k : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%Out_tile : tensor<?x?xf32>) -> tensor<?x?xf32>

              %Out_acc_k_next = tensor.insert_slice %MatmulTile into %Out_acc_k[0, 0][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

              scf.yield %Out_acc_k_next : tensor<?x?xf32>
            } {ascendc.prologue = "lhs:A1->A2,rhs:B1->B2",
               ascendc.epilogue = "acc:CO1->VECIN"}

            // ---------------- Stage 2: Bias Add (Vector) ----------------

            %Bias_inner = tensor.extract_slice %Bias_outer[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

            %AddResult = linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"}
              ins(%OutAfterK, %Bias_inner : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out_inner : tensor<?x?xf32>) -> tensor<?x?xf32>

            // ---------------- Stage 3: ReLU (Vector) ----------------

            %Zero_inner = tensor.extract_slice %Zero_outer[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

            %ReluResult = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"}
              ins(%AddResult, %Zero_inner : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out_acc_block : tensor<?x?xf32>) -> tensor<?x?xf32>

            %Out_block_next = tensor.insert_slice %ReluResult into %Out_acc_block[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

            scf.yield %Out_block_next : tensor<?x?xf32>
          }
          scf.yield %OutAfterNInner : tensor<?x?xf32>
        }

        %Out_tile_next = tensor.insert_slice %OutAfterInner into %Out_acc_tile[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

        scf.yield %Out_tile_next : tensor<?x?xf32>
      } {ascendc.parallel = true,
         ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
         ascendc.epilogue = "result:VECOUT->GM"}

      scf.yield %OutAfterNOuter : tensor<?x?xf32>
    } {ascendc.parallel = true}

    return %OutAfterOuter : tensor<?x?xf32>
  }
}
```

#### 2. Bufferization







#### 6. Vectorization



#### 7. 生成Kernel



#### 8. 生成TilingFunc



#### 9. JIT优化

1. memref扩展硬件信息
2. memref联合affine优化
3. 扩展同步信息（直接用pyasc的op）

