# LinalgTransformOps.cpp 技术原理详解

## 一、整体架构与设计目标

**LinalgTransformOps** 是 MLIR Transform Dialect 的核心组件，实现了约 30+ 个针对线性代数操作（Linalg ops）的可组合变换操作。其核心设计理念：

1. **分离关注点**：变换逻辑与被变换的 IR 解耦
2. **声明式编程**：通过 Transform Operations 描述优化序列
3. **可组合性**：多个变换可链式组合
4. **可追踪性**：通过 Transform State 系统跟踪变换结果

## 二、核心技术机制

### 1. 双层应用模式（Two-Tiered Application Pattern）

所有变换操作遵循统一模式：

```
apply()/applyToOne() [协调层]
    ↓
linalg::transformFunction() [实现层]
    ↓
返回变换后的操作集合
```

**单操作模式**示例（FuseOp）：
```cpp
DiagnosedSilenceableFailure FuseOp::apply(
    TransformRewriter &rewriter,
    TransformResults &transformResults,
    TransformState &state) {

  // 委托给底层 tileConsumerAndFuseProducersUsingSCF
  auto result = tileConsumerAndFuseProducersUsingSCF(
      rewriter, tilingInterfaceOp, tileAndFuseOptions);

  // 设置结果句柄
  transformResults.set(getFusedOp().cast<OpResult>(),
                       {result.tiledAndFusedOps.front()});
}
```

**迭代模式**示例（VectorizeOp）：
```cpp
DiagnosedSilenceableFailure VectorizeOp::applyToOne(
    transform::TransformRewriter &rewriter,
    LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  // 对单个 payload 操作应用
  FailureOr<VectorizationResult> vectorResults =
      linalg::vectorize(rewriter, target, vectorSizes, ...);
}
```

### 2. 参数解析系统（Mixed Static-Dynamic Parameters）

支持三种参数类型：
- **静态属性**（Static Attributes）：编译期常量
- **Transform 参数**（Transform Parameters）：运行时值
- **操作句柄**（Operation Handles）：指向 payload 操作的引用

核心工具函数 `unpackSingleIndexResultPayloadOperations()`：
```cpp
// 统一处理三种形式：
// 1. IntegerAttr 直接值
// 2. 参数句柄 (ParamType)
// 3. 操作句柄 (必须返回单个 index 类型结果)
DiagnosedSilenceableFailure unpackSingleIndexResultPayloadOperations(
    TransformState &state,
    TransformOpInterface transformOp,
    SmallVector<OpFoldResult> &result,
    ArrayRef<OpFoldResult> ofrs)
```

### 3. 新操作追踪机制（NewOpsListener）

通过监听器模式跟踪变换过程中创建的操作：

```cpp
class NewOpsListener : public RewriterBase::ForwardingListener {
  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint previous) {
    if (previous.isSet()) return;  // 跳过非新创建操作
    newOps.insert(op);
  }

  void notifyOperationErased(Operation *op) {
    op->walk([&](Operation *op) { newOps.erase(op); });
  }

  DenseSet<Operation *> newOps;  // 保存所有新创建的操作
};
```

这对于 `BufferizeToAllocationOp` 等需要返回辅助操作句柄的变换至关重要。

### 4. 容错机制（DiagnosedSilenceableFailure）

所有变换返回可静默的失败类型，支持优雅降级：

```cpp
DiagnosedSilenceableFailure VectorizeOp::apply(...) {
  if (!linalg::hasVectorizationImpl(target)) {
    return mlir::emitSilenceableFailure(target->getLoc())
           << "不支持的操作类型，无法向量化";
  }

  FailureOr<VectorizationResult> vectorResults =
      linalg::vectorize(rewriter, target, vectorSizes, ...);

  if (failed(vectorResults)) {
    return mlir::emitSilenceableFailure(target->getLoc())
           << "向量化尝试失败";
  }

  return DiagnosedSilenceableFailure::success();
}
```

### 5. 模式重写集成（Pattern-Based Rewriting）

泛型模板实现类型安全的模式应用：

```cpp
template <typename PatternTy, typename... Args>
static FailureOr<LinalgOp> tryApply(Operation *operation, Args &&...args) {
  auto op = dyn_cast<OpTy>(operation);
  if (!op) return failure();

  PatternTy pattern(operation->getContext(), std::forward<Args>(args)...);
  TrivialPatternRewriter rewriter(operation->getContext());
  auto result = pattern.returningMatchAndRewrite(op, rewriter);

  return cast<LinalgOp>(result->getOperation());
}
```

## 三、关键变换操作分类

### 结构变换类
- **TileUsingForallOp**：并行循环切分（scf.forall）
- **FuseOp**：切分并融合消费者-生产者对
- **SplitReductionOp**：规约维度分割以并行化

### 数据布局优化类
- **PackOp/PackGreedilyOp**：数据打包（布局优化）
- **PadOp**：填充操作以支持对齐/向量化
- **PromoteOp**：提升操作数到更快的内存空间（如 GPU 共享内存）
- **HoistPadOp**：循环外提升填充操作

### 降级与分解类
- **VectorizeOp**：向量化为 vector.transfer 操作
- **DecomposeOp**：分解实现 AggregatedOpInterface 的操作
- **ConvertToLoopsOp**：降级为显式 scf.for 循环
- **GeneralizeOp/SpecializeOp**：通用/特化形式转换

### 卷积专用优化
- **ConvertConv2DToImg2ColOp**：转换为 im2col 格式
- **WinogradConv2DOp/DecomposeWinogradOp**：Winograd 算法变换
- **TransposeConv2DOp**：卷积布局转换

## 四、设计模式应用

### 1. 模板方法模式
所有操作遵循：`apply()` → 委托专用函数 → 返回结果

### 2. 策略模式
通过选项结构配置变换行为：
```cpp
scf::SCFTilingOptions tilingOptions;
tilingOptions.interchangeVector = tileInterchange;
scf::SCFTileAndFuseOptions tileAndFuseOptions;
tileAndFuseOptions.tilingOptions = tilingOptions;
```

### 3. 类型分派模式
使用 TypeSwitch 实现操作特化：
```cpp
auto maybeTransformed =
    TypeSwitch<Operation *, FailureOr<...>>(target)
        .Case<Conv2DNhwcHwcfOp>([&](auto op) {
            return rewriteInIm2Col(rewriter, op);
        })
        .Default([&](Operation *op) {
            return emitSilenceableFailure(op) << "不支持的卷积类型";
        });
```

### 4. 构建器模式
自定义 build() 方法简化操作构造：
```cpp
void TileReductionUsingForOp::build(
    OpBuilder &builder, OperationState &result,
    Value target, ArrayRef<int64_t> staticTileSizes)
```

## 五、关键集成点

### TilingInterface 集成
- 检查目标是否实现 `TilingInterface`
- 实现跨不同操作类型的泛型切分
- 核心可扩展性抽象点

### Transform State 管理
```
TransformState 映射：
  Handle Values → Payload Operations
  Param Values → Attribute values
```

### 内存副作用追踪
```cpp
void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
```

支持变换的优化和依赖分析。

## 六、架构优势

1. **可组合性**：通过句柄数据流自然链接
2. **可扩展性**：新变换可复用现有工具
3. **类型安全**：模板和 TypeSwitch 编译期检查
4. **调试友好**：错误消息包含源位置和上下文
5. **优雅降级**：可静默失败支持部分变换

## 七、实现亮点

- **混合参数处理**：编译时/运行时切分大小统一处理
- **清理模式**：变换后自动应用规范化（如 FuseOp 的 `apply_cleanup`）
- **GPU 内存映射**：PromoteOp 特殊处理 `gpu::GPUMemorySpaceMappingAttr`
- **部分规约策略**：TileReductionOp 使用 `PartialReductionOuterReduction` 策略
- **Winograd 支持**：专用卷积优化路径

## 八、关键类与函数索引

### 核心变换操作类

| 操作类 | 文件位置 | 功能描述 | 关键参数 |
|--------|---------|---------|---------|
| **TileUsingForallOp** | LinalgTransformOps.cpp:XXX | 并行循环切分 | tile_sizes, num_threads, mapping |
| **FuseOp** | LinalgTransformOps.cpp:XXX | 切分并融合 | tile_sizes, tile_interchange |
| **PackOp** | LinalgTransformOps.cpp:XXX | 数据打包 | packed_sizes |
| **VectorizeOp** | LinalgTransformOps.cpp:XXX | 向量化 | vector_sizes, scalable_sizes |
| **PadOp** | LinalgTransformOps.cpp:XXX | 填充操作 | padding_dimensions, padding_values |
| **PromoteOp** | LinalgTransformOps.cpp:XXX | 内存提升 | operands_to_promote, memory_space |

### 核心辅助函数

- **applyTilingToAll()**: 协调多操作的切分和融合
- **tileToForallOpImpl()**: forall 切分核心实现
- **unpackSingleIndexResultPayloadOperations()**: 混合参数解析
- **reifyMixedParamAndHandleResults()**: 参数/句柄具体化
- **tryApply\<PatternTy\>()**: 泛型模式应用

## 九、典型变换流程示例

### 示例 1: TileUsingForallOp 切分流程

```
1. 用户代码:
   transform.structured.tile_using_forall %target tile_sizes [32, 32]

2. TileUsingForallOp::apply() 调用
   ├─ 解析 tile_sizes 参数
   ├─ 构造 scf::SCFTilingOptions
   └─ 调用 tileToForallOpImpl()

3. tileToForallOpImpl() 执行
   ├─ 检查 TilingInterface 实现
   ├─ 调用 scf::tileUsingSCFForallOp()
   └─ 返回 TilingResult

4. 设置变换结果
   └─ transformResults.set(getTiledOp(), {tilingResult.tiledOps})
```

### 示例 2: FuseOp 融合流程

```
1. 用户代码:
   transform.structured.fuse %target tile_sizes [8, 8]

2. FuseOp::apply() 调用
   ├─ 解析切分参数和融合选项
   ├─ 构造 SCFTileAndFuseOptions
   └─ 调用 tileConsumerAndFuseProducersUsingSCF()

3. 融合执行
   ├─ 切分消费者操作
   ├─ 识别生产者操作
   ├─ 逐个融合生产者到消费者循环
   └─ 可选应用清理模式

4. 返回融合结果
   ├─ fused_op: 主要融合操作
   ├─ new_producer_ops: 新创建的生产者
   └─ loops: 生成的循环嵌套
```

## 十、扩展指南

### 添加新的变换操作

1. **定义 ODS 规范** (LinalgTransformOps.td):
```tablegen
def MyTransformOp : Op<Transform_Dialect, "linalg.my_transform",
    [TransformOpInterface, TransformEachOpTrait]> {
  let arguments = (ins TransformHandleTypeInterface:$target,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$my_params);
  let results = (outs TransformHandleTypeInterface:$transformed);
}
```

2. **实现 apply/applyToOne 方法** (LinalgTransformOps.cpp):
```cpp
DiagnosedSilenceableFailure MyTransformOp::applyToOne(
    transform::TransformRewriter &rewriter,
    LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  // 实现变换逻辑
  FailureOr<Operation*> transformed = myTransformFunction(rewriter, target);

  if (failed(transformed))
    return emitSilenceableFailure(target) << "变换失败";

  results.push_back(*transformed);
  return DiagnosedSilenceableFailure::success();
}
```

3. **实现内存副作用追踪**:
```cpp
void MyTransformOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getTransformed(), effects);
  modifiesPayload(effects);
}
```

## 十一、调试技巧

### 1. 启用变换追踪
```bash
mlir-opt --debug-only=transform-dialect input.mlir
```

### 2. 使用 transform.print 调试
```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
  transform.print %0 {name = "Matched ops"}
  %1 = transform.structured.tile_using_forall %0 tile_sizes [32, 32]
  transform.print %1 {name = "After tiling"}
}
```

### 3. 检查变换失败原因
变换操作会返回详细的失败诊断信息，包含源位置和失败原因。

## 十二、性能考虑

1. **切分大小选择**：影响缓存利用率和并行度
2. **融合深度**：过度融合可能增加寄存器压力
3. **向量化宽度**：应匹配目标硬件的 SIMD 宽度
4. **内存提升**：平衡共享内存使用和全局内存访问

## 十三、参考资源

- **源文件位置**: `mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp`
- **ODS 定义**: `mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td`
- **接口定义**: `mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h`
- **相关文档**: [MLIR Transform Dialect Documentation](https://mlir.llvm.org/docs/Dialects/Transform/)

---

这套架构代表了成熟的声明式编译器变换系统，具有强大的组合性和可扩展性，适合构建领域特定的 IR 变换框架。
