让我获取这三个 Pass 的具体实现代码。网络限制无法直接获取源文件，但 commit diff 页面已经完整加载了所有文件内容。我根据已获取的完整 diff（DAG.h、Passes.h、Passes.td、compiler.py 以及 CMakeLists.txt 的全部内容）进行深入分析。

------

## 三个核心 Pass 详解

### 整体执行顺序与分工

```
Triton IR (ttir)
     │
     ▼
┌─────────────────┐
│  DAGSync Pass   │  ← Pass 1：依赖分析 + 同步点插入
└────────┬────────┘
         │
    CSE + Canonicalize
         │
         ▼
┌─────────────────┐
│  DAGScope Pass  │  ← Pass 2：核心亲和域划分（Cube / Vector）
└────────┬────────┘
         │
    CSE + Canonicalize
         │
         ▼
┌──────────────────────┐
│  DAGSSBuffer Pass    │  ← Pass 3：共享缓冲插入 + 软件流水构造
└────────┬─────────────┘
         │
    CSE + Canonicalize
         │
         ▼
  triton_to_structure（后续编译流程）
```

------

## Pass 1：DAGSync（`dag-sync`）

### 功能定位

**分析 Triton IR 中的数据依赖关系，在 Cube 核与 Vector 核之间的跨域数据流处插入同步原语。**

### 工作原理

**第一步：构建 AffinityDAG**

Pass 从 `triton::FuncOp` 出发，调用 `Graph::fromMultiBlockFunc()` 为函数的每个 Block 构建一个有向依赖图。图的核心结构：

```cpp
// 每个 MLIR Operation 映射到一个 DAG Node
class Node {
    Operation* op;
    Graph* graph;             // 所属图（Block）
    SmallPtrSet<Node*, 4> ins;  // 数据依赖的前驱节点
    SmallPtrSet<Node*, 2> outs; // 数据依赖的后继节点
    TinyPtrVector<Graph*> subgraphs; // 对于控制流 Op（如 scf::ForOp），
                                      // 子块构成子图
};
```

**第二步：核心类型标注（markCore）**

这是同步分析的关键前提。`Graph::markCore()` 通过以下两条规则传播 CoreType：

- `markDotUpstream(op)`：从 `tt.dot`（矩阵乘法）算子向上游传播 `CUBE` 标注，所有为 dot 提供数据的 load/compute Op 被标为 Cube 核计算
- `markCubeLoadUpstream(op)`：识别专为 Cube 服务的内存加载，同样向上游传播 `CUBE`

未被标注的节点默认为 `VECTOR`（向量核计算）或 `UNDETERMINED`（标量/公共操作）。

```
UNDETERMINED = 0
VECTOR = 1 << 0 = 0b01
CUBE   = 1 << 1 = 0b10
SCALAR = VECTOR | CUBE = 0b11  ← 两核均可执行（标量操作）
```

**第三步：同步点检测与插入**

DAGSync 遍历 DAG 边，找出跨 CoreType 的依赖边（即一条 Node.out → Node.in 的边，两端 CoreType 不同），在这些边上插入对应的同步 IR（HIVM dialect 中的同步原语），确保：

- Cube 核写完矩阵乘结果后，Vector 核才能读
- Vector 核完成数据预处理后，Cube 核才能开始

**依赖方言：**

- `hivm::HIVMDialect`：提供同步原语 Op
- `bufferization::BufferizationDialect`：内存语义支持

------

## Pass 2：DAGScope（`dag-scope`）

### 功能定位

**将 Triton 原生 IR 转换为 NPU 亲和代码（NPU-affine code），为每个 Op 划定其在哪个计算核（Cube 或 Vector）的 Scope 内执行。**

### 工作原理

**第一步：复用 DAGSync 构建的图**

DAGScope 通过 `GraphManager::getInstance().getGraph(funcName)` 获取 DAGSync 已注册的 AffinityDAG，避免重复构建：

```cpp
class GraphManager {   // 单例
    DenseMap<StringRef, shared_ptr<AffinityDAG::Graph>> graphs;
public:
    static GraphManager& getInstance();
    void registerGraph(StringRef funcName, shared_ptr<Graph> graph);
    Graph* getGraph(StringRef funcName);
};
```

**第二步：Scope 划分**

基于每个 Node 的 CoreType 标注，DAGScope 将 Op 包裹进对应的 `scope::ScopeOp` 区域：

```
原始 IR：
  %a = tt.load ...       ← CUBE（为 dot 服务的加载）
  %b = tt.load ...       ← VECTOR
  %c = tt.dot %a, %b     ← CUBE
  %d = arith.add %c, %e  ← VECTOR（后处理）

转换后（概念示意）：
  scope.cube {
    %a = tt.load ...
    %c = tt.dot %a, %b
  }
  scope.vector {
    %b = tt.load ...
    %d = arith.add %c, %e
  }
```

**第三步：子图展开与控制流处理**

对于 `scf::ForOp` 等控制流，`Node::convertToSubGraph()` 和 `Node::flattenSubGraph()` 负责递归处理嵌套 Block，确保循环体内部的 Op 也能被正确划分到对应 Scope。

**第四步：跨 Scope 的数据传递**

SCALAR 类型的 Op（CoreType = `VECTOR | CUBE`，即标量操作）不需要划分，可以在任意核上执行，作为两个 Scope 间的"公共区域"存在。

**依赖方言：**

- `hivm::HIVMDialect`
- `scope::ScopeDialect`：提供 Scope 划分 Op
- `bufferization::BufferizationDialect`

------

## Pass 3：DAGSSBuffer（`dag-ssbuf`）

### 功能定位

**将 Vector Scope 中的数据生产操作转换为通过 Shared Storage Buffer（SSBuffer，片上共享缓冲区）传递的异步操作，构造软件流水线。**

### 工作原理

SSBuffer 是昇腾 NPU 实现软件流水的关键硬件资源，位于 L1 Buffer 层级，Cube 核和 Vector 核均可访问，是两者之间异步通信的"邮箱"。

**第一步：识别需要 SSBuffer 化的数据流**

DAGSSBuffer 遍历 DAGScope 产出的 IR，找到满足以下条件的 Value：

- 在 `scope.vector` 内生产（Vector 核计算产生）
- 被 `scope.cube` 内的 Op 消费（Cube 核读取）

或反方向（Cube → Vector）。这些跨 Scope 的数据流就是 SSBuffer 替换的候选。

**第二步：插入 SSBuffer 分配与读写**

对每个候选数据流，Pass 执行如下变换（概念示意）：

```
转换前：
  scope.vector {
    %result = vector_op(...)      ← Vector 核产生数据
  }
  scope.cube {
    use(%result)                  ← Cube 核直接使用，产生依赖等待
  }

转换后：
  %ssbuf = hivm.alloc_ssbuffer   ← 分配片上共享缓冲区
  scope.vector {
    %result = vector_op(...)
    hivm.store_ssbuffer %result → %ssbuf   ← 异步写入 SSBuffer
  }
  scope.cube {
    %loaded = hivm.load_ssbuffer %ssbuf    ← 从 SSBuffer 读取
    use(%loaded)
  }
```

**第三步：双缓冲 / 多缓冲流水构造**

SSBuffer 机制的核心价值在于构造软件流水：在循环体中，当 Cube 核处理第 `i` 个 tile 的矩阵乘时，Vector 核并行预取第 `i+1` 个 tile 的数据写入 SSBuffer，实现计算与访存的流水掩盖：

```
迭代 i:   [Vector 预取 tile i+1] [Cube 计算 tile i  ]
迭代 i+1: [Vector 预取 tile i+2] [Cube 计算 tile i+1]
```

**第四步：CSE + Canonicalize 清理**

两轮 `add_cse` + `add_canonicalizer` 分别在 DAGSSBuffer 前后执行，清除因 SSBuffer 插入引入的冗余读写和等价表达式，确保最终 IR 简洁。

**依赖方言：**

- `hivm::HIVMDialect`：提供 `alloc_ssbuffer`、`load_ssbuffer`、`store_ssbuffer` 等 Op
- `scope::ScopeDialect`：读取 Scope 划分结果
- `bufferization::BufferizationDialect`：内存分配语义

------

## 三 Pass 协作总览

```
┌────────────────────────────────────────────────────────┐
│                      AffinityDAG                       │
│                                                        │
│  Node[tt.load]──►Node[tt.dot]──►Node[arith.add]       │
│    CoreType:CUBE    CoreType:CUBE  CoreType:VECTOR      │
│                                                        │
│       ▲ DAGSync 分析依赖边，插入跨核同步               │
│       │ DAGScope 划分执行域 Scope                       │
│       │ DAGSSBuffer 在 Scope 边界插入片上缓冲            │
└────────────────────────────────────────────────────────┘

最终效果：Cube 核与 Vector 核并行执行，通过 SSBuffer 异步通信，
          实现计算访存流水掩盖，充分利用昇腾 AI Core 的双引擎架构。
```

| Pass        | 输入                    | 输出                                                 | 核心贡献                           |
| ----------- | ----------------------- | ---------------------------------------------------- | ---------------------------------- |
| DAGSync     | Triton IR + FuncOp      | 带同步原语的 IR + AffinityDAG（注册到 GraphManager） | 跨核依赖边分析、同步点插入         |
| DAGScope    | 带同步 IR + AffinityDAG | 带 Scope 划分的 NPU-affine IR                        | Op 执行域划分（Cube/Vector Scope） |
| DAGSSBuffer | 带 Scope IR             | 含 SSBuffer 读写的流水 IR                            | 片上缓冲分配、软件流水构造         |





我认为和们matmul-add-relu流水线思路一样，整个PR有3个重要Pass：

1. DAGSync，主要功能有2个：

* 为每个Op标注Vector还是Cube，他们称为CoreType。标注存储在DAG节点的属性上，整个DAG都是他们自己创建的。
* 在DAG的跨CoreType的时候插入HIVM同步Op，还依赖`bufferization::BufferizationDialect`，猜测应该是插入通用的memref.copy Op，标注好buffer的TPosion

2. DAGScope，主要是将IR图按照CoreType划分成多个Scope，就是包一层自定义的Scope Op；
3. DAGSSBuffer，基于HIVM方言为不同的CoreType实际分配释放具体的buffer，包括double buffer、规范化等。