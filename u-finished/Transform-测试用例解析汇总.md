# Transform方言测试用例解析汇总

**总文件数**: 38
**总用例数**: 345

---

# 1. Foreach操作测试

## 1.1 apply-foreach-nested.mlir

### 1.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --split-input-file --verify-diagnostics  --transform-interpreter
```

**用例输入:**

```mlir
func.func private @bar()

func.func @foo() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // expected-note @below {{ancestor payload op}}
  scf.for %i = %c0 to %c1 step %c10 {
    // expected-note @below {{descendant payload op}}
    scf.for %j = %c0 to %c1 step %c10 {
      func.call @bar() : () -> ()
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.op<"scf.for">
    %1 = transform.test_reverse_payload_ops %0 : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
    // expected-error @below {{transform operation consumes a handle pointing to an ancestor payload operation before its descendant}}
    // expected-note @below {{the ancestor is likely erased or rewritten before the descendant is accessed, leading to undefined behavior}}
    transform.test_consume_operand_each %1 : !transform.op<"scf.for">
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 1.1.2 case_2

**功能介绍:**

No error here, processing ancestors before descendants.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --split-input-file --verify-diagnostics  --transform-interpreter
```

**用例输入:**

```mlir
func.func private @bar()

func.func @foo() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c1 step %c10 {
    scf.for %j = %c0 to %c1 step %c10 {
      func.call @bar() : () -> ()
    }
  }
  return
}

// No error here, processing ancestors before descendants.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.op<"scf.for">
    transform.test_consume_operand_each %0 : !transform.op<"scf.for">
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @bar()
  func.func @foo() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    scf.for %arg0 = %c0 to %c1 step %c10 {
      scf.for %arg1 = %c0 to %c1 step %c10 {
        func.call @bar() : () -> ()
      }
    }
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.op<"scf.for">
      transform.test_consume_operand_each %0 : !transform.op<"scf.for">
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共22行，输出共21行
- transform.named_sequence定义被保留

---

## 1.2 foreach-match.mlir

### 1.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Silenceable diagnostics suppressed.
module attributes { transform.with_named_sequence } {
  func.func @test_loop_peeling_not_beneficial() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 40 : index
    %step = arith.constant 5 : index
    scf.for %i = %lb to %ub step %step {
      arith.addi %i, %i : index
    }
    return
  }

  transform.named_sequence @peel(%arg0: !transform.op<"scf.for"> {transform.consumed}) {
    transform.loop.peel %arg0 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @match_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["scf.for"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence  %root : !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      transform.foreach_match in %arg0
          @match_for -> @peel
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func @test_loop_peeling_not_beneficial() {
    %c0 = arith.constant 0 : index
    %c40 = arith.constant 40 : index
    %c5 = arith.constant 5 : index
    scf.for %arg0 = %c0 to %c40 step %c5 {
      %0 = arith.addi %arg0, %arg0 : index
    }
    return
  }
  transform.named_sequence @peel(%arg0: !transform.op<"scf.for"> {transform.consumed}) {
    %peeled_loop, %remainder_loop = transform.loop.peel %arg0 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield 
  }
  transform.named_sequence @match_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["scf.for"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%arg1: !transform.any_op):
      %updated_root = foreach_match in %arg1 
          @match_for -> @peel : (!transform.any_op) -> !transform.any_op
    }
    transform.yield 
  }
}


```

**重点说明:**

- 输入共31行，输出共27行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 1.2.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Silenceable diagnostics propagated.
module attributes { transform.with_named_sequence } {
  func.func @test_loop_peeling_not_beneficial() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 40 : index
    %step = arith.constant 5 : index
    // expected-note @below {{when applied to this matching payload}}
    scf.for %i = %lb to %ub step %step {
      arith.addi %i, %i : index
    }
    return
  }

  // expected-note @below {{failed to peel the last iteration}}
  transform.named_sequence @peel(%arg0: !transform.op<"scf.for"> {transform.consumed}) {
    transform.loop.peel %arg0 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @match_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["scf.for"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @main_suppress(%root: !transform.any_op) {
    transform.sequence  %root : !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      transform.foreach_match in %arg0
          @match_for -> @peel
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence  %root : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      // expected-error @below {{actions failed}}
      transform.foreach_match in %arg0
          @match_for -> @peel
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 1.2.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{op from within the matcher}}
module attributes { transform.with_named_sequence } {
  // expected-remark @below {{returned root}}
  func.func @foo() {
    return
  }

  transform.named_sequence @match_fail(
      %op: !transform.any_op {transform.readonly},
      %root: !transform.any_op {transform.readonly},
      %param: !transform.param<i64> {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    transform.test_succeed_if_operand_of_op_kind %op, "test.impossible_to_match" : !transform.any_op
    transform.yield %root, %param : !transform.any_op, !transform.param<i64>
  }

  transform.named_sequence @match_succeed(
      %op: !transform.any_op {transform.readonly},
      %root: !transform.any_op {transform.readonly},
      %param: !transform.param<i64> {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    transform.debug.emit_remark_at %root, "op from within the matcher" : !transform.any_op
    // expected-remark @below {{param from within the matcher 42}}
    transform.debug.emit_param_as_remark %param, "param from within the matcher" : !transform.param<i64>
    transform.yield %root, %param : !transform.any_op, !transform.param<i64>
  }

  transform.named_sequence @return(
      %root: !transform.any_op {transform.readonly},
      %param: !transform.param<i64> {transform.readonly}) -> (!transform.param<i64>, !transform.param<i64>, !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield %param, %param, %func : !transform.param<i64>, !transform.param<i64>, !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %param = transform.param.constant 42 : i64 -> !transform.param<i64>
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %func2, %yielded:3 = transform.foreach_match restrict_root in %func, %root, %param
      @match_fail -> @return,
      @match_succeed -> @return
      : (!transform.any_op, !transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.any_op)
    transform.debug.emit_remark_at %yielded#2, "returned root" : !transform.any_op
    // expected-remark @below {{42 : i64, 42 : i64}}
    transform.debug.emit_param_as_remark %yielded#0: !transform.param<i64>
    %num_roots = transform.num_associations %yielded#2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{2 : i64}}
    transform.debug.emit_param_as_remark %num_roots : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func @foo() {
    return
  }
  transform.named_sequence @match_fail(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}, %arg2: !transform.param<i64> {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    transform.test_succeed_if_operand_of_op_kind %arg0, "test.impossible_to_match" : !transform.any_op
    transform.yield %arg1, %arg2 : !transform.any_op, !transform.param<i64>
  }
  transform.named_sequence @match_succeed(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}, %arg2: !transform.param<i64> {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    transform.debug.emit_remark_at %arg1, "op from within the matcher" : !transform.any_op
    transform.debug.emit_param_as_remark %arg2, "param from within the matcher" : !transform.param<i64>
    transform.yield %arg1, %arg2 : !transform.any_op, !transform.param<i64>
  }
  transform.named_sequence @return(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.param<i64> {transform.readonly}) -> (!transform.param<i64>, !transform.param<i64>, !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield %arg1, %arg1, %0 : !transform.param<i64>, !transform.param<i64>, !transform.any_op
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.param.constant 42 : i64 -> !transform.param<i64>
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %updated_root, %yielded, %yielded_0, %yielded_1 = transform.foreach_match restrict_root in %1, %arg0, %0 
        @match_fail -> @return, 
        @match_succeed -> @return : (!transform.any_op, !transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.any_op)
    transform.debug.emit_remark_at %yielded_1, "returned root" : !transform.any_op
    transform.debug.emit_param_as_remark %yielded : !transform.param<i64>
    %2 = transform.num_associations %yielded_1 : (!transform.any_op) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %2 : !transform.param<i64>
    transform.yield 
  }
}


```

**重点说明:**

- 输入共48行，输出共30行
- transform.named_sequence定义被保留

---

### 1.2.4 case_4

**功能介绍:**

2 funcs are yielded for each of the 2 funcs = 4:

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  func.func private @foo()
  func.func private @bar()

  transform.named_sequence @match(
      %op: !transform.any_op {transform.readonly},
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @return(
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %func2, %yielded = transform.foreach_match flatten_results restrict_root in %func, %func
      @match -> @return
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %num = transform.num_associations %yielded : (!transform.any_op) -> !transform.param<i64>
    // 2 funcs are yielded for each of the 2 funcs = 4:
    // expected-remark @below {{4 : i64}}
    transform.debug.emit_param_as_remark %num : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func private @foo()
  func.func private @bar()
  transform.named_sequence @match(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.yield %arg1 : !transform.any_op
  }
  transform.named_sequence @return(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %updated_root, %yielded = transform.foreach_match restrict_root flatten_results in %0, %0 
        @match -> @return : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %1 = transform.num_associations %yielded : (!transform.any_op) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %1 : !transform.param<i64>
    transform.yield 
  }
}


```

**重点说明:**

- 输入共27行，输出共18行
- transform.named_sequence定义被保留

---

### 1.2.5 case_5

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  func.func private @foo()
  func.func private @bar()

  transform.named_sequence @match(
      %op: !transform.any_op {transform.readonly},
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @return(
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{action @return has results associated with multiple payload entities, but flattening was not requested}}
    %func2, %yielded = transform.foreach_match restrict_root in %func, %func
      @match -> @return
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %num = transform.num_associations %yielded : (!transform.any_op) -> !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

# 2. IRDL测试

## 2.1 irdl.mlir

### 2.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.irdl.collect_matching in %arg0 : (!transform.any_op) -> (!transform.any_op){
    ^bb0(%arg1: !transform.any_op):
      irdl.dialect @test {
        irdl.operation @whatever {
          %0 = irdl.is i32
          %1 = irdl.is i64
          %2 = irdl.any_of(%0, %1)
          irdl.results(foo: %2)
        }
      }
    }
    transform.debug.emit_remark_at %0, "matched" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5593300512.mlir:1 offset :1:3: error: unexpected error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  ^
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5593300512.mlir:1 offset :0:0: error: unexpected note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

# 3. PDL扩展测试

## 3.1 test-pdl-extension.mlir

### 3.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.debug.emit_remark_at %0, "matched" : !transform.any_op
      }

      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      pdl.pattern @other : benefit(1) {
        %0 = pdl.operation "test.other_op"
        pdl.rewrite %0 with "transform.dialect"
      }
    }
    transform.yield
  }
}

// expected-remark @below {{matched}}
"test.some_op"() : () -> ()
"test.other_op"() : () -> ()
// expected-remark @below {{matched}}
"test.some_op"() : () -> ()
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.debug.emit_remark_at %0, "matched" : !transform.any_op
        }
        pdl.pattern @some : benefit(1) {
          %0 = operation "test.some_op" 
          rewrite %0 with "transform.dialect"
        }
        pdl.pattern @other : benefit(1) {
          %0 = operation "test.other_op" 
          rewrite %0 with "transform.dialect"
        }
      }
      transform.yield 
    }
  }
  "test.some_op"() : () -> ()
  "test.other_op"() : () -> ()
  "test.some_op"() : () -> ()
}


```

**重点说明:**

- 输入共29行，输出共26行
- transform.named_sequence定义被保留

---

### 3.1.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
      }

      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.apply_native_constraint "verbose_constraint"(%0 : !pdl.operation)
        pdl.rewrite %0 with "transform.dialect"
      }
    }
    transform.yield
  }
}

// expected-warning @below {{from PDL constraint}}
"test.some_op"() : () -> ()
"test.other_op"() : () -> ()
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
        }
        pdl.pattern @some : benefit(1) {
          %0 = operation "test.some_op" 
          apply_native_constraint "verbose_constraint"(%0 : !pdl.operation)
          rewrite %0 with "transform.dialect"
        }
      }
      transform.yield 
    }
  }
  "test.some_op"() : () -> ()
  "test.other_op"() : () -> ()
}


```

**重点说明:**

- 输入共22行，输出共21行
- transform.named_sequence定义被保留

---

# 4. Pass应用测试

## 4.1 test-pass-application.mlir

### 4.1.1 case_1

**功能介绍:**

CHECK:   %[[c5:.*]] = arith.constant 5 : index
CHECK:   return %[[c5]]

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @successful_pass_application(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @successful_pass_application(%arg0: tensor<5xf32>) -> index {
    %c5 = arith.constant 5 : index
    return %c5 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.apply_registered_pass "canonicalize" to %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共13行
- transform.named_sequence定义被保留

---

### 4.1.2 case_2

**功能介绍:**

This pipeline does not do anything. Just make sure that the pipeline is
found and no error is produced.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @pass_pipeline() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // This pipeline does not do anything. Just make sure that the pipeline is
    // found and no error is produced.
    transform.apply_registered_pass "test-options-pass-pipeline" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @pass_pipeline() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.apply_registered_pass "test-options-pass-pipeline" to %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共13行，输出共12行
- transform.named_sequence定义被保留

---

### 4.1.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_pass_name() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{unknown pass or pass pipeline: non-existing-pass}}
    transform.apply_registered_pass "non-existing-pass" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.4 case_4

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @not_isolated_from_above(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  // expected-note @below {{target op}}
  // expected-error @below {{trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'}}
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["tensor.dim"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{pass pipeline failed}}
    transform.apply_registered_pass "canonicalize" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.5 case_5

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_pass_option() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to add pass or pass pipeline to pipeline: canonicalize}}
    // expected-error @below {{<Pass-Options-Parser>: no such option invalid-option}}
    transform.apply_registered_pass "canonicalize"
        with options = { "invalid-option" = 1 } to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.6 case_6

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @valid_pass_option() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize"
        with options = { "top-down" = false } to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @valid_pass_option() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.apply_registered_pass "canonicalize" with options = {"top-down" = false} to %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共13行，输出共12行
- transform.named_sequence定义被保留

---

### 4.1.7 case_7

**功能介绍:**

transform.apply_registered_pass "canonicalize" with options = "top-down=false,max-iterations=10" to %1 : (!transform.any_op) -> !transform.any_op

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @valid_pass_options() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    //transform.apply_registered_pass "canonicalize" with options = "top-down=false,max-iterations=10" to %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize"
        with options = { "top-down" = false, "test-convergence" =true } to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @valid_pass_options() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.apply_registered_pass "canonicalize" with options = {"test-convergence" = true, "top-down" = false} to %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共14行，输出共12行
- transform.named_sequence定义被保留

---

### 4.1.8 case_8

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @valid_pass_options_as_list() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize"
        with options = { "top-down" = false, "max-iterations" = 0 } to %1
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @valid_pass_options_as_list() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.apply_registered_pass "canonicalize" with options = {"max-iterations" = 0 : i64, "top-down" = false} to %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共13行，输出共12行
- transform.named_sequence定义被保留

---

### 4.1.9 case_9

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @valid_dynamic_pass_options() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %max_iter = transform.param.constant 10 -> !transform.any_param
    %max_rewrites = transform.param.constant 1 -> !transform.any_param
    %2 = transform.apply_registered_pass
        "canonicalize"
        with options = { "top-down" = false,
                         "max-iterations" = %max_iter,
                         "test-convergence" = true,
                         "max-num-rewrites" =  %max_rewrites }
        to %1
        : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @valid_dynamic_pass_options() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.param.constant 10 : i64 -> !transform.any_param
      %2 = transform.param.constant 1 : i64 -> !transform.any_param
      %3 = transform.apply_registered_pass "canonicalize" with options = {"max-iterations" = %1, "max-num-rewrites" = %2, "test-convergence" = true, "top-down" = false} to %0 : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共20行，输出共14行
- transform.named_sequence定义被保留

---

### 4.1.10 case_10

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module {
  func.func @valid_multiple_values_as_list_option_single_param() {
    return
  }

  func.func @a() {
    return
  }
  func.func @b() {
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.get_parent_op %1 { deduplicate } : (!transform.any_op) -> !transform.any_op
    %symbol_a = transform.param.constant "a" -> !transform.any_param
    %symbol_b = transform.param.constant "b" -> !transform.any_param
    %multiple_symbol_names = transform.merge_handles %symbol_a, %symbol_b : !transform.any_param
    transform.apply_registered_pass "symbol-privatize"
        with options = { exclude = %multiple_symbol_names } to %2
        : (!transform.any_op, !transform.any_param) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  module {
    func.func private @valid_multiple_values_as_list_option_single_param() {
      return
    }
    func.func @a() {
      return
    }
    func.func @b() {
      return
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_parent_op %0 {deduplicate} : (!transform.any_op) -> !transform.any_op
      %2 = transform.param.constant "a" -> !transform.any_param
      %3 = transform.param.constant "b" -> !transform.any_param
      %4 = transform.merge_handles %2, %3 : !transform.any_param
      %5 = transform.apply_registered_pass "symbol-privatize" with options = {"exclude" = %4} to %1 : (!transform.any_op, !transform.any_param) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共26行，输出共24行
- transform.named_sequence定义被保留

---

### 4.1.11 case_11

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module {
  func.func @valid_array_attr_as_list_option() {
    return
  }

  func.func @a() {
    return
  }
  func.func @b() {
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.get_parent_op %1 { deduplicate } : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "symbol-privatize"
        with options = { exclude = ["a", "b"] } to %2
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  module {
    func.func private @valid_array_attr_as_list_option() {
      return
    }
    func.func @a() {
      return
    }
    func.func @b() {
      return
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_parent_op %0 {deduplicate} : (!transform.any_op) -> !transform.any_op
      %2 = transform.apply_registered_pass "symbol-privatize" with options = {"exclude" = ["a", "b"]} to %1 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共23行，输出共21行
- transform.named_sequence定义被保留

---

### 4.1.12 case_12

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module {
  func.func @valid_array_attr_param_as_list_option() {
    return
  }

  func.func @a() {
    return
  }
  func.func @b() {
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.get_parent_op %1 { deduplicate } : (!transform.any_op) -> !transform.any_op
    %multiple_symbol_names = transform.param.constant ["a","b"] -> !transform.any_param
    transform.apply_registered_pass "symbol-privatize"
        with options = { exclude = %multiple_symbol_names } to %2
        : (!transform.any_op, !transform.any_param) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  module {
    func.func private @valid_array_attr_param_as_list_option() {
      return
    }
    func.func @a() {
      return
    }
    func.func @b() {
      return
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_parent_op %0 {deduplicate} : (!transform.any_op) -> !transform.any_op
      %2 = transform.param.constant ["a", "b"] -> !transform.any_param
      %3 = transform.apply_registered_pass "symbol-privatize" with options = {"exclude" = %2} to %1 : (!transform.any_op, !transform.any_param) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共24行，输出共22行
- transform.named_sequence定义被保留

---

### 4.1.13 case_13

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module {
  func.func @valid_multiple_params_as_single_list_option() {
    return
  }

  func.func @a() {
    return
  }
  func.func @b() {
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.get_parent_op %1 { deduplicate } : (!transform.any_op) -> !transform.any_op
    %symbol_a = transform.param.constant "a" -> !transform.any_param
    %symbol_b = transform.param.constant "b" -> !transform.any_param
    transform.apply_registered_pass "symbol-privatize"
        with options = { exclude = [%symbol_a, %symbol_b] } to %2
        : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  module {
    func.func private @valid_multiple_params_as_single_list_option() {
      return
    }
    func.func @a() {
      return
    }
    func.func @b() {
      return
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_parent_op %0 {deduplicate} : (!transform.any_op) -> !transform.any_op
      %2 = transform.param.constant "a" -> !transform.any_param
      %3 = transform.param.constant "b" -> !transform.any_param
      %4 = transform.apply_registered_pass "symbol-privatize" with options = {"exclude" = [%2, %3]} to %1 : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共25行，输出共23行
- transform.named_sequence定义被保留

---

### 4.1.14 case_14

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_options_as_str() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected '{' in options dictionary}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = "top-down=false" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.15 case_15

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_options_as_pairs_without_braces() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected '{' in options dictionary}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = "top-down"=false to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.16 case_16

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_options_due_to_reserved_attr() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+3 {{the param_operand attribute is a marker reserved for indicating a value will be passed via params and is only used in the generic print format}}
    // expected-error @+2 {{expected a valid attribute or operand as value associated to key 'top-down'}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = { "top-down" = #transform.param_operand<index=0> } to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.17 case_17

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_options_due_duplicated_key() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{duplicate keys found in options dictionary}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = {"top-down"=false,"top-down"=true} to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.18 case_18

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_options_due_invalid_key() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+2 {{expected key to either be an identifier or a string}}
    %2 = transform.apply_registered_pass "canonicalize"
        with options = { @label = 0 } to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.19 case_19

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @invalid_pass_option_bare_param() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pass_options = transform.param.constant 42 -> !transform.any_param
    // expected-error @+2 {{expected '{' in options dictionary}}
    transform.apply_registered_pass "canonicalize"
        with options = %pass_options to %1
        : (!transform.any_op, !transform.any_param) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.20 case_20

**功能介绍:**

duplicate-function-elimination can be applied only to ModuleOps.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  // expected-error @below {{trying to schedule a pass on an unsupported operation}}
  // expected-note @below {{target op}}
  func.func @invalid_target_op_type() {
    return
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // duplicate-function-elimination can be applied only to ModuleOps.
    // expected-error @below {{pass pipeline failed}}
    transform.apply_registered_pass "duplicate-function-elimination" to %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 4.1.21 case_21

**功能介绍:**

///////////////////////////////////////////////////////////////////
Check that the following cases are caugh in the generic format. //
///////////////////////////////////////////////////////////////////
Invalid due to param_operand occurences in options dict not being
one-to-one with the dynamic options provided as params:
param_operand_index out of bounds w.r.t. the number of options provided via params.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
/////////////////////////////////////////////////////////////////////
// Check that the following cases are caugh in the generic format. //
/////////////////////////////////////////////////////////////////////

// Invalid due to param_operand occurences in options dict not being
// one-to-one with the dynamic options provided as params:
//   param_operand_index out of bounds w.r.t. the number of options provided via params.

"builtin.module"() ({
  "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.param.constant"() <{value = 10 : i64}> : () -> !transform.any_param
    // expected-error @below {{dynamic option index 1 is out of bounds for the number of dynamic options: 1}}
    %2 = "transform.apply_registered_pass"(%0, %1) <{
      options = {"max-iterations" = #transform.param_operand<index=1 : i64>,
                 "test-convergence" = true,
                 "top-down" = false},
      pass_name = "canonicalize"}>
    : (!transform.any_op, !transform.any_param) -> !transform.any_op
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()
```

**用例输出:**

执行成功，无输出。

---

### 4.1.22 case_22

**功能介绍:**

Invalid due to param_operand occurences in options dict not being
one-to-one with the dynamic options provided as params:
the first option-param is referred to twice and the second one not at all.
(In the pretty-printed format, if you want to refer to a param SSA-value twice, it counts as two param arguments.)

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Invalid due to param_operand occurences in options dict not being
// one-to-one with the dynamic options provided as params:
//   the first option-param is referred to twice and the second one not at all.
// (In the pretty-printed format, if you want to refer to a param SSA-value twice, it counts as two param arguments.)

"builtin.module"() ({
  "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.param.constant"() <{value = 10 : i64}> : () -> !transform.any_param
    %2 = "transform.param.constant"() <{value = 1 : i64}> : () -> !transform.any_param
    // expected-error @below {{dynamic option index 0 is already used in options}}
    %3 = "transform.apply_registered_pass"(%0, %1, %2) <{
      options = {"max-iterations" = #transform.param_operand<index=0 : i64>,
                 "max-num-rewrites" = #transform.param_operand<index=0 : i64>,
                 "test-convergence" = true,
                 "top-down" = false},
      pass_name = "canonicalize"}>
    : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()
```

**用例输出:**

执行成功，无输出。

---

### 4.1.23 case_23

**功能介绍:**

Invalid due to param_operand occurences in options dict not being
one-to-one with the dynamic options provided as params:
two option-params are provide though only the first one is referred to from the options-dict.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Invalid due to param_operand occurences in options dict not being
// one-to-one with the dynamic options provided as params:
//   two option-params are provide though only the first one is referred to from the options-dict.

"builtin.module"() ({
  "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.param.constant"() <{value = 10 : i64}> : () -> !transform.any_param
    %2 = "transform.param.constant"() <{value = 1 : i64}> : () -> !transform.any_param
    // expected-error @below {{a param operand does not have a corresponding param_operand attr in the options dict}}
    %3 = "transform.apply_registered_pass"(%0, %1, %2) <{
      options = {"max-iterations" = #transform.param_operand<index=0 : i64>,
                 "test-convergence" = true,
                 "top-down" = false},
      pass_name = "canonicalize"}>
    : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()
```

**用例输出:**

执行成功，无输出。

---

# 5. 其他测试

## 5.1 lower-to-llvm-transform-symbol-def.mlir

### 5.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @lower_to_llvm(
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378920976.mlir:1:40: error: expected non-function type
transform.named_sequence @lower_to_llvm(
                                       ^

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 5.2 lower-to-llvm.mlir

### 5.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @lower_to_llvm(
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378922128.mlir:1:40: error: expected non-function type
transform.named_sequence @lower_to_llvm(
                                       ^

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 5.3 definitions-self-contained.mlir

### 5.3.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence private @private_helper(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "message" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378473520.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence private @private_helper(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378473520.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence private @colliding(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding (without suffix)" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4377497088.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence private @colliding(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4377497088.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence private @colliding_0(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_0" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474032.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence private @colliding_0(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474032.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.4 case_4

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence private @colliding_2(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_2" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474288.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence private @colliding_2(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474288.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.5 case_5

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence private @colliding_3(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_3" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474544.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence private @colliding_3(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474544.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.6 case_6

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence private @colliding_4(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_4" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474800.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence private @colliding_4(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378474800.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.7 case_7

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @colliding_5(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_5" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378475056.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @colliding_5(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378475056.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.8 case_8

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
    transform.include @private_helper failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378894928.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378894928.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.9 case_9

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378807408.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378807408.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.10 case_10

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @unannotated(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "unannotated" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378806960.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @unannotated(%arg0: !transform.any_op) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378806960.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.3.11 case_11

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @symbol_user(%arg0: !transform.any_op {transform.readonly}) {
    transform.include @colliding failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_0 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_2 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_3 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_4 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_5 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5325735120.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @symbol_user(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5325735120.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 5.4 definitions-with-unresolved.mlir

### 5.4.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly})
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378910272.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly})
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378910272.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 5.4.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @reference_other_module(%arg0: !transform.any_op) {
    transform.include @print_message failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378473520.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @reference_other_module(%arg0: !transform.any_op) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378473520.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 5.5 infer-effects.mlir

### 5.5.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-infer-effects
```

**用例输入:**

```mlir
transform.named_sequence @infer(%op: !transform.any_op, %other: !transform.any_op, %param: !transform.param<i32>) {
    transform.test_consume_operand %op : !transform.any_op
    transform.debug.emit_remark_at %other, "" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4377582128.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @infer(%op: !transform.any_op, %other: !transform.any_op, %param: !transform.param<i32>) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4377582128.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

# 6. 基础操作测试

## 6.1 multi-arg-top-level-ops.mlir

### 6.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-bind-trailing-args=func.func,func.return})"  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.any_op) {
    transform.debug.emit_remark_at %arg1, "first extra" : !transform.any_op
    transform.debug.emit_remark_at %arg2, "second extra" : !transform.any_op
    transform.yield
  }
}

// expected-remark @below {{first extra}}
func.func @foo() {
  // expected-remark @below {{second extra}}
  return
}

// expected-remark @below {{first extra}}
func.func @bar(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // expected-remark @below {{second extra}}
  return
^bb2:
  // expected-remark @below {{second extra}}
  return
}
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op) {
      transform.debug.emit_remark_at %arg1, "first extra" : !transform.any_op
      transform.debug.emit_remark_at %arg2, "second extra" : !transform.any_op
      transform.yield 
    }
  }
  func.func @foo() {
    return
  }
  func.func @bar(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    return
  ^bb2:  // pred: ^bb0
    return
  }
}


```

**重点说明:**

- 输入共26行，输出共19行
- transform.named_sequence定义被保留

---

### 6.1.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-bind-trailing-args=func.func,func.return})"  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.param<i64>) {
    // expected-error @above {{wrong kind of value provided for top-level parameter}}
    transform.yield
  }
}

func.func @foo() {
  return
}
```

**用例输出:**

执行成功，无输出。

---

### 6.1.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-bind-trailing-args=func.func,func.return})"  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.any_value) {
    // expected-error @above {{wrong kind of value provided for the top-level value handle}}
    transform.yield
  }
}

func.func @foo() {
  return
}
```

**用例输出:**

执行成功，无输出。

---

### 6.1.4 case_4

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-bind-trailing-args=func.func,func.return})"  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  // expected-error @below {{operation expects 1 extra value bindings, but 2 were provided to the interpreter}}
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 6.1.5 case_5

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-bind-trailing-args=func.func,func.return})"  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      %arg2: !transform.any_op) {
    transform.sequence %arg0, %arg1, %arg2 : !transform.any_op, !transform.any_op, !transform.any_op failures(propagate) {
    ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
      transform.debug.emit_remark_at %arg4, "first extra" : !transform.any_op
      transform.debug.emit_remark_at %arg5, "second extra" : !transform.any_op
    }
    transform.yield
  }
}

// expected-remark @below {{first extra}}
func.func @foo() {
  // expected-remark @below {{second extra}}
  return
}

// expected-remark @below {{first extra}}
func.func @bar(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // expected-remark @below {{second extra}}
  return
^bb2:
  // expected-remark @below {{second extra}}
  return
}
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op) {
      transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate) {
      ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
        transform.debug.emit_remark_at %arg4, "first extra" : !transform.any_op
        transform.debug.emit_remark_at %arg5, "second extra" : !transform.any_op
      }
      transform.yield 
    }
  }
  func.func @foo() {
    return
  }
  func.func @bar(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    return
  ^bb2:  // pred: ^bb0
    return
  }
}


```

**重点说明:**

- 输入共29行，输出共22行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

## 6.2 ops.mlir

### 6.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

write /dev/stdout: broken pipe

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    yield %arg1 : !transform.any_op
  }
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb2(%arg2: !transform.any_op):
  }
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb3(%arg3: !transform.any_op):
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.4 case_4

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  transform.sequence %arg0, %arg1, %arg2 : !transform.any_op, !transform.any_op, !transform.any_op failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.5 case_5

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.6 case_6

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.7 case_7

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%op0: !transform.any_op, %val0: !transform.any_value, %par0: !transform.any_param):
  transform.foreach %op0 : !transform.any_op {
  ^bb1(%op1: !transform.any_op):
  }
  transform.foreach %op0, %val0, %par0 : !transform.any_op, !transform.any_value, !transform.any_param {
  ^bb1(%op1: !transform.any_op, %val1: !transform.any_value, %par1: !transform.any_param):
  }
  transform.foreach %op0, %val0, %par0 : !transform.any_op, !transform.any_value, !transform.any_param -> !transform.any_op {
  ^bb1(%op1: !transform.any_op, %val1: !transform.any_value, %par1: !transform.any_param):
    transform.yield %op1 : !transform.any_op
  }
  transform.foreach %op0, %val0, %par0 : !transform.any_op, !transform.any_value, !transform.any_param -> !transform.any_param, !transform.any_value {
  ^bb1(%op1: !transform.any_op, %val1: !transform.any_value, %par1: !transform.any_param):
    transform.yield %par1, %val1 : !transform.any_param, !transform.any_value
  }
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.8 case_8

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = cast %arg0: !transform.any_op to !transform.any_op
  %1 = cast %0: !transform.any_op to !transform.op<"builtin.module">
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.9 case_9

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.print %arg0 : !transform.any_op
  transform.print
  transform.print %arg0 {name = "test"} : !transform.any_op
  transform.print {name = "test"}
  transform.print {name = "test", assume_verified}
  transform.print %arg0 {assume_verified} : !transform.any_op
  transform.print %arg0 {use_local_scope} : !transform.any_op
  transform.print %arg0 {skip_regions} : !transform.any_op
  transform.print %arg0 {assume_verified, use_local_scope, skip_regions} : !transform.any_op
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.10 case_10

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_using_for %0 tile_sizes [4, 4, [4]] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.11 case_11

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_using_for %0 tile_sizes [[2], 4, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 6.2.12 case_12

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> | mlir-opt
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  transform.param.constant "example_string" -> !transform.any_param
}
```

**用例输出:**

```
执行失败: /bin/sh: mlir-opt: command not found

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

# 7. 多参数测试

## 7.1 multi-arg-top-level-params.mlir

### 7.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline='builtin.module(transform-interpreter{ debug-bind-trailing-args=#1;2;3,#42;45})'  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.param<i64>,
      %arg2: !transform.param<i64>) {
    // expected-remark @below {{1 : i64, 2 : i64, 3 : i64}}
    transform.debug.emit_param_as_remark %arg1 : !transform.param<i64>
    // expected-remark @below {{42 : i64, 45 : i64}}
    transform.debug.emit_param_as_remark %arg2 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.param<i64>, %arg2: !transform.param<i64>) {
    transform.debug.emit_param_as_remark %arg1 : !transform.param<i64>
    transform.debug.emit_param_as_remark %arg2 : !transform.param<i64>
    transform.yield 
  }
}


```

**重点说明:**

- 输入共11行，输出共7行
- transform.named_sequence定义被保留

---

### 7.1.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline='builtin.module(transform-interpreter{ debug-bind-trailing-args=#1;2;3,#42;45})'  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_op,
      // expected-error @above {{wrong kind of value provided for top-level operation handle}}
      %arg2: !transform.param<i64>) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 7.1.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline='builtin.module(transform-interpreter{ debug-bind-trailing-args=#1;2;3,#42;45})'  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  // expected-error @below {{operation expects 3 extra value bindings, but 2 were provided to the interpreter}}
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.param<i64>,
      %arg2: !transform.param<i64>, %arg3: !transform.param<i64>) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

## 7.2 multi-arg-top-level-values.mlir

### 7.2.1 case_1

**功能介绍:**

Note that diagnostic checker will merge two diagnostics with the same message
at the same location, so only check the remark once.
Note that diagnostic checker will merge two diagnostics with the same message
at the same location, so only check the remark once.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline='builtin.module(transform-interpreter{ debug-bind-trailing-args=^test.some_returning_op,^test.some_other_returning_op})'  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Note that diagnostic checker will merge two diagnostics with the same message
// at the same location, so only check the remark once.
// 
// expected-remark @below {{first extra}}
// expected-note @below {{value handle points to an op result #0}}
// expected-note @below {{value handle points to an op result #1}}
%0:2 = "test.some_returning_op"() : () -> (i32, i64)

// expected-remark @below {{first extra}}
// expected-note @below {{value handle points to an op result #0}}
%1 = "test.some_returning_op"() : () -> index

// Note that diagnostic checker will merge two diagnostics with the same message
// at the same location, so only check the remark once.
// 
// expected-remark @below {{second extra}}
// expected-note @below {{value handle points to an op result #0}}
// expected-note @below {{value handle points to an op result #1}}
%2:2 = "test.some_other_returning_op"() : () -> (f32, f64)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op, %arg1: !transform.any_value,
      %arg2: !transform.any_value) {
    transform.debug.emit_remark_at %arg1, "first extra" : !transform.any_value
    transform.debug.emit_remark_at %arg2, "second extra" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  %0:2 = "test.some_returning_op"() : () -> (i32, i64)
  %1 = "test.some_returning_op"() : () -> index
  %2:2 = "test.some_other_returning_op"() : () -> (f32, f64)
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.any_value, %arg2: !transform.any_value) {
      transform.debug.emit_remark_at %arg1, "first extra" : !transform.any_value
      transform.debug.emit_remark_at %arg2, "second extra" : !transform.any_value
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共29行，输出共12行
- transform.named_sequence定义被保留

---

### 7.2.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline='builtin.module(transform-interpreter{ debug-bind-trailing-args=^test.some_returning_op,^test.some_other_returning_op})'  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
%0:2 = "test.some_returning_op"() : () -> (i32, i64)
%1 = "test.some_returning_op"() : () -> index

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      // expected-error @below {{wrong kind of value provided for top-level operation handle}}
      %arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_value) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 7.2.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline='builtin.module(transform-interpreter{ debug-bind-trailing-args=^test.some_returning_op,^test.some_other_returning_op})'  --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  // expected-error @below {{operation expects 1 extra value bindings, but 2 were provided to the interpreter}}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.any_value) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

# 8. 库加载测试

## 8.1 preload-library.mlir

### 8.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform操作执行，具体功能参见用例输入。

**执行命令:**

```bash
mlir-opt <input_file>  -transform-preload-library=transform-library-paths=<input_dir>%{fs-sep}include%{fs-sep}test-interpreter-library  -transform-interpreter=entry-point=private_helper  -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{message}}
module {}
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-library:0:0: error: unexpected error: '/Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-library' is neither a file nor a directory
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5604531712.mlir:1 offset :18:4: error: expected remark "message" was not produced
// expected-remark @below {{message}}
   ^~~~~~~

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 8.1.2 case_2

**功能介绍:**

Note: no remark here since local entry point takes precedence.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file>  -transform-preload-library=transform-library-paths=<input_dir>%{fs-sep}include%{fs-sep}test-interpreter-library  -transform-interpreter=entry-point=private_helper  -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// Note: no remark here since local entry point takes precedence.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @private_helper(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }
}
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-library:0:0: error: unexpected error: '/Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-library' is neither a file nor a directory
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4379129520.mlir:1 offset :5:8: error: expected remark "applying transformation" was not produced
    // expected-remark @below {{applying transformation}}
       ^~~~~~~~~~~~
```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

# 9. 循环变换测试

## 9.1 test-loop-transforms.mlir

### 9.1.1 case_1

**功能介绍:**

UNSUPPORTED: target=aarch64-pc-windows-msvc
CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
Obfuscate the IR by inserting at offset %sub instead of 0; both of them
have the same value.
Make sure that the handles are still valid (and were updated in case of
the loop).

**核心原理:**

循环变换操作，包括循环展开、分块、融合等优化。这些操作用于优化循环结构以提高性能。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file  --verify-diagnostics
```

**用例输入:**

```mlir
// UNSUPPORTED: target=aarch64-pc-windows-msvc

//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @test_loop_invariant_subset_hoisting(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  // expected-remark @below{{new loop op}}
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    // Obfuscate the IR by inserting at offset %sub instead of 0; both of them
    // have the same value.
    %3 = tensor.insert_slice %2 into %t[0][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %3 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.extract_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    transform.loop.hoist_loop_invariant_subsets %0 : !transform.any_op
    // Make sure that the handles are still valid (and were updated in case of
    // the loop).

    %p = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.debug.emit_remark_at %0, "new loop op" : !transform.any_op
    %p2 = transform.num_associations %1 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>
    %p3 = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>

    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_loop_invariant_subset_hoisting(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "test.foo"() : () -> index
    %1 = "test.foo"() : () -> index
    %2 = "test.foo"() : () -> index
    %extracted_slice = tensor.extract_slice %arg0[0] [5] [1] : tensor<?xf32> to tensor<5xf32>
    %3:2 = scf.for %arg1 = %0 to %1 step %2 iter_args(%arg2 = %arg0, %arg3 = %extracted_slice) -> (tensor<?xf32>, tensor<5xf32>) {
      %4 = "test.foo"(%arg3) : (tensor<5xf32>) -> tensor<5xf32>
      scf.yield %arg2, %4 : tensor<?xf32>, tensor<5xf32>
    }
    %inserted_slice = tensor.insert_slice %3#1 into %3#0[0] [5] [1] : tensor<5xf32> into tensor<?xf32>
    return %inserted_slice : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["tensor.extract_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %2 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.loop.hoist_loop_invariant_subsets %0 : !transform.any_op
      %3 = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %3 : !transform.param<i64>
      transform.debug.emit_remark_at %0, "new loop op" : !transform.any_op
      %4 = transform.num_associations %1 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %4 : !transform.param<i64>
      %5 = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %5 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共43行，输出共30行
- transform.named_sequence定义被保留

---

### 9.1.2 case_2

**功能介绍:**

Checks that transform ops from LoopExtensionOps and SCFTransformOps can be
used together.

**核心原理:**

循环变换操作，包括循环展开、分块、融合等优化。这些操作用于优化循环结构以提高性能。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file  --verify-diagnostics
```

**用例输入:**

```mlir
// Checks that transform ops from LoopExtensionOps and SCFTransformOps can be
// used together.

func.func @test_mixed_loop_extension_scf_transform(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %1 = "test.foo"(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
    scf.yield %1 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %0 : !transform.any_op
    transform.loop.unroll %0 { factor = 4 } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_mixed_loop_extension_scf_transform(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "test.foo"() : () -> index
    %1 = "test.foo"() : () -> index
    %2 = "test.foo"() : () -> index
    %3 = arith.subi %1, %0 : index
    %c1 = arith.constant 1 : index
    %4 = arith.subi %2, %c1 : index
    %5 = arith.addi %3, %4 : index
    %6 = arith.divui %5, %2 : index
    %c4 = arith.constant 4 : index
    %7 = arith.remsi %6, %c4 : index
    %8 = arith.subi %6, %7 : index
    %9 = arith.muli %8, %2 : index
    %10 = arith.addi %0, %9 : index
    %11 = arith.muli %2, %c4 : index
    %12 = scf.for %arg1 = %0 to %10 step %11 iter_args(%arg2 = %arg0) -> (tensor<?xf32>) {
      %14 = "test.foo"(%arg2) : (tensor<?xf32>) -> tensor<?xf32>
      %15 = "test.foo"(%14) : (tensor<?xf32>) -> tensor<?xf32>
      %16 = "test.foo"(%15) : (tensor<?xf32>) -> tensor<?xf32>
      %17 = "test.foo"(%16) : (tensor<?xf32>) -> tensor<?xf32>
      scf.yield %17 : tensor<?xf32>
    }
    %13 = scf.for %arg1 = %10 to %1 step %2 iter_args(%arg2 = %12) -> (tensor<?xf32>) {
      %14 = "test.foo"(%arg2) : (tensor<?xf32>) -> tensor<?xf32>
      scf.yield %14 : tensor<?xf32>
    }
    return %13 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.loop.hoist_loop_invariant_subsets %0 : !transform.any_op
      transform.loop.unroll %0 {factor = 4 : i64} : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共22行，输出共38行
- transform.named_sequence定义被保留

---

# 10. 扩展测试

## 10.1 test-tune-extension.mlir

### 10.1.1 case_1

**功能介绍:**

Dummy sequence to appease -transform-interpreter invocation

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file  --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_with_nondet_knobs(%arg0: !transform.any_op {transform.readonly}) {
    %heads_or_tails = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    %chosen_category = transform.tune.knob<"animal"> options = ["cat", "dog", unit] -> !transform.any_param
    %chosen_tile_size = transform.tune.knob<"tile_size"> options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    %chosen_constant = transform.tune.knob<"magic_value"> options = [2.0 : f32, 2.25 : f32, 2.5 : f32, 2.75 : f32, 3.0 : f32] -> !transform.any_param
    transform.debug.emit_param_as_remark %heads_or_tails : !transform.any_param
    transform.yield
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // Dummy sequence to appease -transform-interpreter invocation
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_with_nondet_knobs(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    %1 = transform.tune.knob<"animal"> options = ["cat", "dog", unit] -> !transform.any_param
    %2 = transform.tune.knob<"tile_size"> options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    %3 = transform.tune.knob<"magic_value"> options = [2.000000e+00 : f32, 2.250000e+00 : f32, 2.500000e+00 : f32, 2.750000e+00 : f32, 3.000000e+00 : f32] -> !transform.any_param
    transform.debug.emit_param_as_remark %0 : !transform.any_param
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield 
  }
}


```

**重点说明:**

- 输入共14行，输出共13行
- transform.named_sequence定义被保留

---

### 10.1.2 case_2

**功能介绍:**

Schedule where non-determinism on knobs has been resolved by selecting a valid option.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file  --verify-diagnostics
```

**用例输入:**

```mlir
// Schedule where non-determinism on knobs has been resolved by selecting a valid option.

func.func private @payload_for_schedule_with_selected_knobs()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %heads_or_tails = transform.tune.knob<"coin"> = true from options = [true, false] -> !transform.any_param
    // expected-remark@below {{true}}
    transform.debug.emit_param_as_remark %heads_or_tails : !transform.any_param

    %chosen_category = transform.tune.knob<"animal"> = "dog" from options = ["cat", "dog", unit] -> !transform.any_param
    %chosen_tile_size = transform.tune.knob<"tile_size"> = 8 from options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    %chosen_constant = transform.tune.knob<"magic_value"> = 2.5 : f32  from options = [2.0 : f32, 2.25 : f32, 2.5 : f32, 2.75 : f32, 3.0 : f32] -> !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @payload_for_schedule_with_selected_knobs()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.tune.knob<"coin"> = true from options = [true, false] -> !transform.any_param
      transform.debug.emit_param_as_remark %0 : !transform.any_param
      %1 = transform.tune.knob<"animal"> = "dog" from options = ["cat", "dog", unit] -> !transform.any_param
      %2 = transform.tune.knob<"tile_size"> = 8 : i64 from options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
      %3 = transform.tune.knob<"magic_value"> = 2.500000e+00 : f32 from options = [2.000000e+00 : f32, 2.250000e+00 : f32, 2.500000e+00 : f32, 2.750000e+00 : f32, 3.000000e+00 : f32] -> !transform.any_param
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共16行，输出共13行
- transform.named_sequence定义被保留

---

### 10.1.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file  --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @payload_for_schedule_where_selected_knob_being_a_member_of_options_is_unverified()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %value_in_half_range = transform.tune.knob<"bounded"> = 4242 from options = affine_set<(d0) : (d0 - 2 >= 0)>  -> !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
#set = affine_set<(d0) : (d0 - 2 >= 0)>
module {
  func.func private @payload_for_schedule_where_selected_knob_being_a_member_of_options_is_unverified()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.tune.knob<"bounded"> = 4242 : i64 from options = #set -> !transform.any_param
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共8行，输出共10行
- transform.named_sequence定义被保留

---

## 10.2 transform-state-extension-initializer.mlir

### 10.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -test-pass-state-extension-communication -verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{Number of currently registered op: 1}}
    transform.test_initializer_extension "A"
    // expected-remark @below {{Number of currently registered op: 2}}
    transform.test_initializer_extension "B"
    // expected-remark @below {{Number of currently registered op: 3}}
    transform.test_initializer_extension "C"
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378942912.mlir:1:3: error: unexpected error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378942912.mlir:0:0: error: unexpected note: symbol table operation
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378942912.mlir:2:8: error: expected remark "Number of 
```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 10.3 transform-state-extension.mlir

### 10.3.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
// expected-note @below {{associated payload op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{extension absent}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.test_add_test_extension "A"
    // expected-remark @below {{extension present, A}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.test_remove_test_extension
    // expected-remark @below {{extension absent}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.test_add_test_extension "A"
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.test_remove_test_extension
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共14行，输出共10行
- transform.named_sequence定义被保留

---

### 10.3.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
// expected-note @below {{associated payload op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    transform.test_remove_test_extension
    transform.test_add_test_extension "B"
    // expected-remark @below {{extension present, B}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    transform.test_remove_test_extension
    transform.test_add_test_extension "B"
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共11行，输出共9行
- transform.named_sequence定义被保留

---

### 10.3.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
// expected-note @below {{associated payload op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    // expected-remark @below {{extension present, A}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    // expected-note @below {{associated payload op}}
    transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{extension present, A}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    %0 = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共13行，输出共9行
- transform.named_sequence定义被保留

---

### 10.3.4 case_4

**功能介绍:**

This is okay because we are replacing the top-level module operation
(0 results) with this operation that has _more_ (1) results.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
     // This is okay because we are replacing the top-level module operation
     // (0 results) with this operation that has _more_ (1) results.
    %dummy = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    %0 = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共9行，输出共7行
- transform.named_sequence定义被保留

---

### 10.3.5 case_5

**功能介绍:**

This is still okay. Even though we are replacing the previous
operation with (1 result) with this operation that has less (0) results,
there is no handle to the result, hence no issue with value handle update.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    %dummy = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    // This is still okay. Even though we are replacing the previous
    // operation with (1 result) with this operation that has less (0) results,
    // there is no handle to the result, hence no issue with value handle update.
    transform.test_remap_operand_to_self %dummy : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 10.3.6 case_6

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    // expected-error @below {{cannot replace an op with another op producing fewer results while tracking handles}}
    %dummy = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    %valuehandle = transform.get_result %dummy[0] : (!transform.any_op) -> !transform.any_value
    transform.test_remap_operand_to_self %dummy : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 10.3.7 case_7

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter -verify-diagnostics -split-input-file
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{TestTransformStateExtension missing}}
    transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

# 11. 方言注入测试

## 11.1 test-dialect-injection.mlir

### 11.1.1 case_1

**功能介绍:**

These types and ops are defined by a test extension but should be okay to
roundtrip.
Ensure that the extension type is roundtripped correctly.

**核心原理:**

Transform操作成功执行，对IR进行变换或验证。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
// These types and ops are defined by a test extension but should be okay to
// roundtrip.

transform.test_transform_op

%0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op

transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op

// Ensure that the extension type is roundtripped correctly.
%1 = transform.cast %0: !transform.any_op to !transform.test_dialect_op
```

**用例输出:**

```mlir
module {
  transform.test_transform_op
  %0 = transform.test_produce_self_handle_or_forward_operand {foo = "bar"} : () -> !transform.any_op
  transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  %1 = transform.cast %0 : !transform.any_op to !transform.test_dialect_op
}


```

**重点说明:**

- 输入共11行，输出共6行
- 输出被包装在module中

---

# 12. 无效输入测试

## 12.1 definitions-invalid.mlir

### 12.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> --verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence private @private_helper(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error @below {{expected ','}}
    transform.debug.emit_remark_at %arg0 "should have ',' prior to this" : !transform.any_op
  }
```

**用例输出:**

执行成功，无输出。

---

## 12.2 ops-invalid.mlir

### 12.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects the entry block to have at least one argument}}
transform.sequence failures(propagate) {
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.2 case_2

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects the first entry block argument to be of type implementing TransformHandleTypeInterface}}
transform.sequence failures(propagate) {
^bb0(%rag0: i64):
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.3 case_3

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{nested in another possible top-level op}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects operands to be provided for a nested op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.4 case_4

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects to have a terminator in the body}}
"transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
^bb0(%arg0: !transform.any_op):
  transform.apply_patterns to %arg0 {
  } : !transform.any_op
}) : () -> ()
```

**用例输出:**

执行成功，无输出。

---

### 12.2.5 case_5

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{'transform.sequence' op expects trailing entry block arguments to be of type implementing TransformHandleTypeInterface, TransformValueHandleTypeInterface or TransformParamTypeInterface}}
// expected-note @below {{argument #1 does not}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: i64):
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.6 case_6

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expected children ops to implement TransformOpInterface}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-note @below {{op without interface}}
  arith.constant 42.0 : f32
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.7 case_7

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects the types of the terminator operands to match the types of the result}}
%0 = transform.sequence -> !transform.any_op failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-note @below {{terminator}}
  transform.yield
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.8 case_8

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects the type of the block argument to match the type of the operand}}
  transform.sequence %arg0: !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.op<"builtin.module">):
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.9 case_9

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  // expected-error @below {{expected types to be provided for all operands}}
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.10 case_10

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
%0 = "test.generate_something"() : () -> !transform.any_op
// expected-error @below {{does not expect extra operands when used as top-level}}
"transform.sequence"(%0) ({
^bb0(%arg0: !transform.any_op):
  "transform.yield"() : () -> ()
}) {failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 1>} : (!transform.any_op) -> ()
```

**用例输出:**

执行成功，无输出。

---

### 12.2.11 case_11

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{nested in another possible top-level op}}
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects operands to be provided for a nested op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.12 case_12

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects only one non-pattern op in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  // expected-note @below {{first non-pattern op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
  // expected-note @below {{second non-pattern op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.13 case_13

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects only pattern and top-level transform ops in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  // expected-note @below {{offending op}}
  "test.something"() : () -> ()
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.14 case_14

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{parent operation}}
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
   // expected-error @below {{op cannot be nested}}
  transform.with_pdl_patterns %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.15 case_15

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{op expects at least one non-pattern op}}
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.foo"
    pdl.rewrite %0 with "transform.dialect"
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.16 case_16

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{op expects at least one non-pattern op}}
  with_pdl_patterns %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.17 case_17

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects at least one region}}
"transform.test_transform_unrestricted_op_no_interface"() : () -> ()
```

**用例输出:**

执行成功，无输出。

---

### 12.2.18 case_18

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects a single-block region}}
"transform.test_transform_unrestricted_op_no_interface"() ({
^bb0(%arg0: !transform.any_op):
  "test.potential_terminator"() : () -> ()
^bb1:
  "test.potential_terminator"() : () -> ()
}) : () -> ()
```

**用例输出:**

执行成功，无输出。

---

### 12.2.19 case_19

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.20 case_20

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    test_consume_operand_of_op_kind_or_fail %arg1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.21 case_21

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    // expected-note @below {{used here as operand #0}}
    test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.22 case_22

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    transform.sequence %arg1 : !transform.any_op failures(propagate) {
    ^bb2(%arg2: !transform.any_op):
      test_consume_operand_of_op_kind_or_fail %arg2, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    }
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.23 case_23

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  // expected-error @below {{expects at least one region}}
  transform.alternatives
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.24 case_24

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  // expected-error @below {{expects terminator operands to have the same type as results of the operation}}
  %2 = transform.alternatives %arg1 : !transform.any_op -> !transform.any_op {
  ^bb2(%arg2: !transform.any_op):
    transform.yield %arg2 : !transform.any_op
  }, {
  ^bb2(%arg2: !transform.any_op):
    // expected-note @below {{terminator}}
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.25 case_25

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{expects the entry block to have at least one argument}}
transform.alternatives {
^bb0:
  transform.yield
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.26 case_26

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{used here as operand #0}}
  transform.foreach %0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    transform.test_consume_operand %arg1 : !transform.any_op
  }
  // expected-note @below {{used here as operand #0}}
  transform.test_consume_operand %0 : !transform.any_op
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.27 case_27

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
  %op = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-error @below {{op expects the same number of targets as the body has block arguments}}
  transform.foreach %op : !transform.any_op -> !transform.any_op, !transform.any_value {
  ^bb1(%op_arg: !transform.any_op, %val_arg: !transform.any_value):
    transform.yield %op_arg, %val_arg : !transform.any_op, !transform.any_value
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.28 case_28

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
  %op = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-error @below {{op expects co-indexed targets and the body's block arguments to have the same op/value/param type}}
  transform.foreach %op : !transform.any_op -> !transform.any_value {
  ^bb1(%val_arg: !transform.any_value):
    transform.yield %val_arg : !transform.any_value
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.29 case_29

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
  %op = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-error @below {{op expects the same number of results as the yield terminator has operands}}
  transform.foreach %op : !transform.any_op -> !transform.any_op, !transform.any_op {
  ^bb1(%arg_op: !transform.any_op):
    transform.yield %arg_op : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.30 case_30

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
  %op = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  %val = transform.test_produce_value_handle_to_self_operand %op : (!transform.any_op) -> !transform.any_value
  // expected-error @below {{expects co-indexed results and yield operands to have the same op/value/param type}}
  transform.foreach %op, %val : !transform.any_op, !transform.any_value -> !transform.any_op, !transform.any_value {
  ^bb1(%op_arg: !transform.any_op, %val_arg: !transform.any_value):
    transform.yield %val_arg, %op_arg : !transform.any_value, !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.31 case_31

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{TransformOpInterface requires memory effects on operands to be specified}}
  // expected-note @below {{no effects specified for operand #0}}
  transform.test_required_memory_effects %arg0 {modifies_payload} : (!transform.any_op) -> !transform.any_op
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.32 case_32

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{TransformOpInterface requires 'allocate' memory effect to be specified for results}}
  // expected-note @below {{no 'allocate' effect specified for result #0}}
  transform.test_required_memory_effects %arg0 {has_operand_effect, modifies_payload} : (!transform.any_op) -> !transform.any_op
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.33 case_33

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{attribute can only be attached to operations with symbol tables}}
"test.unknown_container"() { transform.with_named_sequence } : () -> ()
```

**用例输出:**

执行成功，无输出。

---

### 12.2.34 case_34

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{expected a non-empty body block}}
  "transform.named_sequence"() ({
  ^bb0:
  }) { sym_name = "external_named_sequence", function_type = () -> () } : () -> ()

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @external_named_sequence failures(propagate) () : () -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.35 case_35

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{recursion not allowed in named sequences}}
  transform.named_sequence @self_recursion() -> () {
    transform.include @self_recursion failures(suppress) () : () -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.36 case_36

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module @mutual_recursion attributes { transform.with_named_sequence } {
  // expected-note @below {{operation on recursion stack}}  
  transform.named_sequence @foo(%arg0: !transform.any_op) -> () {
    transform.include @bar failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }

  // expected-error @below {{recursion not allowed in named sequences}}
  transform.named_sequence @bar(%arg0: !transform.any_op) -> () {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.37 case_37

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{unknown attribute: "transform.unknown_container"}}
module @unknown_attribute attributes { transform.unknown_container } {}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.38 case_38

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{op does not reference a named transform sequence}}
    transform.include @non_existent failures(propagate) () : () -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.39 case_39

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{requires attribute 'target'}}
    "transform.include"() {failure_propagation_mode = 1 : i32} : () -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.40 case_40

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> () {
    transform.yield
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.any_op):
    // expected-error @below {{incorrect number of operands for callee}}
    transform.include @foo failures(suppress) () : () -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.41 case_41

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> () {
    transform.yield
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.op<"builtin.module">):
    // expected-error @below {{operand type mismatch: expected operand type '!transform.any_op', but provided '!transform.op<"builtin.module">' for operand number 0}}
    transform.include @foo failures(suppress) (%arg1) : (!transform.op<"builtin.module">) -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.42 case_42

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg0 : !transform.any_op
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.any_op):
    // expected-error @below {{incorrect number of results for callee}}
    transform.include @foo failures(suppress) (%arg1) : (!transform.any_op) -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.43 case_43

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg0 : !transform.any_op
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.any_op):
    // expected-error @below {{type of result #0 must implement the same transform dialect interface as the corresponding callee result}}
    transform.include @foo failures(suppress) (%arg1) : (!transform.any_op) -> (!transform.any_value)
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.44 case_44

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{symbol table operation}}
module {
  // expected-error @below {{expects the parent symbol table to have the 'transform.with_named_sequence' attribute}}
  transform.named_sequence @parent_has_no_attributes() {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.45 case_45

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence} {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{op symbol's parent must have the SymbolTable trai}}
    transform.named_sequence @nested() {
      transform.yield
    }
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.46 case_46

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence} {
  func.func private @foo()

  // expected-error @below {{expected 'transform.yield' as terminator}}
  transform.named_sequence @nested() {
    // expected-note @below {{terminator}}
    func.call @foo() : () -> ()
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.47 case_47

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence} {
  func.func private @foo()

  transform.named_sequence @nested(%arg0: !transform.any_op) {
    // expected-error @below {{expected terminator to have as many operands as the parent op has results}}
    transform.yield %arg0 : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.48 case_48

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence} {
  func.func private @foo()

  transform.named_sequence @nested(%arg0: !transform.any_op) -> !transform.op<"builtin.module"> {
    // expected-error @below {{the type of the terminator operand #0 must match the type of the corresponding parent op result}}
    transform.yield %arg0 : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.49 case_49

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{must provide consumed/readonly status for arguments of external or called ops}}
  transform.named_sequence @foo(%op: !transform.any_op )
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.50 case_50

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{argument #0 cannot be both readonly and consumed}}
  transform.named_sequence @foo(%op: !transform.any_op { transform.readonly, transform.consumed } )
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.51 case_51

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{must provide consumed/readonly status for arguments of external or called ops}}
  transform.named_sequence @foo(%op: !transform.any_op) {
    transform.debug.emit_remark_at %op, "message" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.52 case_52

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{argument #0 cannot be both readonly and consumed}}
  transform.named_sequence @foo(%op: !transform.any_op {transform.readonly, transform.consumed}) {
    transform.debug.emit_remark_at %op, "message" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.53 case_53

**功能介绍:**

Note that printing a warning doesn't result in verification failures, so this
also checks for the IR being printed back.

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // Note that printing a warning doesn't result in verification failures, so this
  // also checks for the IR being printed back.
  // expected-warning @below {{argument #0 is not consumed in the body but is marked as consume}}
  transform.named_sequence @emit_warning_only(%op: !transform.any_op {transform.consumed}) {
    transform.debug.emit_remark_at %op, "message" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @emit_warning_only failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @emit_warning_only(%arg0: !transform.any_op {transform.consumed}) {
    transform.debug.emit_remark_at %arg0, "message" : !transform.any_op
    transform.yield 
  }
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @emit_warning_only failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}


```

**重点说明:**

- 输入共15行，输出共10行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 12.2.54 case_54

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-error @below {{argument #0 is consumed in the body but is not marked as such}}
  transform.named_sequence @foo(%op: !transform.any_op {transform.readonly}) {
    transform.test_consume_operand %op : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.55 case_55

**功能介绍:**

Checking that consumptions annotations are used correctly in invocation checks.

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
// Checking that consumptions annotations are used correctly in invocation checks.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%op: !transform.any_op { transform.consumed } )

  // expected-error @below {{'transform.sequence' op block argument #0 has more than one potential consumer}}
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-note @below {{used here as operand #0}}
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    // expected-note @below {{used here as operand #0}}
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.56 case_56

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved matcher symbol @foo}}
    transform.foreach_match in %root
      @foo -> @bar : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.57 case_57

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  func.func private @foo()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved matcher symbol @foo}}
    transform.foreach_match in %root
      @foo -> @bar : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.58 case_58

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved action symbol @bar}}
    transform.foreach_match in %root
      @match -> @bar : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.59 case_59

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  func.func private @bar()
  transform.named_sequence @match()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved action symbol @bar}}
    transform.foreach_match in %root
      @match -> @bar : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.60 case_60

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match(!transform.any_op {transform.readonly}, !transform.any_op {transform.readonly}) -> !transform.any_op
  transform.named_sequence @action(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{the number of operands (1) doesn't match the number of matcher arguments (2) for @match}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.61 case_61

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match(!transform.any_op {transform.readonly}, !transform.any_op {transform.consumed}) -> !transform.any_op
  transform.named_sequence @action(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    %r = transform.replicate num(%root) %root : !transform.any_op, !transform.any_op
    // expected-error @below {{'transform.foreach_match' op does not expect matcher symbol to consume its operand #1}}
    transform.foreach_match in %root, %r
      @match -> @action : (!transform.any_op, !transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.62 case_62

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match(!transform.any_op {transform.readonly}, !transform.any_op {transform.readonly}) -> !transform.any_op
  transform.named_sequence @action(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    %r = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    // expected-error @below {{mismatching type interfaces for operand and matcher argument #1 of matcher @match}}
    transform.foreach_match in %root, %r
      @match -> @action : (!transform.any_op, !transform.any_value) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.63 case_63

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly}) -> !transform.any_op
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @action(!transform.any_op {transform.readonly}) -> !transform.any_op

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{the number of action results (1) for @action doesn't match the number of extra op results (0)}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.64 case_64

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly}) -> !transform.any_op
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @action(!transform.any_op {transform.readonly}) -> !transform.any_op

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{mismatching type interfaces for action result #0 of action @action and op result}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.65 case_65

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly}) -> !transform.any_op
  transform.named_sequence @action()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{mismatching number of matcher results and action arguments between @match (1) and @action (0)}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.66 case_66

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly})
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @action() -> !transform.any_op

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{the number of action results (1) for @action doesn't match the number of extra op results (0)}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.67 case_67

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match()
  transform.named_sequence @action()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{the number of operands (1) doesn't match the number of matcher arguments (0) for @match}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.68 case_68

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match(!transform.any_op {transform.consumed})
  transform.named_sequence @action()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{'transform.foreach_match' op does not expect matcher symbol to consume its operand #0}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.69 case_69

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expected children ops to implement PatternDescriptorOpInterface}}
  transform.apply_patterns to %arg0 {
    // expected-note @below {{op without interface}}
    transform.named_sequence @foo()
  } : !transform.any_op
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.70 case_70

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expected the type of the parameter attribute ('i64') to match the parameter type ('i32')}}
  transform.num_associations %arg0 : (!transform.any_op) -> !transform.param<i32>
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.71 case_71

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{unresolved matcher symbol @missing_symbol}}
    transform.collect_matching @missing_symbol in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.72 case_72

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected the matcher to take one operation handle argument}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher() {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.73 case_73

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected the matcher argument to be marked readonly}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.74 case_74

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected the matcher to yield as many values as op has results (1), got 0}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.75 case_75

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{mismatching type interfaces for matcher result and op result #0}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_value
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.yield %arg0 : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.2.76 case_76

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_matmul(%entry: !transform.any_op) -> () {
    %c3 = transform.param.constant 1 : i64 -> !transform.param<i64>
    // expected-error @below {{op operand #0 must be TransformHandleTypeInterface instance}}
    transform.print %c3 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

## 12.3 preload-library-invalid.mlir

### 12.3.1 case_1

**功能介绍:**

This test checks if the preload mechanism fails gracefully when passed an
invalid transform file.

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file>  -transform-preload-library=transform-library-paths=<input_dir>%{fs-sep}include%{fs-sep}test-interpreter-library-invalid  -transform-interpreter=entry-point=private_helper  -verify-diagnostics
```

**用例输入:**

```mlir
// This test checks if the preload mechanism fails gracefully when passed an
// invalid transform file.
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-library-invalid:0:0: error: unexpected error: '/Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-library-invalid' is neither a file nor a directory

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 12.4 test-tune-extension-invalid.mlir

### 12.4.1 case_1

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error@below {{provided `selected` attribute is not an element of `options` array of attributes}}
    %heads_or_tails = transform.tune.knob<"coin"> = 1 from options = [true, false] -> !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 12.4.2 case_2

**功能介绍:**

无描述

**核心原理:**

测试Transform操作的验证逻辑，确保无效输入被正确拒绝。这些测试验证了编译时的类型检查和约束验证。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @f()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // expected-error@below {{non-deterministic choice "coin" is only resolved through providing a `selected` attr}}
    %heads_or_tails = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

# 13. 检查测试

## 13.1 check-use-after-free.mlir

### 13.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @use_after_free_branching_control_flow() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
    }
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}
```

**用例输出:**

```mlir
module {
  func.func @use_after_free_branching_control_flow() {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_transform_op_with_regions {
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }, {
      "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
    ^bb1:  // pred: ^bb0
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
      "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
    ^bb2:  // pred: ^bb0
      "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
    ^bb3:  // 2 preds: ^bb1, ^bb2
      transform.sequence %0 : !transform.any_op failures(propagate) {
      ^bb0(%arg0: !transform.any_op):
      }
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }
    return
  }
}


```

**重点说明:**

- 输入共24行，输出共21行
- transform.sequence结构被保留并规范化
- 输出被包装在module中

---

### 13.1.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @use_after_free_in_nested_op() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{freed here}}
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  // expected-warning @below {{operand #0 may be used after free}}
  transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
  }
  return
}
```

**用例输出:**

```mlir
module {
  func.func @use_after_free_in_nested_op() {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_transform_op_with_regions {
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }, {
      "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
    ^bb1:  // pred: ^bb0
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
      "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
    ^bb2:  // pred: ^bb0
      "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
    ^bb3:  // 2 preds: ^bb1, ^bb2
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
    }
    return
  }
}


```

**重点说明:**

- 输入共24行，输出共21行
- transform.sequence结构被保留并规范化
- 输出被包装在module中

---

### 13.1.3 case_3

**功能介绍:**

`transform.sequence` has recursive side effects so it has the same "free"
as the child op it contains.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @use_after_free_recursive_side_effects() {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-note @below {{allocated here}}
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) attributes { ord = 1 } {
    ^bb1(%arg1: !transform.any_op):
      yield %arg1 : !transform.any_op
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 2 } {
    ^bb2(%arg2: !transform.any_op):
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 3 } {
    ^bb3(%arg3: !transform.any_op):
    }

    // `transform.sequence` has recursive side effects so it has the same "free"
    // as the child op it contains.
    // expected-note @below {{freed here}}
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 4 } {
    ^bb4(%arg4: !transform.any_op):
      test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    }
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 5 } {
    ^bb3(%arg3: !transform.any_op):
    }
  }
  return
}
```

**用例输出:**

```mlir
module {
  func.func @use_after_free_recursive_side_effects() {
    transform.sequence  failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      %0 = sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) attributes {ord = 1 : i64} {
      ^bb0(%arg1: !transform.any_op):
        yield %arg1 : !transform.any_op
      }
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 2 : i64} {
      ^bb0(%arg1: !transform.any_op):
      }
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 3 : i64} {
      ^bb0(%arg1: !transform.any_op):
      }
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 4 : i64} {
      ^bb0(%arg1: !transform.any_op):
        test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
      }
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 5 : i64} {
      ^bb0(%arg1: !transform.any_op):
      }
    }
    return
  }
}


```

**重点说明:**

- 输入共29行，输出共25行
- transform.sequence结构被保留并规范化
- 输出被包装在module中

---

### 13.1.4 case_4

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @use_after_free() {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-note @below {{allocated here}}
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) attributes { ord = 1 } {
    ^bb1(%arg1: !transform.any_op):
      yield %arg1 : !transform.any_op
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 2 } {
    ^bb2(%arg2: !transform.any_op):
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 3 } {
    ^bb3(%arg3: !transform.any_op):
    }

    // expected-note @below {{freed here}}
    test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 5 } {
    ^bb3(%arg3: !transform.any_op):
    }
  }
  return
}
```

**用例输出:**

```mlir
module {
  func.func @use_after_free() {
    transform.sequence  failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      %0 = sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) attributes {ord = 1 : i64} {
      ^bb0(%arg1: !transform.any_op):
        yield %arg1 : !transform.any_op
      }
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 2 : i64} {
      ^bb0(%arg1: !transform.any_op):
      }
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 3 : i64} {
      ^bb0(%arg1: !transform.any_op):
      }
      test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
      sequence %0 : !transform.any_op failures(propagate) attributes {ord = 5 : i64} {
      ^bb0(%arg1: !transform.any_op):
      }
    }
    return
  }
}


```

**重点说明:**

- 输入共24行，输出共22行
- transform.sequence结构被保留并规范化
- 输出被包装在module中

---

### 13.1.5 case_5

**功能介绍:**

In the case of a control flow cycle, the operation that uses the value may
precede the one that frees it in the same block. Both operations should
be reported as use-after-free.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// In the case of a control flow cycle, the operation that uses the value may
// precede the one that frees it in the same block. Both operations should
// be reported as use-after-free.
func.func @use_after_free_self_cycle() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
  ^bb1:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
    }
    // expected-warning @below {{operand #0 may be used after free}}
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}
```

**用例输出:**

```mlir
module {
  func.func @use_after_free_self_cycle() {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_transform_op_with_regions {
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }, {
      "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
    ^bb1:  // 2 preds: ^bb0, ^bb1
      transform.sequence %0 : !transform.any_op failures(propagate) {
      ^bb0(%arg0: !transform.any_op):
      }
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
      "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
    ^bb2:  // pred: ^bb1
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }
    return
  }
}


```

**重点说明:**

- 输入共26行，输出共19行
- transform.sequence结构被保留并规范化
- 输出被包装在module中

---

### 13.1.6 case_6

**功能介绍:**

Check that the "free" that happens in a cycle is also reported as potential
use-after-free.

**核心原理:**

Transform操作成功执行，对IR进行变换或验证。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Check that the "free" that happens in a cycle is also reported as potential
// use-after-free.
func.func @use_after_free_cycle() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-warning @below {{operand #0 may be used after free}}
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb2, ^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
  ^bb3:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}
```

**用例输出:**

```mlir
module {
  func.func @use_after_free_cycle() {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_transform_op_with_regions {
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }, {
      "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
    ^bb1:  // 2 preds: ^bb0, ^bb2
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
      "transform.test_branching_transform_op_terminator"()[^bb2, ^bb3] : () -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
    ^bb3:  // pred: ^bb1
      "transform.test_branching_transform_op_terminator"() : () -> ()
    }
    return
  }
}


```

**重点说明:**

- 输入共23行，输出共18行
- 输出被包装在module中

---

### 13.1.7 case_7

**功能介绍:**

This should not crash.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// This should not crash.

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  alternatives %arg0 : !transform.any_op {
  ^bb0(%arg1: !transform.any_op):
  }
}
```

**用例输出:**

```mlir
module {
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    alternatives %arg0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
    }
  }
}


```

**重点说明:**

- 输入共8行，输出共8行
- transform.sequence结构被保留并规范化
- 输出被包装在module中

---

### 13.1.8 case_8

**功能介绍:**

This should not crash.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-dialect-check-uses --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// This should not crash.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共11行，输出共9行
- transform.named_sequence定义被保留

---

## 13.2 expensive-checks.mlir

### 13.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor payload op}}
func.func @func() {
  // expected-note @below {{nested payload op}}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @return : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.return"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        // expected-note @below {{handle to invalidated ops}}
        %0 = pdl_match @return in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
        test_consume_operand %1 : !transform.any_op
        // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
        transform.debug.emit_remark_at %0, "remark" : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.2 case_2

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
func.func @func1() {
  // expected-note @below {{repeated target op}}
  return
}
func.func private @func2()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @func : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }
      pdl.pattern @return : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.return"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @func in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = pdl_match @return in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = replicate num(%0) %1 : !transform.any_op, !transform.any_op
        // expected-error @below {{a handle passed as operand #0 and consumed by this operation points to a payload entity more than once}}
        test_consume_operand %2 : !transform.any_op
        transform.debug.emit_remark_at %0, "remark" : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%0: !transform.any_op) {
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.test_copy_payload %0 : (!transform.any_op) ->!transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.4 case_4

**功能介绍:**

Consuming two handles in the same operation is invalid if they point
to overlapping sets of payload IR ops.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%0: !transform.any_op) {
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // Consuming two handles in the same operation is invalid if they point
    // to overlapping sets of payload IR ops.
    //
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities}}
    transform.test_consume_operand %1, %2 : !transform.any_op, !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.5 case_5

**功能介绍:**

Deduplication attribute allows "merge_handles" to take repeated operands.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// Deduplication attribute allows "merge_handles" to take repeated operands.

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%0: !transform.any_op) {
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    transform.merge_handles %1, %2 { deduplicate } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_copy_payload %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.test_copy_payload %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.merge_handles deduplicate %0, %1 : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共11行，输出共8行
- transform.named_sequence定义被保留

---

### 13.2.6 case_6

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{payload value}}
%0 = "test.match_anchor"() : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated handle}}
    %4 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates handles to the same values as associated with it}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %4 : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.7 case_7

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor op associated with the consumed handle}}
// expected-note @below {{payload value}}
// expected-note @below {{op defining the value as result #0}}
%0 = "test.match_anchor"() : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %2 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.8 case_8

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor op associated with the consumed handle}}
"test.match_anchor_1"() ({
^bb0:
  // expected-note @below {{op defining the value as result #0}}
  // expected-note @below {{payload value}}
  %0 = "test.match_anchor_2"() : () -> (i32)
  "test.region_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.9 case_9

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor op associated with the consumed handle}}
// expected-note @below {{op defining the value as block argument #0 of block #0 in region #0}}
"test.match_anchor_1"() ({
// expected-note @below {{payload value}}
^bb0(%arg0: i32):
  %0 = "test.match_anchor_2"() : () -> (i32)
  "test.region_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.10 case_10

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor op associated with the consumed handle}}
"test.match_anchor_1"() ({
^bb:
  // expected-note @below {{op defining the value as block argument #0 of block #0 in region #0}}
  "test.op_with_regions"() ({
  // expected-note @below {{payload value}}
  ^bb0(%arg0: i32):
    %0 = "test.match_anchor_2"() : () -> (i32)
    "test.region_terminator"() : () -> ()
  }): () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.11 case_11

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
// expected-note @below {{consumed handle points to this payload value}}
%0 = "test.match_anchor"() : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op 
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.12 case_12

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// expected-note @below {{ancestor payload op}}
// expected-note @below {{consumed handle points to this payload value}}
%0 = "test.match_anchor_1"() ({
^bb0:
  // expected-note @below {{nested payload op}}
  "test.match_anchor_2"() : () -> ()
  "test.region_terminator"() : () -> ()
}) : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.13 case_13

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
"test.match_anchor_1"() ({
// expected-note @below {{consumed handle points to this payload value}}
^bb0(%arg0: f32):
  // expected-note @below {{ancestor payload op}}
  // expected-note @below {{nested payload op}}
  "test.match_anchor_2"() : () -> ()
  "test.region_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.14 case_14

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
"test.op_with_regions"() ({
// expected-note @below {{consumed handle points to this payload value}}
^bb(%arg0: i32):
  // expected-note @below {{ancestor payload op}}
  "test.op_with_regions"() ({
  ^bb0:
    // expected-note @below {{nested payload op}}
    "test.match_anchor_2"() : () -> ()
    "test.region_terminator"() : () -> ()
  }): () -> ()
  "test.match_anchor_1"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %1, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.15 case_15

**功能介绍:**

Removing a block argument does not invalidate handles to operations in another block.
Not expecting an error here.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// Removing a block argument does not invalidate handles to operations in another block.
// Not expecting an error here.

"test.op_with_regions"() ({
^bb1(%arg0: i32):
  "test.match_anchor_1"() : () -> ()
^bb2:
  "test.match_anchor_2"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %1, 0 : (!transform.any_op) -> !transform.any_value
    transform.test_consume_operand %3 : !transform.any_value
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  "test.op_with_regions"() ({
  ^bb0(%arg0: i32):
    "test.match_anchor_1"() : () -> ()
  ^bb1:  // no predecessors
    "test.match_anchor_2"() : () -> ()
  }) : () -> ()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %2 = transform.test_produce_value_handle_to_argument_of_parent_block %0, 0 : (!transform.any_op) -> !transform.any_value
      transform.test_consume_operand %2 : !transform.any_value
      transform.test_consume_operand %1 : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共20行，输出共18行
- transform.named_sequence定义被保留

---

### 13.2.16 case_16

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_empty_payload : !transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    transform.test_consume_operand %0 : !transform.any_op
    // expected-error @below {{uses a handle associated with empty payload and invalidated by a previously executed transform op}}
    transform.debug.emit_remark_at %0, "remark" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.17 case_17

**功能介绍:**

Make sure we properly report a use-after-consume error when repeated handles
are allowed in the consuming op. We still want to report handles consumed by
_previous_ operations, just not by this one. To bypass the quick static check
of repeated consumption, create a handle to the transform operation and
invalidate the handle to the root module thus invalidating all other handles.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// Make sure we properly report a use-after-consume error when repeated handles
// are allowed in the consuming op. We still want to report handles consumed by
// _previous_ operations, just not by this one. To bypass the quick static check
// of repeated consumption, create a handle to the transform operation and
// invalidate the handle to the root module thus invalidating all other handles.

// expected-note @below {{ancestor payload op}}
module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{handle to invalidated ops}}
    // expected-note @below {{nested payload op}}
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %arg0 : !transform.any_op
    // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %0 { allow_repeated_handles } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.18 case_18

**功能介绍:**

Re-entering the region should not trigger the consumption error from previous
execution of the region.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// Re-entering the region should not trigger the consumption error from previous
// execution of the region.

module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_re_enter_region {
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand %0 : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_re_enter_region {
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand %0 : !transform.any_op
      transform.yield 
    }
    transform.yield 
  }
}


```

**重点说明:**

- 输入共13行，输出共10行
- transform.named_sequence定义被保留

---

### 13.2.19 case_19

**功能介绍:**

Re-entering the region should not trigger the consumption error from previous
execution of the region.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// Re-entering the region should not trigger the consumption error from previous
// execution of the region.

module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_re_enter_region %0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.test_consume_operand %arg1 : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_re_enter_region %0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.test_consume_operand %arg1 : !transform.any_op
      transform.yield 
    }
    transform.yield 
  }
}


```

**重点说明:**

- 输入共14行，输出共11行
- transform.named_sequence定义被保留

---

### 13.2.20 case_20

**功能介绍:**

Consuming the same handle repeatedly in the region should trigger an error.

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
// Consuming the same handle repeatedly in the region should trigger an error.
module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{payload op}}
    // expected-note @below {{handle to invalidated ops}}
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_re_enter_region {
      // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
      // expected-note @below {{invalidated by this transform op}}
      transform.test_consume_operand %0 : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 13.2.21 case_21

**功能介绍:**

Consuming this handle removes the mapping from the current stack frame
mapping and from the caller's stack frame mapping. (If this were not
be the case, the "expensive checks" caching mechanism for op names
would throw an error saying that an op is mapped but not in the cache.)

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt --transform-interpreter --split-input-file --verify-diagnostics <input_file>
```

**用例输入:**

```mlir
module @named_inclusion_and_consumption attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.consumed}) -> () {
    // Consuming this handle removes the mapping from the current stack frame
    // mapping and from the caller's stack frame mapping. (If this were not
    // be the case, the "expensive checks" caching mechanism for op names
    // would throw an error saying that an op is mapped but not in the cache.)
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

```mlir
module @named_inclusion_and_consumption attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.consumed}) {
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield 
  }
}


```

**重点说明:**

- 输入共16行，输出共10行
- transform.named_sequence定义被保留

---

# 14. 模式应用测试

## 14.1 test-pattern-application.mlir

### 14.1.1 case_1

**功能介绍:**

CHECK:   "test.container"() ({
CHECK:     %0 = "test.foo"() {annotated} : () -> i32
CHECK:   }) : () -> ()
Add an attribute to %1, which is now mapped to a new op.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.foo"() {annotated} : () -> i32
//       CHECK:   }) : () -> ()
func.func @update_tracked_op_mapping() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.foo"} : () -> (i32)
  }) : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    // Add an attribute to %1, which is now mapped to a new op.
    transform.annotate %1 "annotated" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @update_tracked_op_mapping() {
    "test.container"() ({
      %0 = "test.foo"() {annotated} : () -> i32
    }) : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.container"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.foo"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.transform.test_patterns
      } : !transform.any_op
      transform.annotate %1 "annotated" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共22行，输出共19行
- transform.named_sequence定义被保留

---

### 14.1.2 case_2

**功能介绍:**

Only one is replaced.
Pattern application will fail because of the upper limit, wrap in
sequence to suppress the error message.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @limited_updates() {
  "test.container"() ({
    // Only one is replaced.
    %0 = "test.foo"() {replace_with_new_op = "test.foo"} : () -> (i32)
    %1 = "test.foo"() {replace_with_new_op = "test.foo"} : () -> (i32)
  }) : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Pattern application will fail because of the upper limit, wrap in
    // sequence to suppress the error message.
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%arg1: !transform.any_op):
      %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.transform.test_patterns
      }  {max_num_rewrites = 1} : !transform.any_op
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @limited_updates() {
    "test.container"() ({
      %0 = "test.foo"() {replace_with_new_op = "test.foo"} : () -> i32
      %1 = "test.foo"() : () -> i32
    }) : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.sequence %arg0 : !transform.any_op failures(suppress) {
      ^bb0(%arg1: !transform.any_op):
        %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        apply_patterns to %0 {
          transform.apply_patterns.transform.test_patterns
        } {max_num_rewrites = 1 : i64} : !transform.any_op
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共24行，输出共22行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 14.1.3 case_3

**功能介绍:**

%1 must be used in some way. If no replacement payload op could be found,
an error is thrown only if the handle is not dead.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @replacement_op_not_found() {
  "test.container"() ({
    // expected-note @below {{[0] replaced op}}
    // expected-note @below {{[0] replacement value 0}}
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  }) : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{replacement is required because this handle must be updated}}
    %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{tracking listener failed to find replacement op during application of this transform op}}
    // expected-note @below {{ran out of suitable replacement values}}
    transform.apply_patterns to %0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    // %1 must be used in some way. If no replacement payload op could be found,
    // an error is thrown only if the handle is not dead.
    transform.annotate %1 "annotated" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.4 case_4

**功能介绍:**

CHECK:   "test.container"() ({
CHECK:     %0 = "test.bar"() : () -> i32
CHECK:   }) : () -> ()
No error because %1 is dead.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.bar"() : () -> i32
//       CHECK:   }) : () -> ()
func.func @replacement_op_for_dead_handle_not_found() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  }) : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // No error because %1 is dead.
    transform.apply_patterns to %0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @replacement_op_for_dead_handle_not_found() {
    "test.container"() ({
      %0 = "test.bar"() : () -> i32
    }) : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.container"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.foo"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.transform.test_patterns
      } : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共21行，输出共18行
- transform.named_sequence定义被保留

---

### 14.1.5 case_5

**功能介绍:**

CHECK:   "test.container"() ({
CHECK:     %0 = "test.bar"() : () -> i32
CHECK:   }) : () -> ()

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.bar"() : () -> i32
//       CHECK:   }) : () -> ()
func.func @replacement_op_not_found_silenced() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  }) : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.transform.test_patterns
    } {transform.silence_tracking_failures} : !transform.any_op
    transform.annotate %1 "annotated" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @replacement_op_not_found_silenced() {
    "test.container"() ({
      %0 = "test.bar"() : () -> i32
    }) : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.container"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.foo"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.transform.test_patterns
      } {transform.silence_tracking_failures} : !transform.any_op
      transform.annotate %1 "annotated" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共21行，输出共19行
- transform.named_sequence定义被保留

---

### 14.1.6 case_6

**功能介绍:**

CHECK:   %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> i32

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> i32
func.func @patterns_apply_only_to_target_body() {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
  %0 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @patterns_apply_only_to_target_body() {
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> i32
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.foo"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.transform.test_patterns
      } : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 14.1.7 case_7

**功能介绍:**

CHECK:   "test.container"() ({
CHECK-NEXT:   ^bb0:
CHECK-NEXT:   }) : () -> ()
No marker should be printed.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   "test.container"() ({
//  CHECK-NEXT:   ^bb0:
//  CHECK-NEXT:   }) : () -> ()
func.func @erase_tracked_op() {
  "test.container"() ({
    // expected-remark @below {{matched op}}
    %0 = "test.erase_op"() {replace_with_new_op = "test.foo"} : () -> (i32)
  }) : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.erase_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %1, "matched op" : !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    // No marker should be printed.
    transform.debug.emit_remark_at %1, "op was deleted" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @erase_tracked_op() {
    "test.container"() ({
    ^bb0:
    }) : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.container"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.erase_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %1, "matched op" : !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.transform.test_patterns
      } : !transform.any_op
      transform.debug.emit_remark_at %1, "op was deleted" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共24行，输出共20行
- transform.named_sequence定义被保留

---

### 14.1.8 case_8

**功能介绍:**

CHECK:   "test.container"() ({
CHECK-NEXT:   ^bb0:
CHECK-NEXT:   }) : () -> ()
No marker should be printed.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   "test.container"() ({
//  CHECK-NEXT:   ^bb0:
//  CHECK-NEXT:   }) : () -> ()
module attributes {transform.with_named_sequence} {
  func.func @erase_tracked_op_in_named_sequence() {
    "test.container"() ({
      // expected-remark @below {{matched op}}
      %0 = "test.erase_op"() {replace_with_new_op = "test.foo"} : () -> (i32)
    }) : () -> ()
    return
  }

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.erase_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %1, "matched op" : !transform.any_op
    transform.include @foo failures(propagate) (%0) : (!transform.any_op) -> ()
    // No marker should be printed.
    transform.debug.emit_remark_at %1, "op was deleted" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func @erase_tracked_op_in_named_sequence() {
    "test.container"() ({
    ^bb0:
    }) : () -> ()
    return
  }
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) {
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.erase_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %1, "matched op" : !transform.any_op
    transform.include @foo failures(propagate) (%0) : (!transform.any_op) -> ()
    transform.debug.emit_remark_at %1, "op was deleted" : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共29行，输出共22行
- transform.named_sequence定义被保留

---

### 14.1.9 case_9

**功能介绍:**

CHECK:   %[[c5:.*]] = arith.constant 5 : index
CHECK:   return %[[c5]]

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @canonicalization(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["tensor.dim"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @canonicalization(%arg0: tensor<5xf32>) -> index {
    %c5 = arith.constant 5 : index
    return %c5 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["tensor.dim"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %1 {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共18行，输出共16行
- transform.named_sequence定义被保留

---

### 14.1.10 case_10

**功能介绍:**

无描述

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below{{target payload op}}
module {
  func.func @invalid_pattern_application_to_transform_ir() {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
      // expected-error @below {{cannot apply transform to itself (or one of its ancestors)}}
      transform.apply_patterns to %arg1 {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
      transform.yield
    }
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.11 case_11

**功能介绍:**

CHECK-NOT:   memref.subview
CHECK-NOT:   memref.copy

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//   CHECK-NOT:   memref.subview
//   CHECK-NOT:   memref.copy
func.func @canonicalization_and_cse(%m: memref<5xf32>) {
  %c2 = arith.constant 2 : index
  %s0 = memref.subview %m[1] [2] [1] : memref<5xf32> to memref<2xf32, strided<[1], offset: 1>>
  %s1 = memref.subview %m[1] [%c2] [1] : memref<5xf32> to memref<?xf32, strided<[1], offset: 1>>
  memref.copy %s0, %s1 : memref<2xf32, strided<[1], offset: 1>> to memref<?xf32, strided<[1], offset: 1>>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @canonicalization_and_cse(%arg0: memref<5xf32>) {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %0 {
        transform.apply_patterns.canonicalization
      } {apply_cse} : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共19行，输出共14行
- transform.named_sequence定义被保留

---

### 14.1.12 case_12

**功能介绍:**

CHECK-NEXT:   %[[m:.*]] = "test.new_op"() : () -> memref<5xf32>
CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
CHECK-NEXT:   return %[[cast]]

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//  CHECK-NEXT:   %[[m:.*]] = "test.new_op"() : () -> memref<5xf32>
//  CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
//  CHECK-NEXT:   return %[[cast]]
func.func @full_dialect_conversion() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  return %0 : tensor<5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.transform.test_conversion_patterns
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {legal_ops = ["func.func", "func.return", "test.new_op"]}
        : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @full_dialect_conversion() -> tensor<5xf32> {
    %0 = "test.new_op"() : () -> memref<5xf32>
    %1 = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
    return %1 : tensor<5xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_conversion_patterns to %0 {
        transform.apply_conversion_patterns.transform.test_conversion_patterns
      } with type_converter {
        transform.apply_conversion_patterns.transform.test_type_converter
      } {legal_ops = ["func.func", "func.return", "test.new_op"]} : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共20行，输出共18行
- transform.named_sequence定义被保留

---

### 14.1.13 case_13

**功能介绍:**

Full dialect conversion fails because test.bar is not replaced and not legal.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Full dialect conversion fails because test.bar is not replaced and not legal.

// expected-note @below{{target op}}
func.func @full_dialect_conversion_failed() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  // expected-error @below{{failed to legalize operation 'test.bar'}}
  "test.bar"() : () -> ()
  return %0 : tensor<5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{dialect conversion failed}}
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.transform.test_conversion_patterns
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {legal_ops = ["func.func", "func.return", "test.new_op"]}
        : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.14 case_14

**功能介绍:**

Partial dialect conversion succeeds because test.bar is not explicitly
illegal.
CHECK-NEXT:   %[[m:.*]] = "test.new_op"() : () -> memref<5xf32>
CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
CHECK-NEXT:   "test.bar"
CHECK-NEXT:   return %[[cast]]

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Partial dialect conversion succeeds because test.bar is not explicitly
// illegal.

//  CHECK-NEXT:   %[[m:.*]] = "test.new_op"() : () -> memref<5xf32>
//  CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
//  CHECK-NEXT:   "test.bar"
//  CHECK-NEXT:   return %[[cast]]
func.func @partial_dialect_conversion() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  "test.bar"() : () -> ()
  return %0 : tensor<5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.transform.test_conversion_patterns
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {legal_ops = ["func.func", "func.return", "test.new_op"],
       partial_conversion} : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @partial_dialect_conversion() -> tensor<5xf32> {
    %0 = "test.new_op"() : () -> memref<5xf32>
    %1 = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
    "test.bar"() : () -> ()
    return %1 : tensor<5xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_conversion_patterns to %0 {
        transform.apply_conversion_patterns.transform.test_conversion_patterns
      } with type_converter {
        transform.apply_conversion_patterns.transform.test_type_converter
      } {legal_ops = ["func.func", "func.return", "test.new_op"], partial_conversion} : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共25行，输出共19行
- transform.named_sequence定义被保留

---

### 14.1.15 case_15

**功能介绍:**

无描述

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{pattern descriptor does not specify type converter and apply_conversion_patterns op has no default type converter}}
    transform.apply_conversion_patterns to %0 {
      // expected-note @below{{pattern descriptor op}}
      transform.apply_conversion_patterns.transform.test_conversion_patterns
    } {illegal_ops = ["test.foo"]} : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.16 case_16

**功能介绍:**

无描述

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      // expected-error @below{{expected LLVMTypeConverter}}
      transform.apply_conversion_patterns.dialect_to_llvm "test"
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {illegal_ops = ["test.foo"],
       legal_ops = ["func.func", "func.return", "test.new_op"]}
        : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.17 case_17

**功能介绍:**

无描述

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      // expected-error @below{{unknown dialect or dialect not loaded: this_dialect_does_not_exist}}
      transform.apply_conversion_patterns.dialect_to_llvm "this_dialect_does_not_exist"
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
    } {illegal_ops = ["test.foo"],
       legal_ops = ["func.func", "func.return", "test.new_op"]}
        : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.18 case_18

**功能介绍:**

无描述

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      // expected-error @below{{dialect does not implement ConvertToLLVMPatternInterface or extension was not loaded: transform}}
      transform.apply_conversion_patterns.dialect_to_llvm "transform"
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
    } {illegal_ops = ["test.foo"],
       legal_ops = ["func.func", "func.return", "test.new_op"]}
        : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 14.1.19 case_19

**功能介绍:**

No op replacement can be found, but there are no handles that must be
updated. No error should be reported.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  func.func @replacement_op_not_found() {
    // No op replacement can be found, but there are no handles that must be
    // updated. No error should be reported.
    "test.container"() ({
      %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
    }) : () -> ()
    return
  }

  transform.named_sequence @patterns(%container: !transform.any_op {transform.readonly}) {
    transform.apply_patterns to %container {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.annotate %1 "annotated" : !transform.any_op
    transform.include @patterns failures(propagate) (%0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func @replacement_op_not_found() {
    "test.container"() ({
      %0 = "test.bar"() : () -> i32
    }) : () -> ()
    return
  }
  transform.named_sequence @patterns(%arg0: !transform.any_op {transform.readonly}) {
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.transform.test_patterns
    } : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["test.container"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.foo"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.annotate %1 "annotated" : !transform.any_op
    transform.include @patterns failures(propagate) (%0) : (!transform.any_op) -> ()
    transform.yield 
  }
}


```

**重点说明:**

- 输入共25行，输出共21行
- transform.named_sequence定义被保留

---

### 14.1.20 case_20

**功能介绍:**

"test.foo" is tracked and replaced with "test.new_op" during a dialect
conversion. Make sure that the handle is updated accordingly.
CHECK-NEXT:   %[[m:.*]] = "test.new_op"() {annotated} : () -> memref<5xf32>
CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
CHECK-NEXT:   return %[[cast]]
Add an attribute to %1, which is now mapped to a new op.

**核心原理:**

模式应用操作，使用PDL(Pattern Description Language)定义和应用重写模式。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// "test.foo" is tracked and replaced with "test.new_op" during a dialect
// conversion. Make sure that the handle is updated accordingly.

//  CHECK-NEXT:   %[[m:.*]] = "test.new_op"() {annotated} : () -> memref<5xf32>
//  CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
//  CHECK-NEXT:   return %[[cast]]
func.func @dialect_conversion_tracking() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  return %0 : tensor<5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.foo"]} in %0 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.transform.test_conversion_patterns
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {legal_ops = ["func.func", "func.return", "test.new_op"], preserve_handles}
        : !transform.any_op
    // Add an attribute to %1, which is now mapped to a new op.
    transform.annotate %1 "annotated" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @dialect_conversion_tracking() -> tensor<5xf32> {
    %0 = "test.new_op"() {annotated} : () -> memref<5xf32>
    %1 = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
    return %1 : tensor<5xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.foo"]} in %0 : (!transform.any_op) -> !transform.any_op
      transform.apply_conversion_patterns to %0 {
        transform.apply_conversion_patterns.transform.test_conversion_patterns
      } with type_converter {
        transform.apply_conversion_patterns.transform.test_type_converter
      } {legal_ops = ["func.func", "func.return", "test.new_op"], preserve_handles} : !transform.any_op
      transform.annotate %1 "annotated" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共26行，输出共20行
- transform.named_sequence定义被保留

---

# 15. 解释器测试

## 15.1 test-interpreter-external-concurrent-source.mlir

### 15.1.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @func_return : benefit(1) {
        %0 = pdl.operation "func.return"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @func_return in %arg1 : (!transform.any_op) -> !transform.op<"func.return">
        transform.debug.emit_remark_at %0, "matched" : !transform.op<"func.return">
      }
    }
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5325735120.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5325735120.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.2 test-interpreter-external-source.mlir

### 15.2.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "outer" : !transform.any_op
    transform.sequence %arg0 : !transform.any_op failures(propagate) attributes {transform.target_tag="transform"} {
    ^bb1(%arg1: !transform.any_op):
      transform.debug.emit_remark_at %arg1, "inner" : !transform.any_op
    }
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378976304.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378976304.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.3 test-interpreter-external-symbol-def-invalid.mlir

### 15.3.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "message" : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378865952.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378865952.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 15.3.2 case_2

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file>
```

**用例输入:**

```mlir
transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378806960.mlir:1:3: error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
  ^
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378806960.mlir:0:0: note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.4 interpreter-entry-point.mlir

### 15.4.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter=entry-point=entry_point  -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence @entry_point(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378591568.mlir:1 offset :1:3: error: unexpected error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @entry_point(!transform.any_op {transform.readonly}) {
  ^
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378591568.mlir:1 offset :0:0: error: unexpected note: symbol table operation
within split at /Volumes/GM9/code/llvm-project/u-u
```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 15.4.2 case_2

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter=entry-point=entry_point  -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    transform.test_transform_op // Note: does not yield remark.
    transform.yield
  }
```

**用例输出:**

```
执行失败: within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4377497088.mlir:1 offset :1:3: error: unexpected error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @__transform_main(!transform.any_op {transform.readonly}) {
  ^
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4377497088.mlir:1 offset :0:0: error: unexpected note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.5 interpreter.mlir

### 15.5.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter  -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }
```

**用例输出:**

```
执行失败: within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378591568.mlir:1 offset :1:3: error: unexpected error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @__transform_main(!transform.any_op {transform.readonly}) {
  ^
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378591568.mlir:1 offset :0:0: error: unexpected note: symbol table operation
within split at /Volumes/GM9/code/llvm-projec
```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

### 15.5.2 case_2

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> -transform-interpreter  -split-input-file -verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence @entry_point(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    transform.test_transform_op // Note: does not yield remark.
    transform.yield
  }
```

**用例输出:**

```
执行失败: within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378473520.mlir:1 offset :1:3: error: unexpected error: expects the parent symbol table to have the 'transform.with_named_sequence' attribute
  transform.named_sequence @entry_point(!transform.any_op {transform.readonly}) {
  ^
within split at /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378473520.mlir:1 offset :0:0: error: unexpected note: symbol table operation

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.6 test-interpreter-debug.mlir

### 15.6.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-payload-root-tag=payload  entry-point=transform})"  --allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{could not find the operation with transform.target_tag="payload" attribute}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @transform(%arg0: !transform.any_op) {
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.6.2 case_2

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-payload-root-tag=payload  entry-point=transform})"  --allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{could not find a nested named sequence with name: transform}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @not_transform(%arg0: !transform.any_op) {
    transform.yield
  }

  module attributes {transform.target_tag="payload"} {}
}
```

**用例输出:**

执行成功，无输出。

---

### 15.6.3 case_3

**功能介绍:**

This will not be executed.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module(transform-interpreter{ debug-payload-root-tag=payload  entry-point=transform})"  --allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @transform(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "payload" : !transform.any_op
    transform.yield
  }

  // This will not be executed.
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "some other text that is not printed" : !transform.any_op
    transform.yield
  }

  module {
    module {}
    // expected-remark @below {{payload}}
    module attributes {transform.target_tag="payload"} {}
    module {}
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @transform(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "payload" : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "some other text that is not printed" : !transform.any_op
    transform.yield 
  }
  module {
    module {
    }
    module attributes {transform.target_tag = "payload"} {
    }
    module {
    }
  }
}


```

**重点说明:**

- 输入共19行，输出共18行
- transform.named_sequence定义被保留

---

## 15.7 test-interpreter-external-concurrent.mlir

### 15.7.1 case_1

**功能介绍:**

Exercising the pass on multiple functions of different lengths that may be
processed concurrently. This should expose potential races.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module( transform-preload-library{transform-library-paths=<input_dir>%{fs-sep}include%{fs-sep}test-interpreter-external-concurrent-source.mlir}, func.func(transform-interpreter))"  --verify-diagnostics
```

**用例输入:**

```mlir
// Exercising the pass on multiple functions of different lengths that may be
// processed concurrently. This should expose potential races.

func.func @f1() {
  // expected-remark @below {{matched}}
  return
}

func.func @f2() {
  // expected-remark @below {{matched}}
  return
}

func.func @f3() {
  call @f2() : () -> ()
  call @f2() : () -> ()
  call @f5() : () -> ()
  call @f7() : () -> ()
  call @f5() : () -> ()
  call @f5() : () -> ()
  // expected-remark @below {{matched}}
  return
}

func.func @f4() {
  call @f3() : () -> ()
  call @f3() : () -> ()
  // expected-remark @below {{matched}}
  return
}

func.func @f5() {
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f1() : () -> ()
  call @f1() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  call @f7() : () -> ()
  // expected-remark @below {{matched}}
  return
}

func.func @f6() {
  // expected-remark @below {{matched}}
  return
}

func.func @f7() {
  // expected-remark @below {{matched}}
  return
}
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-external-concurrent-source.mlir:0:0: error: unexpected error: '/Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-external-concurrent-source.mlir' is neither a file nor a directory
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5343719936.mlir:10:6: error: expected remark "matched" was not produced
  // expected-remark @below {{matched}}
     ^~~~~~~
/Volumes/GM9/code
```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.8 test-interpreter-external.mlir

### 15.8.1 case_1

**功能介绍:**

The schedule in the separate file emits remarks at the payload root.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --pass-pipeline="builtin.module( transform-preload-library{transform-library-paths=<input_dir>%{fs-sep}include%{fs-sep}test-interpreter-external-source.mlir}, transform-interpreter)"  --verify-diagnostics
```

**用例输入:**

```mlir
// The schedule in the separate file emits remarks at the payload root.

// expected-remark @below {{outer}}
// expected-remark @below {{inner}}
module {}
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-external-source.mlir:0:0: error: unexpected error: '/Volumes/GM9/code/llvm-project/u-unread/temp%{fs-sep}include%{fs-sep}test-interpreter-external-source.mlir' is neither a file nor a directory
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_4378941424.mlir:8:4: error: expected remark "outer" was not produced
// expected-remark @below {{outer}}
   ^~~~~
/Volumes/GM9/code/llvm-project/u-unread/temp/temp_
```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.9 test-interpreter-printing.mlir

### 15.9.1 case_1

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --allow-unregistered-dialect --verify-diagnostics
```

**用例输入:**

```mlir
transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.print {name = "START"}

    transform.print {name = "Local scope", use_local_scope}

    %baz = transform.structured.match ops{["test.baz"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    transform.print %baz : !transform.any_op

    transform.print %baz {name = "Baz"} : !transform.any_op

    transform.print %baz {name = "No region", skip_regions} : !transform.any_op

    transform.test_produce_invalid_ir %baz : !transform.any_op
    transform.print %baz {name = "No verify", assume_verified} : !transform.any_op

    transform.print {name = "END"}
    transform.yield
  }
}
```

**用例输出:**

```
执行失败: /Volumes/GM9/code/llvm-project/u-unread/temp/temp_68304_5604531712.mlir:40:4: error: unexpected error: expected operation name in quotes
  }
   ^

```

**重点说明:** 此用例执行失败，可能包含预期错误或需要特殊环境配置。

---

## 15.10 test-interpreter.mlir

### 15.10.1 case_1

**功能介绍:**

UNSUPPORTED: target=aarch64-pc-windows-msvc

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// UNSUPPORTED: target=aarch64-pc-windows-msvc

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_transform_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共9行，输出共6行
- transform.named_sequence定义被保留

---

### 15.10.2 case_2

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand {foo = "bar"} : () -> !transform.any_op
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共8行，输出共7行
- transform.named_sequence定义被保留

---

### 15.10.3 case_3

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op
    // expected-error @below {{expected the operand to be associated a payload op of kind transform.sequence got transform.test_produce_self_handle_or_forward_operand}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.4 case_4

**功能介绍:**

It is okay to have multiple handles to the same payload op as long
as only one of them is consumed. The expensive checks mode is necessary
to detect double-consumption.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// It is okay to have multiple handles to the same payload op as long
// as only one of them is consumed. The expensive checks mode is necessary
// to detect double-consumption.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand {foo = "bar"} : () -> !transform.any_op
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共12行，输出共8行
- transform.named_sequence定义被保留

---

### 15.10.5 case_5

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      // expected-remark @below {{applying transformation "a"}}
      test_transform_op "a"
      // expected-remark @below {{applying transformation "b"}}
      test_transform_op "b"
      // expected-remark @below {{applying transformation "c"}}
      test_transform_op "c"
    }
    // expected-remark @below {{applying transformation "d"}}
    transform.test_transform_op "d"
    // expected-remark @below {{applying transformation "e"}}
    transform.test_transform_op "e"
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      test_transform_op "a"
      test_transform_op "b"
      test_transform_op "c"
    }
    transform.test_transform_op "d"
    transform.test_transform_op "e"
    transform.yield 
  }
}


```

**重点说明:**

- 输入共18行，输出共13行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.6 case_6

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      // expected-remark @below {{succeeded}}
      test_consume_operand_of_op_kind_or_fail %arg1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      test_consume_operand_of_op_kind_or_fail %arg1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    }
    transform.yield 
  }
}


```

**重点说明:**

- 输入共11行，输出共10行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.7 case_7

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      %1 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      yield %1 : !transform.any_op
    }
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      %1 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      yield %1 : !transform.any_op
    }
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield 
  }
}


```

**重点说明:**

- 输入共12行，输出共11行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.8 case_8

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{parent function}}
func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

// expected-remark @below {{parent function}}
func.func @bar() {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @const : benefit(1) {
        %r = pdl.types
        %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
        pdl.rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %f = pdl_match @const in %arg1 : (!transform.any_op) -> !transform.any_op
        %m = get_parent_op %f {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.debug.emit_remark_at %m, "parent function" : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo() {
    %c0_i32 = arith.constant 0 : i32
    return
  }
  func.func @bar() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @const : benefit(1) {
          %0 = types
          %1 = operation "arith.constant"  -> (%0 : !pdl.range<type>)
          rewrite %1 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @const in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
          transform.debug.emit_remark_at %1, "parent function" : !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共33行，输出共30行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.9 case_9

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @test_get_nth_parent() {
  "test.foo"() ({
    // expected-remark @below{{2nd parent}}
    "test.foo"() ({
      "test.qux"() ({
        // expected-remark @below{{1st parent}}
        "test.foo"() ({
          "test.bar"() : () -> ()
        }) : () -> ()
      }) : () -> ()
    }) : () -> ()
  }) : () -> ()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %f = transform.structured.match ops{["test.bar"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %parent = transform.get_parent_op %f {nth_parent = 1, op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %parent, "1st parent" : !transform.any_op
    %parent2 = transform.get_parent_op %f {nth_parent = 2, op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %parent2, "2nd parent" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_get_nth_parent() {
    "test.foo"() ({
      "test.foo"() ({
        "test.qux"() ({
          "test.foo"() ({
            "test.bar"() : () -> ()
          }) : () -> ()
        }) : () -> ()
      }) : () -> ()
    }) : () -> ()
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.bar"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_parent_op %0 {op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %1, "1st parent" : !transform.any_op
      %2 = transform.get_parent_op %0 {nth_parent = 2 : i64, op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %2, "2nd parent" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共24行，输出共23行
- transform.named_sequence定义被保留

---

### 15.10.10 case_10

**功能介绍:**

This is necessary to run the transformation on something other than the
top-level module, "alternatives" cannot be run on that.
This operation fails, which triggers the next alternative without
reporting the error.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_func : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        // This is necessary to run the transformation on something other than the
        // top-level module, "alternatives" cannot be run on that.
        %0 = pdl_match @match_func in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.alternatives %0 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %1 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
          // This operation fails, which triggers the next alternative without
          // reporting the error.
          transform.test_consume_operand_of_op_kind_or_fail %1, "transform.sequence" : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %1 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
          // expected-remark @below {{succeeded}}
          transform.test_consume_operand_of_op_kind_or_fail %1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
        }
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo() {
    %c0_i32 = arith.constant 0 : i32
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @match_func : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "func.func"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @match_func in %arg2 : (!transform.any_op) -> !transform.any_op
          alternatives %0 : !transform.any_op {
          ^bb0(%arg3: !transform.any_op):
            %1 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
            transform.test_consume_operand_of_op_kind_or_fail %1, "transform.sequence" : !transform.any_op
          }, {
          ^bb0(%arg3: !transform.any_op):
            %1 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
            transform.test_consume_operand_of_op_kind_or_fail %1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
          }
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共38行，输出共33行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.11 case_11

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @bar()

func.func @foo() {
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{all alternatives failed}}
        transform.alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
        }
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.12 case_12

**功能介绍:**

This alternative succeeds.
This alternative is never run, so we must not have a remark here.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @bar()

func.func @foo() {
  // expected-remark @below {{still here}}
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.debug.emit_remark_at %2, "still here" : !transform.any_op
          // This alternative succeeds.
        }, {
        ^bb2(%arg2: !transform.any_op):
          // This alternative is never run, so we must not have a remark here.
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.test_emit_remark_and_erase_operand %2, "should not happen" {fail_after_erase} : !transform.any_op
        }
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @bar()
  func.func @foo() {
    call @bar() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @match_call : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "func.call"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
          alternatives %1 : !transform.any_op {
          ^bb0(%arg3: !transform.any_op):
            %2 = transform.pdl_match @match_call in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
          }, {
          ^bb0(%arg3: !transform.any_op):
            %2 = transform.pdl_match @match_call in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.debug.emit_remark_at %2, "still here" : !transform.any_op
          }, {
          ^bb0(%arg3: !transform.any_op):
            %2 = transform.pdl_match @match_call in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.test_emit_remark_and_erase_operand %2, "should not happen" {fail_after_erase} : !transform.any_op
          }
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共44行，输出共39行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.13 case_13

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @bar()

func.func @erase_call() {
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying second time}}
          transform.test_emit_remark_and_erase_operand %2, "applying second time" : !transform.any_op
        }
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @bar()
  func.func @erase_call() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @match_call : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "func.call"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
          alternatives %1 : !transform.any_op {
          ^bb0(%arg3: !transform.any_op):
            %2 = transform.pdl_match @match_call in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
          }, {
          ^bb0(%arg3: !transform.any_op):
            %2 = transform.pdl_match @match_call in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.test_emit_remark_and_erase_operand %2, "applying second time" : !transform.any_op
          }
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共38行，输出共34行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.14 case_14

**功能介绍:**

The first alternative failed, so the returned value is taken from the
second alternative, associated test_produce_self_handle_or_forward_operand rather
than pdl_match.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @bar()

func.func @foo() {
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        %2 = transform.alternatives %1 : !transform.any_op -> !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %3 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %3, "applying" {fail_after_erase} : !transform.any_op
          %4 = transform.test_produce_self_handle_or_forward_operand %3 : (!transform.any_op) -> !transform.any_op
          transform.yield %4 : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %4 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
          transform.yield %4 : !transform.any_op
        }
        // The first alternative failed, so the returned value is taken from the
        // second alternative, associated test_produce_self_handle_or_forward_operand rather
        // than pdl_match.
        // expected-remark @below {{succeeded}}
        transform.test_consume_operand_of_op_kind_or_fail %2, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @bar()
  func.func @foo() {
    call @bar() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @match_call : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "func.call"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
          %2 = alternatives %1 : !transform.any_op -> !transform.any_op {
          ^bb0(%arg3: !transform.any_op):
            %3 = transform.pdl_match @match_call in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.test_emit_remark_and_erase_operand %3, "applying" {fail_after_erase} : !transform.any_op
            %4 = transform.test_produce_self_handle_or_forward_operand %3 : (!transform.any_op) -> !transform.any_op
            transform.yield %4 : !transform.any_op
          }, {
          ^bb0(%arg3: !transform.any_op):
            %3 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
            transform.yield %3 : !transform.any_op
          }
          test_consume_operand_of_op_kind_or_fail %2, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共44行，输出共38行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.15 case_15

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{scope}}
module attributes {transform.with_named_sequence} {
  func.func @foo() {
    %0 = arith.constant 0 : i32
    return
  }

  func.func @bar() {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    return
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    // expected-error @below {{scope must not contain the transforms being applied}}
    transform.alternatives %arg1 : !transform.any_op {
    ^bb2(%arg2: !transform.any_op):
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    }, {
    ^bb2(%arg2: !transform.any_op):
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.16 case_16

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo(%arg0: index, %arg1: index, %arg2: index) {
  // expected-note @below {{scope}}
  scf.for %i = %arg0 to %arg1 step %arg2 {
    %0 = arith.constant 0 : i32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_const : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.constant"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }


      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = transform.pdl_match @match_const in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{only isolated-from-above ops can be alternative scopes}}
        alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
        }
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.17 case_17

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{application of transform.test_wrong_number_of_results expected to produce 3 results (actually produced 1).}}
        // expected-note @below {{if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
        transform.test_wrong_number_of_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.18 case_18

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  "op" () : () -> ()
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{application of transform.test_wrong_number_of_multi_results expected to produce 1 results (actually produced 0)}}
        // expected-note @below {{if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
        transform.test_wrong_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op)
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.19 case_19

**功能介绍:**

Transform matches 3 ops and produces 2 results.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  "op" () : () -> ()
  "op" () : () -> ()
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // Transform matches 3 ops and produces 2 results.
        %1:2 = transform.test_correct_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo() {
    "foo"() : () -> ()
    "foo"() : () -> ()
    "op"() : () -> ()
    "foo"() : () -> ()
    "foo"() : () -> ()
    "op"() : () -> ()
    "foo"() : () -> ()
    "foo"() : () -> ()
    "op"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @some : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "op"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          %result1, %result2 = test_correct_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共28行，输出共33行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.20 case_20

**功能介绍:**

Transform fails to match any but still produces 2 results.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  "wrong_op_name" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // Transform fails to match any but still produces 2 results.
        %1:2 = transform.test_correct_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo() {
    "wrong_op_name"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @some : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "op"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          %result1, %result2 = test_correct_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共26行，输出共25行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.21 case_21

**功能介绍:**

This should not fail.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// This should not fail.

func.func @foo() {
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.test_mixed_null_and_non_null_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo() {
    "foo"() : () -> ()
    "op"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @some : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "op"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          %null, %non_null = test_mixed_null_and_non_null_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共27行，输出共26行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.22 case_22

**功能介绍:**

Expecting to match all operations by merging the handles that matched addi
and subi separately.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Expecting to match all operations by merging the handles that matched addi
// and subi separately.
func.func @foo(%arg0: index) {
  // expected-remark @below {{matched}}
  %0 = arith.addi %arg0, %arg0 : index
  // expected-remark @below {{matched}}
  %1 = arith.subi %arg0, %arg0 : index
  // expected-remark @below {{matched}}
  %2 = arith.addi %0, %1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @addi : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.addi"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }
      pdl.pattern @subi : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.subi"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @addi in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = pdl_match @subi in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = merge_handles %0, %1 : !transform.any_op
        transform.debug.emit_remark_at %2, "matched" : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo(%arg0: index) {
    %0 = arith.addi %arg0, %arg0 : index
    %1 = arith.subi %arg0, %arg0 : index
    %2 = arith.addi %0, %1 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @addi : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "arith.addi"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        pdl.pattern @subi : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "arith.subi"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @addi in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = pdl_match @subi in %arg2 : (!transform.any_op) -> !transform.any_op
          %2 = merge_handles %0, %1 : !transform.any_op
          transform.debug.emit_remark_at %2, "matched" : !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共40行，输出共35行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.23 case_23

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo(%arg0: index) {
  %0 = arith.addi %arg0, %arg0 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @addi : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.addi"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @addi in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = pdl_match @addi in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = merge_handles deduplicate %0, %1 : !transform.any_op
        %3 = num_associations %2 : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{1}}
        transform.debug.emit_param_as_remark  %3 : !transform.param<i64>
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo(%arg0: index) {
    %0 = arith.addi %arg0, %arg0 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @addi : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "arith.addi"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @addi in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = pdl_match @addi in %arg2 : (!transform.any_op) -> !transform.any_op
          %2 = merge_handles deduplicate %0, %1 : !transform.any_op
          %3 = num_associations %2 : (!transform.any_op) -> !transform.param<i64>
          transform.debug.emit_param_as_remark %3 : !transform.param<i64>
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共29行，输出共28行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.24 case_24

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  "op" () { target_me } : () -> ()
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{failed to apply}}
        transform.test_mixed_success_and_silenceable %0 : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.25 case_25

**功能介绍:**

Not expecting error here because we are suppressing it.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(suppress) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // Not expecting error here because we are suppressing it.
        // expected-remark @below {{foo}}
        test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase} : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @foo() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @some : benefit(1) {
          %0 = operands
          %1 = types
          %2 = operation "op"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
          rewrite %2 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(suppress) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase} : !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共27行，输出共24行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.26 case_26

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @foo() {
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{silenceable error}}
        // expected-remark @below {{foo}}
        test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase} : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.27 case_27

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func private @foo()
  func.func private @bar()

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {

    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @func : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @func in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = replicate num(%0) %arg1 : !transform.any_op, !transform.any_op
        %p = num_associations %1 : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{2}}
        transform.debug.emit_param_as_remark  %p : !transform.param<i64>
        %2 = replicate num(%0) %1 : !transform.any_op, !transform.any_op
        %p2 = num_associations %2 : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{4}}
        transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  func.func private @foo()
  func.func private @bar()
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.with_pdl_patterns %arg0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      pdl.pattern @func : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.func"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }
      sequence %arg1 : !transform.any_op failures(propagate) {
      ^bb0(%arg2: !transform.any_op):
        %0 = pdl_match @func in %arg2 : (!transform.any_op) -> !transform.any_op
        %1 = replicate num(%0) %arg2 : !transform.any_op, !transform.any_op
        %2 = num_associations %1 : (!transform.any_op) -> !transform.param<i64>
        transform.debug.emit_param_as_remark %2 : !transform.param<i64>
        %3 = replicate num(%0) %1 : !transform.any_op, !transform.any_op
        %4 = num_associations %3 : (!transform.any_op) -> !transform.param<i64>
        transform.debug.emit_param_as_remark %4 : !transform.param<i64>
      }
    }
    transform.yield 
  }
}


```

**重点说明:**

- 输入共31行，输出共26行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.28 case_28

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @bar() {
  // expected-remark @below {{transform applied}}
  %0 = arith.constant 0 : i32
  // expected-remark @below {{transform applied}}
  %1 = arith.constant 1 : i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @const : benefit(1) {
        %r = pdl.types
        %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
        pdl.rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %f = pdl_match @const in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %f : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %p = transform.num_associations %arg2 : (!transform.any_op) -> !transform.param<i64>
          // expected-remark @below {{1}}
          transform.debug.emit_param_as_remark  %p : !transform.param<i64>
          transform.debug.emit_remark_at %arg2, "transform applied" : !transform.any_op
        }
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @bar() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @const : benefit(1) {
          %0 = types
          %1 = operation "arith.constant"  -> (%0 : !pdl.range<type>)
          rewrite %1 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @const in %arg2 : (!transform.any_op) -> !transform.any_op
          foreach %0 : !transform.any_op {
          ^bb0(%arg3: !transform.any_op):
            %1 = transform.num_associations %arg3 : (!transform.any_op) -> !transform.param<i64>
            transform.debug.emit_param_as_remark %1 : !transform.param<i64>
            transform.debug.emit_remark_at %arg3, "transform applied" : !transform.any_op
          }
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共33行，输出共30行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.29 case_29

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %results, %types = transform.foreach %0 : !transform.any_op -> !transform.any_value, !transform.any_param {
    ^bb0(%op0 : !transform.any_op):
      %result = transform.get_result %op0[0] : (!transform.any_op) -> !transform.any_value
      %type = transform.get_type elemental %result  : (!transform.any_value) -> !transform.any_param
      transform.yield %result, %type : !transform.any_value, !transform.any_param
    }
    transform.debug.emit_remark_at %results, "result selected" : !transform.any_value
    transform.debug.emit_param_as_remark %types, "elemental types" at %0 : !transform.any_param, !transform.any_op

    transform.yield
  }
}

func.func @payload(%lhs: tensor<10x20xf16>,
                   %rhs: tensor<20x15xf32>) -> (tensor<10x15xf64>, tensor<10x15xf32>) {
  %cst64 = arith.constant 0.0 : f64
  %empty64 = tensor.empty() : tensor<10x15xf64>
  %fill64 = linalg.fill ins(%cst64 : f64) outs(%empty64 : tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-remark @below {{result selected}}
  // expected-note @below {{value handle points to an op result #0}}
  // expected-remark @below {{elemental types f64, f32}}
  %result64 = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>)
                         outs(%fill64: tensor<10x15xf64>) -> tensor<10x15xf64>

  %cst32 = arith.constant 0.0 : f32
  %empty32 = tensor.empty() : tensor<10x15xf32>
  %fill32 = linalg.fill ins(%cst32 : f32) outs(%empty32 : tensor<10x15xf32>) -> tensor<10x15xf32>
  // expected-remark @below {{result selected}}
  // expected-note @below {{value handle points to an op result #0}}
  // expected-remark @below {{elemental types f64, f32}}
  %result32 = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>)
                           outs(%fill32: tensor<10x15xf32>) -> tensor<10x15xf32>

  return %result64, %result32 : tensor<10x15xf64>, tensor<10x15xf32>

}
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1:2 = transform.foreach %0 : !transform.any_op -> !transform.any_value, !transform.any_param {
      ^bb0(%arg1: !transform.any_op):
        %2 = transform.get_result %arg1[0] : (!transform.any_op) -> !transform.any_value
        %3 = transform.get_type elemental %2 : (!transform.any_value) -> !transform.any_param
        transform.yield %2, %3 : !transform.any_value, !transform.any_param
      }
      transform.debug.emit_remark_at %1#0, "result selected" : !transform.any_value
      transform.debug.emit_param_as_remark %1#1, "elemental types" at %0 : !transform.any_param, !transform.any_op
      transform.yield 
    }
  }
  func.func @payload(%arg0: tensor<10x20xf16>, %arg1: tensor<20x15xf32>) -> (tensor<10x15xf64>, tensor<10x15xf32>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty() : tensor<10x15xf64>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<10x15xf64>) -> tensor<10x15xf64>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf16>, tensor<20x15xf32>) outs(%1 : tensor<10x15xf64>) -> tensor<10x15xf64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %3 = tensor.empty() : tensor<10x15xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<10x15xf32>) -> tensor<10x15xf32>
    %5 = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf16>, tensor<20x15xf32>) outs(%4 : tensor<10x15xf32>) -> tensor<10x15xf32>
    return %2, %5 : tensor<10x15xf64>, tensor<10x15xf32>
  }
}


```

**重点说明:**

- 输入共39行，输出共27行
- transform.named_sequence定义被保留

---

### 15.10.30 case_30

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @two_const_ops() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %two_ops = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one_param = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    // expected-error @below {{prior targets' payload size (2) differs from payload size (1) of target}}
    transform.foreach %two_ops, %one_param : !transform.any_op, !transform.test_dialect_param {
    ^bb2(%op: !transform.any_op, %param: !transform.test_dialect_param):
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.31 case_31

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @one_const_op() {
  %0 = arith.constant 0 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %one_op = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one_val = transform.test_produce_value_handle_to_self_operand %one_op : (!transform.any_op) -> !transform.any_value
    %param_one = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    %param_two = transform.param.constant 2 : i32 -> !transform.test_dialect_param
    %two_params = transform.merge_handles %param_one, %param_two : !transform.test_dialect_param

    // expected-error @below {{prior targets' payload size (1) differs from payload size (2) of target}}
    transform.foreach %one_val, %one_op, %two_params : !transform.any_value, !transform.any_op, !transform.test_dialect_param {
    ^bb2(%val: !transform.any_value, %op: !transform.any_op, %param: !transform.test_dialect_param):
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.32 case_32

**功能介绍:**

CHECK-NEXT:   return

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//  CHECK-NEXT:   return
func.func @consume_in_foreach() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index
  %3 = arith.constant 3 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %f = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %f : !transform.any_op {
    ^bb2(%arg2: !transform.any_op):
      // expected-remark @below {{erasing}}
      transform.test_emit_remark_and_erase_operand %arg2, "erasing" : !transform.any_op
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @consume_in_foreach() {
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.constant"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.foreach %0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        transform.test_emit_remark_and_erase_operand %arg1, "erasing" : !transform.any_op
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共20行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.33 case_33

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @bar() {
  scf.execute_region {
    // expected-remark @below {{transform applied}}
    %0 = arith.constant 0 : i32
    scf.yield
  }

  scf.execute_region {
    // expected-remark @below {{transform applied}}
    %1 = arith.constant 1 : i32
    // expected-remark @below {{transform applied}}
    %2 = arith.constant 2 : i32
    scf.yield
  }

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @const : benefit(1) {
        %r = pdl.types
        %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
        pdl.rewrite %0 with "transform.dialect"
      }

      pdl.pattern @execute_region : benefit(1) {
        %r = pdl.types
        %0 = pdl.operation "scf.execute_region" -> (%r : !pdl.range<type>)
        pdl.rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %f = pdl_match @execute_region in %arg1 : (!transform.any_op) -> !transform.any_op
        %results = transform.foreach %f : !transform.any_op -> !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %g = transform.pdl_match @const in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.yield %g : !transform.any_op
        }

        %p = transform.num_associations %results : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{3}}
        transform.debug.emit_param_as_remark  %p : !transform.param<i64>
        transform.debug.emit_remark_at %results, "transform applied" : !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @bar() {
    scf.execute_region {
      %c0_i32 = arith.constant 0 : i32
      scf.yield
    }
    scf.execute_region {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      scf.yield
    }
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @const : benefit(1) {
          %0 = types
          %1 = operation "arith.constant"  -> (%0 : !pdl.range<type>)
          rewrite %1 with "transform.dialect"
        }
        pdl.pattern @execute_region : benefit(1) {
          %0 = types
          %1 = operation "scf.execute_region"  -> (%0 : !pdl.range<type>)
          rewrite %1 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @execute_region in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = foreach %0 : !transform.any_op -> !transform.any_op {
          ^bb0(%arg3: !transform.any_op):
            %3 = transform.pdl_match @const in %arg3 : (!transform.any_op) -> !transform.any_op
            transform.yield %3 : !transform.any_op
          }
          %2 = num_associations %1 : (!transform.any_op) -> !transform.param<i64>
          transform.debug.emit_param_as_remark %2 : !transform.param<i64>
          transform.debug.emit_remark_at %1, "transform applied" : !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共52行，输出共44行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.34 case_34

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-remark @below {{found muli}}
  %0 = arith.muli %arg0, %arg1 : index
  arith.addi %0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %muli = transform.get_producer_of_operand %addi[0] : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %muli, "found muli" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.addi %0, %arg1 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_producer_of_operand %0[0] : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %1, "found muli" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.35 case_35

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  %0 = arith.muli %arg0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not find a producer for operand number: 0 of}}
    %bbarg = transform.get_producer_of_operand %muli[0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.36 case_36

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_consumer(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  // expected-remark @below {{found addi}}
  arith.addi %0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %addi = transform.get_consumers_of_result %muli[0] : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %addi, "found addi" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_consumer(%arg0: index, %arg1: index) {
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.addi %0, %arg1 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.muli"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_consumers_of_result %0[0] : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %1, "found addi" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.37 case_37

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_consumer_fail_1(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  %1 = arith.muli %arg0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{handle must be mapped to exactly one payload op}}
    %bbarg = transform.get_consumers_of_result %muli[0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.38 case_38

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_consumer_fail_2(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{result number overflow}}
    %bbarg = transform.get_consumers_of_result %muli[1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.39 case_39

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    %h:2 = transform.split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected to contain 3 payloads but it contains 2 payloads}}
    %h_2:3 = transform.split_handle %muli_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.40 case_40

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op failures(suppress) {
    ^bb1(%fun: !transform.any_op):
      %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
      %h:2 = split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
      // expected-remark @below {{1}}
      transform.debug.emit_param_as_remark  %p : !transform.param<i64>
      %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
      // Silenceable failure and all handles are now empty.
      %h_2:3 = split_handle %muli_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %p2 = transform.num_associations %h_2#0 : (!transform.any_op) -> !transform.param<i64>
      // expected-remark @below {{0}}
      transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @split_handle(%arg0: index, %arg1: index, %arg2: index) {
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.muli %arg0, %arg2 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.sequence %arg0 : !transform.any_op failures(suppress) {
      ^bb0(%arg1: !transform.any_op):
        %0 = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %1:2 = split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %2 = num_associations %1#0 : (!transform.any_op) -> !transform.param<i64>
        transform.debug.emit_param_as_remark %2 : !transform.param<i64>
        %3 = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %4:3 = split_handle %3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %5 = num_associations %4#0 : (!transform.any_op) -> !transform.param<i64>
        transform.debug.emit_param_as_remark %5 : !transform.param<i64>
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共25行，输出共23行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.41 case_41

**功能介绍:**

No error, last result handle is empty.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    // No error, last result handle is empty.
    %h:3 = transform.split_handle %muli_2 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    %p2 = transform.num_associations %h#1 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
    %p3 = transform.num_associations %h#2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{0}}
    transform.debug.emit_param_as_remark  %p3 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @split_handle(%arg0: index, %arg1: index, %arg2: index) {
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.muli %arg0, %arg2 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.muli"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1:3 = transform.split_handle %0 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %2 = transform.num_associations %1#0 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %2 : !transform.param<i64>
      %3 = transform.num_associations %1#1 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %3 : !transform.param<i64>
      %4 = transform.num_associations %1#2 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %4 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共23行，输出共20行
- transform.named_sequence定义被保留

---

### 15.10.42 case_42

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  %2 = arith.muli %a, %c : index
  %3 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    %h:2 = transform.split_handle %muli_2 {overflow_result = 0} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{3}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    %p2 = transform.num_associations %h#1 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @split_handle(%arg0: index, %arg1: index, %arg2: index) {
    %0 = arith.muli %arg0, %arg1 : index
    %1 = arith.muli %arg0, %arg2 : index
    %2 = arith.muli %arg0, %arg2 : index
    %3 = arith.muli %arg0, %arg2 : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.muli"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1:2 = transform.split_handle %0 {overflow_result = 0 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %2 = transform.num_associations %1#0 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %2 : !transform.param<i64>
      %3 = transform.num_associations %1#1 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %3 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共21行，输出共20行
- transform.named_sequence定义被保留

---

### 15.10.43 case_43

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @opaque() -> (i32, i32)

func.func @split_handle() {
  func.call @opaque() : () -> (i32, i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %op = transform.structured.match ops{["func.call"]} in %fun : (!transform.any_op) -> !transform.any_op
    %val = transform.get_result %op[all] : (!transform.any_op) -> !transform.any_value
    %p = transform.num_associations %val : (!transform.any_value) -> !transform.any_param
    // expected-remark @below {{total 2}}
    transform.debug.emit_param_as_remark %p, "total" : !transform.any_param
    %h:2 = transform.split_handle %val : (!transform.any_value) -> (!transform.any_value, !transform.any_value)
    %p1 = transform.num_associations %h#0 : (!transform.any_value) -> !transform.any_param
    %p2 = transform.num_associations %h#1 : (!transform.any_value) -> !transform.any_param
    // expected-remark @below {{first 1}}
    transform.debug.emit_param_as_remark %p1, "first" : !transform.any_param
    // expected-remark @below {{second 1}}
    transform.debug.emit_param_as_remark %p1, "second" : !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @opaque() -> (i32, i32)
  func.func @split_handle() {
    %0:2 = call @opaque() : () -> (i32, i32)
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_result %0[all] : (!transform.any_op) -> !transform.any_value
      %2 = transform.num_associations %1 : (!transform.any_value) -> !transform.any_param
      transform.debug.emit_param_as_remark %2, "total" : !transform.any_param
      %3:2 = transform.split_handle %1 : (!transform.any_value) -> (!transform.any_value, !transform.any_value)
      %4 = transform.num_associations %3#0 : (!transform.any_value) -> !transform.any_param
      %5 = transform.num_associations %3#1 : (!transform.any_value) -> !transform.any_param
      transform.debug.emit_param_as_remark %4, "first" : !transform.any_param
      transform.debug.emit_param_as_remark %4, "second" : !transform.any_param
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共24行，输出共21行
- transform.named_sequence定义被保留

---

### 15.10.44 case_44

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @opaque() -> (i32, i32)

func.func @split_handle() {
  func.call @opaque() : () -> (i32, i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %op = transform.structured.match ops{["func.call"]} in %fun : (!transform.any_op) -> !transform.any_op
    %val = transform.get_result %op[all] : (!transform.any_op) -> !transform.any_value
    %type = transform.get_type %val : (!transform.any_value) -> !transform.any_param
    %p = transform.num_associations %type : (!transform.any_param) -> !transform.any_param
    // expected-remark @below {{total 2}}
    transform.debug.emit_param_as_remark %p, "total" : !transform.any_param
    %h:2 = transform.split_handle %type : (!transform.any_param) -> (!transform.any_param, !transform.any_param)
    %p1 = transform.num_associations %h#0 : (!transform.any_param) -> !transform.any_param
    %p2 = transform.num_associations %h#1 : (!transform.any_param) -> !transform.any_param
    // expected-remark @below {{first 1}}
    transform.debug.emit_param_as_remark %p1, "first" : !transform.any_param
    // expected-remark @below {{second 1}}
    transform.debug.emit_param_as_remark %p1, "second" : !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @opaque() -> (i32, i32)
  func.func @split_handle() {
    %0:2 = call @opaque() : () -> (i32, i32)
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_result %0[all] : (!transform.any_op) -> !transform.any_value
      %2 = transform.get_type %1 : (!transform.any_value) -> !transform.any_param
      %3 = transform.num_associations %2 : (!transform.any_param) -> !transform.any_param
      transform.debug.emit_param_as_remark %3, "total" : !transform.any_param
      %4:2 = transform.split_handle %2 : (!transform.any_param) -> (!transform.any_param, !transform.any_param)
      %5 = transform.num_associations %4#0 : (!transform.any_param) -> !transform.any_param
      %6 = transform.num_associations %4#1 : (!transform.any_param) -> !transform.any_param
      transform.debug.emit_param_as_remark %5, "first" : !transform.any_param
      transform.debug.emit_param_as_remark %5, "second" : !transform.any_param
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共25行，输出共22行
- transform.named_sequence定义被保留

---

### 15.10.45 case_45

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    // expected-error @below {{op expects result types to implement the same transform interface as the operand type}}
    transform.split_handle %fun : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.46 case_46

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = transform.cast %0 : !transform.any_op to !transform.test_dialect_op
        transform.cast %2 : !transform.test_dialect_op to !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  "test.some_op"() : () -> ()
  "other_dialect.other_op"() : () -> ()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @some : benefit(1) {
          %0 = operation "test.some_op" 
          rewrite %0 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = cast %0 : !transform.any_op to !transform.test_dialect_op
          %2 = cast %1 : !transform.test_dialect_op to !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共22行，输出共22行
- transform.named_sequence定义被保留

---

### 15.10.47 case_47

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @other : benefit(1) {
        %0 = pdl.operation "other_dialect.other_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @other in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{expected the payload operation to belong to the 'test' dialect}}
        %2 = transform.cast %0 : !transform.any_op to !transform.test_dialect_op
        transform.cast %2 : !transform.test_dialect_op to !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.48 case_48

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = transform.cast %0 : !transform.any_op to !transform.op<"test.some_op">
        transform.cast %2 : !transform.op<"test.some_op"> to !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  "test.some_op"() : () -> ()
  "other_dialect.other_op"() : () -> ()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @some : benefit(1) {
          %0 = operation "test.some_op" 
          rewrite %0 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = cast %0 : !transform.any_op to !transform.op<"test.some_op">
          %2 = cast %1 : !transform.op<"test.some_op"> to !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共22行，输出共22行
- transform.named_sequence定义被保留

---

### 15.10.49 case_49

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
"test.some_op"() : () -> ()
// expected-note @below {{payload operation}}
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @other : benefit(1) {
        %0 = pdl.operation "other_dialect.other_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @other in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{incompatible payload operation name}}
        %2 = transform.cast %0 : !transform.any_op to !transform.op<"test.some_op">
        transform.cast %2 : !transform.op<"test.some_op"> to !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.50 case_50

**功能介绍:**

here, the handles nested under are {%root, %arg0, %arg1, %0}
here, the handles nested under are only {%root, %arg0, %arg1}

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // here, the handles nested under are {%root, %arg0, %arg1, %0}
        // expected-remark @below {{4 handles nested under}}
        transform.test_report_number_of_tracked_handles_nested_under %arg1 : !transform.any_op
        // expected-remark @below {{erased}}
        transform.test_emit_remark_and_erase_operand %0, "erased" : !transform.any_op
        // here, the handles nested under are only {%root, %arg0, %arg1}
        // expected-remark @below {{3 handles nested under}}
        transform.test_report_number_of_tracked_handles_nested_under %arg1 : !transform.any_op
      }

      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }
    }
    transform.yield
  }
}

"test.some_op"() : () -> ()
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @some in %arg2 : (!transform.any_op) -> !transform.any_op
          test_report_number_of_tracked_handles_nested_under %arg2 : !transform.any_op
          test_emit_remark_and_erase_operand %0, "erased" : !transform.any_op
          test_report_number_of_tracked_handles_nested_under %arg2 : !transform.any_op
        }
        pdl.pattern @some : benefit(1) {
          %0 = operation "test.some_op" 
          rewrite %0 with "transform.dialect"
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共27行，输出共21行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.51 case_51

**功能介绍:**

/ Test that yield does not crash in the presence of silenceable error in
/ propagate mode.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op -> !transform.any_op failures(propagate) {
    ^bb1(%fun: !transform.any_op):
      %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{expected to contain 3 payloads but it contains 2 payloads}}
      %h_2:3 = split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      /// Test that yield does not crash in the presence of silenceable error in
      /// propagate mode.
      yield %fun : !transform.any_op
    }
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.52 case_52

**功能介绍:**

Edge case propagating empty handles in splitting.
Test does not crash when accessing the empty handle.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      %muli = transform.structured.match ops{["arith.muli"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // Edge case propagating empty handles in splitting.
      %0:3 = split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      // Test does not crash when accessing the empty handle.
      yield %0#0 : !transform.any_op
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %2:3 = split_handle %1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      yield %2#0 : !transform.any_op
    }
    transform.yield 
  }
}


```

**重点说明:**

- 输入共13行，输出共11行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 15.10.53 case_53

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_param (0 : i32) : !transform.test_dialect_param
    // expected-remark @below {{0 : i32}}
    transform.debug.emit_param_as_remark  %0 : !transform.test_dialect_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_param(0 : i32) : !transform.test_dialect_param
    transform.debug.emit_param_as_remark %0 : !transform.test_dialect_param
    transform.yield 
  }
}


```

**重点说明:**

- 输入共8行，输出共7行
- transform.named_sequence定义被保留

---

### 15.10.54 case_54

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected the type of the parameter attribute ('i32') to match the parameter type ('i64')}}
    transform.test_produce_param (0 : i32) : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.55 case_55

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_add_to_param 40
    %1 = transform.test_add_to_param %0, 2
    // expected-remark @below {{42 : i32}}
    transform.debug.emit_param_as_remark  %1 : !transform.test_dialect_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_add_to_param 40
    %1 = transform.test_add_to_param %0, 2
    transform.debug.emit_param_as_remark %1 : !transform.test_dialect_param
    transform.yield 
  }
}


```

**重点说明:**

- 输入共9行，输出共8行
- transform.named_sequence定义被保留

---

### 15.10.56 case_56

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.test_produce_param_with_number_of_test_ops %0 : !transform.any_op
    // expected-remark @below {{1 : i32, 3 : i32}}
    transform.debug.emit_param_as_remark  %1 : !transform.test_dialect_param
    %2 = transform.test_add_to_param %1, 100
    // expected-remark @below {{101 : i32, 103 : i32}}
    transform.debug.emit_param_as_remark  %2 : !transform.test_dialect_param
    transform.yield
  }
}

func.func private @one_test_op(%arg0: i32) {
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  return
}

func.func private @three_test_ops(%arg0: i32) {
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  return
}
```

**用例输出:**

```mlir
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.test_produce_param_with_number_of_test_ops %0 : !transform.any_op
      transform.debug.emit_param_as_remark %1 : !transform.test_dialect_param
      %2 = transform.test_add_to_param %1, 100
      transform.debug.emit_param_as_remark %2 : !transform.test_dialect_param
      transform.yield 
    }
  }
  func.func private @one_test_op(%arg0: i32) {
    %0 = "test.op_a"(%arg0) <{attr = 0 : i32}> : (i32) -> i32
    return
  }
  func.func private @three_test_ops(%arg0: i32) {
    %0 = "test.op_a"(%arg0) <{attr = 0 : i32}> : (i32) -> i32
    %1 = "test.op_a"(%arg0) <{attr = 0 : i32}> : (i32) -> i32
    %2 = "test.op_a"(%arg0) <{attr = 0 : i32}> : (i32) -> i32
    return
  }
}


```

**重点说明:**

- 输入共24行，输出共22行
- transform.named_sequence定义被保留

---

### 15.10.57 case_57

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{when applied to this op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected to produce an Operation * for result #0}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { first_result_is_param }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.58 case_58

**功能介绍:**

Should not fail.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Should not fail.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_produce_transform_param_or_forward_operand %arg0
      { first_result_is_null }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %out, %param = transform.test_produce_transform_param_or_forward_operand %arg0 {first_result_is_null} : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield 
  }
}


```

**重点说明:**

- 输入共10行，输出共6行
- transform.named_sequence定义被保留

---

### 15.10.59 case_59

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{when applied to this op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected to produce an Attribute for result #1}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { second_result_is_handle }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.60 case_60

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{when applied to this op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected to produce a Value for result #0}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { second_result_is_handle }
      : (!transform.any_op) -> (!transform.any_value, !transform.param<i64>)
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.61 case_61

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{attempting to assign a null payload op to this transform value}}
    %0 = transform.test_produce_null_payload : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.62 case_62

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{attempting to assign a null parameter to this transform value}}
    %0 = transform.test_produce_null_param : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.63 case_63

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{attempting to assign a null payload value to this transform handle}}
    %0 = transform.test_produce_null_value : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.64 case_64

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below {{could not find a nested named sequence with name: __transform_main}}
module {
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.65 case_65

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  // expected-remark @below {{value handle}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %0, "value handle" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %0, "value handle" : !transform.any_value
    transform.yield 
  }
}


```

**重点说明:**

- 输入共9行，输出共7行
- transform.named_sequence定义被保留

---

### 15.10.66 case_66

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{result handle}}
// expected-note @below {{value handle points to an op result #1}}
%0:2 = "test.get_two_results"() : () -> (i32, i32)
// expected-remark @below {{result handle}}
// expected-note @below {{value handle points to an op result #1}}
%1:3 = "test.get_three_results"() : () -> (i32, i32, f32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.get_two_results", "test.get_three_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %2, 1 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %3, "result handle" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  %0:2 = "test.get_two_results"() : () -> (i32, i32)
  %1:3 = "test.get_three_results"() : () -> (i32, i32, f32)
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %2 = transform.structured.match ops{["test.get_two_results", "test.get_three_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %3 = transform.test_produce_value_handle_to_result %2, 1 : (!transform.any_op) -> !transform.any_value
      transform.debug.emit_remark_at %3, "result handle" : !transform.any_value
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共12行
- transform.named_sequence定义被保留

---

### 15.10.67 case_67

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
"test.op_with_regions"() ({
^bb0:
  "test.regon_terminator"() : () -> ()
}, {
^bb1:
  "test.regon_terminator"() : () -> ()
// expected-remark @below {{block argument handle}}
// expected-note @below {{value handle points to a block argument #2 in block #1 in region #1}}
^bb2(%arg0: i32, %arg1: f64, %arg3: index):
  "test.match_anchor"() : () -> ()
  "test.regon_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 2 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %3, "block argument handle" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  "test.op_with_regions"() ({
    "test.regon_terminator"() : () -> ()
  }, {
    "test.regon_terminator"() : () -> ()
  ^bb1(%0: i32, %1: f64, %2: index):  // no predecessors
    "test.match_anchor"() : () -> ()
    "test.regon_terminator"() : () -> ()
  }) : () -> ()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.test_produce_value_handle_to_argument_of_parent_block %0, 2 : (!transform.any_op) -> !transform.any_value
      transform.debug.emit_remark_at %1, "block argument handle" : !transform.any_value
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共21行，输出共18行
- transform.named_sequence定义被保留

---

### 15.10.68 case_68

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{value defined here with type '!transform.test_dialect_param'}}
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    // expected-error @below {{unexpectedly consumed a value that is not a handle as operand #0}}
    transform.test_consume_operand %0 : !transform.test_dialect_param
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.69 case_69

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{addi operand}}
// expected-note @below {{value handle points to a block argument #0}}
func.func @get_operand_of_op(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %operand = transform.get_operand %addi[0] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %operand, "addi operand" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_operand_of_op(%arg0: index, %arg1: index) -> index {
    %0 = arith.addi %arg0, %arg1 : index
    return %0 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
      transform.debug.emit_remark_at %1, "addi operand" : !transform.any_value
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共14行
- transform.named_sequence定义被保留

---

### 15.10.70 case_70

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_out_of_bounds_operand_of_op(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{while considering positions of this payload operation}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{position overflow 2 (updated from 2) for maximum 2}}
    %operand = transform.get_operand %addi[2] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %operand, "addi operand" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.71 case_71

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{addi operand}}
// expected-note @below {{value handle points to a block argument #1}}
func.func @get_inverted_operand_of_op(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %operand = transform.get_operand %addi[except(0)] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %operand, "addi operand" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_inverted_operand_of_op(%arg0: index, %arg1: index) -> index {
    %0 = arith.addi %arg0, %arg1 : index
    return %0 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_operand %0[except(0)] : (!transform.any_op) -> !transform.any_value
      transform.debug.emit_remark_at %1, "addi operand" : !transform.any_value
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共14行
- transform.named_sequence定义被保留

---

### 15.10.72 case_72

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_multiple_operands_of_op(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addui = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %operands = transform.get_operand %addui[all] : (!transform.any_op) -> !transform.any_value
    %p = transform.num_associations %operands : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_multiple_operands_of_op(%arg0: index, %arg1: index) -> index {
    %0 = arith.addi %arg0, %arg1 : index
    return %0 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_operand %0[all] : (!transform.any_op) -> !transform.any_value
      %2 = transform.num_associations %1 : (!transform.any_value) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %2 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.73 case_73

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{addi result}}
  // expected-note @below {{value handle points to an op result #0}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %result, "addi result" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
    %0 = arith.addi %arg0, %arg1 : index
    return %0 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_result %0[0] : (!transform.any_op) -> !transform.any_value
      transform.debug.emit_remark_at %1, "addi result" : !transform.any_value
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共14行
- transform.named_sequence定义被保留

---

### 15.10.74 case_74

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_out_of_bounds_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{while considering positions of this payload operation}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{position overflow 1 (updated from 1) for maximum 1}}
    %result = transform.get_result %addi[1] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %result, "addi result" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.75 case_75

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{matched}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
    %op = transform.get_defining_op %result : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %op, "matched" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
    %0 = arith.addi %arg0, %arg1 : index
    return %0 : index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_result %0[0] : (!transform.any_op) -> !transform.any_value
      %2 = transform.get_defining_op %1 : (!transform.any_value) -> !transform.any_op
      transform.debug.emit_remark_at %2, "matched" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.76 case_76

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_multiple_result_of_op(%arg0: index, %arg1: index) -> (index, i1) {
  %r, %b = arith.addui_extended %arg0, %arg1 : index, i1
  return %r, %b : index, i1
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addui = transform.structured.match ops{["arith.addui_extended"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %results = transform.get_result %addui[all] : (!transform.any_op) -> !transform.any_value
    %p = transform.num_associations %results : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_multiple_result_of_op(%arg0: index, %arg1: index) -> (index, i1) {
    %sum, %overflow = arith.addui_extended %arg0, %arg1 : index, i1
    return %sum, %overflow : index, i1
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.addui_extended"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_result %0[all] : (!transform.any_op) -> !transform.any_value
      %2 = transform.num_associations %1 : (!transform.any_value) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %2 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.77 case_77

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{target value}}
func.func @get_result_of_op_bbarg(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %bbarg = transform.test_produce_value_handle_to_argument_of_parent_block %addi, 0 : (!transform.any_op) -> !transform.any_value
    // expected-error @below {{cannot get defining op of block argument}}
    %op = transform.get_defining_op %bbarg : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %op, "matched" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.78 case_78

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module @named_inclusion attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "a"}}
    transform.test_transform_op "a"
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

```mlir
module @named_inclusion attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_transform_op "a"
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield 
  }
}


```

**重点说明:**

- 输入共13行，输出共10行
- transform.named_sequence定义被保留

---

### 15.10.79 case_79

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module @named_inclusion_in_named attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "a"}}
    transform.test_transform_op "a"
    transform.yield
  }

  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "b"}}
    transform.test_transform_op "b"
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @bar failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
```

**用例输出:**

```mlir
module @named_inclusion_in_named attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_transform_op "a"
    transform.yield 
  }
  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_transform_op "b"
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @bar failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.yield 
  }
}


```

**重点说明:**

- 输入共20行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.80 case_80

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{operation}}
module @named_operands attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly},
                                %arg1: !transform.any_value {transform.readonly}) -> () {
    transform.debug.emit_remark_at %arg0, "operation" : !transform.any_op
    transform.debug.emit_remark_at %arg1, "value" : !transform.any_value
    transform.yield
  }

  // expected-remark @below {{value}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.include @foo failures(propagate) (%arg0, %0) : (!transform.any_op, !transform.any_value) -> ()
    transform.yield
  }
}
```

**用例输出:**

```mlir
module @named_operands attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_value {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "operation" : !transform.any_op
    transform.debug.emit_remark_at %arg1, "value" : !transform.any_value
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.include @foo failures(propagate) (%arg0, %0) : (!transform.any_op, !transform.any_value) -> ()
    transform.yield 
  }
}


```

**重点说明:**

- 输入共18行，输出共12行
- transform.named_sequence定义被保留

---

### 15.10.81 case_81

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-remark @below {{operation}}
module @named_return attributes { transform.with_named_sequence } {

  // expected-remark @below {{value}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_value) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_value
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0:2 = transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
    transform.debug.emit_remark_at %0#0, "operation" : !transform.any_op
    transform.debug.emit_remark_at %0#1, "value" : !transform.any_value
    transform.yield
  }
}
```

**用例输出:**

```mlir
module @named_return attributes {transform.with_named_sequence} {
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_value) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_value
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0:2 = transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
    transform.debug.emit_remark_at %0#0, "operation" : !transform.any_op
    transform.debug.emit_remark_at %0#1, "value" : !transform.any_value
    transform.yield 
  }
}


```

**重点说明:**

- 输入共17行，输出共12行
- transform.named_sequence定义被保留

---

### 15.10.82 case_82

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match1(%current: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.test_succeed_if_operand_of_op_kind %current, "test.some_op" : !transform.any_op
    transform.yield %current : !transform.any_op
  }

  transform.named_sequence @match2(%current: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.test_succeed_if_operand_of_op_kind %current, "func.func" : !transform.any_op
    transform.yield %current : !transform.any_op
  }

  transform.named_sequence @action1(%current: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %current, "matched1" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @action2(%current: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %current, "matched2" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.foreach_match in %root
        @match1 -> @action1,
        @match2 -> @action2
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }

  // expected-remark @below {{matched2}}
  func.func private @foo()
  // expected-remark @below {{matched2}}
  func.func private @bar()
  "test.testtest"() : () -> ()
  // expected-remark @below {{matched1}}
  "test.some_op"() : () -> ()
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @match1(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.test_succeed_if_operand_of_op_kind %arg0, "test.some_op" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @match2(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.test_succeed_if_operand_of_op_kind %arg0, "func.func" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @action1(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched1" : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @action2(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched2" : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %updated_root = transform.foreach_match in %arg0 
        @match1 -> @action1, 
        @match2 -> @action2 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
  func.func private @foo()
  func.func private @bar()
  "test.testtest"() : () -> ()
  "test.some_op"() : () -> ()
}


```

**重点说明:**

- 输入共36行，输出共28行
- transform.named_sequence定义被保留

---

### 15.10.83 case_83

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly})
  transform.named_sequence @action()

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // expected-error @below {{unresolved external symbol @match}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.84 case_84

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @action()

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // expected-error @below {{unresolved external symbol @action}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.85 case_85

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) {
    // expected-error @below {{expected operations in the match part to implement MatchOpInterface}}
    "test.unknown_op"() : () -> ()
    transform.yield
  }
  transform.named_sequence @action() {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.86 case_86

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_func(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @print_func(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched func" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.foreach_match in %arg0 @match_func -> @print_func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // expected-remark @below {{matched func}}
  func.func @payload() {
    return
  }

  // expected-remark @below {{matched func}}
  func.func private @declaration()

  "test.something_else"() : () -> ()
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_func(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @print_func(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched func" : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %updated_root = transform.foreach_match in %arg0 
        @match_func -> @print_func : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
  func.func @payload() {
    return
  }
  func.func private @declaration()
  "test.something_else"() : () -> ()
}


```

**重点说明:**

- 输入共27行，输出共20行
- transform.named_sequence定义被保留

---

### 15.10.87 case_87

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @eq_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi eq %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched == 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @ne_0(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 0 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi ne %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched != 0" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @gt_m1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant -1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi gt %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched > -1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @ge_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi ge %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched >= 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @lt_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi lt %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched < 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @le_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi le %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched <= 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.foreach_match in %arg0 @eq_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 @ne_0 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %2 = transform.foreach_match in %1 @gt_m1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %3 = transform.foreach_match in %2 @ge_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %4 = transform.foreach_match in %3 @lt_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %5 = transform.foreach_match in %4 @le_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // expected-remark @below {{matched > -1}}
  // expected-remark @below {{matched < 1}}
  // expected-remark @below {{matched <= 1}}
  func.func private @declaration()

  // expected-remark @below {{matched == 1}}
  // expected-remark @below {{matched != 0}}
  // expected-remark @below {{matched > -1}}
  // expected-remark @below {{matched >= 1}}
  // expected-remark @below {{matched <= 1}}
  func.func @definition() {
    "test.something"() : () -> ()
    return
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @eq_1(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi eq %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched == 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @ne_0(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 0 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi ne %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched != 0" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @gt_m1(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant -1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi gt %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched > -1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @ge_1(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi ge %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched >= 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @lt_1(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi lt %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched < 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @le_1(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi le %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched <= 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield 
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %updated_root = transform.foreach_match in %arg0 
        @eq_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %updated_root_0 = transform.foreach_match in %updated_root 
        @ne_0 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %updated_root_1 = transform.foreach_match in %updated_root_0 
        @gt_m1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %updated_root_2 = transform.foreach_match in %updated_root_1 
        @ge_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %updated_root_3 = transform.foreach_match in %updated_root_2 
        @lt_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %updated_root_4 = transform.foreach_match in %updated_root_3 
        @le_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
  func.func private @declaration()
  func.func @definition() {
    "test.something"() : () -> ()
    return
  }
}


```

**重点说明:**

- 输入共90行，输出共73行
- transform.named_sequence定义被保留

---

### 15.10.88 case_88

**功能介绍:**

CHECK-NEXT:   transform.test_dummy_payload_op  {new_op} : () -> i1
CHECK-NEXT:   transform.test_dummy_payload_op  {new_op} : () -> i1
CHECK-NEXT:   return
CHECK-NEXT: }
One replacement op (test.drop_mapping) is dropped from the mapping.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//  CHECK-NEXT:   transform.test_dummy_payload_op  {new_op} : () -> i1
//  CHECK-NEXT:   transform.test_dummy_payload_op  {new_op} : () -> i1
//  CHECK-NEXT:   return
//  CHECK-NEXT: }
func.func @test_tracked_rewrite() {
  %0 = transform.test_dummy_payload_op {replace_me} : () -> (i1)
  %1 = transform.test_dummy_payload_op {erase_me} : () -> (i1)
  %2 = transform.test_dummy_payload_op {replace_me} : () -> (i1)
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["transform.test_dummy_payload_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{2 iterations}}
    transform.test_tracked_rewrite %0 : (!transform.any_op) -> ()
    // One replacement op (test.drop_mapping) is dropped from the mapping.
    %p = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_tracked_rewrite() {
    %0 = transform.test_dummy_payload_op  {new_op} : () -> i1
    %1 = transform.test_dummy_payload_op  {new_op} : () -> i1
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["transform.test_dummy_payload_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.test_tracked_rewrite %0 : (!transform.any_op) -> ()
      %1 = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %1 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共23行，输出共16行
- transform.named_sequence定义被保留

---

### 15.10.89 case_89

**功能介绍:**

Parameter deduplication happens by value

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// Parameter deduplication happens by value

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.param.constant 1 -> !transform.param<i64>
    %2 = transform.param.constant 1 -> !transform.param<i64>
    %3 = transform.param.constant 2 -> !transform.param<i64>
    %4 = transform.merge_handles %1, %2 { deduplicate } : !transform.param<i64>
    %p = transform.num_associations %4 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    %5 = transform.merge_handles %1, %1 { deduplicate } : !transform.param<i64>
    %p2 = transform.num_associations %5 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>

    %6 = transform.merge_handles %1, %3 { deduplicate } : !transform.param<i64>
    %p3 = transform.num_associations %6 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>

    %7 = transform.merge_handles %1, %1, %2, %3 : !transform.param<i64>
    %p4 = transform.num_associations %7 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{4}}
    transform.debug.emit_param_as_remark %p4 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.param.constant 1 : i64 -> !transform.param<i64>
    %1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    %2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    %3 = transform.merge_handles deduplicate %0, %1 : !transform.param<i64>
    %4 = transform.num_associations %3 : (!transform.param<i64>) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %4 : !transform.param<i64>
    %5 = transform.merge_handles deduplicate %0, %0 : !transform.param<i64>
    %6 = transform.num_associations %5 : (!transform.param<i64>) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %6 : !transform.param<i64>
    %7 = transform.merge_handles deduplicate %0, %2 : !transform.param<i64>
    %8 = transform.num_associations %7 : (!transform.param<i64>) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %8 : !transform.param<i64>
    %9 = transform.merge_handles %0, %0, %1, %2 : !transform.param<i64>
    %10 = transform.num_associations %9 : (!transform.param<i64>) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %10 : !transform.param<i64>
    transform.yield 
  }
}


```

**重点说明:**

- 输入共30行，输出共20行
- transform.named_sequence定义被保留

---

### 15.10.90 case_90

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
%0:3 = "test.get_two_results"() : () -> (i32, i32, f32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.get_two_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
    %3 = transform.test_produce_value_handle_to_result %1, 1 : (!transform.any_op) -> !transform.any_value

    %4 = transform.merge_handles %2, %2 { deduplicate } : !transform.any_value
    %p = transform.num_associations %4 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    %5 = transform.merge_handles %2, %3 { deduplicate } : !transform.any_value
    %p2 = transform.num_associations %5 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>

    %6 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
    %7 = transform.merge_handles %2, %6 { deduplicate } : !transform.any_value
    %p3 = transform.num_associations %6 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>

    %8 = transform.merge_handles %2, %2, %3, %4 : !transform.any_value
    %p4 = transform.num_associations %8 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{4}}
    transform.debug.emit_param_as_remark %p4 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  %0:3 = "test.get_two_results"() : () -> (i32, i32, f32)
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %1 = transform.structured.match ops{["test.get_two_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %2 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
      %3 = transform.test_produce_value_handle_to_result %1, 1 : (!transform.any_op) -> !transform.any_value
      %4 = transform.merge_handles deduplicate %2, %2 : !transform.any_value
      %5 = transform.num_associations %4 : (!transform.any_value) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %5 : !transform.param<i64>
      %6 = transform.merge_handles deduplicate %2, %3 : !transform.any_value
      %7 = transform.num_associations %6 : (!transform.any_value) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %7 : !transform.param<i64>
      %8 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
      %9 = transform.merge_handles deduplicate %2, %8 : !transform.any_value
      %10 = transform.num_associations %8 : (!transform.any_value) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %10 : !transform.param<i64>
      %11 = transform.merge_handles %2, %2, %3, %4 : !transform.any_value
      %12 = transform.num_associations %11 : (!transform.any_value) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %12 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共31行，输出共24行
- transform.named_sequence定义被保留

---

### 15.10.91 case_91

**功能介绍:**

CHECK-NEXT:   "test.annotate_me"()
CHECK-SAME:                        any_attr = "example"
CHECK-SAME:                        broadcast_attr = 2 : i64
CHECK-SAME:                        new_attr = 1 : i32
CHECK-SAME:                        unit_attr
CHECK-NEXT:   "test.annotate_me"()
CHECK-SAME:                        any_attr = "example"
CHECK-SAME:                        broadcast_attr = 2 : i64
CHECK-SAME:                        existing_attr = "test"
CHECK-SAME:                        new_attr = 1 : i32
CHECK-SAME:                        unit_attr
CHECK-NEXT:   "test.annotate_me"()
CHECK-SAME:                        any_attr = "example"
CHECK-SAME:                        broadcast_attr = 2 : i64
CHECK-SAME:                        new_attr = 1 : i32
CHECK-SAME:                        unit_attr

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//  CHECK-NEXT:   "test.annotate_me"()
//  CHECK-SAME:                        any_attr = "example"
//  CHECK-SAME:                        broadcast_attr = 2 : i64
//  CHECK-SAME:                        new_attr = 1 : i32
//  CHECK-SAME:                        unit_attr
//  CHECK-NEXT:   "test.annotate_me"()
//  CHECK-SAME:                        any_attr = "example"
//  CHECK-SAME:                        broadcast_attr = 2 : i64
//  CHECK-SAME:                        existing_attr = "test"
//  CHECK-SAME:                        new_attr = 1 : i32
//  CHECK-SAME:                        unit_attr
//  CHECK-NEXT:   "test.annotate_me"()
//  CHECK-SAME:                        any_attr = "example"
//  CHECK-SAME:                        broadcast_attr = 2 : i64
//  CHECK-SAME:                        new_attr = 1 : i32
//  CHECK-SAME:                        unit_attr
func.func @test_annotation() {
  %0 = "test.annotate_me"() : () -> (i1)
  %1 = "test.annotate_me"() {existing_attr = "test"} : () -> (i1)
  %2 = "test.annotate_me"() {new_attr = 0} : () -> (i1)
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["test.annotate_me"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.test_produce_param_with_number_of_test_ops %0 : !transform.any_op
    transform.annotate %0 "new_attr" = %1 : !transform.any_op, !transform.test_dialect_param

    %2 = transform.param.constant 2 -> !transform.param<i64>
    transform.annotate %0 "broadcast_attr" = %2 : !transform.any_op, !transform.param<i64>
    transform.annotate %0 "unit_attr" : !transform.any_op

    %3 = transform.param.constant "example" -> !transform.any_param
    transform.annotate %0 "any_attr" = %3 : !transform.any_op, !transform.any_param
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_annotation() {
    %0 = "test.annotate_me"() {any_attr = "example", broadcast_attr = 2 : i64, new_attr = 1 : i32, unit_attr} : () -> i1
    %1 = "test.annotate_me"() {any_attr = "example", broadcast_attr = 2 : i64, existing_attr = "test", new_attr = 1 : i32, unit_attr} : () -> i1
    %2 = "test.annotate_me"() {any_attr = "example", broadcast_attr = 2 : i64, new_attr = 1 : i32, unit_attr} : () -> i1
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.annotate_me"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.test_produce_param_with_number_of_test_ops %0 : !transform.any_op
      transform.annotate %0 "new_attr" = %1 : !transform.any_op, !transform.test_dialect_param
      %2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      transform.annotate %0 "broadcast_attr" = %2 : !transform.any_op, !transform.param<i64>
      transform.annotate %0 "unit_attr" : !transform.any_op
      %3 = transform.param.constant "example" -> !transform.any_param
      transform.annotate %0 "any_attr" = %3 : !transform.any_op, !transform.any_param
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共37行，输出共20行
- transform.named_sequence定义被保留

---

### 15.10.92 case_92

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @notify_payload_op_replaced(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 {original} : index
  // expected-remark @below{{updated handle}}
  %1 = arith.muli %arg0, %arg1 {replacement} : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match attributes{original} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes{replacement} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.test_notify_payload_op_replaced %0, %1 : (!transform.any_op, !transform.any_op) -> ()
    transform.debug.emit_remark_at %0, "updated handle" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @notify_payload_op_replaced(%arg0: index, %arg1: index) {
    %0 = arith.muli %arg0, %arg1 {original} : index
    %1 = arith.muli %arg0, %arg1 {replacement} : index
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match attributes {original} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match attributes {replacement} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.test_notify_payload_op_replaced %0, %1 : (!transform.any_op, !transform.any_op) -> ()
      transform.debug.emit_remark_at %0, "updated handle" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共16行，输出共16行
- transform.named_sequence定义被保留

---

### 15.10.93 case_93

**功能介绍:**

CHECK:   %[[const:.*]] = arith.constant 0 : index
CHECK:   %[[ex1:.*]] = scf.execute_region -> index {
CHECK:     scf.yield %[[const]]
CHECK:   }
CHECK:   %[[ex2:.*]] = scf.execute_region -> index {
CHECK:     scf.yield %[[const]]
CHECK:   }
CHECK:   return %[[const]], %[[ex1]], %[[ex2]]
There are 3 arith.constant ops.
"deduplicate" has no effect because these are 3 different ops.
Apply CSE.
The handle is still mapped to 3 arith.constant ops.
But they are all the same op.
The other handles were also updated.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   %[[const:.*]] = arith.constant 0 : index
//       CHECK:   %[[ex1:.*]] = scf.execute_region -> index {
//       CHECK:     scf.yield %[[const]]
//       CHECK:   }
//       CHECK:   %[[ex2:.*]] = scf.execute_region -> index {
//       CHECK:     scf.yield %[[const]]
//       CHECK:   }
//       CHECK:   return %[[const]], %[[ex1]], %[[ex2]]
func.func @test_apply_cse() -> (index, index, index) {
  // expected-remark @below{{eliminated 1}}
  // expected-remark @below{{eliminated 2}}
  %0 = arith.constant 0 : index
  %1 = scf.execute_region -> index {
    %2 = arith.constant 0 : index
    scf.yield %2 : index
  } {first}
  %3 = scf.execute_region -> index {
    %4 = arith.constant 0 : index
    scf.yield %4 : index
  } {second}
  return %0, %1, %3 : index, index, index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %first = transform.structured.match attributes{first} in %0 : (!transform.any_op) -> !transform.any_op
    %elim_first = transform.structured.match ops{["arith.constant"]} in %first : (!transform.any_op) -> !transform.any_op
    %second = transform.structured.match attributes{first} in %0 : (!transform.any_op) -> !transform.any_op
    %elim_second = transform.structured.match ops{["arith.constant"]} in %first : (!transform.any_op) -> !transform.any_op

    // There are 3 arith.constant ops.
    %all = transform.structured.match ops{["arith.constant"]} in %0 : (!transform.any_op) -> !transform.any_op
    %p = transform.num_associations %all : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{3}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    // "deduplicate" has no effect because these are 3 different ops.
    %merged_before = transform.merge_handles deduplicate %all : !transform.any_op
    %p2 = transform.num_associations %merged_before : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{3}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>

    // Apply CSE.
    transform.apply_cse to %0 : !transform.any_op

    // The handle is still mapped to 3 arith.constant ops.
    %p3 = transform.num_associations %all : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{3}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>
    // But they are all the same op.
    %merged_after = transform.merge_handles deduplicate %all : !transform.any_op
    %p4 = transform.num_associations %merged_after : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p4 : !transform.param<i64>

    // The other handles were also updated.
    transform.debug.emit_remark_at %elim_first, "eliminated 1" : !transform.any_op
    %p5 = transform.num_associations %elim_first : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p5 : !transform.param<i64>
    transform.debug.emit_remark_at %elim_second, "eliminated 2" : !transform.any_op
    %p6 = transform.num_associations %elim_second : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p6 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_apply_cse() -> (index, index, index) {
    %c0 = arith.constant 0 : index
    %0 = scf.execute_region -> index {
      scf.yield %c0 : index
    } {first}
    %1 = scf.execute_region -> index {
      scf.yield %c0 : index
    } {second}
    return %c0, %0, %1 : index, index, index
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match attributes {first} in %0 : (!transform.any_op) -> !transform.any_op
      %2 = transform.structured.match ops{["arith.constant"]} in %1 : (!transform.any_op) -> !transform.any_op
      %3 = transform.structured.match attributes {first} in %0 : (!transform.any_op) -> !transform.any_op
      %4 = transform.structured.match ops{["arith.constant"]} in %1 : (!transform.any_op) -> !transform.any_op
      %5 = transform.structured.match ops{["arith.constant"]} in %0 : (!transform.any_op) -> !transform.any_op
      %6 = transform.num_associations %5 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %6 : !transform.param<i64>
      %7 = transform.merge_handles deduplicate %5 : !transform.any_op
      %8 = transform.num_associations %7 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %8 : !transform.param<i64>
      transform.apply_cse to %0 : !transform.any_op
      %9 = transform.num_associations %5 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %9 : !transform.param<i64>
      %10 = transform.merge_handles deduplicate %5 : !transform.any_op
      %11 = transform.num_associations %10 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %11 : !transform.param<i64>
      transform.debug.emit_remark_at %2, "eliminated 1" : !transform.any_op
      %12 = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %12 : !transform.param<i64>
      transform.debug.emit_remark_at %4, "eliminated 2" : !transform.any_op
      %13 = transform.num_associations %4 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %13 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共67行，输出共40行
- transform.named_sequence定义被保留

---

### 15.10.94 case_94

**功能介绍:**

CHECK:   arith.muli
CHECK:   scf.for {{.*}} {
CHECK:     vector.print
CHECK:   }

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//       CHECK:   arith.muli
//       CHECK:   scf.for {{.*}} {
//       CHECK:     vector.print
//       CHECK:   }
func.func @test_licm(%arg0: index, %arg1: index, %arg2: index) {
  scf.for %iv = %arg0 to %arg1 step %arg2 {
    %0 = arith.muli %arg0, %arg1 : index
    vector.print %0 : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %0 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @test_licm(%arg0: index, %arg1: index, %arg2: index) {
    %0 = arith.muli %arg0, %arg1 : index
    scf.for %arg3 = %arg0 to %arg1 step %arg2 {
      vector.print %0 : index
    }
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_licm to %0 : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共19行，输出共16行
- transform.named_sequence定义被保留

---

### 15.10.95 case_95

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below{{when applied to this op}}
module attributes {transform.with_named_sequence} {
  func.func @test_licm_invalid() {
    return
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    // expected-error @below{{transform applied to the wrong op kind}}
    transform.apply_licm to %arg1 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.96 case_96

**功能介绍:**

Get parent by name.
Get immediate parent.
Deduplicate results.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @get_parent_op() {
  // expected-remark @below{{found test.foo parent}}
  "test.foo"() ({
    // expected-remark @below{{direct parent}}
    "test.bar"() ({
      "test.qux"() : () -> ()
      "test.qux"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.qux"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Get parent by name.
    %1 = transform.get_parent_op %0 {op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %1, "found test.foo parent" : !transform.any_op

    // Get immediate parent.
    %2 = transform.get_parent_op %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %2, "direct parent" : !transform.any_op
    %p = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{2}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    // Deduplicate results.
    %3 = transform.structured.match ops{["test.qux"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {deduplicate} : (!transform.any_op) -> !transform.any_op
    %p2 = transform.num_associations %4 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @get_parent_op() {
    "test.foo"() ({
      "test.bar"() ({
        "test.qux"() : () -> ()
        "test.qux"() : () -> ()
      }) : () -> ()
    }) : () -> ()
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["test.qux"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_parent_op %0 {op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %1, "found test.foo parent" : !transform.any_op
      %2 = transform.get_parent_op %0 : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %2, "direct parent" : !transform.any_op
      %3 = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %3 : !transform.param<i64>
      %4 = transform.structured.match ops{["test.qux"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %5 = transform.get_parent_op %4 {deduplicate} : (!transform.any_op) -> !transform.any_op
      %6 = transform.num_associations %5 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %6 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共35行，输出共26行
- transform.named_sequence定义被保留

---

### 15.10.97 case_97

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-note @below {{target op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below{{could not find a parent op that matches all requirements}}
    %3 = transform.get_parent_op %arg0 {op_name = "builtin.module"} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.98 case_98

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @cast(%arg0: f32) -> f64 {
  // expected-remark @below{{f64}}
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg0 : (!transform.any_op) -> !transform.op<"arith.extf">
    %1 = transform.get_result %0[0] : (!transform.op<"arith.extf">) -> !transform.any_value
    %2 = transform.get_type %1 : (!transform.any_value) -> !transform.type
    transform.debug.emit_param_as_remark %2 at %0 : !transform.type, !transform.op<"arith.extf">
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @cast(%arg0: f32) -> f64 {
    %0 = arith.extf %arg0 : f32 to f64
    return %0 : f64
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["arith.extf"]} in %arg0 : (!transform.any_op) -> !transform.op<"arith.extf">
      %1 = transform.get_result %0[0] : (!transform.op<"arith.extf">) -> !transform.any_value
      %2 = transform.get_type %1 : (!transform.any_value) -> !transform.type
      transform.debug.emit_param_as_remark %2 at %0 : !transform.type, !transform.op<"arith.extf">
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共15行，输出共15行
- transform.named_sequence定义被保留

---

### 15.10.99 case_99

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected type attribute, got 0 : i32}}
    transform.test_produce_param (0 : i32) : !transform.type
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.100 case_100

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected affine map attribute, got 0 : i32}}
    transform.test_produce_param (0 : i32) : !transform.affine_map
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.101 case_101

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @type_param_anchor()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_produce_param(f32) : !transform.type
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func private @type_param_anchor()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.test_produce_param(f32) : !transform.type
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共8行，输出共9行
- transform.named_sequence定义被保留

---

### 15.10.102 case_102

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func private @affine_map_param_anchor()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_produce_param(affine_map<(d0) -> ()>) : !transform.affine_map
    transform.yield
  }
}
```

**用例输出:**

```mlir
#map = affine_map<(d0) -> ()>
module {
  func.func private @affine_map_param_anchor()
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.test_produce_param(#map) : !transform.affine_map
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共8行，输出共10行
- transform.named_sequence定义被保留

---

### 15.10.103 case_103

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @verify_success(%arg0: f64) -> f64 {
  return %arg0 : f64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.verify %0 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @verify_success(%arg0: f64) -> f64 {
    return %arg0 : f64
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.verify %0 : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共11行，输出共12行
- transform.named_sequence定义被保留

---

### 15.10.104 case_104

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
// expected-error @below{{fail_to_verify is set}}
// expected-note @below{{payload op}}
func.func @verify_failure(%arg0: f64) -> f64 {
  return %arg0 : f64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.test_produce_invalid_ir %0 : !transform.any_op
    // expected-error @below{{failed to verify payload op}}
    transform.verify %0 : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.105 case_105

**功能介绍:**

Match all ops inside the function (including the function itself).
Select "test.foo".
Select "test.bar".

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @select() {
  // expected-remark @below{{found foo}}
  "test.foo"() : () -> ()
  // expected-remark @below{{found bar}}
  "test.bar"() : () -> ()
  // expected-remark @below{{found foo}}
  "test.foo"() : () -> ()
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Match all ops inside the function (including the function itself).
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %0 = transform.structured.match in %func_op : (!transform.any_op) -> !transform.any_op
    %p = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{5}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    // Select "test.foo".
    %foo = transform.select "test.foo" in %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %foo, "found foo" : !transform.any_op

    // Select "test.bar".
    %bar = transform.select "test.bar" in %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %bar, "found bar" : !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @select() {
    "test.foo"() : () -> ()
    "test.bar"() : () -> ()
    "test.foo"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match in %0 : (!transform.any_op) -> !transform.any_op
      %2 = transform.num_associations %1 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %2 : !transform.param<i64>
      %3 = transform.select "test.foo" in %1 : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %3, "found foo" : !transform.any_op
      %4 = transform.select "test.bar" in %1 : (!transform.any_op) -> !transform.any_op
      transform.debug.emit_remark_at %4, "found bar" : !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共29行，输出共21行
- transform.named_sequence定义被保留

---

### 15.10.106 case_106

**功能介绍:**

CHECK-NEXT:   memref.store
CHECK-NEXT:   return
Two dead ops, interleaved with a non-dead op.

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
//  CHECK-NEXT:   memref.store
//  CHECK-NEXT:   return
func.func @apply_dce(%f: f32, %m: memref<5xf32>, %idx: index) {
  // Two dead ops, interleaved with a non-dead op.
  %0 = tensor.empty() : tensor<5xf32>
  memref.store %f, %m[%idx] : memref<5xf32>
  %1 = tensor.insert %f into %0[%idx] : tensor<5xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %empty_op = transform.structured.match ops{["tensor.empty"]} in %func_op : (!transform.any_op) -> !transform.any_op
    transform.apply_dce to %func_op : !transform.any_op

    %p = transform.num_associations %empty_op : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{0}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @apply_dce(%arg0: f32, %arg1: memref<5xf32>, %arg2: index) {
    memref.store %arg0, %arg1[%arg2] : memref<5xf32>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["tensor.empty"]} in %0 : (!transform.any_op) -> !transform.any_op
      transform.apply_dce to %0 : !transform.any_op
      %2 = transform.num_associations %1 : (!transform.any_op) -> !transform.param<i64>
      transform.debug.emit_param_as_remark %2 : !transform.param<i64>
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共22行，输出共16行
- transform.named_sequence定义被保留

---

### 15.10.107 case_107

**功能介绍:**

Match `arith.constant`s that are not nested under a `scf.for` and ensure
there are none in the program

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @no_constant_under_loop(%lb: index, %ub: index, %step: index) {
  scf.for %i= %lb to %ub step %step {
    arith.constant 0 : index
  }
  return
}

module @named_inclusion attributes { transform.with_named_sequence } {
// Match `arith.constant`s that are not nested under a `scf.for` and ensure
// there are none in the program

  transform.named_sequence @print(%root: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %root, "matched func" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_constant_not_under_scf_for(%root: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %root ["arith.constant"] : !transform.any_op
    %for = transform.get_parent_op %root { op_name = "scf.for", allow_empty_results }
      : (!transform.any_op) -> (!transform.any_op)
    transform.match.operation_empty %for : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.foreach_match in %arg0
        @match_constant_not_under_scf_for -> @print
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @no_constant_under_loop(%arg0: index, %arg1: index, %arg2: index) {
    scf.for %arg3 = %arg0 to %arg1 step %arg2 {
      %c0 = arith.constant 0 : index
    }
    return
  }
  module @named_inclusion attributes {transform.with_named_sequence} {
    transform.named_sequence @print(%arg0: !transform.any_op {transform.readonly}) {
      transform.debug.emit_remark_at %arg0, "matched func" : !transform.any_op
      transform.yield 
    }
    transform.named_sequence @match_constant_not_under_scf_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["arith.constant"] : !transform.any_op
      %0 = transform.get_parent_op %arg0 {allow_empty_results, op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
      transform.match.operation_empty %0 : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %updated_root = transform.foreach_match in %arg0 
          @match_constant_not_under_scf_for -> @print : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共32行，输出共25行
- transform.named_sequence定义被保留

---

### 15.10.108 case_108

**功能介绍:**

Match `arith.constant`s that are not nested under a `scf.for` and ensure
there are none in the program

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
func.func @no_constant_under_loop(%lb: index, %ub: index, %step: index) {
  // expected-remark @below {{no parent scf.for}}
  arith.constant 0 : index
  return
}

module @named_inclusion attributes { transform.with_named_sequence } {
  // Match `arith.constant`s that are not nested under a `scf.for` and ensure
  // there are none in the program

  transform.named_sequence @print(%root: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %root, "no parent scf.for" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_constant_not_under_scf_for(%root: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %root ["arith.constant"] : !transform.any_op
    %for = transform.get_parent_op %root { op_name = "scf.for", allow_empty_results }
      : (!transform.any_op) -> (!transform.any_op)
    transform.match.operation_empty %for : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.foreach_match in %arg0
        @match_constant_not_under_scf_for -> @print
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
```

**用例输出:**

```mlir
module {
  func.func @no_constant_under_loop(%arg0: index, %arg1: index, %arg2: index) {
    %c0 = arith.constant 0 : index
    return
  }
  module @named_inclusion attributes {transform.with_named_sequence} {
    transform.named_sequence @print(%arg0: !transform.any_op {transform.readonly}) {
      transform.debug.emit_remark_at %arg0, "no parent scf.for" : !transform.any_op
      transform.yield 
    }
    transform.named_sequence @match_constant_not_under_scf_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["arith.constant"] : !transform.any_op
      %0 = transform.get_parent_op %arg0 {allow_empty_results, op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
      transform.match.operation_empty %0 : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %updated_root = transform.foreach_match in %arg0 
          @match_constant_not_under_scf_for -> @print : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共31行，输出共23行
- transform.named_sequence定义被保留

---

### 15.10.109 case_109

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{result #0, associated with 2 payload objects, expected 1}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.merge_handles %arg0, %arg0 : !transform.any_op
    transform.yield %0 : !transform.any_op
  }
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.110 case_110

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{unresolved external symbol @matcher}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
}
```

**用例输出:**

执行成功，无输出。

---

### 15.10.111 case_111

**功能介绍:**

无描述

**核心原理:**

Transform解释器执行变换序列，通过named_sequence定义可重用的变换操作。解释器会解析transform IR并执行相应的变换操作。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics
```

**用例输入:**

```mlir
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{matched}}
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{matched}}
    transform.debug.emit_remark_at %0, "matched" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["transform.debug.emit_remark_at", "transform.collect_matching"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
}
```

**用例输出:**

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %0, "matched" : !transform.any_op
    transform.yield 
  }
  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["transform.debug.emit_remark_at", "transform.collect_matching"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
}


```

**重点说明:**

- 输入共14行，输出共11行
- transform.named_sequence定义被保留

---

# 16. 选择性目标测试

## 16.1 selective-targeting.mlir

### 16.1.1 case_1

**功能介绍:**

This operation is marked for tiling only.
This operation is marked f
This operation is marked for tiling and vectorization.
This operation is marked for vectorization only.
Match matmul operations inside @matmul_tensors with test.attrA set.
TODO: we don't want this, but it is the required terminator for pdl.pattern
Match matmul operations inside @matmul_tensors with test.attrC set.
TODO: we don't want this, but it is the required terminator for pdl.pattern

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file
```

**用例输入:**

```mlir
func.func @matmul_tensors_1(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  // This operation is marked for tiling only.
  %0 = linalg.matmul { test.attrA }
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @matmul_tensors_2(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  // This operation is marked f
  // This operation is marked for tiling and vectorization.
  %0 = linalg.matmul { test.attrA, test.attrC }
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @matmul_tensors_3(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  // This operation is marked for vectorization only.
  %0 = linalg.matmul { test.attrC }
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      // Match matmul operations inside @matmul_tensors with test.attrA set.
      pdl.pattern @pdl_target_attrA : benefit(1) {
        %args = operands
        %results = types
        %attr = attribute
        %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
        // TODO: we don't want this, but it is the required terminator for pdl.pattern
        rewrite %0 with "transform.dialect"
      }

      // Match matmul operations inside @matmul_tensors with test.attrC set.
      pdl.pattern @pdl_target_attrC : benefit(1) {
        %args = operands
        %results = types
        %attr = attribute
        %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrC" = %attr}-> (%results : !pdl.range<type>)
        // TODO: we don't want this, but it is the required terminator for pdl.pattern
        rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @pdl_target_attrA in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.structured.tile_using_for %0 tile_sizes [4, 4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %1 = pdl_match @pdl_target_attrC in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = get_parent_op %1 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.structured.vectorize_children_and_apply_patterns %2 : (!transform.any_op) -> !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_tensors_1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_2 = arith.constant 128 : index
    %c128_3 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c4_4 = arith.constant 4 : index
    %c4_5 = arith.constant 4 : index
    %0 = scf.for %arg3 = %c0 to %c128 step %c4 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
      %1 = scf.for %arg5 = %c0_0 to %c128_2 step %c4_4 iter_args(%arg6 = %arg4) -> (tensor<128x128xf32>) {
        %2 = scf.for %arg7 = %c0_1 to %c128_3 step %c4_5 iter_args(%arg8 = %arg6) -> (tensor<128x128xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %extracted_slice_6 = tensor.extract_slice %arg1[%arg7, %arg5] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %extracted_slice_7 = tensor.extract_slice %arg8[%arg3, %arg5] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %3 = linalg.matmul {test.attrA} ins(%extracted_slice, %extracted_slice_6 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%extracted_slice_7 : tensor<4x4xf32>) -> tensor<4x4xf32>
          %inserted_slice = tensor.insert_slice %3 into %arg8[%arg3, %arg5] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice : tensor<128x128xf32>
        }
        scf.yield %2 : tensor<128x128xf32>
      }
      scf.yield %1 : tensor<128x128xf32>
    }
    return %0 : tensor<128x128xf32>
  }
  func.func @matmul_tensors_2(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %1 = scf.for %arg3 = %c0 to %c128 step %c4 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
      %2 = scf.for %arg5 = %c0 to %c128 step %c4 iter_args(%arg6 = %arg4) -> (tensor<128x128xf32>) {
        %3 = scf.for %arg7 = %c0 to %c128 step %c4 iter_args(%arg8 = %arg6) -> (tensor<128x128xf32>) {
          %4 = vector.transfer_read %arg0[%arg3, %arg7], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<4x4xf32>
          %5 = vector.transfer_read %arg1[%arg7, %arg5], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<4x4xf32>
          %6 = vector.transfer_read %arg8[%arg3, %arg5], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<4x4xf32>
          %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %6 : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
          %8 = vector.transfer_write %7, %arg8[%arg3, %arg5] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<128x128xf32>
          scf.yield %8 : tensor<128x128xf32>
        }
        scf.yield %3 : tensor<128x128xf32>
      }
      scf.yield %2 : tensor<128x128xf32>
    }
    return %1 : tensor<128x128xf32>
  }
  func.func @matmul_tensors_3(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : index
    %0 = ub.poison : f32
    %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %3 = vector.transfer_read %arg2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<128x128xf32>, vector<128x128xf32> into vector<128x128xf32>
    %5 = vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf32>, tensor<128x128xf32>
    return %5 : tensor<128x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @pdl_target_attrA : benefit(1) {
          %0 = operands
          %1 = types
          %2 = attribute
          %3 = operation "linalg.matmul"(%0 : !pdl.range<value>)  {"test.attrA" = %2} -> (%1 : !pdl.range<type>)
          rewrite %3 with "transform.dialect"
        }
        pdl.pattern @pdl_target_attrC : benefit(1) {
          %0 = operands
          %1 = types
          %2 = attribute
          %3 = operation "linalg.matmul"(%0 : !pdl.range<value>)  {"test.attrC" = %2} -> (%1 : !pdl.range<type>)
          rewrite %3 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @pdl_target_attrA in %arg2 : (!transform.any_op) -> !transform.any_op
          %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [4, 4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
          %1 = pdl_match @pdl_target_attrC in %arg2 : (!transform.any_op) -> !transform.any_op
          %2 = get_parent_op %1 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
          %3 = transform.structured.vectorize_children_and_apply_patterns %2 : (!transform.any_op) -> !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共73行，输出共92行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 16.1.2 case_2

**功能介绍:**

TODO: we don't want this, but it is the required terminator for pdl.pattern

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file
```

**用例输入:**

```mlir
func.func @vectorize_one(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @vectorize_none(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @pdl_target : benefit(1) {
        %args = operands
        %results = types
        %attr = attribute
        %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
        // TODO: we don't want this, but it is the required terminator for pdl.pattern
        rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @pdl_target in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
      }
    }
    transform.yield
  }
}
```

**用例输出:**

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @vectorize_one(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : index
    %0 = ub.poison : f32
    %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %3 = vector.transfer_read %arg2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<128x128xf32>, vector<128x128xf32> into vector<128x128xf32>
    %5 = vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf32>, tensor<128x128xf32>
    return %5 : tensor<128x128xf32>
  }
  func.func @vectorize_none(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.with_pdl_patterns %arg0 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        pdl.pattern @pdl_target : benefit(1) {
          %0 = operands
          %1 = types
          %2 = attribute
          %3 = operation "linalg.matmul"(%0 : !pdl.range<value>)  {"test.attrA" = %2} -> (%1 : !pdl.range<type>)
          rewrite %3 with "transform.dialect"
        }
        sequence %arg1 : !transform.any_op failures(propagate) {
        ^bb0(%arg2: !transform.any_op):
          %0 = pdl_match @pdl_target in %arg2 : (!transform.any_op) -> !transform.any_op
          %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
          %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
        }
      }
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共44行，输出共40行
- transform.sequence结构被保留并规范化
- transform.named_sequence定义被保留

---

### 16.1.3 case_3

**功能介绍:**

无描述

**核心原理:**

transform.sequence是Transform方言的核心操作，用于定义一系列变换操作的执行序列。支持失败处理策略(propagate/suppress)。

**执行命令:**

```bash
mlir-opt <input_file> --transform-interpreter --split-input-file
```

**用例输入:**

```mlir
func.func @vectorize_all(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %1 = linalg.matmul ins(%arg0, %0: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.vectorize_children_and_apply_patterns %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

**用例输出:**

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @vectorize_all(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : index
    %0 = ub.poison : f32
    %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %3 = vector.transfer_read %arg2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<128x128xf32>, vector<128x128xf32> into vector<128x128xf32>
    %5 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %6 = vector.transfer_read %arg3[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x128xf32>, vector<128x128xf32>
    %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %4, %6 : vector<128x128xf32>, vector<128x128xf32> into vector<128x128xf32>
    %8 = vector.transfer_write %7, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf32>, tensor<128x128xf32>
    return %8 : tensor<128x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.vectorize_children_and_apply_patterns %arg0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


```

**重点说明:**

- 输入共20行，输出共24行
- transform.named_sequence定义被保留

---

