# Lowering

In the [previous tutorial](../02_mlir_opt/README.md) we saw how `mlir-opt` can parse and validate MLIR files, but the real power of MLIR is transforming IR from one dialect to another; this process is called **lowering**.

In this tutorial we will lower a simple `linalg.matmul` operation all the way to the LLVM Dialect, which is the last abstraction level before actual machine code.

## The starting point

First, let's get a grasp of where we are starting from:

```mlir
module {
  func.func @matmul(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>, %Cinit: tensor<2x4xf32>) -> tensor<2x4xf32> { 
    %C = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>) outs(%Cinit : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %C : tensor<2x4xf32>
  }
}
```

At this stage we have a tensor dialect (which describes data as abstract SSA values), the linalg dialect (which describes the matmul as a named structured operation), and the func dialect (which defines the function itself).

The **Makefile** in this folder describes the steps of the lowering pipeline; To run the whole pipeline, just use the command:

```bash
make all
```

This will create the following files in this directory:

```text
03_lowering/
    linalg_on_buffers.mlir
    loops.mlir
    host_device_code.mlir
    llvm_dialect.mlir
```

We will now look at them one-by-one and try to understand what's going on.

## Step 1 - from tensors to buffers

Hardware does not think in terms of abstract SSA values. It thinks in terms of memory locations. The first step makes this explicit by converting tensor<...> types into memref<...> types. This is called **bufferization**. 

To bufferize, let's run the following command:

```bash
mlir-opt linalg_on_tensors.mlir \
  -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  -o linalg_on_buffers.mlir
```

The output we obtain is:

```mlir
module {
  func.func @matmul(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) -> memref<2x4xf32> {
    linalg.matmul ins(%arg0, %arg1 : memref<2x3xf32>, memref<3x4xf32>) outs(%arg2 : memref<2x4xf32>)
    return %arg2 : memref<2x4xf32>
  }
}
```

We can see that `tensor<2x3xf32>` became `memref<2x3xf32>`. A memref is (duh) a reference to memory — it carries the same shape information but represents actual storage instead of an SSA value.


## Step 2 - from linalg to loops

A linalg.matmul is a high-level operation that hides the actual loop structure. Most backends don't understand "matmul", they understand loads, stores, and arithmetic. This step expands the matmul into explicit loops.

Let's run:

```bash
mlir-opt linalg_on_buffers.mlir \
  -convert-linalg-to-loops \
  -o loops.mlir
```


And inspect the obtained file.

```mlir
module {
  func.func @matmul(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) -> memref<2x4xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c3 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<2x3xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<3x4xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<2x4xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<2x4xf32>
        }
      }
    }
    return %arg2 : memref<2x4xf32>
  }
}
```

The `linalg` dialect disappeared and, in it's place, we have the `scf` dialect (or Structured Control Flow) and the `arith` dialect.

In this file it's easy to see the actual matmul triple loop>

```mlir
scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c3 step %c1 {
```


## Step 3 - from loops to flat control flow

As said before, the s in `scf` stands for **structured**, which means Regions still exist. Some backends prefere unstructured control flow.

Let's run the following command:

```bash
mlir-opt loops.mlir \
  -convert-scf-to-cf \
  -o host_device_code.mlir
```

Which leads to:

```mlir
module {
  func.func @matmul(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) -> memref<2x4xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
    %1 = arith.cmpi slt, %0, %c2 : index
    cf.cond_br %1, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
    %3 = arith.cmpi slt, %2, %c4 : index
    cf.cond_br %3, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c0 : index)
  ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
    %5 = arith.cmpi slt, %4, %c3 : index
    cf.cond_br %5, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %6 = memref.load %arg0[%0, %4] : memref<2x3xf32>
    %7 = memref.load %arg1[%4, %2] : memref<3x4xf32>
    %8 = memref.load %arg2[%0, %2] : memref<2x4xf32>
    %9 = arith.mulf %6, %7 : f32
    %10 = arith.addf %8, %9 : f32
    memref.store %10, %arg2[%0, %2] : memref<2x4xf32>
    %11 = arith.addi %4, %c1 : index
    cf.br ^bb5(%11 : index)
  ^bb7:  // pred: ^bb5
    %12 = arith.addi %2, %c1 : index
    cf.br ^bb3(%12 : index)
  ^bb8:  // pred: ^bb3
    %13 = arith.addi %0, %c1 : index
    cf.br ^bb1(%13 : index)
  ^bb9:  // pred: ^bb1
    return %arg2 : memref<2x4xf32>
  }
}
```

Each `scf.for` becomes a set of basic blocks connected by `cf.br` (unconditional branch) and `cf.cond_br` (conditional branch). The loop structure is still there, but it's now explicit in the control-flow graph. The `scf` dialect is gone.


## Step 4 - from flat control flow to LLVM

Finally, we lower everything to the LLVM dialect — MLIR's near-1:1 representation of LLVM IR:

```bash
mlir-opt host_device_code.mlir \
  -expand-strided-metadata \
  -finalize-memref-to-llvm \
  -convert-arith-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o llvm_dialect.mlir
```

```mlir
module {
  llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.br ^bb1(%0, %1 : i64, i64)
  ^bb1(%arg3: i64, %arg4: i64):
    ...
  }
}
```

Every dialect has been lowered to llvm:
- memref → !llvm.ptr
- arith → llvm.add, llvm.mul, etc.
- cf → llvm.br, llvm.cond_br
- func → llvm.func

At this point you could run mlir-translate --mlir-to-llvmir llvm_dialect.mlir to get proper LLVM IR, and then feed it to llc to generate machine code.