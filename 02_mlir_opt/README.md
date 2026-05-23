# mlir-opt

**mlir-opt** is the transformation tool in the upstream MLIR. It allows the user to utilize Passes of the officially included Dialects.

## Using mlir-opt

As our hands on tutorial, we will use mlir-opt on the linalg_on_tensors.mlir file. To use it, just run the command

```bash
mlir-opt linalg_on_tensors.mlir
```

The effect of this is mostly nothing; the mlir file should be printed in the terminal basically unmodified.

If you are too lazy to run the command don't worry, the output is shown here below:

```mlir
module {
  func.func @matmul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%arg2 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
```

When inspecting the file we can notice the Value names aren't the ones from the original file; for example, names like `%A`, `%B`, and `%Cinit` may became `%arg0`, `%arg1`, and `%arg2`.

In MLIR, SSA value names are not semantically important. They are mostly just textual names used to make the IR readable. When `mlir-opt` reads the file, it parses the textual IR into an internal compiler data structure. When it prints the IR back, it is free to choose new names for the values.

The other behaviour of mlir-opt can be observed wehen trying to give a file that has bad syntax of unregistered Dialects. To show what happens, i personally used my amazing imagination to create an operation in a fake dialect called `fake.fake`. The file containing this operation is called fake.mlir.

If we try to use mlir-opt on this file, we will get:

```bash
fake.mlir:8:10: error: Dialect `fake' not found for custom op 'fake.fake' 
    %C = fake.fake
```

Internally, when calling mlir-opt, a check of the registered dialects is performed when parsinng. To allow unregistered dialects, we can just run the mlir-opt command with the --allow-unregistered-dialects flag.

```mlir
module {
  func.func @matmul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "fake.fake"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
```

Note that parsing of unregistered Dialects requires the operations to have standard **assembly formats** (more on this in future turorials), which basically means that MLIR needs to know how to parse them in the standard format operation, namely:

```text
"dialectName.operationName"(inputs): (input types) -> result types
```

If you are curious, using the --help flag shows every flag that mlir-opt allows (but remember curiosity kelled the cat...).