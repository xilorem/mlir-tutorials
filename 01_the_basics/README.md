# The basics

## Operations, Values, Regions and Blocks

At a high level an MLIR program is only composed of four main components:

- **Operations**
- **Values**
- **Regions**
- **Blocks**

Operations and Values can be seen respectively as nodes and edges of a graph. 

Blocks are linear subgraphs; they represent a straight-line sequence of operations, while regions are scoped areas owned by operations. 

Now, let's look at some MLIR code to better understand how these component work with each other.

```mlir
func.func @example(%x: i32, %cond: i1) -> i32 {
  %result = scf.if %cond -> (i32) {
    %c1 = arith.constant 1 : i32
    %a = arith.addi %x, %c1 : i32
    scf.yield %a : i32
  } else {
    %c2 = arith.constant 2 : i32
    %b = arith.addi %x, %c2 : i32
    scf.yield %b : i32
  }

  return %result : i32
}
```

The first line is a `func` Operation; it's purpose is just to define a function in MLIR, basically the same idea of `def func(){}` in C. 

`%result` is a Value. In MLIR Values are SSA, or _Single Static Assignment_, which means that, once a value with a certain name is assigned to something, it cannot be assigned to anything else.

The curly brackets define Regions. As we previously said, regions are _scoped_, which means Values defined inside of them cannot normally be used outside. This is why `yield` operations exist: they allow assignment of SSA Values from the inside of a Region to the outside. For example, the scoped `%a` Value (inside the first region of `scf.if`) gets assigned to the `%result` Value (outside of scope) through the `scf.yield` operation.

The first and second Regions of `scf.if` both have a single Block inside of them; for example, The first Region's block is:

```mlir
%c1 = arith.constant 1 : i32
%a = arith.addi %x, %c1 : i32
scf.yield %a : i32
```

Don't be fooled: a Region can have more than one block; When doing _Lowering_ (more on that later) many IRs loose structure and have conditional branches inside a single Region, using Blocks.

## Dialects

A **Dialect** is a domain-specific extension of MLIR used to represent operations at a specific abstraction level. Importantly, many Dialects can coexist in the same MLIR program, which means that MLIR does not force the whole program to be represented in only one way at a time.

Dialects define a unique namespace for identification. In the example in the last section we already saw the `func`, the `scf`, and the `arith` Dialects. An operation defined in a Dialect will be used in a program following the `dialectNamespace.operationName` structure.


## Passes

Passes in MLIR are transformations or analyses over MLIR IR. Based of the input and outputs, they take different names

- **Translation**: a Pass from a non-MLIR language (for example, C) to a set of MLIR Dialects
- **Export**: a Pass from a set of MLIR Dialects to a non-MLIR language
- **Conversion**: a Pass from a set of MLIR Dialects to another set of MLIR Dialects (for example, from `scf` to `cf`)
- **Optimization**: a Pass that doesn't change the set of MLIR Dialects used, but rewrites the IR usually to improve some objective