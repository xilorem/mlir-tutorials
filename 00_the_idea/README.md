# The Idea

Citing the [original work](https://arxiv.org/abs/2002.11054), **MLIR** is a compiler infrastructure centered around reusability and extensibility.

The core idea of MLIR is to define various _Intermediate Representations_ (aka IRs) in order to better target specific goals; To clarify this concept, let's make an example:

Our test subject is Jim. Jim wants to run a Matmul on a GPU. Jim knows a bit of python code, but knows nothing about GPUs, he just wants Matmul go fast.

So Jim writes something  like `C = A @ B`, and hopes the computer will take care of the rest. However, between Jim’s innocent Python line and the actual GPU execution, there is a huge gap. The GPU does not naturally think in terms of “matrix multiplication”. It thinks in terms of threads, memory loads, memory stores, synchronization, registers, blocks, warps, and many other scary things Jim definitely did not sign up for.

Instead of immediately translating Jim’s Matmul into very low-level GPU code, MLIR can represent the same computation through different IRs, each one useful for a different step of the journey. At the beginning, the program can be represented with a high-level IR that still clearly says: “this is a Matmul”. Later, it can be transformed into an IR that exposes loops, so the compiler can tile the computation. Then it can move to an IR that talks about memory buffers, so the compiler can decide where the data should live. Finally, it can be lowered to a GPU-specific IR, where the computation is mapped to GPU blocks and threads.

In other words, MLIR lets the compiler gradually change the way it looks at a program, avoiding inserting low-level details to the IR when they are not needed yet, hugely decreasing complexity.

What if now Jim wants to use a CPU? Well, some passes (like, for example, the looping pass) can be reused, up until the CPU-relevant details need to be added; at that point, we can just "attach" the CPU-specific IR to the lowering pipeline.