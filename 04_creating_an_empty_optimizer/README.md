# Creating an empty optimizer

So, our first real code writing experience will be hand-writing an optimizer tool like mlir-opt that takes an mlir module and returns it. Cool right?

To accomplish our goal, we are gonna create a file named `play-opt.cpp`. If you don't wanna write along, the full example is already written in the current folder.

Now, first of all we are gonna write the main entry point of our program.


```cpp
using namespace mlir;

int main(int argc, char **argv) {

}
```

This main function should return an `MlirOptMain` function, which is the template mlir function for a custom optimize; this function takes a `registry` as the last argument: this is a register of all the Dialects that our optimizer will recognize. If a Dialect that is not registered is seen in the module that we are trying to optimize, the optimizer will return an error (can be handled, we will talk about this in future tutorials).


```cpp
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


using namespace mlir;

int main(int argc, char **argv) {
    DialectRegistry registry;

    return asMainReturnCode(MlirOptMain(argc, argv, "opt description \n", registry))
}
```

Dialects can be added one by one. We are lazy, so we are gonna use the `RegisterAllDialects` function, that appends all upstream dialects to the registry.


```cpp
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"


using namespace mlir;

int main(int argc, char **argv) {


    DialectRegistry registry;
    registerAllDialects(registry);
  
    return asMainReturnCode(MlirOptMain(argc, argv, "MLIR playground optimizer\n", registry));
}
```

Next step is creating a CMakeLists.txt in the same directory.

```cmake
cmake_minimum_required(VERSION 3.20)

project(MLIRPlayground LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(play-opt play-opt.cpp)

llvm_update_compile_flags(play-opt)

target_link_libraries(play-opt
  PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
)
```

This is pretty straighforward, only pay attention to the link libraries that we are adding and to the executable and compile flags name.

Now we are ready to build our optimizer! To configure cmake use:

```bash
cmake -S . -B $(BUILD_DIR) -G Ninja \
    -DMLIR_DIR=$(MLIR_DIR) \
    -DLLVM_DIR=$(LLVM_DIR) \
    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
```

where:

- BUILD_DIR is your preferred build directory, build is fine
- MLIR_DIR and LLVM_DIR point to your build-mlir/cmake libraries
- BUILD_TYPE can be selected as Debug

Then you can do:

```bash
$BUILD_DIR/play-opt linalg_on_tensors.mlir
```