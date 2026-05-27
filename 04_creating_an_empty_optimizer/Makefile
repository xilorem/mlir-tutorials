
MLIR_DIR := /home/ivan/repos/llvm-project/build-mlir/lib/cmake/mlir
LLVM_DIR := /home/ivan/repos/llvm-project/build-mlir/lib/cmake/llvm
BUILD_DIR := build
BUILD_TYPE := Debug

.PHONY: all configure build clean

all: configure build

configure:
	cmake -S . -B $(BUILD_DIR) -G Ninja \
	  -DMLIR_DIR=$(MLIR_DIR) \
	  -DLLVM_DIR=$(LLVM_DIR) \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build:
	cmake --build build

clean:
	rm -rf build