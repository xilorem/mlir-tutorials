
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"

#include "llvm/Support/raw_ostream.h"


using namespace mlir;

int main(int argc, char **argv) {


    DialectRegistry registry;
    registerAllDialects(registry);
  
    return asMainReturnCode(
        MlirOptMain(argc, argv, "MLIR playground optimizer\n", registry));
}