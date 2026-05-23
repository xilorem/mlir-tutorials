#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"


using namespace mlir;

namespace {


struct PrintOpNamesPass
  : public PassWrapper<PrintOpNamesPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpNamesPass)

      StringRef getArgument() const final { return "print-op-names"; }

      StringRef getDescription() const final {
        return "Print all operation names in the module";
      }

      void runOnOperation() override  {
        ModuleOp module = getOperation();

        module.walk([](Operation *op){
            llvm::outs() << "op: " << op->getName() << "\n";
            });
      }
      
  };


struct PrintAttrNamesPass : public PassWrapper<PrintAttrNamesPass, OperationPass<ModuleOp>>{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintAttrNamesPass)

    StringRef getArgument() const final {
      return "print-attr-op-names";
    }

    StringRef getDescription() const final {
      return "Print attributes attached to each operation";
    }

    void runOnOperation() override {
      ModuleOp module = getOperation();

      module.walk([](Operation *op){
          if (op->getAttrs().empty())
          return;

          llvm::outs() << "op: " << op->getName() << "\n";

          for (NamedAttribute attr: op->getAttrs()) {
            llvm::outs() << "  " << attr.getName() << " = ";
            attr.getValue().print(llvm::outs());
            llvm::outs() << "\n";
          }
    });
    }
};


struct RemoveZeroAddPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // x + 0 -> x
    if (matchPattern(rhs, m_Zero())) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    // 0 + x -> x
    if (matchPattern(lhs, m_Zero())) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    return failure();
  }
};

struct RemoveZeroAddPass
    : public PassWrapper<RemoveZeroAddPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveZeroAddPass)

  StringRef getArgument() const final {
    return "remove-zero-add";
  }

  StringRef getDescription() const final {
    return "Remove integer additions by zero";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    patterns.add<RemoveZeroAddPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};





} // namespace




int main(int argc, char **argv) {

  static PassRegistration<PrintOpNamesPass> printOpsPass; // register the pass
  static PassRegistration<PrintAttrNamesPass> printAttrsPass;
  static PassRegistration<RemoveZeroAddPass> removeZeroPass;


  DialectRegistry registry;
  registry.insert<
      arith::ArithDialect,
      func::FuncDialect,
      scf::SCFDialect,
      cf::ControlFlowDialect,
      memref::MemRefDialect,
      tensor::TensorDialect,
      linalg::LinalgDialect,
      bufferization::BufferizationDialect>();
  
  return asMainReturnCode(
      MlirOptMain(argc, argv, "MLIR playground optimizer\n", registry));
}


